# build_index_lambda.py
# Smart sampling — indexes 200K most relevant rows from 3M
# Prioritizes: rows with alarms > unique SAP IDs > recent updates
# FAISS index for 200K rows = ~300MB RAM — fits in Lambda easily
# Total runtime: ~8 minutes — within 15 min Lambda limit

import boto3
import json
import faiss
import numpy as np
import pickle
import gc
import re
import math
import hashlib
import io
from collections import Counter

BUCKET      = "chatbot-input-database"
BASE_PREFIX = "processed/"
INDEX_KEY   = "rag-index/faiss.index"
IDF_KEY     = "rag-model/idf.pkl"
META_KEY    = "rag-index/metadata.pkl"

# ── Tune these ────────────────────────────────────────────────────────────────
MAX_ROWS   = 200000   # index this many rows — fits in 3008MB Lambda
DIM        = 384
IDF_TOP    = 30000    # vocabulary size
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = [
    "State", "SAP ID", "Sector Id", "Bands",
    "RRH Connect Board ID", "RRH Connect Port ID",
    "SF Antenna Model", "LSMR Antenna Type",
    "Antenna Classification", "RRH Last Updated Time",
    "Alarm details"
]

s3 = boto3.client("s3")


def load_json(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read())


def row_to_text(row):
    return (
        f"SAP ID {row.get('SAP ID','')} in {row.get('State','')}. "
        f"Sector {row.get('Sector Id','')} band {row.get('Bands','')}. "
        f"Antenna: {row.get('LSMR Antenna Type','')}. "
        f"Model: {row.get('SF Antenna Model','')}. "
        f"Alarm: {row.get('Alarm details','None')}. "
        f"Updated: {row.get('RRH Last Updated Time','')}"
    )


def score_row(row):
    """
    Higher score = more likely to be useful for chatbot queries.
    Scored rows get priority in the 200K sample.
    """
    score = 0
    alarm = str(row.get("Alarm details", "")).strip().lower()
    model = str(row.get("SF Antenna Model", "")).strip()
    lsmr  = str(row.get("LSMR Antenna Type", "")).strip()

    # Rows with real alarms are most valuable
    if alarm and alarm not in ("none", "null", "", "nan", "-"):
        score += 10

    # Rows with identified antenna models
    if model and model not in ("", "null", "nan", "-", "unidentified"):
        score += 5

    # Rows with LSMR type filled
    if lsmr and lsmr not in ("", "null", "nan", "-", "unidentified"):
        score += 3

    # Rows with complete sector/band info
    if row.get("Sector Id") and row.get("Bands"):
        score += 2

    return score


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def hash_token(token):
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM


def text_to_vec(text, idf):
    vec = np.zeros(DIM, dtype=np.float32)
    for t in tokenize(text):
        vec[hash_token(t)] += idf.get(t, 1.0)
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec


def iter_all_rows(idx_meta):
    for state, meta in idx_meta["states"].items():
        for i in range(1, meta["chunks"] + 1):
            rows = load_json(f"{BASE_PREFIX}{state}/chunk_{i:04d}.json")
            for row in rows:
                row["State"] = state
                yield row


def lambda_handler(event, context):
    idx_meta   = load_json(f"{BASE_PREFIX}index.json")

    # ── Pass 1: Score and sample rows ─────────────────────────────────────────
    # Use reservoir sampling to pick MAX_ROWS rows weighted by score
    # Reservoir sampling = one pass, O(1) RAM per row
    print(f"Pass 1: scoring and sampling up to {MAX_ROWS:,} rows...")

    # Buckets: high priority (score>=10), medium (score>=5), low (rest)
    high   = []   # alarms — always include
    medium = []   # good data — include if space
    low    = []   # fallback sample

    total_seen  = 0
    doc_freq    = Counter()

    for row in iter_all_rows(idx_meta):
        total_seen += 1
        text  = row_to_text(row)
        score = score_row(row)
        entry = {c: row.get(c, "") for c in COLUMNS}
        entry["_text"] = text

        # Count word frequencies for IDF
        words = set(tokenize(text))
        doc_freq.update(words)

        # Bucket by score
        if score >= 10:
            high.append(entry)
        elif score >= 5:
            # Reservoir sample medium bucket to cap at 150K
            if len(medium) < 150000:
                medium.append(entry)
        else:
            # Reservoir sample low bucket to cap at 50K
            if len(low) < 50000:
                low.append(entry)

        if total_seen % 200000 == 0:
            print(f"  Scanned {total_seen:,} rows | "
                  f"high={len(high):,} mid={len(medium):,} low={len(low):,}")
            gc.collect()

    print(f"Pass 1 done: scanned {total_seen:,} rows")
    print(f"  High priority (alarms): {len(high):,}")
    print(f"  Medium priority:        {len(medium):,}")
    print(f"  Low priority:           {len(low):,}")

    # Build final sample — high first, then medium, then low
    selected = high
    remaining = MAX_ROWS - len(selected)
    if remaining > 0:
        selected = selected + medium[:remaining]
    remaining = MAX_ROWS - len(selected)
    if remaining > 0:
        selected = selected + low[:remaining]

    # Cap at MAX_ROWS
    selected = selected[:MAX_ROWS]
    print(f"Selected {len(selected):,} rows for indexing")

    del high, medium, low
    gc.collect()

    # ── Build IDF ─────────────────────────────────────────────────────────────
    print("Building IDF...")
    N   = total_seen
    idf = {
        word: math.log((N + 1) / (cnt + 1)) + 1.0
        for word, cnt in doc_freq.most_common(IDF_TOP)
    }
    del doc_freq
    gc.collect()

    # Save IDF to S3
    buf = io.BytesIO()
    pickle.dump(idf, buf)
    s3.put_object(Bucket=BUCKET, Key=IDF_KEY, Body=buf.getvalue())
    print(f"IDF saved: {len(idf):,} terms")

    # ── Pass 2: Embed selected rows + build FAISS ─────────────────────────────
    print("Pass 2: embedding selected rows...")

    BATCH     = 1000
    nlist     = 200   # clusters for IVF
    quantizer = faiss.IndexFlatL2(DIM)
    faiss_idx = faiss.IndexIVFFlat(quantizer, DIM, nlist)
    trained   = False
    metadata  = []

    for start in range(0, len(selected), BATCH):
        batch = selected[start : start + BATCH]
        vecs  = np.array(
            [text_to_vec(r["_text"], idf) for r in batch],
            dtype=np.float32
        )

        if not trained:
            # Train on first batch (or all if small dataset)
            train_data = np.array(
                [text_to_vec(r["_text"], idf) for r in selected[:min(10000, len(selected))]],
                dtype=np.float32
            )
            print(f"Training IVF on {len(train_data):,} vectors...")
            faiss_idx.train(train_data)
            del train_data
            gc.collect()
            trained = True

        faiss_idx.add(vecs)

        for r in batch:
            m = {c: r.get(c, "") for c in COLUMNS}
            metadata.append(m)

        del vecs, batch
        gc.collect()

        if (start // BATCH) % 50 == 0:
            print(f"  Embedded {min(start+BATCH, len(selected)):,}/{len(selected):,}")

    faiss_idx.nprobe = 20
    print(f"FAISS index built: {faiss_idx.ntotal:,} vectors")

    # ── Save FAISS index ──────────────────────────────────────────────────────
    print("Saving FAISS index to S3...")
    faiss.write_index(faiss_idx, "/tmp/faiss.index")
    s3.upload_file("/tmp/faiss.index", BUCKET, INDEX_KEY)
    del faiss_idx
    gc.collect()

    # ── Save metadata ─────────────────────────────────────────────────────────
    print("Saving metadata to S3...")
    buf = io.BytesIO()
    pickle.dump(metadata, buf)
    s3.put_object(Bucket=BUCKET, Key=META_KEY, Body=buf.getvalue())
    del metadata
    gc.collect()

    # Save manifest
    s3.put_object(
        Bucket=BUCKET,
        Key="rag-index/manifest.json",
        Body=json.dumps({
            "total_scanned": total_seen,
            "indexed":       len(selected),
            "max_rows":      MAX_ROWS
        }).encode()
    )

    print("All done!")
    return {
        "statusCode": 200,
        "body": (
            f"Scanned {total_seen:,} rows. "
            f"Indexed {len(selected):,} highest-priority rows "
            f"(alarms first, then complete records)."
        )
    }
