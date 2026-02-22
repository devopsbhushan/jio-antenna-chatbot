# build_index_lambda.py
# Three-pass streaming approach — never holds all data in RAM at once
# Pass 1: count word frequencies (streaming, low RAM)
# Pass 2: embed + add to FAISS (streaming, low RAM)
# Peak RAM: ~300MB

import boto3
import json
import faiss
import numpy as np
import pickle
import gc
import re
import math
import hashlib
from collections import Counter

BUCKET       = "chatbot-input-database"
BASE_PREFIX  = "processed/"
INDEX_KEY    = "rag-index/faiss.index"
METADATA_KEY = "rag-index/metadata.pkl"
IDF_KEY      = "rag-model/idf.pkl"

DIM    = 384
BATCH  = 200   # rows embedded at once — keep low

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


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def hash_token(token):
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM


def text_to_vec(text, idf):
    tokens = tokenize(text)
    vec    = np.zeros(DIM, dtype=np.float32)
    for t in tokens:
        vec[hash_token(t)] += idf.get(t, 1.0)
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec


def iter_all_rows(idx_meta):
    """Generator — yields (row_dict) one at a time. Never holds all rows."""
    for state, meta in idx_meta["states"].items():
        for i in range(1, meta["chunks"] + 1):
            rows = load_json(f"{BASE_PREFIX}{state}/chunk_{i:04d}.json")
            for row in rows:
                row["State"] = state
                yield row


def lambda_handler(event, context):
    idx_meta = load_json(f"{BASE_PREFIX}index.json")

    # ── Pass 1: Count doc frequencies (streaming — no list accumulation) ─────
    print("Pass 1: counting word frequencies...")
    doc_freq  = Counter()
    total_rows = 0

    for row in iter_all_rows(idx_meta):
        words = set(tokenize(row_to_text(row)))
        doc_freq.update(words)
        total_rows += 1
        if total_rows % 10000 == 0:
            print(f"  Counted {total_rows} rows...")

    print(f"Total rows: {total_rows}, unique words: {len(doc_freq)}")

    # Build IDF dict — keep only top 50K words to save RAM
    TOP_WORDS = 50000
    common    = doc_freq.most_common(TOP_WORDS)
    idf       = {
        word: math.log((total_rows + 1) / (cnt + 1)) + 1.0
        for word, cnt in common
    }
    del doc_freq, common
    gc.collect()
    print(f"IDF built: {len(idf)} terms")

    # Save IDF to S3
    with open("/tmp/idf.pkl", "wb") as f:
        pickle.dump(idf, f)
    s3.upload_file("/tmp/idf.pkl", BUCKET, IDF_KEY)
    print("IDF saved to S3")

    # ── Pass 2: Embed + build FAISS (streaming — batch by batch) ─────────────
    print("Pass 2: embedding and indexing...")

    nlist     = max(10, min(100, total_rows // 100))
    quantizer = faiss.IndexFlatL2(DIM)
    faiss_idx = faiss.IndexIVFFlat(quantizer, DIM, nlist)
    trained   = False

    batch_texts = []
    batch_meta  = []
    all_meta    = []
    indexed     = 0

    for row in iter_all_rows(idx_meta):
        text = row_to_text(row)
        meta = {c: row.get(c, "") for c in COLUMNS}
        batch_texts.append(text)
        batch_meta.append(meta)

        if len(batch_texts) >= BATCH:
            # Embed batch
            vecs = np.array(
                [text_to_vec(t, idf) for t in batch_texts],
                dtype=np.float32
            )

            # Train on first batch
            if not trained:
                print(f"Training IVF on {len(vecs)} vectors...")
                faiss_idx.train(vecs)
                trained = True

            faiss_idx.add(vecs)
            all_meta.extend(batch_meta)
            indexed += len(batch_texts)

            # Clear batch immediately
            batch_texts = []
            batch_meta  = []
            del vecs
            gc.collect()

            if indexed % 10000 == 0:
                print(f"  Indexed {indexed}/{total_rows}")

    # Process remaining rows
    if batch_texts:
        vecs = np.array(
            [text_to_vec(t, idf) for t in batch_texts],
            dtype=np.float32
        )
        if not trained:
            faiss_idx.train(vecs)
        faiss_idx.add(vecs)
        all_meta.extend(batch_meta)
        indexed += len(batch_texts)
        del vecs
        gc.collect()

    faiss_idx.nprobe = 10
    print(f"Indexing complete: {indexed} rows")

    # ── Save FAISS index ──────────────────────────────────────────────────────
    print("Saving FAISS index...")
    faiss.write_index(faiss_idx, "/tmp/faiss.index")
    s3.upload_file("/tmp/faiss.index", BUCKET, INDEX_KEY)

    # ── Save metadata in chunks to avoid RAM spike ────────────────────────────
    print("Saving metadata...")
    with open("/tmp/metadata.pkl", "wb") as f:
        pickle.dump(all_meta, f)
    s3.upload_file("/tmp/metadata.pkl", BUCKET, METADATA_KEY)

    print("All done!")
    return {
        "statusCode": 200,
        "body": f"Indexed {indexed} rows successfully"
    }
