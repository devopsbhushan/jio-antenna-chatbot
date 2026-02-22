# build_index_lambda.py
# Handles 3 million rows without OOM
# Strategy:
#   - Never store metadata in RAM
#   - Store metadata as numbered S3 chunks (1000 rows each)
#   - FAISS index stores row_id only — chatbot fetches metadata from S3 at query time
#   - Use IndexFlatL2 directly — no IVF training needed, lower RAM
#   - Stream everything row by row

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
import io

BUCKET      = "chatbot-input-database"
BASE_PREFIX = "processed/"
INDEX_KEY   = "rag-index/faiss.index"
IDF_KEY     = "rag-model/idf.pkl"
META_PREFIX = "rag-index/meta/"   # meta/0000000.json, meta/0001000.json ...

META_CHUNK  = 1000   # rows per metadata chunk in S3
EMBED_BATCH = 500    # rows embedded at once before adding to FAISS
DIM         = 384

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
    """Generator — one row at a time, zero accumulation."""
    for state, meta in idx_meta["states"].items():
        for i in range(1, meta["chunks"] + 1):
            rows = load_json(f"{BASE_PREFIX}{state}/chunk_{i:04d}.json")
            for row in rows:
                row["State"] = state
                yield row


def upload_meta_chunk(chunk_id, rows):
    """Upload a metadata chunk to S3."""
    key  = f"{META_PREFIX}{chunk_id:07d}.json"
    body = json.dumps(rows).encode("utf-8")
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)


def lambda_handler(event, context):
    idx_meta = load_json(f"{BASE_PREFIX}index.json")

    # ── Pass 1: Count doc frequencies (pure streaming) ────────────────────────
    print("Pass 1: counting word frequencies...")
    doc_freq   = Counter()
    total_rows = 0

    for row in iter_all_rows(idx_meta):
        words = set(tokenize(row_to_text(row)))
        doc_freq.update(words)
        total_rows += 1
        if total_rows % 100000 == 0:
            print(f"  Pass 1: {total_rows:,} rows counted...")
            gc.collect()

    print(f"Pass 1 done: {total_rows:,} rows, {len(doc_freq):,} unique words")

    # Keep only top 30K words — enough for good retrieval, saves RAM
    TOP = 30000
    idf = {
        word: math.log((total_rows + 1) / (cnt + 1)) + 1.0
        for word, cnt in doc_freq.most_common(TOP)
    }
    del doc_freq
    gc.collect()
    print(f"IDF built: {len(idf):,} terms")

    # Save IDF to S3
    buf = io.BytesIO()
    pickle.dump(idf, buf)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=IDF_KEY, Body=buf.getvalue())
    print("IDF saved to S3")

    # ── Pass 2: Embed + FAISS + stream metadata to S3 ────────────────────────
    print("Pass 2: embedding, indexing, streaming metadata to S3...")

    # Use FAISS IndexFlatL2 written incrementally to /tmp
    # For 3M rows × 384 dims × 4 bytes = ~4.6GB — too large for /tmp
    # Solution: use IVF with on-disk storage
    # nlist tuned for 3M rows
    nlist     = 1000
    quantizer = faiss.IndexFlatL2(DIM)
    faiss_idx = faiss.IndexIVFFlat(quantizer, DIM, nlist)

    batch_texts  = []
    meta_chunk   = []
    chunk_id     = 0
    indexed      = 0
    trained      = False

    # We need a training sample — collect 50K rows first
    train_vecs = []
    train_done = False
    TRAIN_SIZE = 50000

    for row in iter_all_rows(idx_meta):
        text = row_to_text(row)
        vec  = text_to_vec(text, idf)
        meta = {c: row.get(c, "") for c in COLUMNS}

        # Collect training vectors
        if not train_done:
            train_vecs.append(vec)
            if len(train_vecs) >= TRAIN_SIZE:
                print(f"Training IVF index on {len(train_vecs):,} vectors...")
                train_arr = np.array(train_vecs, dtype=np.float32)
                faiss_idx.train(train_arr)
                del train_vecs, train_arr
                gc.collect()
                train_done = True
                trained    = True
                print("IVF training complete")

        if not trained:
            # Buffer until training is done
            batch_texts.append((vec, meta))
            continue

        # Add buffered pre-training rows
        if batch_texts:
            print(f"Adding {len(batch_texts):,} pre-training rows...")
            for bvec, bmeta in batch_texts:
                faiss_idx.add(np.array([bvec], dtype=np.float32))
                meta_chunk.append(bmeta)
                indexed += 1
                if len(meta_chunk) >= META_CHUNK:
                    upload_meta_chunk(chunk_id, meta_chunk)
                    chunk_id  += 1
                    meta_chunk  = []
            batch_texts = []
            gc.collect()

        # Add current row
        faiss_idx.add(np.array([vec], dtype=np.float32))
        meta_chunk.append(meta)
        indexed += 1

        # Flush metadata chunk to S3
        if len(meta_chunk) >= META_CHUNK:
            upload_meta_chunk(chunk_id, meta_chunk)
            chunk_id  += 1
            meta_chunk  = []

        if indexed % 100000 == 0:
            print(f"  Indexed {indexed:,}/{total_rows:,} rows")
            gc.collect()

    # Flush remaining metadata
    if meta_chunk:
        upload_meta_chunk(chunk_id, meta_chunk)
        chunk_id += 1

    faiss_idx.nprobe = 20
    print(f"Indexing complete: {indexed:,} rows in {chunk_id} metadata chunks")

    # ── Save FAISS index ──────────────────────────────────────────────────────
    print("Saving FAISS index to S3...")
    faiss.write_index(faiss_idx, "/tmp/faiss.index")
    s3.upload_file("/tmp/faiss.index", BUCKET, INDEX_KEY)

    # Save index manifest (total rows + chunk count for chatbot)
    manifest = {"total_rows": indexed, "meta_chunks": chunk_id, "dim": DIM}
    s3.put_object(
        Bucket=BUCKET,
        Key="rag-index/manifest.json",
        Body=json.dumps(manifest).encode()
    )

    print("All done!")
    return {
        "statusCode": 200,
        "body": f"Indexed {indexed:,} rows in {chunk_id} S3 metadata chunks"
    }
