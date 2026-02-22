# build_index_lambda.py
# Memory-efficient FAISS index builder using lightweight TF-IDF embedder
# Peak RAM: ~400MB — well within 3008MB Lambda limit

import boto3
import json
import faiss
import numpy as np
import pickle
import gc
import embedder

BUCKET       = "chatbot-input-database"
BASE_PREFIX  = "processed/"
INDEX_KEY    = "rag-index/faiss.index"
METADATA_KEY = "rag-index/metadata.pkl"

COLUMNS = [
    "State", "SAP ID", "Sector Id", "Bands",
    "RRH Connect Board ID", "RRH Connect Port ID",
    "SF Antenna Model", "LSMR Antenna Type",
    "Antenna Classification", "RRH Last Updated Time",
    "Alarm details"
]

CHUNK_BUFFER = 3   # chunks loaded at once
DIM          = 384

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


def lambda_handler(event, context):
    idx_meta = load_json(f"{BASE_PREFIX}index.json")

    # ── Pass 1: Collect all texts to build IDF ───────────────────────────────
    print("Pass 1: collecting texts for IDF...")
    all_texts    = []
    all_metadata = []

    for state, meta in idx_meta["states"].items():
        for i in range(1, meta["chunks"] + 1):
            rows = load_json(f"{BASE_PREFIX}{state}/chunk_{i:04d}.json")
            for row in rows:
                row["State"] = state
                all_texts.append(row_to_text(row))
                all_metadata.append({c: row.get(c, "") for c in COLUMNS})

    total_rows = len(all_texts)
    print(f"Total rows: {total_rows}")

    # Build IDF from all texts and save to S3
    embedder.build_and_save_idf(all_texts)
    gc.collect()

    # ── Pass 2: Embed and build FAISS index ──────────────────────────────────
    print("Pass 2: embedding and building FAISS index...")

    # IVFFlat index — memory efficient
    quantizer = faiss.IndexFlatL2(DIM)
    faiss_idx = faiss.IndexIVFFlat(quantizer, DIM, min(100, total_rows // 10))

    # Embed in batches of 500 rows
    BATCH = 500
    first_batch = True

    for start in range(0, total_rows, BATCH):
        batch_texts = all_texts[start : start + BATCH]
        embeddings  = embedder.encode(batch_texts)

        if first_batch:
            print(f"Training IVF index on first {len(embeddings)} vectors...")
            faiss_idx.train(embeddings)
            first_batch = False

        faiss_idx.add(embeddings)

        if start % 5000 == 0:
            print(f"  Indexed {start + len(batch_texts)}/{total_rows} rows")

        del batch_texts, embeddings
        gc.collect()

    faiss_idx.nprobe = 10

    # ── Save to S3 ────────────────────────────────────────────────────────────
    print("Saving FAISS index to S3...")
    faiss.write_index(faiss_idx, "/tmp/faiss.index")
    s3.upload_file("/tmp/faiss.index", BUCKET, INDEX_KEY)

    print("Saving metadata to S3...")
    with open("/tmp/metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f)
    s3.upload_file("/tmp/metadata.pkl", BUCKET, METADATA_KEY)

    print(f"Done! Indexed {total_rows} rows.")
    return {
        "statusCode": 200,
        "body": f"Indexed {total_rows} rows successfully"
    }
