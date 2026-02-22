# build_index_lambda.py
# Reads all S3 antenna data, creates FAISS index using lightweight ONNX embeddings

import boto3
import json
import faiss
import numpy as np
import pickle
import embedder   # <-- our lightweight ONNX embedder, no torch/CUDA needed

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
    texts    = []
    metadata = []

    for state, meta in idx_meta["states"].items():
        for i in range(1, meta["chunks"] + 1):
            rows = load_json(f"{BASE_PREFIX}{state}/chunk_{i:04d}.json")
            for row in rows:
                row["State"] = state
                texts.append(row_to_text(row))
                metadata.append({c: row.get(c, "") for c in COLUMNS})

    print(f"Indexing {len(texts)} rows...")

    # Embed using lightweight ONNX MiniLM (no torch/CUDA)
    embeddings = embedder.encode(texts, batch_size=64)   # shape: (N, 384)

    # Build FAISS L2 index
    faiss_idx = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_idx.add(embeddings)

    # Save FAISS index to S3
    faiss.write_index(faiss_idx, "/tmp/faiss.index")
    s3.upload_file("/tmp/faiss.index", BUCKET, INDEX_KEY)

    # Save metadata to S3
    with open("/tmp/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    s3.upload_file("/tmp/metadata.pkl", BUCKET, METADATA_KEY)

    return {
        "statusCode": 200,
        "body": f"Indexed {len(texts)} rows successfully"
    }
