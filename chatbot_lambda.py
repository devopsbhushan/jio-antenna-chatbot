# chatbot_lambda.py
# Fetches metadata from S3 chunks at query time — no metadata in RAM
# Handles 3M row dataset within 1024MB Lambda memory

import boto3
import json
import faiss
import numpy as np
import pickle
import urllib.request
import os
import re
import math
import hashlib
import io

BUCKET      = os.environ["BUCKET"]
INDEX_KEY   = "rag-index/faiss.index"
IDF_KEY     = "rag-model/idf.pkl"
META_PREFIX = "rag-index/meta/"
META_CHUNK  = 1000
DIM         = 384

GROQ_KEY = os.environ["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

s3 = boto3.client("s3")

# Module-level cache
_faiss_index = None
_idf         = None

CORS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Content-Type":                 "application/json"
}


# ── Embedding (same logic as index builder) ───────────────────────────────────
def _load_idf():
    global _idf
    if _idf is not None:
        return
    print("Loading IDF from S3...")
    obj  = s3.get_object(Bucket=BUCKET, Key=IDF_KEY)
    _idf = pickle.loads(obj["Body"].read())
    print(f"IDF loaded: {len(_idf):,} terms")


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def hash_token(token):
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM


def encode(text):
    _load_idf()
    vec = np.zeros(DIM, dtype=np.float32)
    for t in tokenize(text):
        vec[hash_token(t)] += _idf.get(t, 1.0)
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec.reshape(1, -1)


# ── FAISS index ───────────────────────────────────────────────────────────────
def load_index():
    global _faiss_index
    if _faiss_index is None:
        print("Loading FAISS index from S3...")
        s3.download_file(BUCKET, INDEX_KEY, "/tmp/faiss.index")
        _faiss_index = faiss.read_index("/tmp/faiss.index")
        if hasattr(_faiss_index, "nprobe"):
            _faiss_index.nprobe = 20
        print(f"FAISS index loaded: {_faiss_index.ntotal:,} vectors")
    return _faiss_index


# ── Metadata fetch from S3 ────────────────────────────────────────────────────
def fetch_metadata(row_ids):
    """Fetch metadata for specific row IDs from S3 chunks."""
    # Group row IDs by which S3 chunk they belong to
    chunks_needed = {}
    for row_id in row_ids:
        chunk_id = (row_id // META_CHUNK) * META_CHUNK
        if chunk_id not in chunks_needed:
            chunks_needed[chunk_id] = []
        chunks_needed[chunk_id].append(row_id % META_CHUNK)

    results = {}
    for chunk_start, offsets in chunks_needed.items():
        key = f"{META_PREFIX}{chunk_start:07d}.json"
        try:
            obj   = s3.get_object(Bucket=BUCKET, Key=key)
            chunk = json.loads(obj["Body"].read())
            for offset in offsets:
                if offset < len(chunk):
                    results[chunk_start + offset] = chunk[offset]
        except Exception as e:
            print(f"Could not fetch chunk {key}: {e}")

    return [results.get(rid, {}) for rid in row_ids]


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query, top_k=20):
    idx    = load_index()
    vec    = encode(query)
    D, I   = idx.search(vec, top_k)
    valid  = [int(i) for i in I[0] if i != -1]
    return fetch_metadata(valid)


# ── Groq LLM ─────────────────────────────────────────────────────────────────
def ask_groq(question, docs, history):
    context = "\n".join([
        f"SAP:{r.get('SAP ID','')} | State:{r.get('State','')} |"
        f" Sector:{r.get('Sector Id','')} | Band:{r.get('Bands','')} |"
        f" Antenna:{r.get('LSMR Antenna Type','')} |"
        f" Alarm:{r.get('Alarm details','None')} |"
        f" Updated:{r.get('RRH Last Updated Time','')}"
        for r in docs if r
    ])

    system = (
        "You are a Jio telecom antenna inventory assistant. "
        "Answer ONLY using the retrieved data. "
        "Be specific — mention SAP IDs, sectors, bands. "
        "If data is insufficient, say so."
    )

    messages = history + [{
        "role":    "user",
        "content": f"Retrieved antenna data:\n{context}\n\nQuestion: {question}"
    }]

    payload = json.dumps({
        "model":       "llama3-8b-8192",
        "messages":    [{"role": "system", "content": system}] + messages,
        "max_tokens":  1024,
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_URL, data=payload,
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type":  "application/json"
        }
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


# ── Handler ───────────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS, "body": ""}

    try:
        body = json.loads(event.get("body") or "{}")
    except Exception:
        body = {}

    user_msg = body.get("message", "").strip()
    history  = body.get("history", [])[-10:]

    if user_msg == "ping":
        load_index()
        _load_idf()
        return {"statusCode": 200, "headers": CORS,
                "body": json.dumps({"reply": "warm"})}

    if not user_msg:
        return {"statusCode": 400, "headers": CORS,
                "body": json.dumps({"error": "No message"})}

    docs   = retrieve(user_msg, top_k=20)
    answer = ask_groq(user_msg, docs, history)

    updated_history = (history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": answer}
    ])[-10:]

    return {
        "statusCode": 200,
        "headers":    CORS,
        "body": json.dumps({
            "reply":     answer,
            "history":   updated_history,
            "retrieved": len(docs)
        })
    }
