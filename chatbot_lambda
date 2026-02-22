# chatbot_lambda.py
# RAG chatbot handler using ONNX MiniLM embeddings + FAISS + Groq Llama 3

import boto3
import json
import faiss
import numpy as np
import pickle
import urllib.request
import os
import embedder   # <-- our lightweight ONNX embedder

BUCKET       = os.environ["BUCKET"]
INDEX_KEY    = "rag-index/faiss.index"
METADATA_KEY = "rag-index/metadata.pkl"
GROQ_KEY     = os.environ["GROQ_API_KEY"]
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

s3 = boto3.client("s3")

# Module-level cache — survives across warm Lambda invocations
_faiss_index = None
_metadata    = None

CORS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Content-Type":                 "application/json"
}


def load_index():
    """Load FAISS index and metadata from S3 (cached after first load)."""
    global _faiss_index, _metadata

    if _faiss_index is None:
        print("Loading FAISS index from S3...")
        s3.download_file(BUCKET, INDEX_KEY, "/tmp/faiss.index")
        _faiss_index = faiss.read_index("/tmp/faiss.index")

    if _metadata is None:
        print("Loading metadata from S3...")
        s3.download_file(BUCKET, METADATA_KEY, "/tmp/metadata.pkl")
        with open("/tmp/metadata.pkl", "rb") as f:
            _metadata = pickle.load(f)

    return _faiss_index, _metadata


def retrieve(query, top_k=20):
    """Embed query and find top-K similar antenna records."""
    faiss_idx, meta = load_index()
    vec = embedder.encode([query])              # shape: (1, 384)
    _, indices = faiss_idx.search(vec, top_k)
    return [meta[i] for i in indices[0] if i != -1]


def ask_groq(question, docs, history):
    """Send retrieved context + question to Groq Llama 3."""
    context = "\n".join([
        f"SAP:{r.get('SAP ID','')} | State:{r.get('State','')} |"
        f" Sector:{r.get('Sector Id','')} | Band:{r.get('Bands','')} |"
        f" Antenna:{r.get('LSMR Antenna Type','')} |"
        f" Alarm:{r.get('Alarm details','None')} |"
        f" Updated:{r.get('RRH Last Updated Time','')}"
        for r in docs
    ])

    system = (
        "You are a Jio telecom antenna inventory assistant. "
        "Answer ONLY using the retrieved data provided. "
        "Be specific — mention SAP IDs, sectors, bands. "
        "If data is insufficient, say so clearly."
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
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def lambda_handler(event, context):

    # Handle CORS preflight
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS, "body": ""}

    # Parse body
    try:
        body = json.loads(event.get("body") or "{}")
    except Exception:
        body = {}

    user_msg = body.get("message", "").strip()
    history  = body.get("history", [])[-10:]

    # Warm ping — keeps Lambda warm, preloads index
    if user_msg == "ping":
        load_index()
        embedder._load()   # preload ONNX model too
        return {"statusCode": 200, "headers": CORS, "body": json.dumps({"reply": "warm"})}

    if not user_msg:
        return {
            "statusCode": 400, "headers": CORS,
            "body": json.dumps({"error": "No message provided"})
        }

    # RAG pipeline
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
