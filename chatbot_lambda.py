# chatbot_lambda.py
# No CORS headers in code — handled entirely by Lambda Function URL config

import boto3, json, faiss, numpy as np, pickle
import os, re, hashlib, ssl
import urllib.request, urllib.error

BUCKET   = os.environ["BUCKET"]
GROQ_KEY = os.environ["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DIM      = 384

s3 = boto3.client("s3")
_index = _idf = _meta = None


def _load():
    global _index, _idf, _meta
    if _index is None:
        print("Loading FAISS index...")
        s3.download_file(BUCKET, "rag-index/faiss.index", "/tmp/faiss.index")
        _index = faiss.read_index("/tmp/faiss.index")
        if hasattr(_index, "nprobe"):
            _index.nprobe = 20
        print(f"FAISS loaded: {_index.ntotal:,} vectors")
    if _idf is None:
        obj  = s3.get_object(Bucket=BUCKET, Key="rag-model/idf.pkl")
        _idf = pickle.loads(obj["Body"].read())
        print(f"IDF loaded: {len(_idf):,} terms")
    if _meta is None:
        print("Loading metadata...")
        obj   = s3.get_object(Bucket=BUCKET, Key="rag-index/metadata.pkl")
        _meta = pickle.loads(obj["Body"].read())
        print(f"Metadata loaded: {len(_meta):,} rows")


def tokenize(t):
    t = re.sub(r'[^a-z0-9\s]', ' ', t.lower())
    return [w for w in t.split() if len(w) > 1]


def encode(text):
    vec = np.zeros(DIM, dtype=np.float32)
    for t in tokenize(text):
        idx = int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM
        vec[idx] += _idf.get(t, 1.0)
    n = np.linalg.norm(vec)
    if n > 1e-9:
        vec /= n
    return vec.reshape(1, -1)


def retrieve(query, top_k=20):
    _load()
    _, I = _index.search(encode(query), top_k)
    return [_meta[i] for i in I[0] if i != -1 and i < len(_meta)]


def ask_groq(question, docs, history):
    ctx = "\n".join([
        f"SAP:{r.get('SAP ID','')} | State:{r.get('State','')} | "
        f"Sector:{r.get('Sector Id','')} | Band:{r.get('Bands','')} | "
        f"Antenna:{r.get('LSMR Antenna Type','')} | "
        f"Alarm:{r.get('Alarm details','None')} | "
        f"Updated:{r.get('RRH Last Updated Time','')}"
        for r in docs if r
    ])
    payload = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content":
             "You are a Jio antenna inventory assistant. "
             "Answer using only the retrieved data. "
             "Mention SAP IDs, sectors, bands specifically."}
        ] + history + [
            {"role": "user", "content": f"Data:\n{ctx}\n\nQuestion: {question}"}
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }).encode("utf-8")

    ctx_ssl = ssl.create_default_context()
    try:
        req = urllib.request.Request(
            GROQ_URL, data=payload,
            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type":  "application/json",
                "User-Agent":    "python-urllib/3.11"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30, context=ctx_ssl) as r:
            print(f"Groq status: {r.status}")
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"Groq HTTP error {e.code}: {body}")
        raise Exception(f"Groq error {e.code}: {body[:200]}")
    except urllib.error.URLError as e:
        print(f"Groq URL error: {e.reason}")
        raise Exception(f"Cannot reach Groq: {e.reason}")


def lambda_handler(event, context):
    # No CORS headers here — Lambda Function URL config handles all CORS
    method = event.get("requestContext", {}).get("http", {}).get("method", "")

    # OPTIONS preflight — return empty 200 (Function URL adds CORS headers)
    if method == "OPTIONS":
        return {"statusCode": 200, "body": ""}

    try:
        body = json.loads(event.get("body") or "{}")
    except Exception:
        body = {}

    msg     = body.get("message", "").strip()
    history = body.get("history", [])[-10:]

    if msg == "ping":
        _load()
        return {"statusCode": 200,
                "body": json.dumps({"reply": "warm"})}

    if not msg:
        return {"statusCode": 400,
                "body": json.dumps({"error": "No message"})}

    try:
        docs   = retrieve(msg)
        answer = ask_groq(msg, docs, history)
    except Exception as e:
        print(f"Error: {e}")
        return {"statusCode": 500,
                "body": json.dumps({"error": str(e)})}

    h2 = (history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": answer}
    ])[-10:]

    return {
        "statusCode": 200,
        "body": json.dumps({
            "reply":     answer,
            "history":   h2,
            "retrieved": len(docs)
        })
    }
