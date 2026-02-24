# chatbot_lambda.py
# Returns both raw_records (all columns) and ai_summary separately

import boto3, json, faiss, numpy as np, pickle
import os, re, hashlib, ssl
import urllib.request, urllib.error

BUCKET   = os.environ["BUCKET"]
GROQ_KEY = os.environ["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DIM      = 384

s3 = boto3.client("s3")
_index = _idf = _meta = None

COLUMNS = [
    "State", "SAP ID", "Sector Id", "Bands",
    "RRH Connect Board ID", "RRH Connect Port ID",
    "SF Antenna Model", "LSMR Antenna Type",
    "Antenna Classification", "RRH Last Updated Time",
    "Alarm details"
]


def _load():
    global _index, _idf, _meta
    if _index is None:
        s3.download_file(BUCKET, "rag-index/faiss.index", "/tmp/faiss.index")
        _index = faiss.read_index("/tmp/faiss.index")
        if hasattr(_index, "nprobe"):
            _index.nprobe = 20
    if _idf is None:
        obj  = s3.get_object(Bucket=BUCKET, Key="rag-model/idf.pkl")
        _idf = pickle.loads(obj["Body"].read())
    if _meta is None:
        obj   = s3.get_object(Bucket=BUCKET, Key="rag-index/metadata.pkl")
        _meta = pickle.loads(obj["Body"].read())


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
    # Send compact record summary to Groq for AI summary only
    ctx = "\n".join([
        f"SAP:{r.get('SAP ID','N/A')} State:{r.get('State','N/A')} "
        f"Sector:{r.get('Sector Id','N/A')} Band:{r.get('Bands','N/A')} "
        f"Model:{r.get('SF Antenna Model','N/A')} "
        f"LSMR:{r.get('LSMR Antenna Type','N/A')} "
        f"Alarm:{r.get('Alarm details','N/A')}"
        for r in docs if r
    ])

    payload = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content":
             "You are a Jio antenna inventory assistant. "
             "The user can already see the full raw data table. "
             "Your job is to write a SHORT 3-5 line AI summary that: "
             "1) Directly answers the user's question "
             "2) Highlights key patterns (common alarms, states, bands) "
             "3) Flags anything notable (unidentified antennas, missing data) "
             "Be concise. No need to list every record — the table shows that."}
        ] + history + [
            {"role": "user",
             "content": f"Question: {question}\n\nData:\n{ctx}"}
        ],
        "max_tokens": 512,
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
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise Exception(f"Groq error {e.code}: {body[:200]}")
    except urllib.error.URLError as e:
        raise Exception(f"Cannot reach Groq: {e.reason}")


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "")

    if method == "OPTIONS":
        return {"statusCode": 200, "body": ""}

    try:
        body = json.loads(event.get("body") or "{}")
    except Exception:
        body = {}

    msg     = body.get("message", "").strip()
    history = body.get("history", [])[-6:]

    if msg == "ping":
        _load()
        return {"statusCode": 200,
                "body": json.dumps({"reply": "warm"})}

    if not msg:
        return {"statusCode": 400,
                "body": json.dumps({"error": "No message"})}

    try:
        docs    = retrieve(msg)
        summary = ask_groq(msg, docs, history)

        # Return raw records (all columns) + AI summary separately
        raw_records = [
            {col: r.get(col, "") for col in COLUMNS}
            for r in docs if r
        ]

    except Exception as e:
        print(f"Error: {e}")
        return {"statusCode": 500,
                "body": json.dumps({"error": str(e)})}

    h2 = (history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": summary}
    ])[-6:]

    return {
        "statusCode": 200,
        "body": json.dumps({
            "summary":     summary,      # AI summary (short)
            "records":     raw_records,  # Full raw data table
            "columns":     COLUMNS,      # Column headers
            "retrieved":   len(docs),
            "history":     h2
        })
    }
