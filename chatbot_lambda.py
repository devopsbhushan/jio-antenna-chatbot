# chatbot_lambda.py

import boto3, json, faiss, numpy as np, pickle
import os, re, hashlib, ssl
import urllib.request, urllib.error

BUCKET   = os.environ["BUCKET"]
GROQ_KEY = os.environ["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DIM      = 384

s3 = boto3.client("s3")
_index   = None
_idf     = None
_meta    = None
_sap_map = None   # SAP ID -> list of row dicts (instant lookup)

COLUMNS = [
    "State", "SAP ID", "Sector Id", "Bands",
    "RRH Connect Board ID", "RRH Connect Port ID",
    "SF Antenna Model", "LSMR Antenna Type",
    "Antenna Classification", "RRH Last Updated Time",
    "Alarm details"
]


def _load_base():
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


def _load_sap_map():
    global _sap_map
    if _sap_map is None:
        print("Loading SAP map...")
        obj      = s3.get_object(Bucket=BUCKET, Key="rag-index/sap_map.pkl")
        _sap_map = pickle.loads(obj["Body"].read())
        print(f"SAP map loaded: {len(_sap_map):,} unique SAP IDs")


def tokenize(t):
    t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
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


def extract_sap_id(msg):
    m = re.search(r"I-[A-Z]{2}-[A-Z0-9]+-ENB-[0-9A-Z]+", msg.upper())
    return m.group(0) if m else None


def retrieve_semantic(query, top_k=20):
    _load_base()
    _, I = _index.search(encode(query), top_k)
    return [_meta[i] for i in I[0] if i != -1 and i < len(_meta)]


def ask_groq(question, docs, history, sap_id=None):
    ctx = "\n".join([
        f"Sector:{r.get('Sector Id','N/A')} | "
        f"Band:{r.get('Bands','N/A')} | "
        f"SF Model:{r.get('SF Antenna Model','N/A')} | "
        f"LSMR:{r.get('LSMR Antenna Type','N/A')} | "
        f"Classification:{r.get('Antenna Classification','N/A')} | "
        f"Alarm:{r.get('Alarm details','N/A')} | "
        f"Updated:{r.get('RRH Last Updated Time','N/A')}"
        for r in docs if r
    ])
    state = docs[0].get("State","N/A") if docs else "N/A"
    site  = sap_id or "this site"

    if sap_id:
        system = (
            "You are a Jio antenna inventory assistant. "
            f"Write a site report for Site: {site} | State: {state}\n\n"
            "List each sector-band combination:\n"
            "- Sector X | Band Y | Antenna: [model] | Alarm: [alarm or None]\n\n"
            "End with: Total X sectors, Y alarms active."
        )
    else:
        system = (
            "You are a Jio antenna inventory assistant. "
            "Write a SHORT 3-5 line summary answering the question. "
            "Highlight key patterns: alarms, states, bands. "
            "Be concise — user sees full table separately."
        )

    payload = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "system", "content": system}]
            + history
            + [{"role": "user", "content": f"Question: {question}\n\nData:\n{ctx}"}],
        "max_tokens": 512,
        "temperature": 0.1
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
        _load_base()
        return {"statusCode": 200, "body": json.dumps({"reply": "warm"})}

    if not msg:
        return {"statusCode": 400, "body": json.dumps({"error": "No message"})}

    try:
        sap_id = extract_sap_id(msg)

        if sap_id:
            # Instant lookup from SAP map — no S3 scanning needed
            _load_sap_map()
            docs = _sap_map.get(sap_id, [])
            print(f"SAP map lookup: {sap_id} -> {len(docs)} records")
            if not docs:
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "summary":   f"No records found for SAP ID {sap_id}.",
                        "records":   [],
                        "columns":   COLUMNS,
                        "retrieved": 0,
                        "history":   history
                    })
                }
        else:
            # Semantic FAISS search for general questions
            docs = retrieve_semantic(msg, top_k=20)

        summary     = ask_groq(msg, docs, history, sap_id)
        raw_records = [{col: r.get(col,"") for col in COLUMNS} for r in docs]

    except Exception as e:
        print(f"Error: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    h2 = (history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": summary}
    ])[-6:]

    return {
        "statusCode": 200,
        "body": json.dumps({
            "summary":   summary,
            "records":   raw_records,
            "columns":   COLUMNS,
            "retrieved": len(docs),
            "history":   h2
        })
    }
