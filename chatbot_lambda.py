import boto3, json, faiss, numpy as np, pickle
import os, re, hashlib, ssl, io
import urllib.request, urllib.error

BUCKET     = os.environ["BUCKET"]
GROQ_KEY   = os.environ["GROQ_API_KEY"]
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
SAP_PREFIX    = "rag-index/sap/"
BLANK_RET_KEY = "rag-exports/Blank_RET_All_States.csv"
DIM        = 384

s3 = boto3.client("s3")
_index     = None
_idf       = None
_meta      = None
_sap_cache = {}

COLUMNS = [
    "State","JioCenter","Cluster ID","SAP ID","ENB ID","Sector Id","Cell ID","Bands",
    "RRH Connect Board ID","RRH Connect Port ID",
    "SF Antenna Model","SF Antenna Type",
    "RET Connect Board ID","RET Connect Port ID","RET ANT ID",
    "Vendor Code","Serial Number","Ant Model Number","Ant Operating Band",
    "LSMR Tilt Value","LSMR Tilt Date","LSMR Antenna Type","LSMR Antenna Category",
    "Antenna Classification","Additional Remarks","5G Exclusive Antenna",
    "Alarm Status","Alarm details","RRH Last Updated Time","RET Source"
]

# Short columns for normal display
DISPLAY_COLUMNS = [
    "State","SAP ID","Sector Id","Bands",
    "RRH Connect Board ID","RRH Connect Port ID",
    "SF Antenna Model","LSMR Antenna Type",
    "Antenna Classification","RRH Last Updated Time","Alarm details"
]

STATE_CODE_MAP = {
    "MH": "MAHARASHTRA", "DL": "DELHI", "KA": "KARNATAKA",
    "TN": "TAMIL_NADU",  "GJ": "GUJARAT", "RJ": "RAJASTHAN",
    "UP": "UTTAR_PRADESH_EAST", "UW": "UTTAR_PRADESH_WEST",
    "WB": "WEST_BENGAL", "AP": "ANDHRA_PRADESH", "TS": "TELANGANA",
    "MP": "MADHYA_PRADESH", "PB": "PUNJAB", "HR": "HARYANA",
    "KL": "KERALA", "OR": "ODISHA", "BR": "BIHAR",
    "JH": "JHARKHAND", "AS": "ASSAM", "JK": "JAMMU_KASHMIR",
    "HP": "HIMACHAL_PRADESH", "UK": "UTTARAKHAND", "CH": "CHATTISGARH",
    "NE": "NORTH_EAST", "MN": "MANIPUR", "TR": "TRIPURA",
    "ML": "MEGHALAYA", "SK": "SIKKIM", "GA": "GOA",
    "HM": "HIMACHAL_PRADESH", "NK": "KARNATAKA"
}

# Reverse map: full state name -> state name as stored in S3
# Also allow partial matching e.g. "maharashtra" -> "MAHARASHTRA"
STATE_NAME_LIST = [
    "MAHARASHTRA","DELHI","KARNATAKA","TAMIL_NADU","GUJARAT","RAJASTHAN",
    "UTTAR_PRADESH_EAST","UTTAR_PRADESH_WEST","WEST_BENGAL","ANDHRA_PRADESH",
    "TELANGANA","MADHYA_PRADESH","PUNJAB","HARYANA","KERALA","ODISHA","BIHAR",
    "JHARKHAND","ASSAM","JAMMU_KASHMIR","HIMACHAL_PRADESH","UTTARAKHAND",
    "CHATTISGARH","NORTH_EAST","MANIPUR","TRIPURA","MEGHALAYA","SIKKIM","GOA"
]

# Common aliases
STATE_ALIASES = {
    "UP EAST": "UTTAR_PRADESH_EAST", "UP WEST": "UTTAR_PRADESH_WEST",
    "J&K": "JAMMU_KASHMIR", "JK": "JAMMU_KASHMIR",
    "TN": "TAMIL_NADU", "AP": "ANDHRA_PRADESH", "WB": "WEST_BENGAL",
    "MP": "MADHYA_PRADESH", "HP": "HIMACHAL_PRADESH", "UK": "UTTARAKHAND",
    "MH": "MAHARASHTRA", "DL": "DELHI", "KA": "KARNATAKA",
    "GJ": "GUJARAT", "RJ": "RAJASTHAN", "PB": "PUNJAB",
    "HR": "HARYANA", "KL": "KERALA", "OR": "ODISHA", "BR": "BIHAR",
    "JH": "JHARKHAND", "AS": "ASSAM", "CH": "CHATTISGARH",
    "GA": "GOA", "TS": "TELANGANA"
}


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


def _load_state_sap(state_name):
    if state_name in _sap_cache:
        return _sap_cache[state_name]
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{SAP_PREFIX}{state_name}.pkl")
        sap = pickle.loads(obj["Body"].read())
        _sap_cache[state_name] = sap
        print(f"Loaded SAP map {state_name}: {len(sap)} SAP IDs")
        return sap
    except Exception as e:
        print(f"Could not load SAP map {state_name}: {e}")
        return {}


def tokenize(t):
    t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
    return [w for w in t.split() if len(w) > 1]


def encode(text):
    _load_base()
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


def extract_state_name(msg):
    """Extract state name from message, return S3 key name or None."""
    upper = msg.upper()

    # Check aliases first (short codes)
    for alias, state in STATE_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", upper):
            return state

    # Check full state names (with underscore or space)
    for state in STATE_NAME_LIST:
        readable = state.replace("_", " ")
        if readable in upper or state in upper:
            return state

    return None


def is_blank_ret_query(msg):
    """Detect queries asking for blank/missing RET data."""
    lower = msg.lower()
    patterns = [
        r"blank.{0,10}ret",
        r"missing.{0,10}ret",
        r"empty.{0,10}ret",
        r"no.{0,10}ret.{0,10}data",
        r"ret.{0,10}not.{0,10}filled",
        r"ret.{0,10}missing",
        r"ret.{0,10}blank",
        r"antenna classification.{0,20}blank",
    ]
    return any(re.search(p, lower) for p in patterns)


def is_blank(val):
    return not val or val.strip() in ("", "-", "N/A", "null", "None", "nan", "0")


def get_blank_ret_records(state_name):
    """
    Filter:
    Antenna Classification = 'RET'  (case-insensitive)
    AND RRH Connect Board ID = blank/'-'
    AND RRH Connect Port ID  = blank/'-'
    Returns ALL columns + State column.
    """
    sap_map = _load_state_sap(state_name)
    results = []
    total   = 0
    sample_classes = set()

    for sap_id, records in sap_map.items():
        for r in records:
            total += 1
            ret_board = r.get("RET Connect Board ID", "").strip()
            ret_port  = r.get("RET Connect Port ID",  "").strip()
            ant_class = r.get("Antenna Classification","").strip()

            # Always collect sample values for CloudWatch debug
            sample_classes.add(repr(ant_class))

            if ret_board == "-" and ret_port == "-" and ant_class.upper() == "RET":
                row = dict(r)               # ALL original columns
                row["State"] = state_name   # ensure State is set
                results.append(row)

    print(f"{state_name}: {total} records checked, {len(results)} matched")
    # Print ALL unique Antenna Classification values seen
    print(f"ALL Antenna Classification values in {state_name}: {sorted(sample_classes)}")
    # Also print board/port blank stats
    blank_board = sum(1 for sap_id, recs in sap_map.items()
                      for r in recs if is_blank(r.get("RRH Connect Board ID","").strip()))
    blank_port  = sum(1 for sap_id, recs in sap_map.items()
                      for r in recs if is_blank(r.get("RRH Connect Port ID","").strip()))
    print(f"Records with blank Board ID: {blank_board}, blank Port ID: {blank_port}")
    return results


def lookup_sap(sap_id):
    parts      = sap_id.upper().split("-")
    state_code = parts[1] if len(parts) > 1 else ""
    state_name = STATE_CODE_MAP.get(state_code, "")
    if state_name:
        sap_map = _load_state_sap(state_name)
        records = sap_map.get(sap_id.upper(), [])
        if records:
            return records
    # Fallback scan
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=SAP_PREFIX)
        for obj in resp.get("Contents", []):
            key   = obj["Key"]
            sname = key.replace(SAP_PREFIX, "").replace(".pkl", "")
            smap  = _load_state_sap(sname)
            recs  = smap.get(sap_id.upper(), [])
            if recs:
                return recs
    except Exception as e:
        print(f"Fallback scan error: {e}")
    return []


def retrieve_semantic(query, top_k=20):
    _load_base()
    _, I = _index.search(encode(query), top_k)
    return [_meta[i] for i in I[0] if i != -1 and i < len(_meta)]


def ask_groq(question, docs, history):
    ctx = "\n".join([
        f"Sector:{r.get('Sector Id','N/A')} | "
        f"Band:{r.get('Bands','N/A')} | "
        f"LSMR:{r.get('LSMR Antenna Type','N/A')} | "
        f"Alarm:{r.get('Alarm details','N/A')}"
        for r in docs[:20] if r
    ])
    payload = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content":
             "You are a Jio antenna inventory assistant. "
             "Write a SHORT 3-5 line summary answering the question. "
             "Highlight key patterns: alarms, states, bands. Be concise."}
        ] + history + [
            {"role": "user", "content": f"Question: {question}\n\nData:\n{ctx}"}
        ],
        "max_tokens": 300, "temperature": 0.1
    }).encode("utf-8")
    ctx_ssl = ssl.create_default_context()
    try:
        req = urllib.request.Request(GROQ_URL, data=payload,
            headers={"Authorization": f"Bearer {GROQ_KEY}",
                     "Content-Type": "application/json",
                     "User-Agent": "python-urllib/3.11"}, method="POST")
        with urllib.request.urlopen(req, timeout=30, context=ctx_ssl) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Summary unavailable: {e}"


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
        sap_id     = extract_sap_id(msg)
        state_name = extract_state_name(msg)
        blank_ret  = is_blank_ret_query(msg)

        # ── Blank RET download ──
        if blank_ret:
            # Detect "all states" request
            all_states_requested = bool(re.search(
                r"all.{0,10}state|every.{0,10}state|pan.{0,5}india|all.{0,10}circle",
                msg, re.IGNORECASE
            ))

            if all_states_requested:
                # Pre-built at index time — just return presigned URL instantly
                try:
                    url = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": BUCKET, "Key": BLANK_RET_KEY},
                        ExpiresIn=3600
                    )
                    return {"statusCode": 200, "body": json.dumps({
                        "summary":       "Blank RET data for all states is ready. Click below to download.",
                        "records":       [],
                        "columns":       COLUMNS,
                        "retrieved":     0,
                        "history":       history,
                        "download":      False,
                        "presigned_url": url,
                        "filename":      "Blank_RET_All_States.csv"
                    })}
                except Exception as e:
                    return {"statusCode": 500, "body": json.dumps({"error": f"Could not generate download link: {e}"})}

            elif state_name:
                docs     = get_blank_ret_records(state_name)
                label    = state_name
                filename = f"Blank_RET_{state_name}.xlsx"
            else:
                return {"statusCode": 200, "body": json.dumps({
                    "summary":   "Please specify a state name or 'all states'. E.g. 'blank RET data for Maharashtra'",
                    "records":   [], "columns": COLUMNS,
                    "retrieved": 0,  "history": history,
                    "download":  False
                })}

            if not docs:
                return {"statusCode": 200, "body": json.dumps({
                    "summary":   f"No blank RET records found for {label}.",
                    "records":   [], "columns": COLUMNS,
                    "retrieved": 0,  "history": history,
                    "download":  False
                })}

            return {"statusCode": 200, "body": json.dumps({
                "summary":   f"Found {len(docs):,} blank RET records for {label}. Excel downloading now.",
                "records":   docs,
                "columns":   COLUMNS,
                "retrieved": len(docs),
                "history":   history,
                "download":  True,
                "excel":     True,
                "state":     label,
                "filename":  filename
            })}

        # ── SAP ID lookup ──
        elif sap_id:
            docs = lookup_sap(sap_id)
            if not docs:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"No records found for SAP ID {sap_id}.",
                    "records": [], "columns": COLUMNS,
                    "retrieved": 0, "history": history, "download": False
                })}
            return {"statusCode": 200, "body": json.dumps({
                "summary":   "", "records": docs, "columns": COLUMNS,
                "retrieved": len(docs), "history": history, "download": False
            })}

        # ── Semantic search ──
        else:
            docs    = retrieve_semantic(msg, top_k=20)
            summary = ask_groq(msg, docs, history)
            h2 = (history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": summary}
            ])[-6:]
            return {"statusCode": 200, "body": json.dumps({
                "summary": summary, "records": docs, "columns": COLUMNS,
                "retrieved": len(docs), "history": h2, "download": False
            })}

    except Exception as e:
        print(f"Error: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
