import boto3, json, faiss, numpy as np, pickle
import os, re, hashlib, ssl, io, csv
import urllib.request, urllib.error
from datetime import datetime, timedelta

# ── LangChain imports (zero cost — open source, uses Groq free tier) ──
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough

BUCKET     = os.environ["BUCKET"]
GROQ_KEY   = os.environ["GROQ_API_KEY"]
SAP_PREFIX    = "rag-index/sap/"
BLANK_RET_KEY = "rag-exports/Blank_RET_All_States.csv"
DIM        = 384

s3 = boto3.client("s3")
_index     = None
_idf       = None
_meta      = None
_sap_cache = {}

# ── LangChain LLM — Groq free tier, zero cost ──
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=GROQ_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=300
        )
    return _llm

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

STATE_NAME_LIST = [
    "MAHARASHTRA","DELHI","KARNATAKA","TAMIL_NADU","GUJARAT","RAJASTHAN",
    "UTTAR_PRADESH_EAST","UTTAR_PRADESH_WEST","WEST_BENGAL","ANDHRA_PRADESH",
    "TELANGANA","MADHYA_PRADESH","PUNJAB","HARYANA","KERALA","ODISHA","BIHAR",
    "JHARKHAND","ASSAM","JAMMU_KASHMIR","HIMACHAL_PRADESH","UTTARAKHAND",
    "CHATTISGARH","NORTH_EAST","MANIPUR","TRIPURA","MEGHALAYA","SIKKIM","GOA"
]

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


# ════════════════════════════════════════════════════════════
#  BASE LOADERS
# ════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════
#  CODE MAP — SAP code -> actual S3 state filename(s)
# ════════════════════════════════════════════════════════════

_code_map_cache = {}

def _load_code_map():
    global _code_map_cache
    if _code_map_cache:
        return _code_map_cache
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{SAP_PREFIX}code_map.json")
        _code_map_cache = json.loads(obj["Body"].read())
        print(f"Loaded code_map: {_code_map_cache}")
    except Exception as e:
        print(f"code_map.json not found, using STATE_CODE_MAP fallback: {e}")
        _code_map_cache = {}
    return _code_map_cache


def _resolve_states(state_code):
    code_map = _load_code_map()
    if state_code in code_map:
        val = code_map[state_code]
        return val if isinstance(val, list) else [val]
    fallback = STATE_CODE_MAP.get(state_code, "")
    return [fallback] if fallback else []


# ════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════

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
    upper = msg.upper()
    for alias, state in STATE_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", upper):
            return state
    for state in STATE_NAME_LIST:
        readable = state.replace("_", " ")
        if readable in upper or state in upper:
            return state
    return None


def is_blank_ret_query(msg):
    lower = msg.lower()
    if re.search(r"\b(why|reason|explain|cause|what is|what are|how|tell me about|describe)\b", lower):
        return False
    download_intent = bool(re.search(
        r"\b(show|get|download|give|list|export|fetch|find|extract|pull|share|provide)\b"
        r"|blank ret data|missing ret data|empty ret data", lower
    ))
    if not download_intent:
        return False
    ret_patterns = [
        r"blank.{0,15}ret", r"missing.{0,15}ret", r"empty.{0,15}ret",
        r"ret.{0,15}blank",  r"ret.{0,15}missing", r"ret.{0,15}not.{0,15}filled",
        r"no.{0,10}ret.{0,10}data",
    ]
    return any(re.search(p, lower) for p in ret_patterns)


def is_blank(val):
    return not val or val.strip() in ("", "-", "N/A", "null", "None", "nan", "0")


def parse_date(val):
    if not val:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    return None


def filter_and_sort(results):
    if not results:
        return results
    all_dates = [parse_date(r.get("RRH Last Updated Time", "")) for r in results
                 if parse_date(r.get("RRH Last Updated Time", ""))]
    if all_dates:
        latest_date = max(all_dates)
        cutoff_date = latest_date - timedelta(days=15)
        results = [r for r in results
                   if parse_date(r.get("RRH Last Updated Time", "")) and
                      parse_date(r.get("RRH Last Updated Time", "")) >= cutoff_date]
    results.sort(key=lambda x: (
        x.get("SAP ID", ""),
        int(x.get("Sector Id", 0)) if str(x.get("Sector Id", "")).isdigit() else 999
    ))
    return results


# ════════════════════════════════════════════════════════════
#  SAP LOOKUP
# ════════════════════════════════════════════════════════════

def lookup_sap(sap_id):
    parts      = sap_id.upper().split("-")
    state_code = parts[1] if len(parts) > 1 else ""
    state_list = _resolve_states(state_code)
    print(f"SAP {sap_id}: code={state_code} -> states={state_list}")
    for state_name in state_list:
        if not state_name:
            continue
        sap_map = _load_state_sap(state_name)
        records = sap_map.get(sap_id.upper(), [])
        if records:
            print(f"SAP {sap_id} found in {state_name}: {len(records)} records")
            return records
        print(f"SAP {sap_id} not in {state_name}, trying next...")
    print(f"SAP {sap_id} not found in any of {state_list}")
    return []


# ════════════════════════════════════════════════════════════
#  BLANK RET
# ════════════════════════════════════════════════════════════

def get_blank_ret_records(state_name):
    code_map = _load_code_map()
    all_state_files = []
    for v in code_map.values():
        if isinstance(v, list):
            all_state_files.extend(v)
        else:
            all_state_files.append(v)
    all_state_files = list(set(all_state_files))
    resolved = state_name
    if state_name not in all_state_files:
        for name in all_state_files:
            if name.upper() == state_name.upper():
                resolved = name
                break
    print(f"Blank RET: resolving '{state_name}' -> '{resolved}'")
    sap_map = _load_state_sap(resolved)
    results = []
    total   = 0
    sample_classes = set()
    for sap_id, records in sap_map.items():
        for r in records:
            total += 1
            ret_board = r.get("RET Connect Board ID", "").strip()
            ret_port  = r.get("RET Connect Port ID",  "").strip()
            ant_class = r.get("Antenna Classification","").strip()
            sample_classes.add(repr(ant_class))
            if ret_board == "-" and ret_port == "-" and ant_class.upper() == "RET":
                row = dict(r)
                row["State"] = resolved
                results.append(row)
    print(f"{resolved}: {total} records checked, {len(results)} matched")
    print(f"ALL Antenna Classification values: {sorted(sample_classes)}")
    return results


# ════════════════════════════════════════════════════════════
#  LANGCHAIN SEMANTIC SEARCH + RAG
# ════════════════════════════════════════════════════════════

def retrieve_semantic(query, top_k=20, score_threshold=1.5):
    """FAISS vector search — unchanged, LangChain wraps the LLM only."""
    _load_base()
    D, I = _index.search(encode(query), top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1 or idx >= len(_meta):
            continue
        if dist <= score_threshold:
            results.append(_meta[idx])
    print(f"Semantic search: {len(results)}/{top_k} within threshold "
          f"(distances: {D[0][:5].tolist()})")
    return results


def build_context(docs):
    """Format retrieved docs into a context string for the LLM."""
    return "\n".join([
        f"State:{r.get('State','N/A')} | SAP:{r.get('SAP ID','N/A')} | "
        f"Sector:{r.get('Sector Id','N/A')} | Band:{r.get('Bands','N/A')} | "
        f"LSMR:{r.get('LSMR Antenna Type','N/A')} | Alarm:{r.get('Alarm details','N/A')}"
        for r in docs[:20] if r
    ])


def ask_langchain(question, docs, history):
    """
    LangChain-powered RAG:
    - ChatGroq LLM (Groq free tier — zero cost)
    - ConversationBufferWindowMemory (last 3 turns)
    - Strict anti-hallucination system prompt
    - Context injected from FAISS-retrieved docs
    """
    if not docs:
        return "No matching records found in the database for this query."

    llm     = _get_llm()
    context = build_context(docs)

    # Build message list from history + system + context + question
    messages = [
        SystemMessage(content=(
            "You are a Jio antenna inventory assistant. "
            "Answer ONLY using the database records provided below. "
            "Do NOT invent, assume, or add any information not present in the data. "
            "If the data does not contain enough information to answer, say so clearly. "
            "Be concise: 2-4 lines max. No bullet points.\n\n"
            f"Database records:\n{context}"
        ))
    ]

    # Add conversation history (LangChain message objects)
    for h in history:
        role    = h.get("role", "")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Add current question
    messages.append(HumanMessage(content=question))

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"LangChain/Groq error: {e}")
        return f"Summary unavailable: {e}"


# ════════════════════════════════════════════════════════════
#  LAMBDA HANDLER
# ════════════════════════════════════════════════════════════

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

        # ── Blank RET download ──────────────────────────────────────────────
        if blank_ret:
            all_states_requested = bool(re.search(
                r"all.{0,10}state|every.{0,10}state|pan.{0,5}india|all.{0,10}circle",
                msg, re.IGNORECASE
            ))
            if all_states_requested:
                try:
                    url = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": BUCKET, "Key": BLANK_RET_KEY},
                        ExpiresIn=3600
                    )
                    return {"statusCode": 200, "body": json.dumps({
                        "summary": "Blank RET data for all states is ready. Click below to download.",
                        "records": [], "columns": COLUMNS, "retrieved": 0,
                        "history": history, "download": False,
                        "presigned_url": url, "filename": "Blank_RET_All_States.csv"
                    })}
                except Exception as e:
                    return {"statusCode": 500, "body": json.dumps({"error": f"Download link error: {e}"})}

            elif state_name:
                docs     = get_blank_ret_records(state_name)
                label    = state_name
                filename = f"Blank_RET_{state_name}.csv"
            else:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": "Please specify a state. E.g. 'show blank RET data for Maharashtra'",
                    "records": [], "columns": COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}

            if not docs:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"No blank RET records found for {label}.",
                    "records": [], "columns": COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}

            csv_key = f"exports/Blank_RET_{label}.csv"
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in docs:
                writer.writerow({c: row.get(c, "") for c in COLUMNS})
            s3.put_object(Bucket=BUCKET, Key=csv_key,
                          Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
            presigned_url = s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": csv_key}, ExpiresIn=3600)
            print(f"CSV saved: {csv_key}, {len(docs)} rows")
            return {"statusCode": 200, "body": json.dumps({
                "summary": f"Found {len(docs):,} blank RET records for {label}. Click the link to download.",
                "records": [], "columns": COLUMNS, "retrieved": len(docs),
                "history": history, "download": False,
                "presigned_url": presigned_url, "filename": filename
            })}

        # ── SAP ID lookup ───────────────────────────────────────────────────
        elif sap_id:
            docs = filter_and_sort(lookup_sap(sap_id))
            if not docs:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"No records found for SAP ID {sap_id}.",
                    "records": [], "columns": COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}
            return {"statusCode": 200, "body": json.dumps({
                "summary": "", "records": docs, "columns": COLUMNS,
                "retrieved": len(docs), "history": history, "download": False
            })}

        # ── LangChain semantic search + RAG ─────────────────────────────────
        else:
            docs = retrieve_semantic(msg, top_k=20, score_threshold=1.5)
            if not docs:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": "No closely matching records found. Try a SAP ID, state name, band, or alarm type.",
                    "records": [], "columns": DISPLAY_COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}

            summary = ask_langchain(msg, docs, history)
            h2 = (history + [
                {"role": "user",      "content": msg},
                {"role": "assistant", "content": summary}
            ])[-6:]
            return {"statusCode": 200, "body": json.dumps({
                "summary":   summary,
                "records":   [{c: r.get(c, "") for c in DISPLAY_COLUMNS} for r in docs],
                "columns":   DISPLAY_COLUMNS,
                "retrieved": len(docs),
                "history":   h2,
                "download":  False
            })}

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
