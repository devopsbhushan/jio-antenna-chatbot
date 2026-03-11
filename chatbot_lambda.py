import boto3, json, faiss, numpy as np, pickle
import os, re, hashlib, ssl, io, csv
import urllib.request, urllib.error
from datetime import datetime, timedelta

# LangChain imported lazily inside _get_llm() to avoid Lambda crash on import error
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


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

# ── LangChain LLM — lazy import so Lambda doesn't crash if package missing ──
_llm          = None
_langchain_ok = None   # None=untested, True=working, False=failed

def _get_llm():
    """
    Lazy-import LangChain. If import fails (version conflict, missing package),
    sets _langchain_ok=False so callers fall back to direct urllib Groq call.
    """
    global _llm, _langchain_ok
    if _langchain_ok is False:
        return None
    if _llm is not None:
        return _llm
    try:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            api_key=GROQ_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=300
        )
        _langchain_ok = True
        print("LangChain ChatGroq initialised successfully")
    except Exception as e:
        _langchain_ok = False
        print(f"LangChain unavailable ({e}), falling back to direct Groq API")
    return _llm


def detect_and_translate(msg):
    """
    Default language: English.
    Translation disabled — returns original message as-is.
    """
    return msg, None

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
    "MH": "MAHARASHTRA",        "DL": "DELHI",              "KA": "KARNATAKA",
    "TN": "TAMIL_NADU",         "GJ": "GUJARAT",            "RJ": "RAJASTHAN",
    "UP": "UTTAR_PRADESH_EAST", "UW": "UTTAR_PRADESH_WEST", "WB": "WEST_BENGAL",
    "AP": "ANDHRA_PRADESH",     "TS": "TELANGANA",          "MP": "MADHYA_PRADESH",
    "PB": "PUNJAB",             "HR": "HARYANA",            "KL": "KERALA",
    "OR": "ODISHA",             "BR": "BIHAR",              "JH": "JHARKHAND",
    "AS": "ASSAM",              "JK": "JAMMU_KASHMIR",      "HP": "HIMACHAL_PRADESH",
    "UK": "UTTARAKHAND",        "CH": "CHATTISGARH",        "NE": "NORTH_EAST",
    "MN": "MANIPUR",            "TR": "TRIPURA",            "ML": "MEGHALAYA",
    "SK": "SIKKIM",             "GA": "GOA",               "GO": "GOA",
    "MU": "MUMBAI",             "KO": "KOLKATA",            "HY": "HYDERABAD",
    "BL": "BANGALORE",          "PU": "PUNE",               "NK": "KARNATAKA"
}

STATE_NAME_LIST = [
    "MAHARASHTRA","DELHI","KARNATAKA","TAMIL_NADU","GUJARAT","RAJASTHAN",
    "UTTAR_PRADESH_EAST","UTTAR_PRADESH_WEST","WEST_BENGAL","ANDHRA_PRADESH",
    "TELANGANA","MADHYA_PRADESH","PUNJAB","HARYANA","KERALA","ODISHA","BIHAR",
    "JHARKHAND","ASSAM","JAMMU_KASHMIR","HIMACHAL_PRADESH","UTTARAKHAND",
    "CHATTISGARH","NORTH_EAST","MANIPUR","TRIPURA","MEGHALAYA","SIKKIM","GOA",
    "MUMBAI","KOLKATA","CHENNAI","HYDERABAD"
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

def is_alarm_query(msg):
    """
    Detect queries that want to DOWNLOAD alarm records.
    Must have download intent + alarm keyword.
    Excludes explain/why questions.
    """
    lower = msg.lower()
    if re.search(r"\b(why|reason|explain|cause|what is|what are|how|tell me about|describe)\b", lower):
        return False
    download_intent = bool(re.search(
        r"\b(show|get|download|give|list|export|fetch|find|extract|pull|share|provide)\b"
        r"|alarm data|alarm list|alarm records|alarm report",
        lower
    ))
    if not download_intent:
        return False
    alarm_patterns = [
        r"\balarm\b", r"\balarms\b", r"\balert\b", r"\bfault\b",
        r"\bvswr\b", r"\bald\b", r"\bret not calibrated\b",
    ]
    return any(re.search(p, lower) for p in alarm_patterns)


def get_alarm_records(state_name):
    """Return all records where Alarm Status=='Y' and Alarm details != '-'."""
    sap_map = _load_state_sap(state_name)
    results = []
    total   = 0
    for records in sap_map.values():
        for r in records:
            total += 1
            alarm_status  = str(r.get("Alarm Status",  "")).strip()
            alarm_details = str(r.get("Alarm details", "")).strip()
            if alarm_status == "Y" and alarm_details != "-":
                row = dict(r)
                row["State"] = state_name
                results.append(row)
    print(f"{state_name}: {total} records checked, {len(results)} alarm matched")
    return results


def is_state_data_query(msg):
    """
    Detect queries requesting full state-level data download.
    Examples: "download Maharashtra data", "get all data for Delhi",
              "extract Gujarat state data", "full data for Karnataka"
    Must NOT be blank RET or alarm query.
    """
    lower = msg.lower()
    if re.search(r"\b(why|reason|explain|cause|what is|what are|how|tell me about|describe)\b", lower):
        return False
    # Must have download intent
    download_intent = bool(re.search(
        r"\b(show|get|download|give|list|export|fetch|find|extract|pull|share|provide|all data|full data|entire|complete)\b",
        lower
    ))
    if not download_intent:
        return False
    # Must have data keyword
    data_keyword = bool(re.search(
        r"\b(data|records|sites|inventory|antennas|all sites|all records)\b", lower
    ))
    return data_keyword


def get_state_data_15days(state_name):
    """
    Extract ALL records for a state filtered to latest 15 days
    (same logic as filter_and_sort used in summary display).
    Returns (records, latest_date_str, cutoff_date_str)
    """
    sap_map = _load_state_sap(state_name)
    all_records = []
    for records in sap_map.values():
        for r in records:
            row = dict(r)
            row["State"] = state_name
            all_records.append(row)

    if not all_records:
        return [], None, None

    # Find latest RRH Last Updated Time across ALL records
    all_dates = [
        parse_date(r.get("RRH Last Updated Time", ""))
        for r in all_records
        if parse_date(r.get("RRH Last Updated Time", ""))
    ]

    if not all_dates:
        # No dates — return all records
        return all_records, None, None

    latest_date = max(all_dates)
    cutoff_date = latest_date - timedelta(days=15)

    filtered = [
        r for r in all_records
        if parse_date(r.get("RRH Last Updated Time", "")) and
           parse_date(r.get("RRH Last Updated Time", "")) >= cutoff_date
    ]

    # Sort by SAP ID then Sector Id
    filtered.sort(key=lambda x: (
        x.get("SAP ID", ""),
        int(x.get("Sector Id", 0)) if str(x.get("Sector Id", "")).isdigit() else 999
    ))

    latest_str = latest_date.strftime("%Y-%m-%d")
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    print(f"{state_name}: {len(all_records)} total, {len(filtered)} within 15 days "
          f"({cutoff_str} to {latest_str})")
    return filtered, latest_str, cutoff_str


def extract_jc_name(msg):
    """
    Extract JioCenter ID from message.
    JC IDs look like: GO-PNJI-JC01-0227, MH-PUNE-JC02-0001, etc.
    Also handles plain names like MUMBAI, PUNE after 'for'/'at'.
    """
    # Priority 1: structured JC ID pattern (XX-XXXX-JC##-####)
    jc_id_match = re.search(
        r"\b([A-Z]{2,3}-[A-Z0-9]{2,8}-JC\d{2}-\d{4})\b",
        msg, re.IGNORECASE
    )
    if jc_id_match:
        return jc_id_match.group(1).strip().upper()

    # Priority 2: partial JC code like GO-PNJI or MH-PUNE (state-city prefix)
    jc_prefix_match = re.search(
        r"\b([A-Z]{2}-[A-Z]{2,6})\b",
        msg, re.IGNORECASE
    )
    if jc_prefix_match:
        val = jc_prefix_match.group(1).upper()
        # Exclude SAP ID patterns (those have ENB in them)
        if "ENB" not in msg.upper():
            return val

    # Priority 3: JC/JioCenter followed by a name (not a generic word)
    SKIP = {"level","data","report","records","all","entire","complete","full",
            "in","for","of","the","a","antenna","antennas","inventory","sites"}
    patterns = [
        r"\bjc\b.*?\b(?:at|for|of)\s+([A-Za-z0-9_\-]+)",
        r"\bjio\s*center\b.*?\b(?:at|for|of)\s+([A-Za-z0-9_\-]+)",
        r"\bjio\s*center\s+([A-Za-z0-9_\-]+)",
        r"\bjc\s+([A-Za-z0-9_\-]+)",
    ]
    for p in patterns:
        m = re.search(p, msg, re.IGNORECASE)
        if m:
            name = m.group(1).strip().upper()
            if name.lower() not in SKIP:
                return name
    return None


def resolve_state_from_jc(jc_name):
    """
    Auto-detect state from JC ID prefix.
    GO-PNJI-JC01-0227 -> state code 'GO' -> resolve to state name via code_map.
    Returns state_name string or None.
    """
    if not jc_name:
        return None
    # Extract first segment before '-' as state code
    parts = jc_name.split("-")
    if len(parts) >= 1:
        state_code = parts[0].upper()
        states = _resolve_states(state_code)
        if states:
            print(f"JC '{jc_name}' -> state_code '{state_code}' -> {states[0]}")
            return states[0]
    return None

def extract_jc_id_and_state(msg):
    """
    Extract structured JC ID like GO-PNJI-JC01-0227 from message.
    Also infers state from the 2-letter prefix (e.g. GO -> GOA, MH -> MAHARASHTRA).
    Returns (jc_id, inferred_state) or (None, None).
    """
    jc_id_pattern = r"\b([A-Z]{2}-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+)\b"
    m = re.search(jc_id_pattern, msg.upper())
    if m:
        jc_id = m.group(1)
        prefix = jc_id.split("-")[0]
        inferred_state = STATE_CODE_MAP.get(prefix)
        return jc_id, inferred_state
    return None, None


def is_jc_data_query(msg):
    """
    Detect queries requesting JioCenter level data download.
    Examples: "JC level report for Mumbai", "download JC data for Pune",
              "JioCenter report for NAGPUR", "get JC DELHI data"
    """
    lower = msg.lower()
    if re.search(r"\b(why|reason|explain|cause|what is|what are|how|tell me about|describe)\b", lower):
        return False
    has_jc = bool(re.search(r"\b(jc|jiocenter|jio center)\b", lower))
    if not has_jc:
        return False
    download_intent = bool(re.search(
        r"\b(show|get|download|give|list|export|fetch|find|extract|pull|share|provide|report|data|records)\b",
        lower
    ))
    return download_intent


def get_jc_data_15days(state_name, jc_name):
    """
    Extract records for a specific JioCenter within a state, filtered to latest 15 days.
    For structured IDs (GO-PNJI-JC01-0227): exact match on JioCenter column.
    For plain names (MUMBAI, PUNE): partial case-insensitive match.
    """
    sap_map = _load_state_sap(state_name)
    all_records = []
    is_structured_id = bool(re.match(r"^[A-Z]{2,3}-[A-Z0-9]", jc_name, re.IGNORECASE))
    for records in sap_map.values():
        for r in records:
            jc_val = str(r.get("JioCenter", "")).strip().upper()
            if is_structured_id:
                match = (jc_val == jc_name or jc_name in jc_val)
            else:
                match = (jc_name in jc_val or jc_val in jc_name)
            if match:
                row = dict(r)
                row["State"] = state_name
                all_records.append(row)

    print(f"JC {jc_name} in {state_name}: {len(all_records)} records matched")
    if not all_records:
        return [], None, None

    # Apply 15-day filter
    all_dates = [
        parse_date(r.get("RRH Last Updated Time", ""))
        for r in all_records
        if parse_date(r.get("RRH Last Updated Time", ""))
    ]
    if not all_dates:
        return all_records, None, None

    latest_date = max(all_dates)
    cutoff_date = latest_date - timedelta(days=15)
    filtered = [
        r for r in all_records
        if parse_date(r.get("RRH Last Updated Time", "")) and
           parse_date(r.get("RRH Last Updated Time", "")) >= cutoff_date
    ]
    filtered.sort(key=lambda x: (
        x.get("SAP ID", ""),
        int(x.get("Sector Id", 0)) if str(x.get("Sector Id", "")).isdigit() else 999
    ))
    print(f"JC {jc_name}: {len(all_records)} total -> {len(filtered)} within 15 days")
    return filtered, latest_date.strftime("%Y-%m-%d"), cutoff_date.strftime("%Y-%m-%d")


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
            ret_board = str(r.get("RET Connect Board ID", "")).strip()
            ret_port  = str(r.get("RET Connect Port ID",  "")).strip()
            ant_class = str(r.get("Antenna Classification","")).strip()
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


def _groq_direct(system_prompt, history, user_msg):
    """Direct Groq API call via urllib — fallback when LangChain unavailable."""
    import ssl
    ctx = ssl.create_default_context()
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})
    payload = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.0
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            GROQ_URL, data=payload,
            headers={"Authorization": f"Bearer {GROQ_KEY}",
                     "Content-Type": "application/json",
                     "User-Agent": "python-urllib/3.11"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Summary unavailable: {e}"


def ask_langchain(question, docs, history, lang=None):
    """
    LangChain-powered RAG with automatic fallback to direct Groq API.
    - Tries ChatGroq (LangChain) first
    - Falls back to urllib Groq call if LangChain unavailable
    - Zero cost either way (Groq free tier)
    """
    if not docs:
        return "No matching records found in the database for this query."

    context = build_context(docs)
    lang_instruction = f" Respond in {lang}." if lang else " Respond in English."
    system  = (
        "You are a Jio antenna inventory assistant. "
        "Answer ONLY using the database records provided below. "
        "Do NOT invent, assume, or add any information not present in the data. "
        "If the data does not contain enough information to answer, say so clearly. "
        "Be concise: 2-4 lines max. No bullet points." + lang_instruction + "\n\n"
        "Database records:\n" + context
    )

    llm = _get_llm()

    if llm is not None:
        # ── LangChain path ──
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            messages = [SystemMessage(content=system)]
            for h in history:
                role = h.get("role", "")
                content = h.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=question))
            response = llm.invoke(messages)
            print("LangChain RAG response OK")
            return response.content
        except Exception as e:
            print(f"LangChain invoke error ({e}), falling back to urllib")

    # ── Direct urllib fallback ──
    print("Using direct Groq urllib fallback")
    return _groq_direct(system, history, question)


# ════════════════════════════════════════════════════════════
#  GENERAL QUESTION DETECTOR
# ════════════════════════════════════════════════════════════

def is_greeting(msg):
    """Detect greetings and chitchat that need no data lookup at all."""
    lower = msg.strip().lower()
    greetings = [
        "hi", "hello", "hey", "hii", "helo", "howdy", "namaste",
        "good morning", "good afternoon", "good evening", "good night",
        "thanks", "thank you", "ok", "okay", "bye", "goodbye",
        "who are you", "what can you do", "help me", "help",
    ]
    # Exact match or starts with greeting word
    if lower in greetings:
        return True
    if any(lower.startswith(g + " ") or lower == g for g in greetings):
        return True
    return False


def ask_greeting(msg, history, lang=None):
    """Respond to greetings/chitchat without any data lookup."""
    lang_instruction = f" Respond in {lang}." if lang else " Respond in English."
    system = (
        "You are a Jio antenna inventory assistant chatbot. "
        "Respond warmly and briefly to greetings or chitchat. "
        "Introduce yourself if it is the first message. "
        "Tell the user they can: look up SAP IDs, query alarm data by state, "
        "download blank RET reports, or ask general antenna questions. "
        "Keep response to 2-3 lines max." + lang_instruction
    )
    llm = _get_llm()
    if llm is not None:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            messages = [SystemMessage(content=system)]
            for h in history:
                if h.get("role") == "user":
                    messages.append(HumanMessage(content=h["content"]))
                elif h.get("role") == "assistant":
                    messages.append(AIMessage(content=h["content"]))
            messages.append(HumanMessage(content=msg))
            return llm.invoke(messages).content
        except Exception as e:
            print(f"LangChain greeting error: {e}")
    return _groq_direct(system, history, msg)


def is_general_question(msg):
    """
    Detect questions that are general knowledge / explanatory —
    these do NOT need FAISS search. Answer directly from LLM.
    Examples: "why is RET blank", "what is RET", "explain antenna classification"
    """
    lower = msg.lower()

    # First: if query contains a SAP ID or state+data keyword, treat as data query
    if re.search(r"I-[A-Z]{2}-[A-Z0-9]+-ENB-[0-9A-Z]+", msg.upper()):
        return False
    data_keywords = ["show data", "show records", "get data", "fetch data",
                     "list sites", "find sites", "alarm data", "sites in"]
    if any(kw in lower for kw in data_keywords):
        return False

    general_patterns = [
        r"\bwhy\b",
        r"\bhow\b",                         # catches: how this, how can, how to
        r"\bwhat is\b", r"\bwhat are\b", r"\bwhat can\b", r"\bwhat should\b",
        r"\bexplain\b",
        r"\bwhat does\b", r"\btell me about\b", r"\bdescribe\b",
        r"\bmeaning of\b", r"\bdefinition\b", r"\bpurpose of\b",
        r"\breason\b", r"\bcause\b",
        r"\baction\b", r"\bactions\b",       # "necessary actions"
        r"\bsteps\b", r"\bprocedure\b",      # "steps to fix"
        r"\bcan appear\b", r"\bcan happen\b",# "how this can appear"
        r"\bshould be\b", r"\bneeded\b",
    ]
    return any(re.search(p, lower) for p in general_patterns)


def ask_general(question, history, lang=None):
    """
    Answer general/explanatory questions directly from LLM —
    no FAISS needed, no S3 loads, fast response.
    Uses LangChain if available, falls back to direct Groq urllib.
    """
    lang_instruction = f" Respond in {lang}." if lang else " Respond in English."
    system = (
        "You are a Jio telecom antenna inventory expert. "
        "Answer the question clearly and concisely based on your knowledge "
        "of telecom antenna systems, RET (Remote Electrical Tilt), "
        "RRH (Remote Radio Head), antenna classifications, and Jio network. "
        "Be concise: 3-5 lines max. No bullet points." + lang_instruction
    )

    llm = _get_llm()

    if llm is not None:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            messages = [SystemMessage(content=system)]
            for h in history:
                role = h.get("role", "")
                content = h.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=question))
            response = llm.invoke(messages)
            print("LangChain general response OK")
            return response.content
        except Exception as e:
            print(f"LangChain general error ({e}), falling back to urllib")

    # Direct urllib fallback
    print("Using direct Groq urllib for general question")
    return _groq_direct(system, history, question)


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

    # ── Detect & translate Indian regional language to English ──
    original_msg, detected_lang = detect_and_translate(msg)
    if detected_lang:
        msg = original_msg   # Use English translation for all routing/search

    try:
        sap_id      = extract_sap_id(msg)
        state_name  = extract_state_name(msg)
        jc_name     = extract_jc_name(msg)
        blank_ret   = is_blank_ret_query(msg)
        alarm_query = (not blank_ret) and is_alarm_query(msg)
        jc_query    = (not blank_ret and not alarm_query) and is_jc_data_query(msg) and bool(jc_name)
        state_data  = (not blank_ret and not alarm_query and not jc_query) and is_state_data_query(msg) and bool(state_name)
        greeting    = is_greeting(msg)
        general_q   = (not greeting) and is_general_question(msg)

        # ── Greeting / chitchat ────────────────────────────────────────────
        if greeting:
            reply = ask_greeting(msg, history, lang=detected_lang)

            h2 = (history + [
                {"role": "user",      "content": msg},
                {"role": "assistant", "content": reply}
            ])[-6:]
            return {"statusCode": 200, "body": json.dumps({
                "summary": reply, "records": [], "columns": DISPLAY_COLUMNS,
                "retrieved": 0, "history": h2, "download": False
            })}

        # ── Blank RET download ──────────────────────────────────────────────
        elif blank_ret:
            all_states_requested = bool(re.search(
                r"all.{0,10}state|every.{0,10}state|pan.{0,5}india|all.{0,10}circle",
                msg, re.IGNORECASE
            ))
            if all_states_requested:
                import gc
                code_map = _load_code_map()
                all_state_files = sorted(set(
                    s for v in code_map.values()
                    for s in (v if isinstance(v, list) else [v])
                ))

                tmp_path  = "/tmp/blank_ret_all.csv"
                total_rows = 0

                with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
                    writer.writeheader()

                    for state_file in all_state_files:
                        # Load, filter, write, then CLEAR from cache immediately
                        sap_map = _load_state_sap(state_file)
                        state_rows = 0
                        for records in sap_map.values():
                            for r in records:
                                ret_board = str(r.get("RET Connect Board ID","")).strip()
                                ret_port  = str(r.get("RET Connect Port ID", "")).strip()
                                ant_class = str(r.get("Antenna Classification","")).strip()
                                if ret_board == "-" and ret_port == "-" and ant_class.upper() == "RET":
                                    row = {c: r.get(c,"") for c in COLUMNS}
                                    row["State"] = state_file
                                    writer.writerow(row)
                                    state_rows += 1
                        total_rows += state_rows
                        # Critical: evict from cache to free memory
                        _sap_cache.pop(state_file, None)
                        gc.collect()
                        print(f"All-states: {state_file} -> {state_rows} rows | total so far: {total_rows:,}")

                if total_rows == 0:
                    return {"statusCode": 200, "body": json.dumps({
                        "summary": "No blank RET records found across all states.",
                        "records": [], "columns": COLUMNS, "retrieved": 0,
                        "history": history, "download": False
                    })}

                # Upload /tmp file directly (no in-memory buffer)
                csv_key = BLANK_RET_KEY
                s3.upload_file(tmp_path, BUCKET, csv_key)
                import os; os.remove(tmp_path)

                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": BUCKET, "Key": csv_key},
                    ExpiresIn=3600
                )
                print(f"All-states CSV uploaded: {total_rows:,} rows -> {csv_key}")
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"Blank RET data for all states ready — {total_rows:,} records across {len(all_state_files)} states. Click to download.",
                    "records": [], "columns": COLUMNS, "retrieved": total_rows,
                    "history": history, "download": False,
                    "presigned_url": url, "filename": "Blank_RET_All_States.csv"
                })}
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

        # ── Alarm records download ──────────────────────────────────────────
        elif alarm_query:
            all_states_requested = bool(re.search(
                r"all.{0,10}state|every.{0,10}state|pan.{0,5}india|all.{0,10}circle",
                msg, re.IGNORECASE
            ))
            if all_states_requested:
                import gc
                code_map = _load_code_map()
                all_state_files = sorted(set(
                    s for v in code_map.values()
                    for s in (v if isinstance(v, list) else [v])
                ))
                tmp_path   = "/tmp/alarm_all.csv"
                total_rows = 0
                with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
                    writer.writeheader()
                    for state_file in all_state_files:
                        sap_map = _load_state_sap(state_file)
                        state_rows = 0
                        for records in sap_map.values():
                            for r in records:
                                if (str(r.get("Alarm Status","")).strip() == "Y" and
                                        str(r.get("Alarm details","")).strip() != "-"):
                                    row = {c: r.get(c,"") for c in COLUMNS}
                                    row["State"] = state_file
                                    writer.writerow(row)
                                    state_rows += 1
                        total_rows += state_rows
                        _sap_cache.pop(state_file, None)
                        gc.collect()
                        print(f"Alarm all-states: {state_file} -> {state_rows} rows | total: {total_rows:,}")

                if total_rows == 0:
                    return {"statusCode": 200, "body": json.dumps({
                        "summary": "No alarm records found across all states.",
                        "records": [], "columns": COLUMNS, "retrieved": 0,
                        "history": history, "download": False
                    })}
                alarm_all_key = "rag-exports/Alarm_All_States.csv"
                s3.upload_file(tmp_path, BUCKET, alarm_all_key)
                import os; os.remove(tmp_path)
                url = s3.generate_presigned_url(
                    "get_object", Params={"Bucket": BUCKET, "Key": alarm_all_key}, ExpiresIn=3600)
                print(f"Alarm all-states CSV: {total_rows:,} rows -> {alarm_all_key}")
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"Alarm records for all states ready — {total_rows:,} records across {len(all_state_files)} states. Click to download.",
                    "records": [], "columns": COLUMNS, "retrieved": total_rows,
                    "history": history, "download": False,
                    "presigned_url": url, "filename": "Alarm_All_States.csv"
                })}

            elif state_name:
                docs     = get_alarm_records(state_name)
                label    = state_name
                filename = f"Alarm_{state_name}.csv"
            else:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": "Please specify a state or 'all states'. E.g. 'show alarm records for Maharashtra'",
                    "records": [], "columns": COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}

            if not docs:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"No alarm records found for {label}.",
                    "records": [], "columns": COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}

            csv_key = f"exports/Alarm_{label}.csv"
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in docs:
                writer.writerow({c: row.get(c, "") for c in COLUMNS})
            s3.put_object(Bucket=BUCKET, Key=csv_key,
                          Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
            presigned_url = s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": csv_key}, ExpiresIn=3600)
            print(f"Alarm CSV saved: {csv_key}, {len(docs)} rows")
            return {"statusCode": 200, "body": json.dumps({
                "summary": f"Found {len(docs):,} alarm records for {label}. Click the link to download.",
                "records": [], "columns": COLUMNS, "retrieved": len(docs),
                "history": history, "download": False,
                "presigned_url": presigned_url, "filename": filename
            })}

        # ── JioCenter level data download (latest 15 days) ─────────────────
        elif jc_query:
            # Auto-resolve state from JC ID prefix (GO-PNJI-JC01-0227 -> GO -> GOA)
            resolved_state = state_name or resolve_state_from_jc(jc_name)
            if not resolved_state:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"Could not determine state for JC '{jc_name}'. "
                               f"Please include the state name, e.g. 'JC {jc_name} Goa'.",
                    "records": [], "columns": COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}
            docs, latest_str, cutoff_str = get_jc_data_15days(resolved_state, jc_name)
            if not docs:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"No records found for JioCenter '{jc_name}' in {resolved_state}. "
                               f"Please verify the JC ID.",
                    "records": [], "columns": COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}
            safe_jc = re.sub(r"[^A-Za-z0-9_\-]", "_", jc_name)
            csv_key = f"exports/JC_{safe_jc}_{resolved_state}_15days.csv"
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in docs:
                writer.writerow({c: row.get(c, "") for c in COLUMNS})
            s3.put_object(Bucket=BUCKET, Key=csv_key,
                          Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
            presigned_url = s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": csv_key}, ExpiresIn=3600)
            print(f"JC CSV: {csv_key}, {len(docs)} rows")
            return {"statusCode": 200, "body": json.dumps({
                "summary": f"Downloaded {len(docs):,} records for JioCenter '{jc_name}' "
                           f"in {resolved_state} (latest 15 days). Click to download.",
                "records": [], "columns": COLUMNS, "retrieved": len(docs),
                "history": history, "download": False,
                "presigned_url": presigned_url,
                "filename": f"JC_{safe_jc}_{resolved_state}_15days.csv"
            })}

        # ── Full state data download (latest 15 days) ──────────────────────
        elif state_data:
            docs, latest_str, cutoff_str = get_state_data_15days(state_name)
            if not docs:
                return {"statusCode": 200, "body": json.dumps({
                    "summary": f"No records found for {state_name}.",
                    "records": [], "columns": COLUMNS, "retrieved": 0,
                    "history": history, "download": False
                })}
            # Write CSV to S3
            csv_key = f"exports/StateData_{state_name}_15days.csv"
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in docs:
                writer.writerow({c: row.get(c, "") for c in COLUMNS})
            s3.put_object(Bucket=BUCKET, Key=csv_key,
                          Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
            presigned_url = s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": csv_key}, ExpiresIn=3600)
            print(f"State data CSV: {csv_key}, {len(docs)} rows")
            return {"statusCode": 200, "body": json.dumps({
                "summary": f"Downloaded {len(docs):,} records for {state_name} (latest 15 days). Click to download.",
                "records": [], "columns": COLUMNS, "retrieved": len(docs),
                "history": history, "download": False,
                "presigned_url": presigned_url,
                "filename": f"StateData_{state_name}_15days.csv"
            })}

        # ── SAP ID lookup ───────────────────────────────────────────────────
        elif sap_id:
            docs = filter_and_sort(lookup_sap(sap_id))
            if not docs:
                # Check if state code is simply unknown — give helpful message
                parts      = sap_id.upper().split("-")
                state_code = parts[1] if len(parts) > 1 else ""
                state_list = _resolve_states(state_code)
                if not state_list:
                    msg_out = (f"SAP ID {sap_id} contains unknown state code '{state_code}'. "
                               f"This state may not be in the current index. "
                               f"Please rebuild the index if this is a new state.")
                else:
                    msg_out = f"No records found for SAP ID {sap_id}."
                return {"statusCode": 200, "body": json.dumps({
                    "summary": msg_out, "records": [], "columns": COLUMNS,
                    "retrieved": 0, "history": history, "download": False
                })}
            return {"statusCode": 200, "body": json.dumps({
                "summary": "", "records": docs, "columns": COLUMNS,
                "retrieved": len(docs), "history": history, "download": False
            })}

        # ── General knowledge question (no FAISS needed) ───────────────────
        elif general_q:
            summary = ask_general(msg, history, lang=detected_lang)
            h2 = (history + [
                {"role": "user",      "content": msg},
                {"role": "assistant", "content": summary}
            ])[-6:]
            return {"statusCode": 200, "body": json.dumps({
                "summary":   summary,
                "records":   [],
                "columns":   DISPLAY_COLUMNS,
                "retrieved": 0,
                "history":   h2,
                "download":  False
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

            # ── Post-filter 1: state filter if state mentioned in query ──
            if state_name:
                # state_name is S3 format e.g. MAHARASHTRA — match against State column
                state_filtered = [
                    r for r in docs
                    if state_name.replace("_", " ").upper() in
                       r.get("State", "").upper().replace("_", " ")
                ]
                if state_filtered:
                    print(f"State filter '{state_name}': {len(docs)} -> {len(state_filtered)} records")
                    docs = state_filtered
                else:
                    print(f"State filter '{state_name}' matched 0 — keeping all {len(docs)} records")

            # ── Post-filter 2: alarm filter if query mentions alarm/alert ──
            alarm_query = bool(re.search(
                r"\b(alarm|alert|fault|error|fail|warning|vswr|ald|ret not calibrated)\b",
                msg, re.IGNORECASE
            ))
            if alarm_query:
                alarm_filtered = [
                    r for r in docs
                    if r.get("Alarm details", "").strip() not in ("", "-", "N/A", "null", "None")
                ]
                if alarm_filtered:
                    print(f"Alarm filter: {len(docs)} -> {len(alarm_filtered)} records")
                    docs = alarm_filtered
                else:
                    print(f"Alarm filter matched 0 — keeping all {len(docs)} records")

            summary = ask_langchain(msg, docs, history, lang=detected_lang)
            h2 = (history + [
                {"role": "user",      "content": msg},
                {"role": "assistant", "content": summary}
            ])[-6:]
            return {"statusCode": 200, "body": json.dumps({
                "summary":   summary,
                "records":   [],
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
