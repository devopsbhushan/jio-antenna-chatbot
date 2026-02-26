import boto3, json, faiss, numpy as np, pickle
import gc, re, math, hashlib, io, random
from collections import Counter

BUCKET      = "chatbot-input-database"
BASE_PREFIX = "processed/"
INDEX_KEY   = "rag-index/faiss.index"
IDF_KEY     = "rag-model/idf.pkl"
META_KEY    = "rag-index/metadata.pkl"
SAP_PREFIX  = "rag-index/sap/"   # sap/MAHARASHTRA.pkl, sap/DELHI.pkl ...

MAX_ROWS = 200000
DIM      = 384
IDF_TOP  = 30000
CAP_A    = 150000
CAP_B    =  50000

COLUMNS = [
    "State","SAP ID","Sector Id","Bands",
    "RRH Connect Board ID","RRH Connect Port ID",
    "SF Antenna Model","LSMR Antenna Type",
    "Antenna Classification","RRH Last Updated Time",
    "Alarm details"
]

s3 = boto3.client("s3")

def load_json(key):
    return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())

def row_to_text(row):
    return (
        f"SAP ID {row.get('SAP ID','')} in {row.get('State','')}. "
        f"Sector {row.get('Sector Id','')} band {row.get('Bands','')}. "
        f"Antenna: {row.get('LSMR Antenna Type','')}. "
        f"Alarm: {row.get('Alarm details','None')}."
    )

def has_alarm(row):
    v = str(row.get("Alarm details","")).strip().lower()
    return v not in ("","none","null","nan","-")

def tokenize(text):
    return [w for w in re.sub(r"[^a-z0-9\s]"," ",text.lower()).split() if len(w)>1]

def hash_tok(t):
    return int(hashlib.md5(t.encode()).hexdigest(),16) % DIM

def text_to_vec(text, idf):
    vec = np.zeros(DIM, dtype=np.float32)
    for t in tokenize(text):
        vec[hash_tok(t)] += idf.get(t, 1.0)
    n = np.linalg.norm(vec)
    if n > 1e-9: vec /= n
    return vec

def lambda_handler(event, context):
    def mins_left():
        return context.get_remaining_time_in_millis() / 60000

    idx_meta  = load_json(f"{BASE_PREFIX}index.json")
    bucket_a  = []
    bucket_b  = []
    count_a   = count_b = 0
    doc_freq  = Counter()
    total     = 0
    states_saved = 0

    print("Processing state by state...")

    for state, meta in idx_meta["states"].items():
        # Build SAP map for this state only
        state_sap = {}

        for i in range(1, meta["chunks"]+1):
            rows = load_json(f"{BASE_PREFIX}{state}/chunk_{i:04d}.json")
            for row in rows:
                total += 1
                row["State"] = state
                text = row_to_text(row)
                doc_freq.update(set(tokenize(text)))

                # SAP map for this state
                sap_id = row.get("SAP ID","").strip()
                if sap_id:
                    if sap_id not in state_sap:
                        state_sap[sap_id] = []
                    state_sap[sap_id].append({c: row.get(c,"") for c in COLUMNS})

                # FAISS reservoir sample
                entry = {c: row.get(c,"") for c in COLUMNS}
                entry["_text"] = text
                if has_alarm(row):
                    count_a += 1
                    if len(bucket_a) < CAP_A:
                        bucket_a.append(entry)
                    else:
                        r = random.randint(0, count_a-1)
                        if r < CAP_A: bucket_a[r] = entry
                else:
                    count_b += 1
                    if len(bucket_b) < CAP_B:
                        bucket_b.append(entry)
                    else:
                        r = random.randint(0, count_b-1)
                        if r < CAP_B: bucket_b[r] = entry

        # Save this state SAP map to S3 immediately — free RAM
        buf = io.BytesIO()
        pickle.dump(state_sap, buf)
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{SAP_PREFIX}{state}.pkl",
            Body=buf.getvalue()
        )
        states_saved += 1
        print(f"  Saved {state}: {len(state_sap)} SAP IDs | "
              f"total rows: {total:,} | time left: {mins_left():.1f} min")
        del state_sap, buf
        gc.collect()

    print(f"Pass 1 done: {total:,} rows, {states_saved} states saved")
    print(f"Time left: {mins_left():.1f} min")

    # Build IDF
    N   = total
    idf = {w: math.log((N+1)/(c+1))+1.0 for w,c in doc_freq.most_common(IDF_TOP)}
    del doc_freq
    gc.collect()
    buf = io.BytesIO()
    pickle.dump(idf, buf)
    s3.put_object(Bucket=BUCKET, Key=IDF_KEY, Body=buf.getvalue())
    print(f"IDF saved: {len(idf):,} terms | time left: {mins_left():.1f} min")

    # Build FAISS
    selected = (bucket_a + bucket_b)[:MAX_ROWS]
    del bucket_a, bucket_b
    gc.collect()
    print(f"Building FAISS on {len(selected):,} rows...")

    nlist     = 200
    quantizer = faiss.IndexFlatL2(DIM)
    faiss_idx = faiss.IndexIVFFlat(quantizer, DIM, nlist)
    BATCH     = 1000
    metadata  = []

    train_vecs = np.array(
        [text_to_vec(r["_text"],idf) for r in selected[:10000]],
        dtype=np.float32
    )
    faiss_idx.train(train_vecs)
    del train_vecs
    gc.collect()

    for start in range(0, len(selected), BATCH):
        batch = selected[start:start+BATCH]
        vecs  = np.array([text_to_vec(r["_text"],idf) for r in batch], dtype=np.float32)
        faiss_idx.add(vecs)
        for r in batch:
            metadata.append({c: r.get(c,"") for c in COLUMNS})
        del vecs, batch
        gc.collect()
        if (start//BATCH) % 50 == 0:
            print(f"  FAISS: {min(start+BATCH,len(selected)):,}/{len(selected):,} | "
                  f"time left: {mins_left():.1f} min")

    faiss_idx.nprobe = 20
    faiss.write_index(faiss_idx, "/tmp/faiss.index")
    s3.upload_file("/tmp/faiss.index", BUCKET, INDEX_KEY)
    del faiss_idx
    gc.collect()

    buf = io.BytesIO()
    pickle.dump(metadata, buf)
    s3.put_object(Bucket=BUCKET, Key=META_KEY, Body=buf.getvalue())
    del metadata
    gc.collect()

    print(f"All done! Time left: {mins_left():.1f} min")
    return {
        "statusCode": 200,
        "body": f"Done. {total:,} rows. {states_saved} state SAP maps saved. FAISS: {len(selected):,} rows."
    }
