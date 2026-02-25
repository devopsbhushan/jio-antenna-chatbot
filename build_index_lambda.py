# build_index_lambda.py
# Builds FAISS index + SAP ID lookup map for instant exact lookups

import boto3, json, faiss, numpy as np, pickle
import gc, re, math, hashlib, io, random
from collections import Counter, defaultdict

BUCKET      = "chatbot-input-database"
BASE_PREFIX = "processed/"
INDEX_KEY   = "rag-index/faiss.index"
IDF_KEY     = "rag-model/idf.pkl"
META_KEY    = "rag-index/metadata.pkl"
SAP_MAP_KEY = "rag-index/sap_map.pkl"   # NEW: SAP ID -> list of row dicts

MAX_ROWS  = 200000
DIM       = 384
IDF_TOP   = 30000
CAP_A     = 150000
CAP_B     =  50000

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
        f"Model: {row.get('SF Antenna Model','')}. "
        f"Alarm: {row.get('Alarm details','None')}. "
        f"Updated: {row.get('RRH Last Updated Time','')}"
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


def iter_all_rows(idx_meta):
    for state, meta in idx_meta["states"].items():
        for i in range(1, meta["chunks"]+1):
            rows = load_json(f"{BASE_PREFIX}{state}/chunk_{i:04d}.json")
            for row in rows:
                row["State"] = state
                yield row


def lambda_handler(event, context):
    idx_meta = load_json(f"{BASE_PREFIX}index.json")

    bucket_a  = []
    bucket_b  = []
    count_a   = 0
    count_b   = 0
    doc_freq  = Counter()
    total     = 0
    # SAP map: sap_id -> list of full row dicts (ALL rows, not just sampled)
    sap_map   = defaultdict(list)

    print("Pass 1: sampling + IDF + building SAP map...")

    for row in iter_all_rows(idx_meta):
        total += 1
        text  = row_to_text(row)
        doc_freq.update(set(tokenize(text)))

        entry = {c: row.get(c,"") for c in COLUMNS}
        entry["_text"] = text

        # Build SAP map for ALL rows (not just sampled)
        sap_id = row.get("SAP ID","").strip()
        if sap_id:
            sap_map[sap_id].append({c: row.get(c,"") for c in COLUMNS})

        # Reservoir sampling for FAISS
        if has_alarm(row):
            count_a += 1
            if len(bucket_a) < CAP_A:
                bucket_a.append(entry)
            else:
                r = random.randint(0, count_a-1)
                if r < CAP_A:
                    bucket_a[r] = entry
        else:
            count_b += 1
            if len(bucket_b) < CAP_B:
                bucket_b.append(entry)
            else:
                r = random.randint(0, count_b-1)
                if r < CAP_B:
                    bucket_b[r] = entry

        if total % 300000 == 0:
            print(f"  {total:,} rows | SAP IDs: {len(sap_map):,} | "
                  f"alarm_bucket={len(bucket_a):,} other={len(bucket_b):,}")
            gc.collect()

    print(f"Done: {total:,} rows | unique SAP IDs: {len(sap_map):,}")

    # Save SAP map to S3
    print("Saving SAP map to S3...")
    buf = io.BytesIO()
    pickle.dump(dict(sap_map), buf)
    s3.put_object(Bucket=BUCKET, Key=SAP_MAP_KEY, Body=buf.getvalue())
    print(f"SAP map saved: {len(sap_map):,} unique SAP IDs")
    del sap_map
    gc.collect()

    # Merge FAISS sample
    selected = (bucket_a + bucket_b)[:MAX_ROWS]
    del bucket_a, bucket_b
    gc.collect()
    print(f"FAISS sample: {len(selected):,} rows")

    # Build IDF
    N   = total
    idf = {w: math.log((N+1)/(c+1))+1.0 for w,c in doc_freq.most_common(IDF_TOP)}
    del doc_freq
    gc.collect()
    buf = io.BytesIO()
    pickle.dump(idf, buf)
    s3.put_object(Bucket=BUCKET, Key=IDF_KEY, Body=buf.getvalue())
    print(f"IDF saved: {len(idf):,} terms")

    # Build FAISS
    print("Building FAISS index...")
    nlist     = 200
    quantizer = faiss.IndexFlatL2(DIM)
    faiss_idx = faiss.IndexIVFFlat(quantizer, DIM, nlist)
    BATCH     = 1000
    metadata  = []

    TRAIN_N    = min(10000, len(selected))
    train_vecs = np.array([text_to_vec(r["_text"],idf) for r in selected[:TRAIN_N]], dtype=np.float32)
    print(f"Training IVF on {TRAIN_N:,} vectors...")
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
            print(f"  Embedded {min(start+BATCH,len(selected)):,}/{len(selected):,}")

    faiss_idx.nprobe = 20
    print(f"FAISS built: {faiss_idx.ntotal:,} vectors")

    faiss.write_index(faiss_idx, "/tmp/faiss.index")
    s3.upload_file("/tmp/faiss.index", BUCKET, INDEX_KEY)
    del faiss_idx
    gc.collect()

    buf = io.BytesIO()
    pickle.dump(metadata, buf)
    s3.put_object(Bucket=BUCKET, Key=META_KEY, Body=buf.getvalue())
    del metadata
    gc.collect()

    return {
        "statusCode": 200,
        "body": f"Indexed {len(selected):,} rows. SAP map: {total:,} rows across all SAP IDs."
    }
