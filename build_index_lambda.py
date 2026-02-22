# build_index_lambda.py
# Reservoir sampling — each bucket capped at fixed size from the start
# Never accumulates more than MAX_ROWS rows in RAM at any time
# Peak RAM: ~500MB for 200K rows

import boto3, json, faiss, numpy as np, pickle
import gc, re, math, hashlib, io, random
from collections import Counter

BUCKET      = "chatbot-input-database"
BASE_PREFIX = "processed/"
INDEX_KEY   = "rag-index/faiss.index"
IDF_KEY     = "rag-model/idf.pkl"
META_KEY    = "rag-index/metadata.pkl"

MAX_ROWS  = 200000   # total rows to index
DIM       = 384
IDF_TOP   = 30000

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
    return [w for w in re.sub(r'[^a-z0-9\s]',' ',text.lower()).split() if len(w)>1]


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

    # ── Reservoir sampling — fixed-size buckets ───────────────────────────────
    # Bucket A: rows WITH alarms    (cap: 150K)
    # Bucket B: rows WITHOUT alarms (cap:  50K)
    # Using reservoir sampling so bucket never exceeds cap
    # RAM = cap_A × row_size + cap_B × row_size ← fixed, never grows

    CAP_A = 150000   # alarm rows
    CAP_B =  50000   # non-alarm rows

    bucket_a  = []   # alarm rows     — reservoir
    bucket_b  = []   # no-alarm rows  — reservoir
    count_a   = 0    # total alarm rows seen
    count_b   = 0    # total non-alarm rows seen
    doc_freq  = Counter()
    total     = 0

    print("Pass 1: reservoir sampling + IDF counting...")

    for row in iter_all_rows(idx_meta):
        total += 1
        text  = row_to_text(row)

        # IDF counting — stream, no accumulation
        doc_freq.update(set(tokenize(text)))

        entry = {c: row.get(c,"") for c in COLUMNS}
        entry["_text"] = text

        if has_alarm(row):
            count_a += 1
            # Reservoir sampling for bucket A
            if len(bucket_a) < CAP_A:
                bucket_a.append(entry)
            else:
                # Replace random element with decreasing probability
                r = random.randint(0, count_a - 1)
                if r < CAP_A:
                    bucket_a[r] = entry
        else:
            count_b += 1
            if len(bucket_b) < CAP_B:
                bucket_b.append(entry)
            else:
                r = random.randint(0, count_b - 1)
                if r < CAP_B:
                    bucket_b[r] = entry

        if total % 300000 == 0:
            print(f"  Scanned {total:,} | "
                  f"alarm_bucket={len(bucket_a):,}/{CAP_A} "
                  f"other_bucket={len(bucket_b):,}/{CAP_B}")
            gc.collect()

    print(f"Scan done: {total:,} rows | "
          f"alarm rows: {count_a:,} | other: {count_b:,}")
    print(f"Sampled: {len(bucket_a):,} alarm + {len(bucket_b):,} other "
          f"= {len(bucket_a)+len(bucket_b):,} total")

    # Merge buckets
    selected = bucket_a + bucket_b
    del bucket_a, bucket_b
    gc.collect()
    print(f"Selected {len(selected):,} rows for indexing")

    # ── Build IDF ─────────────────────────────────────────────────────────────
    print("Building IDF...")
    N   = total
    idf = {
        w: math.log((N+1)/(c+1))+1.0
        for w,c in doc_freq.most_common(IDF_TOP)
    }
    del doc_freq
    gc.collect()

    buf = io.BytesIO()
    pickle.dump(idf, buf)
    s3.put_object(Bucket=BUCKET, Key=IDF_KEY, Body=buf.getvalue())
    print(f"IDF saved: {len(idf):,} terms")

    # ── Build FAISS — train then add in batches ───────────────────────────────
    print("Pass 2: embedding + building FAISS index...")

    nlist     = 200
    quantizer = faiss.IndexFlatL2(DIM)
    faiss_idx = faiss.IndexIVFFlat(quantizer, DIM, nlist)
    metadata  = []
    BATCH     = 1000

    # Train on a 10K sample
    TRAIN_N    = min(10000, len(selected))
    print(f"Training IVF on {TRAIN_N:,} vectors...")
    train_vecs = np.array(
        [text_to_vec(r["_text"], idf) for r in selected[:TRAIN_N]],
        dtype=np.float32
    )
    faiss_idx.train(train_vecs)
    del train_vecs
    gc.collect()
    print("IVF trained")

    # Add in batches
    for start in range(0, len(selected), BATCH):
        batch = selected[start : start+BATCH]
        vecs  = np.array(
            [text_to_vec(r["_text"], idf) for r in batch],
            dtype=np.float32
        )
        faiss_idx.add(vecs)
        for r in batch:
            metadata.append({c: r.get(c,"") for c in COLUMNS})
        del vecs, batch
        gc.collect()
        if (start // BATCH) % 50 == 0:
            done = min(start+BATCH, len(selected))
            print(f"  Embedded {done:,}/{len(selected):,}")

    faiss_idx.nprobe = 20
    print(f"FAISS built: {faiss_idx.ntotal:,} vectors")

    # ── Save to S3 ────────────────────────────────────────────────────────────
    print("Saving FAISS index...")
    faiss.write_index(faiss_idx, "/tmp/faiss.index")
    s3.upload_file("/tmp/faiss.index", BUCKET, INDEX_KEY)
    del faiss_idx
    gc.collect()

    print("Saving metadata...")
    buf = io.BytesIO()
    pickle.dump(metadata, buf)
    s3.put_object(Bucket=BUCKET, Key=META_KEY, Body=buf.getvalue())
    del metadata
    gc.collect()

    s3.put_object(
        Bucket=BUCKET, Key="rag-index/manifest.json",
        Body=json.dumps({
            "total_scanned": total,
            "alarm_rows_seen": count_a,
            "other_rows_seen": count_b,
            "indexed": len(selected)
        }).encode()
    )

    return {
        "statusCode": 200,
        "body": (f"Scanned {total:,} rows. "
                 f"Indexed {len(selected):,} rows "
                 f"({count_a:,} had alarms, sampled 150K of those).")
    }
