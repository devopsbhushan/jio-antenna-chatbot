# 📡 Jio Antenna RAG Chatbot

An intelligent chatbot for querying Jio's antenna inventory across all Indian states. Built on a fully serverless AWS architecture using FAISS vector search, Groq Llama 3.1, and LangChain — at near-zero cost.

---

## 🏗️ Architecture

```
User (Browser)
      │
      ▼
index.html  (S3 Static UI)
      │  HTTP POST
      ▼
antenna-chatbot_Lambda  (AWS Lambda · Docker · 3008MB · 300s)
      │
      ├── SAP Lookup      → S3: rag-index/sap/<STATE>.pkl
      ├── FAISS Search    → S3: rag-index/faiss.index
      ├── LLM Response    → Groq API (Llama 3.1 8B)
      └── CSV Export      → S3: exports/*.csv  →  Presigned URL
```

### Data Pipeline

```
S3: input/ (Excel/CSV files)
      │
      ▼
[Step 1] Preprocessor Lambda
      │  Parses Excel → JSON chunks per state
      │  Writes → S3: processed/<STATE>/chunk_XXXX.json
      │
      ▼
[Step 2] Build_FAISS_Index Lambda
      │  Builds per-state SAP maps  → S3: rag-index/sap/<STATE>.pkl
      │  Builds FAISS vector index  → S3: rag-index/faiss.index
      │  Builds IDF weights         → S3: rag-model/idf.pkl
      │  Builds state code map      → S3: rag-index/sap/code_map.json
      │
      ▼
[Step 3] Chatbot Lambda
      Auto-detects new data via S3 ETag comparison (no restart needed)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **SAP ID Lookup** | Instant full record lookup by SAP ID |
| **State Data Download** | All antenna records for a state (latest 15 days) |
| **JC Level Report** | Records filtered by JioCenter ID (e.g. `GO-PNJI-JC01-0227`) |
| **Alarm Records** | Sites with active alarms — single state or all states |
| **Blank RET Report** | RET-class antennas with missing board/port — single state or all states |
| **Semantic Search** | Natural language queries via FAISS + Groq LLM |
| **Regional Languages** | Auto-detects and translates 12 Indian languages |
| **Auto Cache Refresh** | Detects new data via S3 ETag — no manual restart needed |
| **15-Day Filter** | All downloads consistently filtered to latest 15 days |

---

## 🗂️ Repository Structure

```
├── chatbot_lambda.py          # Main chatbot Lambda (routing, search, download)
├── build_index_lambda.py      # FAISS index builder Lambda
├── build_blank_ret_lambda.py  # Standalone blank RET CSV builder
├── build_sap_map_lambda.py    # SAP map builder (legacy)
├── index.html                 # Frontend UI (auto-deployed to S3)
├── build_layer.yml            # GitHub Actions deployment workflow
└── README.md                  # This file
```

---

## ☁️ AWS Resources

| Resource | Name | Purpose |
|---|---|---|
| Lambda | `antenna-chatbot_Lambda` | Main chatbot handler |
| Lambda | `process_antenna_data` | Parses input files → JSON chunks |
| Lambda | `Build_FAISS_Index` | Rebuilds FAISS index from chunks |
| ECR | `antenna-chatbot` | Docker image for chatbot |
| ECR | `antenna-faiss-index` | Docker image for index builder |
| S3 | `chatbot-input-database` | All data storage |
| S3 path | `input/` | Raw Excel/CSV uploads |
| S3 path | `processed/` | JSON chunks (preprocessor output) |
| S3 path | `rag-index/` | FAISS index + pkl files |
| S3 path | `rag-model/` | IDF weights |
| S3 path | `exports/` | Generated CSV downloads |
| S3 path | `ui/index.html` | Frontend UI |

---

## 🚀 Deployment

### Prerequisites — GitHub Secrets

Set these in **Settings → Secrets → Actions**:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `ap-south-1` |
| `S3_BUCKET` | e.g. `chatbot-input-database` |
| `GROQ_API_KEY` | From [console.groq.com](https://console.groq.com) |

### Deploy Code Changes

```
GitHub → Actions → Deploy Lambda (Docker) → Run workflow
```

This builds Docker images, pushes to ECR, updates both Lambda functions, and auto-patches `index.html` with the live Lambda URL.

---

## 🔄 Updating Data (New Input Files)

Follow this exact order every time:

```
1. Upload new Excel/CSV files
   → S3: chatbot-input-database/input/

2. Run Preprocessor Lambda
   AWS Console → Lambda → process_antenna_data → Test → {}
   Wait: ~15-20 minutes

3. Run Build_FAISS_Index Lambda
   AWS Console → Lambda → Build_FAISS_Index → Test → {}
   Wait: ~15 minutes

4. Done ✅
   Chatbot auto-detects new data on next query (ETag check)
   No restart or redeployment needed
```

> ⚠️ **Common mistake:** Running only Step 3 without Step 2 will rebuild indexes from **old chunks** and show stale data.

---

## 💬 Supported Query Types

### SAP ID Lookup
```
Show antenna at SAP ID I-MH-MUMB-ENB-I001
I-UE-AAIT-ENB-9001
```

### State Level Download
```
Download antenna data for Maharashtra state
Get all data for Goa
```

### JC Level Report
```
JC report for GO-PNJI-JC01-0227
JC level data for Mumbai Maharashtra
```

### Alarm Records
```
Alarm records for Maharashtra
Show alarm data for all states
```

### Blank RET Report
```
Blank RET data for Maharashtra
Download blank RET for all states
```

### General / Semantic
```
What is RET in antenna systems?
Explain antenna classification
What does Alarm Status Y mean?
```

### Indian Regional Languages
Queries in Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Punjabi, Bengali, Odia, Assamese, and Urdu are automatically detected, translated to English for search, and responded to in the original language.

---

## 🏷️ 15-Day Data Filter

All CSV downloads apply a consistent **latest 15-day filter**:

- Finds the maximum `RRH Last Updated Time` across all records
- Keeps only records within 15 days of that maximum date
- Applied independently per state for all-states downloads
- Ensures downloaded data matches what is shown in the UI summary

---

## 💰 Cost Breakdown

| Component | Service | Cost |
|---|---|---|
| LLM inference | Groq free tier (14,400 req/day) | $0 |
| Vector search | FAISS (open source) | $0 |
| Compute | Lambda free tier (1M req + 400K GB-sec) | ~$0 |
| Storage | S3 (~23GB) | ~$0.53/month |
| CI/CD | GitHub Actions (2,000 min/month free) | $0 |
| **Total** | | **~$0.53/month** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq · Llama 3.1 8B Instant |
| Orchestration | LangChain (with urllib fallback) |
| Vector Search | FAISS (IndexIVFFlat, 384 dims) |
| Runtime | AWS Lambda · Python 3.11 · Docker |
| Storage | Amazon S3 |
| Frontend | Vanilla HTML/JS (S3 static hosting) |
| CI/CD | GitHub Actions |

---

## 🔧 Lambda Configuration

| Lambda | Memory | Timeout | Trigger |
|---|---|---|---|
| `antenna-chatbot_Lambda` | 3008 MB | 300 sec | Function URL (public) |
| `Build_FAISS_Index` | 3008 MB | 900 sec | Manual test event `{}` |

---

## 📊 Data Scale

- **Total rows:** ~3,090,000+
- **States:** 28
- **Unique SAP IDs:** ~130,000+
- **FAISS index size:** ~134,000 vectors (reservoir sampled)
- **State code map:** 22 codes → 28 state files

---

## 🐛 Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Chatbot shows old data after upload | Preprocessor Lambda not run | Run Step 1 → Step 2 |
| SAP ID not found | Wrong state code in SAP ID | Check `code_map.json` in S3 |
| JC report returns no records | JC ID not matching `JioCenter` column | Verify exact JC ID format |
| Download CSV is empty | All records outside 15-day window | Check `RRH Last Updated Time` in source data |
| Lambda timeout on all-states | Memory pressure | Already uses streaming write — check CloudWatch for OOM |
| `Cache invalidated` not appearing | Lambda still warm with old ETag | Wait 15 min or save Lambda config to force cold start |

---

## 📝 CloudWatch Log Signals

```bash
# Healthy query
Cache valid — no S3 changes detected
SAP I-MH-MUMB-ENB-I001: code=MH -> states=['GOA', 'MAHARASHTRA', 'MUMBAI']
Loaded SAP map MAHARASHTRA: 14230 SAP IDs
filter_and_sort: 9 records, latest=2026-03-18, cutoff=2026-03-03, after_filter=9

# After new data rebuild
Cache invalidated — new data detected. faiss="abc123...", meta="def456...", cmap="ghi789..."

# 15-day filter applied on download
apply_15day_filter: 45230 -> 12840 records (latest=2026-03-18, cutoff=2026-03-03)
```
