# embedder.py
# Drop-in replacement for SentenceTransformer('all-MiniLM-L6-v2')
# Uses ONNX Runtime instead of PyTorch — ~8MB vs 1.7GB
# Same model, same 384-dim output vectors, fully compatible with existing FAISS index

import os
import json
import numpy as np
import urllib.request
import boto3
import onnxruntime as ort
from tokenizers import Tokenizer

# ── Model files stored in S3 (downloaded once on cold start) ─────────────────
BUCKET       = os.environ.get("BUCKET", "chatbot-input-database")
MODEL_PREFIX = "rag-model/"
MODEL_FILES  = ["model.onnx", "tokenizer.json"]
LOCAL_DIR    = "/tmp/minilm/"

# Public HuggingFace ONNX model URLs for all-MiniLM-L6-v2
HF_BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/"

s3 = boto3.client("s3")

_session   = None
_tokenizer = None


def _download_models():
    """Download ONNX model files from S3 (or HuggingFace on first ever run)."""
    os.makedirs(LOCAL_DIR, exist_ok=True)

    for fname in MODEL_FILES:
        local_path = LOCAL_DIR + fname
        if os.path.exists(local_path):
            continue  # already cached in this Lambda instance

        s3_key = MODEL_PREFIX + fname
        try:
            # Try S3 first (fast, free within AWS)
            print(f"Downloading {fname} from S3...")
            s3.download_file(BUCKET, s3_key, local_path)
        except Exception:
            # First time ever — download from HuggingFace and cache to S3
            print(f"Not in S3 yet. Downloading {fname} from HuggingFace...")
            url = HF_BASE + fname
            urllib.request.urlretrieve(url, local_path)
            print(f"Caching {fname} to S3 for future use...")
            s3.upload_file(local_path, BUCKET, s3_key)


def _load():
    """Load ONNX session and tokenizer (cached in module globals)."""
    global _session, _tokenizer
    if _session is not None:
        return

    _download_models()

    _session   = ort.InferenceSession(LOCAL_DIR + "model.onnx")
    _tokenizer = Tokenizer.from_file(LOCAL_DIR + "tokenizer.json")

    # Configure tokenizer padding/truncation to match MiniLM requirements
    _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
    _tokenizer.enable_truncation(max_length=128)

    print("ONNX MiniLM model loaded")


def mean_pooling(token_embeddings, attention_mask):
    """Average token embeddings weighted by attention mask."""
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask       = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def normalize(embeddings):
    """L2-normalize embeddings (matches sentence-transformers output)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


def encode(texts, batch_size=32):
    """
    Encode a list of strings into 384-dim normalized embeddings.
    Returns numpy float32 array of shape (len(texts), 384).
    Drop-in replacement for SentenceTransformer.encode()
    """
    _load()

    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Tokenize
        encoded     = _tokenizer.encode_batch(batch)
        input_ids   = np.array([e.ids              for e in encoded], dtype=np.int64)
        attn_mask   = np.array([e.attention_mask   for e in encoded], dtype=np.int64)
        token_type  = np.zeros_like(input_ids,                        dtype=np.int64)

        # Run ONNX inference
        outputs = _session.run(None, {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type,
        })

        # outputs[0] = token embeddings shape (batch, seq_len, 384)
        embeddings = mean_pooling(outputs[0], attn_mask)
        embeddings = normalize(embeddings)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings).astype(np.float32)
