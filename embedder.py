# embedder.py
# Ultra-lightweight embedder using TF-IDF style hashing
# No model download, no ONNX, no heavy dependencies
# Produces 384-dim vectors compatible with existing FAISS index
# Fits easily within 512MB RAM

import numpy as np
import hashlib
import re
import os
import json
import boto3
import pickle

# ── Vocabulary cached in module globals ──────────────────────────────────────
_idf    = None   # word -> idf weight
_vocab  = None   # word -> index
DIM     = 384

BUCKET   = os.environ.get("BUCKET", "chatbot-input-database")
IDF_KEY  = "rag-model/idf.pkl"

s3 = boto3.client("s3")


def _tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def _hash_token(token, dim=DIM):
    """Map token to a dimension index using hash."""
    h = int(hashlib.md5(token.encode()).hexdigest(), 16)
    return h % dim


def _build_idf(corpus):
    """Build IDF weights from a list of documents."""
    from collections import Counter
    N    = len(corpus)
    df   = Counter()
    for doc in corpus:
        words = set(_tokenize(doc))
        df.update(words)
    idf = {}
    for word, count in df.items():
        idf[word] = np.log((N + 1) / (count + 1)) + 1.0
    return idf


def _load_idf():
    """Load IDF weights from S3 if available."""
    global _idf
    if _idf is not None:
        return
    try:
        s3.download_file(BUCKET, IDF_KEY, "/tmp/idf.pkl")
        with open("/tmp/idf.pkl", "rb") as f:
            _idf = pickle.load(f)
        print(f"IDF loaded: {len(_idf)} terms")
    except Exception:
        # No IDF file yet — will use uniform weights
        _idf = {}
        print("No IDF file found — using uniform weights")


def build_and_save_idf(corpus):
    """Build IDF from corpus and save to S3. Call this from index builder."""
    global _idf
    print(f"Building IDF from {len(corpus)} documents...")
    _idf = _build_idf(corpus)
    with open("/tmp/idf.pkl", "wb") as f:
        pickle.dump(_idf, f)
    s3.upload_file("/tmp/idf.pkl", BUCKET, IDF_KEY)
    print(f"IDF saved to S3: {len(_idf)} terms")


def _encode_one(text):
    """Encode a single text to a DIM-dimensional float32 vector."""
    _load_idf()
    tokens  = _tokenize(text)
    vec     = np.zeros(DIM, dtype=np.float32)

    for token in tokens:
        idx    = _hash_token(token, DIM)
        weight = _idf.get(token, 1.0)
        vec[idx] += weight

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


def encode(texts, batch_size=128):
    """
    Encode list of strings → float32 numpy array shape (N, 384).
    Drop-in replacement for SentenceTransformer.encode()
    """
    if isinstance(texts, str):
        texts = [texts]

    vectors = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        vectors[i] = _encode_one(text)
    return vectors


def _load():
    """Pre-load IDF. Called by warm ping."""
    _load_idf()
