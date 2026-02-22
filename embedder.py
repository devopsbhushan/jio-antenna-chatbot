# embedder.py
# Lightweight TF-IDF embedder — loads IDF from S3
# No ONNX, no model files, no heavy dependencies
# RAM usage: ~50MB (just the IDF dict)

import os
import re
import math
import hashlib
import pickle
import numpy as np
import boto3

BUCKET  = os.environ.get("BUCKET", "chatbot-input-database")
IDF_KEY = "rag-model/idf.pkl"
DIM     = 384

s3   = boto3.client("s3")
_idf = None   # loaded once, cached in module


def _load():
    global _idf
    if _idf is not None:
        return
    try:
        print("Loading IDF from S3...")
        s3.download_file(BUCKET, IDF_KEY, "/tmp/idf.pkl")
        with open("/tmp/idf.pkl", "rb") as f:
            _idf = pickle.load(f)
        print(f"IDF loaded: {len(_idf)} terms")
    except Exception as e:
        print(f"IDF not found, using uniform weights: {e}")
        _idf = {}


def _tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def _hash(token):
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM


def _encode_one(text):
    _load()
    vec = np.zeros(DIM, dtype=np.float32)
    for t in _tokenize(text):
        vec[_hash(t)] += _idf.get(t, 1.0)
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec


def encode(texts, batch_size=128):
    """Encode list of strings → float32 array (N, 384)."""
    if isinstance(texts, str):
        texts = [texts]
    return np.array([_encode_one(t) for t in texts], dtype=np.float32)
