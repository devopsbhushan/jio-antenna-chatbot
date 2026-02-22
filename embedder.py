# embedder.py
# Loads all-MiniLM-L6-v2 ONNX model from S3 — never calls HuggingFace
# Model files are pre-uploaded to S3 by GitHub Actions workflow

import os
import numpy as np
import boto3
import onnxruntime as ort
from tokenizers import Tokenizer

BUCKET     = os.environ.get("BUCKET", "chatbot-input-database")
LOCAL_DIR  = "/tmp/minilm/"

# S3 paths where GitHub Actions uploaded the model files
S3_MODEL     = "rag-model/model.onnx"
S3_TOKENIZER = "rag-model/tokenizer.json"

s3 = boto3.client("s3")

_session   = None
_tokenizer = None


def _download_models():
    """Download ONNX model files from S3 to /tmp."""
    os.makedirs(LOCAL_DIR, exist_ok=True)

    model_path     = LOCAL_DIR + "model.onnx"
    tokenizer_path = LOCAL_DIR + "tokenizer.json"

    if not os.path.exists(model_path):
        print(f"Downloading model.onnx from S3...")
        s3.download_file(BUCKET, S3_MODEL, model_path)
        print(f"model.onnx downloaded ({os.path.getsize(model_path)//1024//1024}MB)")

    if not os.path.exists(tokenizer_path):
        print(f"Downloading tokenizer.json from S3...")
        s3.download_file(BUCKET, S3_TOKENIZER, tokenizer_path)
        print(f"tokenizer.json downloaded")


def _load():
    """Load ONNX session and tokenizer (cached in module globals)."""
    global _session, _tokenizer
    if _session is not None:
        return

    _download_models()

    _session = ort.InferenceSession(
        LOCAL_DIR + "model.onnx",
        providers=["CPUExecutionProvider"]
    )

    _tokenizer = Tokenizer.from_file(LOCAL_DIR + "tokenizer.json")
    _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
    _tokenizer.enable_truncation(max_length=128)

    print("ONNX MiniLM model loaded successfully")


def _mean_pooling(token_embeddings, attention_mask):
    mask     = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed   = np.sum(token_embeddings * mask, axis=1)
    counts   = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


def encode(texts, batch_size=32):
    """
    Encode list of strings → float32 numpy array shape (N, 384).
    Drop-in replacement for SentenceTransformer.encode()
    """
    _load()

    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch   = texts[i : i + batch_size]
        encoded = _tokenizer.encode_batch(batch)

        input_ids  = np.array([e.ids            for e in encoded], dtype=np.int64)
        attn_mask  = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type = np.zeros_like(input_ids,                      dtype=np.int64)

        outputs    = _session.run(None, {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type,
        })

        embeddings = _mean_pooling(outputs[0], attn_mask)
        embeddings = _normalize(embeddings)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings).astype(np.float32)
