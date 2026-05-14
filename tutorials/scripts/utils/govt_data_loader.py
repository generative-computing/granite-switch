"""Load or build the ChromaDB corpus for the govt RAG tutorial.

Kept separate from the notebook so the pipeline stays focused on RAG concepts.

First run: downloads `govt.jsonl.zip` from IBM mt-rag-benchmark (49k passages),
embeds with `ibm-granite/granite-embedding-small-english-r2`, and saves to
`./govt_chroma`. Subsequent runs: loads the persisted index instantly.
"""

import io
import json
import os
import time
import warnings
import zipfile

import chromadb
import httpx
import torch
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL_ID = "ibm-granite/granite-embedding-small-english-r2"
CHROMA_PATH        = "./govt_chroma"
GOVT_JSONL_URL     = "https://github.com/IBM/mt-rag-benchmark/raw/main/corpora/passage_level/govt.jsonl.zip"
GOVT_JSONL_PATH    = "./govt.jsonl"


class GraniteEmbeddingFunction(EmbeddingFunction):
    """ChromaDB EmbeddingFunction backed by ibm-granite/granite-embedding-*-r2."""

    def __init__(self, model_id=EMBEDDING_MODEL_ID, batch_size=64):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device    = device
        self._batch     = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model     = AutoModel.from_pretrained(model_id).to(device).eval()
        print(f"Granite embedding model ready on {device}  ({model_id})")
        if device == "cpu":
            warnings.warn(
                "Embedding ~49k passages on CPU will take hours. "
                "Expected runtime is ~10 min on a single consumer GPU. "
                "Consider running on a GPU host, or sharing a pre-built ./govt_chroma directory.",
                stacklevel=2,
            )

    def __call__(self, input: Documents) -> Embeddings:
        all_embs = []
        for i in range(0, len(input), self._batch):
            batch = list(input[i : i + self._batch])
            enc = self._tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.extend(emb.cpu().float().tolist())
        return all_embs


def load_or_build_govt_chroma(
    chroma_path=CHROMA_PATH,
    jsonl_path=GOVT_JSONL_PATH,
    jsonl_url=GOVT_JSONL_URL,
    embedding_model_id=EMBEDDING_MODEL_ID,
):
    """Return a ready-to-query Chroma collection for the govt corpus.

    Loads from ``chroma_path`` if it already has documents; otherwise downloads
    the source jsonl, embeds, and persists.
    """
    granite_ef = GraniteEmbeddingFunction(model_id=embedding_model_id)
    client     = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name="govt",
        embedding_function=granite_ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"Loaded from {chroma_path}  ({collection.count():,} docs).")
        return collection

    if not os.path.exists(jsonl_path):
        print(f"Downloading {jsonl_url} ...")
        t0 = time.time()
        # Stream into memory with a progress bar - the zip is ~50MB and the
        # unblocked .get() used to leave users staring at a silent cell for minutes.
        # Split timeout: fail fast on connect (10s), allow slow reads (300s).
        timeout = httpx.Timeout(300.0, connect=10.0)
        buf = io.BytesIO()
        with httpx.Client(follow_redirects=True, timeout=timeout) as c:
            with c.stream("GET", jsonl_url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0)) or None
                with tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        buf.write(chunk)
                        bar.update(len(chunk))
        buf.seek(0)
        # Atomic write: extract to a .tmp path then os.replace, so a kill/crash
        # mid-write can't leave a truncated jsonl that later runs silently use.
        tmp_path = jsonl_path + ".tmp"
        with zipfile.ZipFile(buf) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".jsonl"))
            with zf.open(inner) as src, open(tmp_path, "wb") as dst:
                dst.write(src.read())
        os.replace(tmp_path, jsonl_path)
        print(f"Saved {jsonl_path} in {time.time() - t0:.1f}s.")

    print(f"Reading {jsonl_path} -> {chroma_path}...")
    t0 = time.time()
    ids, texts, metas = [], [], []
    with open(jsonl_path) as f:
        for line in f:
            doc  = json.loads(line)
            text = doc.get("text", "").strip()
            if not text:
                continue
            ids.append(doc.get("_id", doc.get("id", str(len(ids)))))
            texts.append(text)
            metas.append({"title": doc.get("title", ""), "url": doc.get("url", "")})
    if not ids:
        raise RuntimeError(
            f"{jsonl_path} yielded zero documents - the file may be empty, truncated, "
            f"or schema-drifted (expected a 'text' field per line). Delete it and rerun "
            f"to re-download."
        )
    print(f"Read {len(ids):,} docs in {time.time() - t0:.1f}s.  Embedding & indexing...")

    t1 = time.time()
    batch = 500
    for i in tqdm(range(0, len(ids), batch), unit="batch", desc="indexing"):
        collection.upsert(
            ids       = ids  [i : i + batch],
            documents = texts[i : i + batch],
            metadatas = metas[i : i + batch],
        )
    print(f"Done. {collection.count():,} docs saved to {chroma_path} in {time.time() - t1:.1f}s.")
    return collection
