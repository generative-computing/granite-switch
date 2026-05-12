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
            print("Notice: running Embedding & indexing step on a cpu might take a very long time.")

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
        print(f"Downloading {jsonl_url} …")
        t0 = time.time()
        with httpx.Client(follow_redirects=True, timeout=120.0) as c:
            resp = c.get(jsonl_url)
            resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".jsonl"))
            with zf.open(inner) as src, open(jsonl_path, "wb") as dst:
                dst.write(src.read())
        print(f"Saved {jsonl_path} in {time.time() - t0:.1f}s.")

    print(f"Reading {jsonl_path} → {chroma_path}…")
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
    print(f"Read {len(ids):,} docs in {time.time() - t0:.1f}s.  Embedding & indexing…")

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
