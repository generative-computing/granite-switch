#!/usr/bin/env python3
"""Build a ChromaDB index from the mt-rag-benchmark govt.jsonl corpus.

The corpus is downloaded automatically from:
    https://github.com/IBM/mt-rag-benchmark/tree/main/corpora/passage_level

Usage (auto-download):
    python tutorials/alora_vs_lora_race/build_govt_chroma.py

Usage (local file):
    python tutorials/alora_vs_lora_race/build_govt_chroma.py \
        --jsonl /tmp/govt.jsonl \
        --output ./tutorials/alora_vs_lora_race/govt_chroma

The resulting index is bit-compatible with govt_chroma: same embedding model
(ibm-granite/granite-embedding-small-english-r2), same mean-pooling, same
cosine space, same document text and IDs.

See also: tutorials/notebooks/02_govt_rag_pipeline.ipynb §2 for the notebook
equivalent of this indexing step.
"""

import argparse
import io
import json
import threading
import urllib.request
import zipfile
from pathlib import Path

import torch
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

EMBEDDING_MODEL_ID = "ibm-granite/granite-embedding-small-english-r2"
BATCH_SIZE = 256
COLLECTION_NAME = "govt"
CORPUS_URL = "https://github.com/IBM/mt-rag-benchmark/raw/main/corpora/passage_level/govt.jsonl.zip"
_HERE = Path(__file__).parent


def _download_corpus(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = cache_dir / "govt.jsonl"
    if jsonl_path.exists():
        print(f"Using cached corpus at {jsonl_path}")
        return jsonl_path
    print(f"Downloading corpus from {CORPUS_URL} ...")
    data, _ = urllib.request.urlretrieve(CORPUS_URL)
    print("Decompressing...")
    with zipfile.ZipFile(data) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
        with zf.open(name) as src, open(jsonl_path, "wb") as dst:
            dst.write(src.read())
    return jsonl_path


class _EmbedFn(EmbeddingFunction):
    def __init__(self, device: str = "cpu"):
        self._device = device
        self._lock = threading.Lock()
        self._tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)
        self._model = AutoModel.from_pretrained(EMBEDDING_MODEL_ID).to(device).eval()

    def __call__(self, input: Documents) -> Embeddings:
        with self._lock:
            enc = self._tok(list(input), return_tensors="pt",
                            truncation=True, max_length=512, padding=True)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return emb.cpu().float().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", default=None,
                        help="Path to govt.jsonl (unzipped); downloaded automatically if omitted")
    parser.add_argument("--output", default=str(_HERE / "govt_chroma"),
                        help="Directory to write the ChromaDB index (default: govt_chroma/)")
    parser.add_argument("--device", default=None,
                        help="Torch device for embeddings (default: cuda if available, else cpu)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.jsonl is None:
        jsonl_path = _download_corpus(_HERE / "govt_corpus")
    else:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            raise FileNotFoundError(jsonl_path)

    print(f"Embedding model : {EMBEDDING_MODEL_ID}")
    print(f"Device          : {args.device}")
    print(f"Source          : {jsonl_path}")
    print(f"Output          : {args.output}")
    print()

    embed_fn = _EmbedFn(device=args.device)
    client = chromadb.PersistentClient(path=args.output)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    lines = jsonl_path.read_text().splitlines()
    total = len(lines)
    print(f"Indexing {total:,} passages in batches of {args.batch_size}...")

    for start in tqdm(range(0, total, args.batch_size), unit="batch"):
        batch = [json.loads(l) for l in lines[start:start + args.batch_size]]
        ids       = [d.get("_id", d.get("id", str(start + i))) for i, d in enumerate(batch)]
        documents = [d["text"] for d in batch]
        metadatas = [{k: v for k, v in d.items() if k not in ("_id", "id", "text")}
                     for d in batch]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count():,} docs.")


if __name__ == "__main__":
    main()
