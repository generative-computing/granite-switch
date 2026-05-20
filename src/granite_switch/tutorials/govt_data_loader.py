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

TUTORIAL_DOC_IDS = ["05537c9ec2dfe15e-1362-3310", "05537c9ec2dfe15e-2821-4679", "0798145372958e67-2-1819", "0798145372958e67-2806-4797", "087417ad420d618c-1327-3164", "087417ad420d618c-2-1730", "087417ad420d618c-2428-4297", "087417ad420d618c-3940-5774", "089882437c965a3e-119809-121676", "089882437c965a3e-121198-123235", "089882437c965a3e-122746-124833", "089882437c965a3e-127429-129378", "089882437c965a3e-157219-159194", "089882437c965a3e-158778-160687", "089882437c965a3e-170699-172699", "089882437c965a3e-177094-179288", "089882437c965a3e-189063-190980", "089882437c965a3e-194311-196074", "089882437c965a3e-50348-52005", "089882437c965a3e-5822-7681", "089882437c965a3e-89493-91300", "09fcd3522f225c08-1470-3262", "09fcd3522f225c08-2-1892", "09fcd3522f225c08-2906-4419", "0c89eb6ce158b539-11637-12717", "0c89eb6ce158b539-12370-13635", "0c89eb6ce158b539-1518-3586", "0c89eb6ce158b539-18523-20370", "0c89eb6ce158b539-3002-5409", "0c89eb6ce158b539-8548-10560", "0ecab3f697d26347-1362-3129", "0f45348143eb4bd0-1387-3413", "0f45348143eb4bd0-2-1982", "0f45348143eb4bd0-3043-4798", "0f45348143eb4bd0-4453-6134", "101d281a2d161497-2-1982", "142cbdf06f6e40d9-4140-6181", "1698f908f9619e8f-1526-3502", "1698f908f9619e8f-2-2034", "198447caa94a927c-1807-4103", "1cda065b868bd23b-2-1936", "1f39ea3b9e5013f8-2-1925", "1ffcde90f3d010a3-1591-3368", "1ffcde90f3d010a3-2-2006", "1ffcde90f3d010a3-4281-6403", "1ffcde90f3d010a3-5411-7580", "1ffcde90f3d010a3-6789-7959", "246104fc4644074c-2-2049", "246104fc4644074c-3282-5480", "270cc2faa0c776d6-1468-3362", "270cc2faa0c776d6-2859-4889", "270cc2faa0c776d6-4217-5062", "290721904d6b8d18-7475-9573", "2ead5535f9d6d3be-1376-3143", "3090260a5d934d78-2225-3536", "33a5169f5e33bad5-1534-3779", "3630bbba71396272-1400-3319", "39b1796bb2d75e7a-1390-3454", "39b1796bb2d75e7a-2-1877", "39b1796bb2d75e7a-2969-5323", "39b1796bb2d75e7a-4773-6791", "40ce723b445ac8eb-1350-3146", "40ce723b445ac8eb-2410-4295", "40ce723b445ac8eb-5372-7150", "4737c8c10d509563-1765-3618", "4737c8c10d509563-2-2269", "4bff9b930f3cc7c6-2-2040", "4c201f242ec49883-1381-3148", "4c5aa13052b32146-6335-8608", "4c5aa13052b32146-8084-10482", "4c5aa13052b32146-9850-12358", "4fa2c24ac435ca8b-4790-7047", "50a24d38902fbdd0-1340-3177", "54ebabfac7902cbd-1180-2748", "54ebabfac7902cbd-3516-4952", "54ebabfac7902cbd-4647-6096", "54ebabfac7902cbd-5676-7062", "5cb1fad7d465d3bd-1602-4009", "5cb1fad7d465d3bd-2-2255", "607505aad360dbec-6305-8416", "681cdbd9fc48fe1f-0-2988", "683fb68df597342b-1712-3504", "683fb68df597342b-2-2183", "6ad5cd6137c6d951-1972-2912", "6ddc73cb3877e2aa-1384-3151", "761ee8348a5fe52b-2-2126", "77de29ffa3c3d800-1352-3553", "77de29ffa3c3d800-2-1946", "78ffc1cfa4706629-11622-12441", "7fe68ab7967494ca-1358-3306", "805784ee2a7a4221-8598-10489", "818e03cc80181db4-2-1767", "824c4c47b2989363-1365-3132", "82b7d497827eb245-1730-3904", "82b7d497827eb245-2-2179", "82f7a783325de97a-1402-3321", "8af768273dd2691d-2-2004", "8b113b68bfa20ba1-1586-3280", "8cd62677aa5dcb92-2-1746", "9413658bb51af449-3382-5729", "9413658bb51af449-5167-7545", "9726fa169575dc43-1331-3168", "9db9da9839056803-2-1880", "a034282d28370877-1524-3173", "a2150d5e7dbf5f53-2-2234", "a35c391085e517ad-5408-5885", "a4a53cb6b6bf326e-1349-3145", "a4a53cb6b6bf326e-2-1780", "a4a53cb6b6bf326e-2409-4294", "a4a53cb6b6bf326e-3921-5691", "a4a53cb6b6bf326e-5362-7156", "a4a53cb6b6bf326e-6689-8701", "a4a53cb6b6bf326e-8201-10002", "a6587cf8dcff1d60-1691-3797", "a6587cf8dcff1d60-2-2166", "a930d03cf0b406fd-23288-25302", "a930d03cf0b406fd-30996-32981", "b89841e0c5f8920f-1799-3938", "baeabe1656c35c30-8479-10328", "bee07c0637d4ea68-1532-3236", "c550156dbbfe212c-1401-3320", "c6dde1589cd32f45-4199-4754", "c6dde1589cd32f45-5539-6114", "cac49d31da2406e4-8520-10369", "cc5a2abf2f4c206d-2733-4805", "cef22c7afe95118f-2-1868", "cef22c7afe95118f-2852-4410", "d371624a0569c75b-1422-2691", "d371624a0569c75b-2-1872", "d66e9c997b458093-7922-9838", "d702a574b85659f6-9864-10288", "e2848c047f58a3fd-1801-4011", "e2848c047f58a3fd-2-1974", "e580ce520db3ff10-124119-126003", "ed17e5bd32458f9c-1347-3143", "ed17e5bd32458f9c-63679-65713", "ed17e5bd32458f9c-65052-66706", "ed17e5bd32458f9c-66188-68348", "eec1e1d0c6764fef-4959-7024", "eec1e1d0c6764fef-5736-7893", "f14d35fd47c9ed59-1352-3148", "f31905a75e94b6e0-1429-3532", "f31905a75e94b6e0-3111-3664", "f7225d77034b8398-1402-3321", "f85a3f7907a25896-1620-3537", "fa0556b613cdc403-1480-3099", "fc19b16721100f97-2-2119", "fc19b16721100f97-4734-6081"]


class GraniteEmbeddingFunction(EmbeddingFunction):
    """ChromaDB EmbeddingFunction backed by ibm-granite/granite-embedding-*-r2."""

    def __init__(self, model_id=EMBEDDING_MODEL_ID, batch_size=64, device = None):
        if device == None:
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
    load_only_tutorial_docs=False,
    device=None,
):
    """Return a ready-to-query Chroma collection for the govt corpus.

    Loads from ``chroma_path`` if it already has documents; otherwise downloads
    the source jsonl, embeds, and persists.

    When ``load_only_tutorial_docs=True``, embed only docs whose ``_id`` is in
    ``TUTORIAL_DOC_IDS`` (the curated subset that the demo queries actually
    retrieve). Cuts the 49k-passage corpus down dramatically so first-run
    embedding takes seconds instead of minutes.
    """
    granite_ef = GraniteEmbeddingFunction(model_id=embedding_model_id, device= device)
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

    keep_ids = set(TUTORIAL_DOC_IDS) if load_only_tutorial_docs else None
    if keep_ids is not None:
        print(f"Filtering to {len(keep_ids)} tutorial doc ids")

    print(f"Reading {jsonl_path} -> {chroma_path}...")
    t0 = time.time()
    ids, texts, metas = [], [], []
    with open(jsonl_path) as f:
        for line in f:
            doc  = json.loads(line)
            text = doc.get("text", "").strip()
            if not text:
                continue
            doc_id = doc.get("_id", doc.get("id", str(len(ids))))
            if keep_ids is not None and doc_id not in keep_ids:
                continue
            ids.append(doc_id)
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
