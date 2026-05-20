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

TUTORIAL_DOC_IDS = ["05537c9ec2dfe15e-1362-3310", "05537c9ec2dfe15e-2-1779", "05537c9ec2dfe15e-2821-4679", "05537c9ec2dfe15e-4280-6252", "087417ad420d618c-1327-3164", "087417ad420d618c-2428-4297", "087417ad420d618c-3940-5774", "089882437c965a3e-113907-115852", "089882437c965a3e-115237-117256", "089882437c965a3e-119809-121676", "089882437c965a3e-121198-123235", "089882437c965a3e-122746-124833", "089882437c965a3e-130164-131917", "089882437c965a3e-1427-3375", "089882437c965a3e-157219-159194", "089882437c965a3e-158778-160687", "089882437c965a3e-170699-172699", "089882437c965a3e-173726-175992", "089882437c965a3e-175465-177577", "089882437c965a3e-177094-179288", "089882437c965a3e-182078-183322", "089882437c965a3e-184664-186341", "089882437c965a3e-190627-192211", "089882437c965a3e-191792-193455", "089882437c965a3e-194311-196074", "089882437c965a3e-2-1955", "089882437c965a3e-42318-44668", "089882437c965a3e-51633-53566", "089882437c965a3e-53014-54918", "089882437c965a3e-85071-87052", "089882437c965a3e-86622-88344", "0ecab3f697d26347-1362-3129", "142cbdf06f6e40d9-1544-3414", "142cbdf06f6e40d9-2-2014", "142cbdf06f6e40d9-4140-6181", "142cbdf06f6e40d9-5655-7824", "19240942bfc0abf5-11151-13247", "19240942bfc0abf5-1354-3015", "2c89b9fe3cfe95ee-1392-3518", "2ead5535f9d6d3be-1376-3143", "3090260a5d934d78-1166-2578", "3090260a5d934d78-2225-3536", "32472b4a577f296f-2-1847", "353067ac7a68e5f0-2-1815", "3630bbba71396272-1400-3319", "3630bbba71396272-4267-6086", "40ce723b445ac8eb-1350-3146", "40ce723b445ac8eb-2-1781", "40ce723b445ac8eb-3922-5642", "40ce723b445ac8eb-5372-7150", "40ce723b445ac8eb-6691-8678", "40ce723b445ac8eb-8241-9800", "4c201f242ec49883-1381-3148", "4c201f242ec49883-5418-7248", "4e1c120aee9a75b6-1369-3165", "50a24d38902fbdd0-1340-3177", "50a24d38902fbdd0-3953-5813", "565fb21ac38feaa1-15852-17699", "5b86a17591806ce5-1532-3330", "60e02c03620cd1ef-9523-11519", "6ddc73cb3877e2aa-1384-3151", "6ddc73cb3877e2aa-2-1801", "77de29ffa3c3d800-1352-3553", "77de29ffa3c3d800-2-1946", "7fe68ab7967494ca-1358-3306", "81478086b28ab210-5831-7806", "818e03cc80181db4-1346-3469", "818e03cc80181db4-2-1767", "818e03cc80181db4-3125-4727", "824c4c47b2989363-1365-3132", "824c4c47b2989363-2-1782", "82f7a783325de97a-1402-3321", "82f7a783325de97a-4269-6188", "882a9cc2bb08bcdf-2-1811", "8cd62677aa5dcb92-2-1746", "9726fa169575dc43-1331-3168", "9726fa169575dc43-2-1734", "9726fa169575dc43-2432-4301", "9726fa169575dc43-3944-5768", "9726fa169575dc43-5394-7430", "9726fa169575dc43-6967-8603", "97e58e54bb79a7fe-3231-5248", "99c7b4f2bfb48b7f-3321-5534", "a005bd5aedbb28e5-33908-36180", "a005bd5aedbb28e5-35687-37469", "a4a53cb6b6bf326e-1349-3145", "a4a53cb6b6bf326e-2-1780", "a4a53cb6b6bf326e-2409-4294", "a4a53cb6b6bf326e-3921-5691", "a4a53cb6b6bf326e-5362-7156", "a4a53cb6b6bf326e-6689-8701", "a4a53cb6b6bf326e-8201-10002", "a930d03cf0b406fd-23288-25302", "a930d03cf0b406fd-30996-32981", "c550156dbbfe212c-1401-3320", "c550156dbbfe212c-16212-18433", "c550156dbbfe212c-29308-31304", "c550156dbbfe212c-30794-33132", "c550156dbbfe212c-32367-34910", "c550156dbbfe212c-37745-39895", "c550156dbbfe212c-39218-41274", "c550156dbbfe212c-40668-42844", "c550156dbbfe212c-42364-44521", "c550156dbbfe212c-44034-46164", "c550156dbbfe212c-45669-47909", "c550156dbbfe212c-47421-49701", "c550156dbbfe212c-9073-11428", "c67a2f65008344fd-2-1909", "c93223e21ee4ecfb-2-1754", "d4c48e9a4029f3e9-1801-3993", "d4edd2b762f5dce9-7713-9881", "e580ce520db3ff10-109466-111339", "e580ce520db3ff10-119467-121417", "e580ce520db3ff10-124119-126003", "e580ce520db3ff10-129933-131969", "e580ce520db3ff10-131480-133562", "e580ce520db3ff10-190530-192253", "e580ce520db3ff10-191857-193702", "e580ce520db3ff10-35813-37462", "e580ce520db3ff10-36974-38756", "e6ea24fa9e962807-1357-3305", "e6ea24fa9e962807-4275-6126", "ed17e5bd32458f9c-1347-3143", "ed17e5bd32458f9c-3919-5735", "f0b48597d0c22d32-2-1647", "f0b48597d0c22d32-2585-4675", "f0b48597d0c22d32-999-3136", "f14d35fd47c9ed59-1352-3148", "f14d35fd47c9ed59-3924-5795", "f14d35fd47c9ed59-5374-7566", "f7225d77034b8398-1402-3321", "f90bb40d57fe7ba5-1469-3644", "f90bb40d57fe7ba5-2-1890", "f90bb40d57fe7ba5-3142-5127", "f90bb40d57fe7ba5-8968-10553", "fcdc09416b6aa645-1276-2982", "fcdc09416b6aa645-2-1649"]


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
    batch = 16 if granite_ef._device == "cpu" else 500
    for i in tqdm(range(0, len(ids), batch), unit="batch", desc="indexing"):
        collection.upsert(
            ids       = ids  [i : i + batch],
            documents = texts[i : i + batch],
            metadatas = metas[i : i + batch],
        )
    print(f"Done. {collection.count():,} docs saved to {chroma_path} in {time.time() - t1:.1f}s.")
    return collection
