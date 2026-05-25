"""Generic ChromaDB loader with sentence-transformers backend.

This module provides a flexible interface for loading or building ChromaDB collections
from various data sources, with improved embedding capabilities using sentence-transformers.

Key improvements over the legacy govt_data_loader:
- sentence-transformers backend for cleaner API and better batching
- Configurable max_length (1024 vs hardcoded 512)
- Configurable batch_size (auto-tuned vs hardcoded 64)
- Per-document progress tracking
- Generic loader supporting MT-RAG corpora and HuggingFace datasets
- Flexible ID filtering via filter_ids parameter

Backward compatibility maintained via load_or_build_govt_chroma() wrapper.
"""

import io
import json
import os
import time
import warnings
import zipfile
from typing import Dict, List, Optional, Set, Tuple

import chromadb
import httpx
import torch
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm.auto import tqdm

# Constants from original govt_data_loader
EMBEDDING_MODEL_ID = "ibm-granite/granite-embedding-small-english-r2"
CHROMA_PATH = "./govt_chroma"
GOVT_JSONL_URL = "https://github.com/IBM/mt-rag-benchmark/raw/main/corpora/passage_level/govt.jsonl.zip"
GOVT_JSONL_PATH = "./govt.jsonl"

# Tutorial subset: 177 docs for T4/CPU-friendly embedding
TUTORIAL_DOC_IDS = set([
    "05537c9ec2dfe15e-1362-3310", "05537c9ec2dfe15e-2-1779", "05537c9ec2dfe15e-2821-4679",
    "05537c9ec2dfe15e-4280-6252", "087417ad420d618c-1327-3164", "087417ad420d618c-2428-4297",
    "087417ad420d618c-3940-5774", "089882437c965a3e-113907-115852", "089882437c965a3e-115237-117256",
    "089882437c965a3e-119809-121676", "089882437c965a3e-121198-123235", "089882437c965a3e-122746-124833",
    "089882437c965a3e-130164-131917", "089882437c965a3e-1427-3375", "089882437c965a3e-157219-159194",
    "089882437c965a3e-158778-160687", "089882437c965a3e-170699-172699", "089882437c965a3e-173726-175992",
    "089882437c965a3e-175465-177577", "089882437c965a3e-177094-179288", "089882437c965a3e-182078-183322",
    "089882437c965a3e-184664-186341", "089882437c965a3e-190627-192211", "089882437c965a3e-191792-193455",
    "089882437c965a3e-194311-196074", "089882437c965a3e-2-1955", "089882437c965a3e-42318-44668",
    "089882437c965a3e-51633-53566", "089882437c965a3e-53014-54918", "089882437c965a3e-85071-87052",
    "089882437c965a3e-86622-88344", "0ecab3f697d26347-1362-3129", "142cbdf06f6e40d9-1544-3414",
    "142cbdf06f6e40d9-2-2014", "142cbdf06f6e40d9-4140-6181", "142cbdf06f6e40d9-5655-7824",
    "19240942bfc0abf5-11151-13247", "19240942bfc0abf5-1354-3015", "2c89b9fe3cfe95ee-1392-3518",
    "2ead5535f9d6d3be-1376-3143", "3090260a5d934d78-1166-2578", "3090260a5d934d78-2225-3536",
    "32472b4a577f296f-2-1847", "353067ac7a68e5f0-2-1815", "3630bbba71396272-1400-3319",
    "3630bbba71396272-4267-6086", "40ce723b445ac8eb-1350-3146", "40ce723b445ac8eb-2-1781",
    "40ce723b445ac8eb-3922-5642", "40ce723b445ac8eb-5372-7150", "40ce723b445ac8eb-6691-8678",
    "40ce723b445ac8eb-8241-9800", "4c201f242ec49883-1381-3148", "4c201f242ec49883-5418-7248",
    "4e1c120aee9a75b6-1369-3165", "50a24d38902fbdd0-1340-3177", "50a24d38902fbdd0-3953-5813",
    "565fb21ac38feaa1-15852-17699", "5b86a17591806ce5-1532-3330", "60e02c03620cd1ef-9523-11519",
    "6ddc73cb3877e2aa-1384-3151", "6ddc73cb3877e2aa-2-1801", "77de29ffa3c3d800-1352-3553",
    "77de29ffa3c3d800-2-1946", "7fe68ab7967494ca-1358-3306", "81478086b28ab210-5831-7806",
    "818e03cc80181db4-1346-3469", "818e03cc80181db4-2-1767", "818e03cc80181db4-3125-4727",
    "824c4c47b2989363-1365-3132", "824c4c47b2989363-2-1782", "82f7a783325de97a-1402-3321",
    "82f7a783325de97a-4269-6188", "882a9cc2bb08bcdf-2-1811", "8cd62677aa5dcb92-2-1746",
    "9726fa169575dc43-1331-3168", "9726fa169575dc43-2-1734", "9726fa169575dc43-2432-4301",
    "9726fa169575dc43-3944-5768", "9726fa169575dc43-5394-7430", "9726fa169575dc43-6967-8603",
    "97e58e54bb79a7fe-3231-5248", "99c7b4f2bfb48b7f-3321-5534", "a005bd5aedbb28e5-33908-36180",
    "a005bd5aedbb28e5-35687-37469", "a4a53cb6b6bf326e-1349-3145", "a4a53cb6b6bf326e-2-1780",
    "a4a53cb6b6bf326e-2409-4294", "a4a53cb6b6bf326e-3921-5691", "a4a53cb6b6bf326e-5362-7156",
    "a4a53cb6b6bf326e-6689-8701", "a4a53cb6b6bf326e-8201-10002", "a930d03cf0b406fd-23288-25302",
    "a930d03cf0b406fd-30996-32981", "c550156dbbfe212c-1401-3320", "c550156dbbfe212c-16212-18433",
    "c550156dbbfe212c-29308-31304", "c550156dbbfe212c-30794-33132", "c550156dbbfe212c-32367-34910",
    "c550156dbbfe212c-37745-39895", "c550156dbbfe212c-39218-41274", "c550156dbbfe212c-40668-42844",
    "c550156dbbfe212c-42364-44521", "c550156dbbfe212c-44034-46164", "c550156dbbfe212c-45669-47909",
    "c550156dbbfe212c-47421-49701", "c550156dbbfe212c-9073-11428", "c67a2f65008344fd-2-1909",
    "c93223e21ee4ecfb-2-1754", "d4c48e9a4029f3e9-1801-3993", "d4edd2b762f5dce9-7713-9881",
    "e580ce520db3ff10-109466-111339", "e580ce520db3ff10-119467-121417", "e580ce520db3ff10-124119-126003",
    "e580ce520db3ff10-129933-131969", "e580ce520db3ff10-131480-133562", "e580ce520db3ff10-190530-192253",
    "e580ce520db3ff10-191857-193702", "e580ce520db3ff10-35813-37462", "e580ce520db3ff10-36974-38756",
    "e6ea24fa9e962807-1357-3305", "e6ea24fa9e962807-4275-6126", "ed17e5bd32458f9c-1347-3143",
    "ed17e5bd32458f9c-3919-5735", "f0b48597d0c22d32-2-1647", "f0b48597d0c22d32-2585-4675",
    "f0b48597d0c22d32-999-3136", "f14d35fd47c9ed59-1352-3148", "f14d35fd47c9ed59-3924-5795",
    "f14d35fd47c9ed59-5374-7566", "f7225d77034b8398-1402-3321", "f90bb40d57fe7ba5-1469-3644",
    "f90bb40d57fe7ba5-2-1890", "f90bb40d57fe7ba5-3142-5127", "f90bb40d57fe7ba5-8968-10553",
    "fcdc09416b6aa645-1276-2982", "fcdc09416b6aa645-2-1649",
])

# MT-RAG corpus metadata
CORPUS_INFO = {
    "govt": {
        "url": GOVT_JSONL_URL,
        "local_path": GOVT_JSONL_PATH,
        "chroma_path": CHROMA_PATH,
        "collection_name": "govt",
    },
}


class GraniteEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function using sentence-transformers backend.

    This class wraps a sentence-transformers model for use with ChromaDB.
    Uses eager loading (model loaded in __init__) for clear upfront waiting time.

    Args:
        model_id: HuggingFace model ID for sentence-transformers
        batch_size: Batch size for encoding (None = auto-tune)
        max_length: Maximum sequence length for embeddings (default 1024)
        device: Device to use ("cpu", "cuda", or None for auto-detect)
    """

    def __init__(
        self,
        model_id: str = EMBEDDING_MODEL_ID,
        batch_size: Optional[int] = None,
        max_length: int = 1024,
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                warnings.warn(
                    "Embedding on CPU will be slow. "
                    "Expected runtime is ~10 min on a single consumer GPU. "
                    "Consider running on a GPU host.",
                    stacklevel=2,
                )

        self.device = device

        # Eager loading: model loaded immediately (clear upfront waiting)
        self.model = SentenceTransformer(model_id, device=device)

        # Set max_seq_length on the model itself (not in encode() kwargs)
        self.model.max_seq_length = max_length

        print(f"Granite embedding model ready on {device}  ({model_id})")

    def __call__(self, input: Documents) -> Embeddings:
        """Embed texts with batching and progress bar."""
        # Build encode kwargs, omitting batch_size if None (let library auto-tune)
        # Note: max_seq_length is set on the model itself in __init__, not here
        encode_kwargs = {
            "show_progress_bar": False,  # Disable internal progress (we track at doc level)
            "convert_to_numpy": True,
        }
        if self.batch_size is not None:
            encode_kwargs["batch_size"] = self.batch_size

        embeddings = self.model.encode(input, **encode_kwargs)
        return embeddings.tolist()


def _download_jsonl_zip(url: str, output_path: str) -> None:
    """Download and extract JSONL from ZIP archive with progress tracking.

    Args:
        url: URL to download ZIP archive from
        output_path: Local path to save extracted JSONL file
    """
    print(f"Downloading {url} ...")
    t0 = time.time()

    # Stream into memory with progress bar
    # Split timeout: fail fast on connect (10s), allow slow reads (300s)
    timeout = httpx.Timeout(300.0, connect=10.0)
    buf = io.BytesIO()
    with httpx.Client(follow_redirects=True, timeout=timeout) as c:
        with c.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    buf.write(chunk)
                    bar.update(len(chunk))
    buf.seek(0)

    # Atomic write: extract to .tmp then replace, so crashes can't leave truncated files
    tmp_path = output_path + ".tmp"
    with zipfile.ZipFile(buf) as zf:
        inner = next(n for n in zf.namelist() if n.endswith(".jsonl"))
        with zf.open(inner) as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())
    os.replace(tmp_path, output_path)
    print(f"Saved {output_path} in {time.time() - t0:.1f}s.")


def _load_records_from_jsonl(
    jsonl_path: str,
    filter_ids: Optional[Set[str]] = None,
    max_docs: Optional[int] = None,
    text_field: str = "text",
    id_field: Optional[str] = None,
) -> Tuple[List[str], List[str], List[Dict]]:
    """Load document records from JSONL file.

    Args:
        jsonl_path: Path to JSONL file
        filter_ids: Set of document IDs to include (None = all)
        max_docs: Maximum documents to load (None = no limit)
        text_field: Field name for document text
        id_field: Field name for document ID (None = use _id or id field)

    Returns:
        Tuple of (ids, texts, metadatas)
    """
    ids, texts, metas = [], [], []

    with open(jsonl_path) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get(text_field, "").strip()
            if not text:
                continue

            # Extract document ID
            if id_field:
                doc_id = doc.get(id_field)
            else:
                doc_id = doc.get("_id", doc.get("id", str(len(ids))))

            # Apply filtering
            if filter_ids is not None and doc_id not in filter_ids:
                continue

            ids.append(doc_id)
            texts.append(text)
            metas.append({
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
            })

            # Respect max_docs limit
            if max_docs is not None and len(ids) >= max_docs:
                break

    if not ids:
        raise RuntimeError(
            f"{jsonl_path} yielded zero documents - the file may be empty, truncated, "
            f"or schema-drifted (expected a '{text_field}' field per line). "
            f"Delete it and rerun to re-download."
        )

    return ids, texts, metas


def _load_records_from_hf(
    dataset_id: str,
    filter_ids: Optional[Set[str]] = None,
    max_docs: Optional[int] = None,
    config: Optional[str] = None,
    split: str = "train",
    text_field: str = "text",
    id_field: Optional[str] = None,
) -> Tuple[List[str], List[str], List[Dict]]:
    """Load document records from HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID
        filter_ids: Set of document IDs to include (None = all)
        max_docs: Maximum documents to load (None = no limit)
        config: Dataset configuration name
        split: Dataset split to load
        text_field: Field name for document text
        id_field: Field name for document ID

    Returns:
        Tuple of (ids, texts, metadatas)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with: "
            "pip install datasets"
        )

    dataset = load_dataset(dataset_id, config, split=split)

    ids, texts, metas = [], [], []
    for i, example in enumerate(dataset):
        text = example.get(text_field, "").strip()
        if not text:
            continue

        # Extract document ID
        if id_field:
            doc_id = example.get(id_field, str(i))
        else:
            doc_id = example.get("_id", example.get("id", str(i)))

        # Apply filtering
        if filter_ids is not None and doc_id not in filter_ids:
            continue

        ids.append(doc_id)
        texts.append(text)
        metas.append({
            "title": example.get("title", ""),
            "url": example.get("url", ""),
        })

        # Respect max_docs limit
        if max_docs is not None and len(ids) >= max_docs:
            break

    if not ids:
        raise RuntimeError(
            f"Dataset {dataset_id} yielded zero documents. "
            f"Check that '{text_field}' field exists."
        )

    return ids, texts, metas


def load_or_build_chroma(
    corpus_name: Optional[str] = None,
    hf_dataset_id: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    jsonl_url: Optional[str] = None,
    chroma_path: Optional[str] = None,
    collection_name: str = "default",
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    batch_size: Optional[int] = None,
    max_length: int = 1024,
    max_docs: Optional[int] = None,
    filter_ids: Optional[Set[str]] = None,
    device: Optional[str] = None,
    query_device: str = "cpu",
    text_field: str = "text",
    id_field: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: str = "train",
) -> chromadb.Collection:
    """Generic ChromaDB loader supporting multiple data sources.

    This function loads or builds a ChromaDB collection from either:
    1. Named MT-RAG corpus (via corpus_name)
    2. Local JSONL file (via jsonl_path + optional jsonl_url)
    3. HuggingFace dataset (via hf_dataset_id)

    Indexing uses `device` (default: GPU if available) for fast batch embedding.
    After indexing, the GPU model is freed and the returned collection uses
    `query_device` (default: "cpu") for query-time embedding — freeing the GPU
    for vLLM. On cache hit the GPU is never loaded at all.

    Args:
        corpus_name: Named MT-RAG corpus ("govt", "fiqa", etc.)
        hf_dataset_id: HuggingFace dataset ID (mutually exclusive with corpus_name)
        jsonl_path: Local JSONL file path (derived from corpus_name if None)
        jsonl_url: URL to download JSONL (derived from corpus_name if None)
        chroma_path: Persistent storage path (derived from corpus_name if None)
        collection_name: ChromaDB collection name (derived from corpus_name if None)
        embedding_model_id: Sentence-transformers model ID
        batch_size: Embedding batch size (None = auto-tune)
        max_length: Maximum sequence length for embeddings
        max_docs: Maximum documents to ingest (None = no limit)
        filter_ids: Set of document IDs to include (None = all docs)
        device: Indexing device — "cpu", "cuda", or None (auto-detect GPU)
        query_device: Query-time embedding device (default "cpu")
        text_field: Field name for document text
        id_field: Field name for document ID
        hf_config: HuggingFace dataset configuration
        hf_split: HuggingFace dataset split

    Returns:
        ChromaDB collection ready for queries (embedding function on query_device)
    """
    # Resolve corpus info if corpus_name provided
    if corpus_name:
        if corpus_name not in CORPUS_INFO:
            raise ValueError(
                f"Unknown corpus '{corpus_name}'. "
                f"Available: {list(CORPUS_INFO.keys())}"
            )
        info = CORPUS_INFO[corpus_name]
        jsonl_url = jsonl_url or info["url"]
        jsonl_path = jsonl_path or info["local_path"]
        chroma_path = chroma_path or info["chroma_path"]
        collection_name = collection_name if collection_name != "default" else info["collection_name"]

    # Validate inputs
    if not chroma_path:
        raise ValueError("chroma_path must be specified")
    if not hf_dataset_id and not jsonl_path:
        raise ValueError("Must specify either hf_dataset_id or jsonl_path")

    # Query-time embedding function (CPU by default — GPU is reserved for vLLM)
    query_ef = GraniteEmbeddingFunction(
        model_id=embedding_model_id,
        batch_size=batch_size,
        max_length=max_length,
        device=query_device,
    )

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=query_ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Cache hit: GPU never needed
    if collection.count() > 0:
        print(f"Loaded from {chroma_path}  ({collection.count():,} docs).")
        return collection

    # Load documents
    if hf_dataset_id:
        print(f"Loading from HuggingFace dataset {hf_dataset_id}...")
        ids, texts, metas = _load_records_from_hf(
            dataset_id=hf_dataset_id,
            filter_ids=filter_ids,
            max_docs=max_docs,
            config=hf_config,
            split=hf_split,
            text_field=text_field,
            id_field=id_field,
        )
    else:
        # Download JSONL if needed
        if not os.path.exists(jsonl_path):
            if not jsonl_url:
                raise ValueError(f"{jsonl_path} not found and no jsonl_url provided")
            _download_jsonl_zip(jsonl_url, jsonl_path)

        if filter_ids is not None:
            print(f"Filtering to {len(filter_ids)} doc IDs")

        print(f"Reading {jsonl_path} -> {chroma_path}...")
        t0 = time.time()
        ids, texts, metas = _load_records_from_jsonl(
            jsonl_path=jsonl_path,
            filter_ids=filter_ids,
            max_docs=max_docs,
            text_field=text_field,
            id_field=id_field,
        )
        print(f"Read {len(ids):,} docs in {time.time() - t0:.1f}s.")

    # Index on GPU: create a separate indexing embedding function
    index_ef = GraniteEmbeddingFunction(
        model_id=embedding_model_id,
        batch_size=batch_size,
        max_length=max_length,
        device=device,  # GPU (or auto-detect)
    )

    print(f"Embedding {len(ids):,} documents on {index_ef.device}...")
    t1 = time.time()

    # Pre-compute all embeddings on the indexing device in batches
    embed_batch = batch_size or 64
    all_embeddings: List = []
    for i in tqdm(range(0, len(ids), embed_batch), unit="batch", desc="embedding"):
        all_embeddings.extend(index_ef(texts[i : i + embed_batch]))

    # Free indexing model from GPU before vLLM launches
    del index_ef.model
    del index_ef
    torch.cuda.empty_cache()
    print(f"Embedding done in {time.time() - t1:.1f}s. GPU memory freed.")

    # Upsert pre-computed embeddings (no re-embedding by ChromaDB)
    upsert_batch = 500
    for i in tqdm(range(0, len(ids), upsert_batch), unit="batch", desc="indexing"):
        collection.upsert(
            ids=ids[i : i + upsert_batch],
            documents=texts[i : i + upsert_batch],
            metadatas=metas[i : i + upsert_batch],
            embeddings=all_embeddings[i : i + upsert_batch],
        )

    print(f"Done. {collection.count():,} docs saved to {chroma_path}.")
    return collection


def load_or_build_govt_chroma(
    chroma_path: str = CHROMA_PATH,
    jsonl_path: str = GOVT_JSONL_PATH,
    jsonl_url: str = GOVT_JSONL_URL,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    load_only_tutorial_docs: bool = False,
    device: Optional[str] = None,
    query_device: str = "cpu",
) -> chromadb.Collection:
    """Backward-compatible govt corpus loader.

    This function maintains the API of the legacy govt_data_loader module
    while using the improved chroma_loader implementation underneath.

    Args:
        chroma_path: Persistent storage path
        jsonl_path: Local JSONL path
        jsonl_url: Download URL
        embedding_model_id: Embedding model ID
        load_only_tutorial_docs: If True, load only 177 tutorial docs (T4-friendly)
        device: Indexing device — "cpu", "cuda", or None (auto-detect GPU)
        query_device: Query-time embedding device (default "cpu")

    Returns:
        ChromaDB collection with govt corpus
    """
    filter_ids = TUTORIAL_DOC_IDS if load_only_tutorial_docs else None

    return load_or_build_chroma(
        corpus_name="govt",
        jsonl_path=jsonl_path,
        jsonl_url=jsonl_url,
        chroma_path=chroma_path,
        embedding_model_id=embedding_model_id,
        filter_ids=filter_ids,
        device=device,
        query_device=query_device,
        max_docs=None,  # NO artificial limit
    )
