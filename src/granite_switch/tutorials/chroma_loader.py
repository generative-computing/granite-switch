"""ChromaDB corpus loader for RAG tutorials.

Supports two dataset sources:
- MT-RAG benchmark corpora (jsonl.zip, auto-downloaded from GitHub)
- HuggingFace datasets (via ``datasets.load_dataset``)

First run embeds and persists; subsequent runs load the persisted index instantly.
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
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

EMBEDDING_MODEL_ID = "ibm-granite/granite-embedding-small-english-r2"
CHROMA_PATH        = "./govt_chroma"
MTRAG_URL_TEMPLATE = "https://github.com/IBM/mt-rag-benchmark/raw/main/corpora/passage_level/{name}.jsonl.zip"

# Backward-compatibility aliases
GOVT_JSONL_URL  = MTRAG_URL_TEMPLATE.format(name="govt")
GOVT_JSONL_PATH = "./govt.jsonl"

TUTORIAL_DOC_IDS = ["05537c9ec2dfe15e-1362-3310", "05537c9ec2dfe15e-2-1779", "05537c9ec2dfe15e-2821-4679", "05537c9ec2dfe15e-4280-6252", "087417ad420d618c-1327-3164", "087417ad420d618c-2428-4297", "087417ad420d618c-3940-5774", "089882437c965a3e-113907-115852", "089882437c965a3e-115237-117256", "089882437c965a3e-119809-121676", "089882437c965a3e-121198-123235", "089882437c965a3e-122746-124833", "089882437c965a3e-130164-131917", "089882437c965a3e-1427-3375", "089882437c965a3e-157219-159194", "089882437c965a3e-158778-160687", "089882437c965a3e-170699-172699", "089882437c965a3e-173726-175992", "089882437c965a3e-175465-177577", "089882437c965a3e-177094-179288", "089882437c965a3e-182078-183322", "089882437c965a3e-184664-186341", "089882437c965a3e-190627-192211", "089882437c965a3e-191792-193455", "089882437c965a3e-194311-196074", "089882437c965a3e-2-1955", "089882437c965a3e-42318-44668", "089882437c965a3e-51633-53566", "089882437c965a3e-53014-54918", "089882437c965a3e-85071-87052", "089882437c965a3e-86622-88344", "0ecab3f697d26347-1362-3129", "142cbdf06f6e40d9-1544-3414", "142cbdf06f6e40d9-2-2014", "142cbdf06f6e40d9-4140-6181", "142cbdf06f6e40d9-5655-7824", "19240942bfc0abf5-11151-13247", "19240942bfc0abf5-1354-3015", "2c89b9fe3cfe95ee-1392-3518", "2ead5535f9d6d3be-1376-3143", "3090260a5d934d78-1166-2578", "3090260a5d934d78-2225-3536", "32472b4a577f296f-2-1847", "353067ac7a68e5f0-2-1815", "3630bbba71396272-1400-3319", "3630bbba71396272-4267-6086", "40ce723b445ac8eb-1350-3146", "40ce723b445ac8eb-2-1781", "40ce723b445ac8eb-3922-5642", "40ce723b445ac8eb-5372-7150", "40ce723b445ac8eb-6691-8678", "40ce723b445ac8eb-8241-9800", "4c201f242ec49883-1381-3148", "4c201f242ec49883-5418-7248", "4e1c120aee9a75b6-1369-3165", "50a24d38902fbdd0-1340-3177", "50a24d38902fbdd0-3953-5813", "565fb21ac38feaa1-15852-17699", "5b86a17591806ce5-1532-3330", "60e02c03620cd1ef-9523-11519", "6ddc73cb3877e2aa-1384-3151", "6ddc73cb3877e2aa-2-1801", "77de29ffa3c3d800-1352-3553", "77de29ffa3c3d800-2-1946", "7fe68ab7967494ca-1358-3306", "81478086b28ab210-5831-7806", "818e03cc80181db4-1346-3469", "818e03cc80181db4-2-1767", "818e03cc80181db4-3125-4727", "824c4c47b2989363-1365-3132", "824c4c47b2989363-2-1782", "82f7a783325de97a-1402-3321", "82f7a783325de97a-4269-6188", "882a9cc2bb08bcdf-2-1811", "8cd62677aa5dcb92-2-1746", "9726fa169575dc43-1331-3168", "9726fa169575dc43-2-1734", "9726fa169575dc43-2432-4301", "9726fa169575dc43-3944-5768", "9726fa169575dc43-5394-7430", "9726fa169575dc43-6967-8603", "97e58e54bb79a7fe-3231-5248", "99c7b4f2bfb48b7f-3321-5534", "a005bd5aedbb28e5-33908-36180", "a005bd5aedbb28e5-35687-37469", "a4a53cb6b6bf326e-1349-3145", "a4a53cb6b6bf326e-2-1780", "a4a53cb6b6bf326e-2409-4294", "a4a53cb6b6bf326e-3921-5691", "a4a53cb6b6bf326e-5362-7156", "a4a53cb6b6bf326e-6689-8701", "a4a53cb6b6bf326e-8201-10002", "a930d03cf0b406fd-23288-25302", "a930d03cf0b406fd-30996-32981", "c550156dbbfe212c-1401-3320", "c550156dbbfe212c-16212-18433", "c550156dbbfe212c-29308-31304", "c550156dbbfe212c-30794-33132", "c550156dbbfe212c-32367-34910", "c550156dbbfe212c-37745-39895", "c550156dbbfe212c-39218-41274", "c550156dbbfe212c-40668-42844", "c550156dbbfe212c-42364-44521", "c550156dbbfe212c-44034-46164", "c550156dbbfe212c-45669-47909", "c550156dbbfe212c-47421-49701", "c550156dbbfe212c-9073-11428", "c67a2f65008344fd-2-1909", "c93223e21ee4ecfb-2-1754", "d4c48e9a4029f3e9-1801-3993", "d4edd2b762f5dce9-7713-9881", "e580ce520db3ff10-109466-111339", "e580ce520db3ff10-119467-121417", "e580ce520db3ff10-124119-126003", "e580ce520db3ff10-129933-131969", "e580ce520db3ff10-131480-133562", "e580ce520db3ff10-190530-192253", "e580ce520db3ff10-191857-193702", "e580ce520db3ff10-35813-37462", "e580ce520db3ff10-36974-38756", "e6ea24fa9e962807-1357-3305", "e6ea24fa9e962807-4275-6126", "ed17e5bd32458f9c-1347-3143", "ed17e5bd32458f9c-3919-5735", "f0b48597d0c22d32-2-1647", "f0b48597d0c22d32-2585-4675", "f0b48597d0c22d32-999-3136", "f14d35fd47c9ed59-1352-3148", "f14d35fd47c9ed59-3924-5795", "f14d35fd47c9ed59-5374-7566", "f7225d77034b8398-1402-3321", "f90bb40d57fe7ba5-1469-3644", "f90bb40d57fe7ba5-2-1890", "f90bb40d57fe7ba5-3142-5127", "f90bb40d57fe7ba5-8968-10553", "fcdc09416b6aa645-1276-2982", "fcdc09416b6aa645-2-1649"]


class GraniteEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function backed by a SentenceTransformer model.

    The model is loaded lazily on the first call, so constructing this object
    is cheap — no GPU/CPU memory is allocated until embedding is actually needed.
    """

    def __init__(
        self,
        model_id=EMBEDDING_MODEL_ID,
        batch_size=64,
        max_length: int | None = None,
        device=None,
    ):
        """
        Parameters:
            model_id: SentenceTransformer model identifier.
            batch_size: Number of texts to encode per forward pass. Default 64.
            max_length: Maximum token sequence length. Defaults to 1024.
            device: Computation device ('cuda' or 'cpu'). Auto-detected if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device     = device
        self._batch      = batch_size
        self._model_id   = model_id
        self._max_length = max_length if max_length is not None else 1024
        self._model      = None  # loaded on first __call__

    def _load_model(self) -> None:
        self._model = SentenceTransformer(self._model_id, device=self._device)
        self._model.max_seq_length = self._max_length
        print(f"Granite embedding model ready on {self._device}  ({self._model_id})")
        if self._device == "cpu":
            warnings.warn(
                "Embedding on CPU will be slow. "
                "Expected runtime is ~2 min on a single consumer GPU for this dataset. "
                "Consider running on a GPU host, or sharing a pre-built ChromaDB directory.",
                stacklevel=3,
            )

    def __call__(self, documents: Documents) -> Embeddings:
        if self._model is None:
            self._load_model()
        return self._model.encode(
            list(documents),
            batch_size=self._batch,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()


def _download_jsonl_zip(url: str, local_path: str) -> None:
    """
    Downloads a ZIP file containing a JSONL file from a given URL and extracts its contents to a local path,
    while showing a progress bar.

    This function performs the download via an HTTP request, measures the time taken, and provides a progress
    indicator for monitoring the download process. It then extracts the JSONL file inside the downloaded ZIP and
    saves it atomically to the specified local path to avoid partial writes in case of failure.

    Parameters:
    url: str
        The URL from which the ZIP file containing the JSONL file will be downloaded.
    local_path: str
        The absolute or relative path where the extracted JSONL file should be saved.

    Returns:
    None
    """
    print(f"Downloading {url} ...")
    t0 = time.time()
    # Split timeout: fail fast on connect (10s), allow slow reads (300s).
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
    # Atomic write via tmp file so a mid-write crash can't leave a truncated jsonl.
    tmp_path = local_path + ".tmp"
    with zipfile.ZipFile(buf) as zf:
        inner = next((n for n in zf.namelist() if n.endswith(".jsonl")), None)
        if inner is None:
            raise ValueError(
                f"No .jsonl entry found in archive from {url}. "
                f"Archive contains: {zf.namelist()}"
            )
        with zf.open(inner) as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())
    os.replace(tmp_path, local_path)
    print(f"Saved {local_path} in {time.time() - t0:.1f}s.")


def _load_records_from_jsonl(
    jsonl_path: str,
    text_field: str = "text",
    id_field: str = "_id",
    title_field: str = "title",
    url_field: str = "url",
    filter_ids: "set | None" = None,
    max_docs: "int | None" = None,
) -> "tuple[list, list, list]":
    """
    Loads records from a JSONL (JSON Lines) file and returns lists of IDs, texts, and metadata.

    This function reads a JSON Lines file where each line represents a JSON object. It processes the
    objects by extracting specific fields to construct lists of document IDs, texts, and metadata.
    Only documents satisfying specific filters are included in the output.

    Parameters:
        jsonl_path (str): Path to the JSONL file to be read.
        text_field (str): The field name in the JSON object that contains the document text. Defaults to "text".
        id_field (str): The field name in the JSON object that contains the document ID. Defaults to "_id".
        title_field (str): The field name in the JSON object that contains the title. Defaults to "title".
        url_field (str): The field name in the JSON object that contains the URL. Defaults to "url".
        filter_ids (set | None): A set of IDs to include in the output. Only documents with IDs in this
            set will be processed. If None, all documents will be included. Defaults to None.
        max_docs (int | None): The maximum number of documents to process. If None, there is no limit.
            Defaults to None.

    Returns:
        tuple[list, list, list]:
            A tuple containing three lists:
            - The first list contains all processed document IDs.
            - The second list contains all processed text values.
            - The third list contains all metadata dictionaries corresponding to the processed documents.
    """
    ids, texts, metas = [], [], []
    with open(jsonl_path) as f:
        for line in f:
            doc  = json.loads(line)
            text = (doc.get(text_field) or "").strip()
            if not text:
                continue
            doc_id = doc.get(id_field, doc.get("id", str(len(ids))))
            if filter_ids is not None and doc_id not in filter_ids:
                continue
            ids.append(doc_id)
            texts.append(text)
            metas.append({"title": doc.get(title_field) or "", "url": doc.get(url_field) or ""})
            if max_docs is not None and len(ids) >= max_docs:
                break
    return ids, texts, metas


def _load_records_from_hf(
    hf_dataset_id: str,
    hf_config: "str | None" = None,
    hf_split: str = "train",
    text_field: str = "text",
    id_field: str = "id",
    title_field: str = "title",
    url_field: str = "url",
    filter_ids: "set | None" = None,
    max_docs: "int | None" = None,
) -> "tuple[list, list, list]":
    """
    Loads records from a HuggingFace dataset and returns filtered and processed data.

    This function loads a dataset from HuggingFace using the specified dataset ID and
    optional configuration. It extracts specific fields from the dataset, filters based
    on IDs if provided, and limits the number of returned records if a maximum limit
    is specified.

    Parameters:
        hf_dataset_id (str): The ID of the HuggingFace dataset to load.
        hf_config (str | None): The configuration name of the dataset (e.g., a specific
            subset) if applicable. Defaults to None.
        hf_split (str): The split of the dataset to load (e.g., "train", "validation",
            or "test"). Defaults to "train".
        text_field (str): The field name in the dataset that contains the main text data.
            Defaults to "text".
        id_field (str): The field name in the dataset used to extract unique document IDs.
            Defaults to "id".
        title_field (str): The field name in the dataset that contains the title metadata.
            Defaults to "title".
        url_field (str): The field name in the dataset that contains the URL metadata.
            Defaults to "url".
        filter_ids (set | None): A set of IDs to filter the dataset by. If provided, only
            records with IDs in the set will be included. Defaults to None.
        max_docs (int | None): The maximum number of records to return. If None, all
            matching records will be returned. Defaults to None.

    Returns:
        tuple[list, list, list]: A tuple containing three lists:
            - ids (list): A list of unique document IDs extracted from the dataset.
            - texts (list): A list of processed text data from the dataset.
            - metas (list): A list of dictionaries containing metadata for each
              corresponding document (e.g., "title" and "url").

    Raises:
        ImportError: If the `datasets` library is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install 'datasets' to load HuggingFace datasets: pip install datasets"
        )
    dataset = load_dataset(hf_dataset_id, hf_config, split=hf_split)
    ids, texts, metas = [], [], []
    for i, row in enumerate(dataset):
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        doc_id = str(row.get(id_field, i))
        if filter_ids is not None and doc_id not in filter_ids:
            continue
        ids.append(doc_id)
        texts.append(text)
        metas.append({"title": str(row.get(title_field, "")), "url": str(row.get(url_field, ""))})
        if max_docs is not None and len(ids) >= max_docs:
            break
    return ids, texts, metas


def _upsert_to_collection(collection, client, ids, texts, metas):
    """Upsert documents into collection in chunks sized to ChromaDB's limit."""
    chunk = client.get_max_batch_size()
    if len(ids) <= chunk:
        collection.upsert(ids=ids, documents=texts, metadatas=metas)
    else:
        for i in tqdm(range(0, len(ids), chunk), unit="batch", desc="indexing"):
            collection.upsert(
                ids       = ids  [i : i + chunk],
                documents = texts[i : i + chunk],
                metadatas = metas[i : i + chunk],
            )


def load_or_build_chroma(
    chroma_path: str,
    collection_name: str,
    *,
    # MT-RAG source (used when hf_dataset_id is not given)
    corpus_name: "str | None" = None,    # e.g. "govt", "finance" — auto-derives URL
    jsonl_path: "str | None" = None,     # local cache; defaults to ./{corpus_name}.jsonl
    jsonl_url: "str | None" = None,      # overrides the auto-derived MT-RAG URL
    # HuggingFace source (mutually exclusive with corpus_name)
    hf_dataset_id: "str | None" = None,  # e.g. "ibm/mt-rag"
    hf_config: "str | None" = None,
    hf_split: str = "train",
    # Field mappings (both sources)
    text_field: str = "text",
    id_field: "str | None" = None,       # defaults: "_id" for jsonl, "id" for HF
    title_field: str = "title",
    url_field: str = "url",
    # Filtering
    filter_ids: "set | None" = None,     # keep only these doc IDs (None = keep all)
    max_docs: "int | None" = None,
    # Embedding
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    device: "str | None" = None,
    embedding_fn: "EmbeddingFunction | None" = None,
    batch_size: "int | None" = None,
    max_length: "int | None" = 1024,
) -> chromadb.Collection:
    """Return a ready-to-query Chroma collection, building and persisting it on first run.

    Loads from ``chroma_path`` if the collection already has documents; otherwise
    fetches, embeds, and persists. The embedding model is loaded lazily — no GPU/CPU
    memory is allocated on cache hits.

    Arguments:
        chroma_path: Path to the ChromaDB storage directory.
        collection_name: Name of the ChromaDB collection to create or load.

    Keyword Arguments:
        corpus_name: MT-RAG corpus name (e.g. ``"govt"``, ``"finance"``); auto-derives
            the download URL. Mutually exclusive with ``hf_dataset_id``.
        jsonl_path: Local cache path for the downloaded jsonl. Defaults to
            ``./{corpus_name}.jsonl``.
        jsonl_url: Override the auto-derived MT-RAG download URL.
        hf_dataset_id: HuggingFace dataset identifier (e.g. ``"ibm/mt-rag"``).
            Mutually exclusive with ``corpus_name``.
        hf_config: HuggingFace dataset configuration name, if applicable.
        hf_split: HuggingFace dataset split to load. Default ``"train"``.
        text_field: Field name for document text. Default ``"text"``.
        id_field: Field name for document ID. Defaults to ``"_id"`` for JSONL
            and ``"id"`` for HuggingFace.
        title_field: Field name for document title. Default ``"title"``.
        url_field: Field name for document URL. Default ``"url"``.
        filter_ids: Keep only documents whose ID is in this set. ``None`` keeps all.
        max_docs: Maximum number of documents to load. ``None`` means no limit.
        embedding_model_id: SentenceTransformer model to use for embedding.
        device: Device for embedding (``"cpu"`` or ``"cuda"``). Auto-detected if None.
        embedding_fn: Custom ChromaDB ``EmbeddingFunction`` to override the default.
        batch_size: Embedding batch size passed to the model's ``encode()`` call.
        max_length: Maximum token sequence length for the embedding model. Default 1024.

    Raises:
        ValueError: If both ``corpus_name`` and ``hf_dataset_id`` are provided.
        RuntimeError: If the source yields zero documents.

    Returns:
        A ChromaDB ``Collection`` ready for upsert and query.

    MT-RAG example::

        load_or_build_chroma("./finance_chroma", "finance", corpus_name="finance")

    HuggingFace example::

        load_or_build_chroma(
            "./my_chroma", "my_collection",
            hf_dataset_id="ibm/mt-rag",
            hf_config="finance",
            text_field="passage",
            id_field="id",
        )
    """
    if hf_dataset_id is not None and corpus_name is not None:
        raise ValueError("Specify corpus_name (MT-RAG) or hf_dataset_id (HuggingFace), not both.")

    ef_kwargs: dict = {"model_id": embedding_model_id, "device": device, "max_length": max_length}
    if batch_size is not None:
        ef_kwargs["batch_size"] = batch_size
    granite_ef = embedding_fn or GraniteEmbeddingFunction(**ef_kwargs)

    client     = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=granite_ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"Loaded from {chroma_path}  ({collection.count():,} docs).")
        return collection

    t0 = time.time()

    if hf_dataset_id is not None:
        print(f"Reading HuggingFace dataset {hf_dataset_id!r} ({hf_split}) -> {chroma_path}...")
        ids, texts, metas = _load_records_from_hf(
            hf_dataset_id, hf_config, hf_split,
            text_field, id_field or "id", title_field, url_field,
            filter_ids, max_docs,
        )
    else:
        _corpus = corpus_name or collection_name
        _url    = jsonl_url  or MTRAG_URL_TEMPLATE.format(name=_corpus)
        _path   = jsonl_path or f"./{_corpus}.jsonl"
        if not os.path.exists(_path):
            _download_jsonl_zip(_url, _path)
        if filter_ids is not None:
            print(f"Filtering to {len(filter_ids)} doc ids")
        print(f"Reading {_path} -> {chroma_path}...")
        ids, texts, metas = _load_records_from_jsonl(
            _path, text_field, id_field or "_id", title_field, url_field,
            filter_ids, max_docs,
        )

    if not ids:
        raise RuntimeError(
            "Dataset yielded zero documents — check field names (text_field, id_field) "
            "and that the source file/dataset is non-empty."
        )

    print(f"Read {len(ids):,} docs in {time.time() - t0:.1f}s.  Embedding & indexing...")
    t1 = time.time()
    _upsert_to_collection(collection, client, ids, texts, metas)
    print(f"Done. {collection.count():,} docs saved to {chroma_path} in {time.time() - t1:.1f}s.")
    return collection


def load_or_build_govt_chroma(
    chroma_path=CHROMA_PATH,
    jsonl_path=GOVT_JSONL_PATH,
    jsonl_url=GOVT_JSONL_URL,
    embedding_model_id=EMBEDDING_MODEL_ID,
    load_only_tutorial_docs=False,
    device=None,
    max_docs=None,
    embedding_fn: "EmbeddingFunction | None" = None,
    batch_size: "int | None" = None,
    max_length: "int | None" = 1024,
) -> chromadb.Collection:
    """
    Loads or builds a Chroma collection for government-related data.

    This function wraps the `load_or_build_chroma` function, using default values
    and specific configurations for the government corpus. It supports loading existing
    collections or building a new one from provided JSONL data. The function is highly
    configurable, allowing users to specify embedding models, devices, batch sizes,
    and filtering options.

    Arguments:
        chroma_path (str): Optional. Path to the directory where Chroma data
            is stored. Defaults to the globally defined `CHROMA_PATH`.
        jsonl_path (str): Optional. Path to the JSONL file containing the government
            data. Defaults to the globally defined `GOVT_JSONL_PATH`.
        jsonl_url (str): Optional. URL to download the JSONL file if `jsonl_path`
            is not available. Defaults to the globally defined `GOVT_JSONL_URL`.
        embedding_model_id (str): Optional. Identifier for the embedding model to
            be used. Defaults to the globally defined `EMBEDDING_MODEL_ID`.
        load_only_tutorial_docs (bool): Optional. Specifies whether to load only
            tutorial documents. Defaults to False.
        device (str | None): Optional. Specifies the device to use for embedding
            (e.g., 'cpu' or 'cuda'). Defaults to None, using available hardware.
        max_docs (int | None): Optional. Maximum number of documents to process.
            Defaults to None, allowing all documents.
        embedding_fn (EmbeddingFunction | None): Optional. Custom embedding function
            to override the default embedding model. Defaults to None.
        batch_size (int | None): Optional. Batch size to use for processing.
            Defaults to None.
        max_length (int | None): Optional. Maximum length of input text for
            embedding. Defaults to 1024.

    Returns:
        chromadb.Collection: The loaded or newly built Chroma collection for
        government-related data.
    """
    return load_or_build_chroma(
        chroma_path        = chroma_path,
        collection_name    = "govt",
        corpus_name        = "govt",
        jsonl_path         = jsonl_path,
        jsonl_url          = jsonl_url,
        filter_ids         = set(TUTORIAL_DOC_IDS) if load_only_tutorial_docs else None,
        embedding_model_id = embedding_model_id,
        device             = device,
        max_docs           = max_docs,
        embedding_fn       = embedding_fn,
        batch_size         = batch_size,
        max_length         = max_length,
    )
