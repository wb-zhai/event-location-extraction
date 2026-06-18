import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers.util.similarity import cos_sim
from tqdm import tqdm

from src.common.log import get_logger
from src.common.models import RetrievedSample
from src.index.documents import Document, DocumentStore

logger = get_logger(__name__)

SEARCH_METRIC_TO_FUNCTION = {
    "matmul": lambda x, y: torch.matmul(x, y.T),
    "cosine": lambda x, y: cos_sim(x, y),
    "euclidean": lambda x, y: torch.cdist(x, y),
}


@dataclass
class IndexerOutput:
    indices: Union[torch.Tensor, np.ndarray]
    distances: Union[torch.Tensor, np.ndarray]


class Indexer:
    """
    Base class for document indexes.

    Args:
        documents (:obj:`str`, :obj:`List[str]`, :obj:`os.PathLike`, :obj:`List[os.PathLike]`, :obj:`DocumentStore`, `optional`):
            The documents to index. If `None`, an empty document store will be created. Defaults to `None`.
        embeddings (:obj:`torch.Tensor`, `optional`):
            The embeddings of the documents. If `None`, the documents will not be indexed. Defaults to `None`.
        name_or_path (:obj:`str`, :obj:`os.PathLike`, `optional`):
            The name or directory of the retriever.
    """

    def __init__(
        self,
        documents: DocumentStore | None = None,
        embeddings: torch.Tensor | None = None,
        metadata_fields: List[str] | None = None,
        add_metadata_keys_to_text: bool = False,
        separator: str = " ",
        device: str | torch.device = "cpu",
        precision: str | int = 32,
    ) -> None:

        if embeddings is not None and documents is not None:
            logger.info("Both documents and embeddings are provided.")
            if len(documents) != embeddings.shape[0]:
                raise ValueError(
                    "The number of documents and embeddings must be the same."
                )

        if metadata_fields is None:
            metadata_fields = []

        self.metadata_fields = metadata_fields
        self.add_metadata_keys_to_text = add_metadata_keys_to_text
        self.separator = separator

        self.document_path: List[str | os.PathLike] = []

        if documents is not None:
            if isinstance(documents, DocumentStore):
                self.documents = documents
            else:
                raise ValueError("`documents` must be a `DocumentStore`.")
        else:
            self.documents = DocumentStore()

        self.embeddings = embeddings

        # store the device in case embeddings are not provided
        self.device_in_init = device

    def __iter__(self):
        # make this class iterable
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.get_passage_from_index(index)

    def to(self, device_or_precision: str | torch.device | torch.dtype) -> "Indexer":
        """
        Move the retriever to the specified device or precision.

        Args:
            device_or_precision (`str` | `torch.device` | `torch.dtype`):
                The device or precision to move the retriever to.

        Returns:
            `Indexer`: The indexer.
        """
        if self.embeddings is not None:
            if isinstance(device_or_precision, torch.dtype) and self.device != "cpu":
                # if the device is a dtype, then we need to move the embeddings to cpu
                # first before converting to the dtype to avoid OOM
                previous_device = self.embeddings.device
                self.embeddings = self.embeddings.cpu()
                self.embeddings = self.embeddings.to(device_or_precision)
                self.embeddings = self.embeddings.to(previous_device)
            else:
                self.embeddings = self.embeddings.to(device_or_precision)
        return self

    @property
    def device(self):
        return (
            self.embeddings.device
            if self.embeddings is not None
            else self.device_in_init
        )

    def index(
        self,
        retriever,
        *args,
        **kwargs,
    ) -> "Indexer":
        raise NotImplementedError

    def search(self, query: Any, k: int = 1, *args, **kwargs) -> List:
        raise NotImplementedError

    def get_document_from_passage(self, passage: str) -> Document | None:
        """
        Get the document label from the passage.

        Args:
            passage (`str`):
                The document to get the label for.

        Returns:
            `str`: The document label.
        """
        # get the text from the document
        if self.separator:
            text = passage.split(self.separator)[0]
        else:
            text = passage
        return self.documents.get_document_from_text(text)

    def get_index_from_passage(self, passage: str) -> int:
        """
        Get the index of the passage.

        Args:
            passage (`str`):
                The document to get the index for.

        Returns:
            `int`: The index of the document.
        """
        # get the text from the document
        doc = self.get_document_from_passage(passage)
        if doc is None:
            raise ValueError(f"Document `{passage}` not found.")
        return doc.id

    def get_document_from_index(self, index: int) -> Document | None:
        """
        Get the document from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `str`: The document.
        """
        return self.documents.get_document_from_index(index)

    def get_passage_from_index(self, index: int) -> str:
        """
        Get the document from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `str`: The document.
        """
        document = self.get_document_from_index(index)
        # build the passage using the metadata fields
        passage = document.text
        for field in self.metadata_fields:
            if field not in document.metadata:
                continue

            value = document.metadata[field]
            if isinstance(value, dict):
                metadata_text = str(value.get("name", value))
            elif isinstance(value, list):
                metadata_text = ", ".join(
                    str(item.get("name", item)) if isinstance(item, dict) else str(item)
                    for item in value
                )
            else:
                metadata_text = str(value)

            if self.add_metadata_keys_to_text:
                passage += f"{self.separator}{field}:{metadata_text}"
            else:
                passage += f"{self.separator}{metadata_text}"

        return passage

    def get_passage_from_document(self, document: Document) -> str:
        passage = document.text
        for field in self.metadata_fields:
            if field not in document.metadata:
                continue

            value = document.metadata[field]
            if isinstance(value, dict):
                metadata_text = str(value.get("name", value))

            elif isinstance(value, list):
                metadata_text = ", ".join(
                    str(item.get("name", item)) if isinstance(item, dict) else str(item)
                    for item in value
                )
            else:
                metadata_text = str(value)

            if self.add_metadata_keys_to_text:
                passage += f"{self.separator}{field}:{metadata_text}"
            else:
                passage += f"{self.separator}{metadata_text}"

        return passage

    def get_embeddings_from_index(self, index: int) -> torch.Tensor:
        """
        Get the document vector from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.embeddings is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        if index >= self.embeddings.shape[0]:
            raise ValueError(
                f"The index {index} is out of bounds. The maximum index is {len(self.embeddings) - 1}."
            )
        return self.embeddings[index]

    def get_embeddings_from_passage(self, document: str) -> torch.Tensor:
        """
        Get the document vector from the document label.

        Args:
            document (`str`):
                The document to get the vector for.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.embeddings is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        return self.get_embeddings_from_index(self.get_index_from_passage(document))

    def get_embeddings_from_document(self, document: str) -> torch.Tensor:
        """
        Get the document vector from the document label.

        Args:
            document (`str`):
                The document to get the vector for.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.embeddings is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        return self.get_embeddings_from_index(self.get_index_from_document(document))

    def get_passages(self, documents: DocumentStore | None = None) -> List[str]:
        """
        Get the passages from the document store.

        Returns:
            `List[str]`: The passages.
        """
        documents = documents or self.documents
        # construct the passages from the documents
        # return [self.get_passage_from_index(i) for i in range(len(documents))]
        return [self.get_passage_from_document(doc) for doc in documents]

    def save_pretrained(
        self,
        dir_path: str | os.PathLike,
        mmap: bool = False,
        dtype: str = "float32",
    ) -> None:
        """
        Save retriever state to a JSONL file.

        Args:
            path (:obj:`str` or :obj:`os.PathLike`):
                The path to the JSONL file where the retriever state will be saved.
            mmap (:obj:`bool`, `optional`):
                Whether to save embeddings to a sidecar mmap file. Defaults to False.
            mmap_path (:obj:`str` or :obj:`os.PathLike`, `optional`):
                The path to the mmap file. Defaults to `<path>.mmap` when `mmap` is True.
            dtype (:obj:`str`, `optional`):
                The dtype to use for the mmap file. Defaults to "float32".
        """

        # Some sanity checks before we start writing files
        if self.embeddings is None:
            raise ValueError("No embeddings found. Please run `index()` first.")

        if len(self.documents) != self.embeddings.shape[0]:
            raise ValueError(
                "Document count does not match embeddings rows: "
                f"{len(self.documents)} != {self.embeddings.shape[0]}"
            )

        # ensure the output directory exists
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        documents_path = dir_path / "documents.jsonl"

        if mmap:
            mmap_path = dir_path / "embeddings.mmap"
            meta_path = dir_path / "embeddings.meta.json"

            with open(documents_path, "w", encoding="utf-8") as f:
                for record in self.documents:
                    payload = (
                        record
                        if isinstance(record, RetrievedSample)
                        else record.to_dict()
                    )
                    f.write(json.dumps(payload) + "\n")

            embeddings_cpu = self.embeddings.detach().cpu().contiguous()
            rows, dim = embeddings_cpu.shape

            np_dtype = np.dtype(dtype)
            embeddings_np = embeddings_cpu.numpy()

            if embeddings_np.dtype != np_dtype:
                embeddings_np = embeddings_np.astype(np_dtype, copy=False)

            mmap_array = np.memmap(
                mmap_path, dtype=np_dtype, mode="w+", shape=(rows, dim)
            )
            mmap_array[:] = embeddings_np
            mmap_array.flush()

            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(
                    {
                        "rows": rows,
                        "dim": dim,
                        "dtype": dtype,
                        "metadata_fields": self.metadata_fields,
                        "add_metadata_keys_to_text": self.add_metadata_keys_to_text,
                        "separator": self.separator,
                    },
                    meta_file,
                )

            logger.info(
                "Saved retriever state with %s records to %s and mmap to %s",
                len(self.documents),
                documents_path,
                mmap_path,
            )
        else:
            with open(documents_path, "w", encoding="utf-8") as f:
                for record, vector in zip(self.documents, self.embeddings):
                    to_dump = (
                        dict(record)
                        if isinstance(record, RetrievedSample)
                        else record.to_dict()
                    )
                    to_dump["vector"] = vector.detach().cpu().tolist()
                    f.write(json.dumps(to_dump) + "\n")

            logger.info(
                f"Saved retriever state with {len(self.documents)} records to {documents_path}"
            )

    @classmethod
    def from_pretrained(
        cls,
        dir_path: str | os.PathLike,
        device: str | torch.device = "cpu",
        precision: str | int | None = None,
        metadata_fields: List[str] | None = None,
        separator: str | None = None,
        add_metadata_keys_to_text: bool | None = None,
        dtype: str = "float32",
    ) -> "Indexer":
        """
        Load retriever state from a JSONL file.

        Args:
            path (:obj:`str` or :obj:`os.PathLike`):
                The path to the JSONL file containing the retriever state.
            device (:obj:`str`, `optional`):
                The device to load the retriever state onto. Defaults to "cpu".
            precision (:obj:`str`, `optional`):
                The precision to use when loading the retriever state. Defaults to None (use default precision).
            metadata_fields (:obj:`List[str]`, `optional`):
                The list of metadata fields to include in the indexer. Defaults to None (no metadata
                fields).
            separator (:obj:`str`, `optional`):
                The separator to use when concatenating metadata fields. Defaults to " ".
            dtype (:obj:`str`, `optional`):
                The dtype to use for the mmap file. Defaults to "float32".

        Returns:
            :obj:`Indexer`: The loaded retriever indexer.
        """

        # default file names are
        # - `<dir_path>/documents.jsonl` for documents
        # - `<dir_path>/embeddings.mmap` for sidecar embeddings
        # - `<dir_path>/embeddings.meta.json` for mmap metadata

        dir_path = Path(dir_path)
        documents_path = dir_path / "documents.jsonl"
        mmap_path = dir_path / "embeddings.mmap"
        meta_path = dir_path / "embeddings.meta.json"

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not documents_path.exists():
            raise FileNotFoundError(f"Documents file not found: {documents_path}")

        use_mmap = mmap_path.exists() and meta_path.exists()

        documents = DocumentStore()

        if use_mmap:
            if not mmap_path.exists() or not meta_path.exists():
                # First pass to determine shape.
                rows = 0
                dim: int | None = None

                with open(documents_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(tqdm(f, desc="Scanning embeddings")):
                        record = json.loads(line)
                        vector = record.get("vector")

                        if vector is None:
                            raise ValueError(
                                "Missing 'vector' field while building mmap at "
                                f"line {i+1}"
                            )

                        if dim is None:
                            dim = len(vector)
                        elif len(vector) != dim:
                            raise ValueError(
                                "Vector length mismatch at line "
                                f"{i+1}: expected {dim}, got {len(vector)}"
                            )

                        rows += 1

                if dim is None:
                    raise ValueError("No vectors found in the JSONL file.")

                mmap_array = np.memmap(
                    mmap_path, dtype=dtype, mode="w+", shape=(rows, dim)
                )

                with open(documents_path, "r", encoding="utf-8") as f:
                    for row_idx, line in enumerate(
                        tqdm(f, desc="Writing mmap embeddings")
                    ):
                        record = json.loads(line)
                        vector = record.pop("vector", None)

                        if vector is None:
                            raise ValueError(
                                "Missing 'vector' field while building mmap at "
                                f"line {row_idx+1}"
                            )

                        mmap_array[row_idx] = np.asarray(vector, dtype=dtype)
                        documents.add_document(Document(**record))

                mmap_array.flush()

                with open(meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump({"rows": rows, "dim": dim, "dtype": dtype}, meta_file)
            else:
                with open(meta_path, "r", encoding="utf-8") as meta_file:
                    meta = json.load(meta_file)

                rows = int(meta["rows"])
                dim = int(meta["dim"])
                dtype = str(meta["dtype"])

                with open(documents_path, "r", encoding="utf-8") as f:
                    for line in tqdm(f, desc="Loading documents"):
                        record = json.loads(line)
                        record.pop("vector", None)
                        documents.add_document(Document(**record))

                mmap_array = np.memmap(
                    mmap_path, dtype=dtype, mode="r+", shape=(rows, dim)
                )

                metadata_fields = meta.get("metadata_fields", None)
                separator = meta.get("separator", None)
                add_metadata_keys_to_text = meta.get("add_metadata_keys_to_text", None)

            embeddings_tensor = torch.from_numpy(mmap_array)

            if precision is not None:
                embeddings_tensor = embeddings_tensor.to(precision)

            if device != "cpu":
                embeddings_tensor = embeddings_tensor.to(device)

        else:
            embeddings: List[torch.Tensor] = []

            with open(documents_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(tqdm(f, desc="Loading embeddings")):
                    record = json.loads(line)
                    vector = record.pop("vector", None)

                    if vector is None:
                        raise ValueError(
                            f"Record is missing 'vector' field at line {i+1}"
                        )

                    embedding = torch.tensor(vector, device=device)
                    if precision is not None:
                        embedding = embedding.to(precision)

                    embeddings.append(embedding)

                    documents.add_document(Document(**record))

            embeddings_tensor = torch.stack(embeddings)

        if metadata_fields is None:
            logger.warning(
                "No metadata fields specified. Defaulting to no metadata fields. Please make sure this is intentional."
            )
        if separator is None:
            logger.warning(
                "No separator specified. Defaulting to space (' '). Please make sure this is intentional."
            )
            separator = " "

        if add_metadata_keys_to_text is None:
            logger.warning(
                "No value specified for add_metadata_keys_to_text. Defaulting to False. Please make sure this is intentional."
            )
            add_metadata_keys_to_text = False

        indexer = cls(
            documents=documents,
            embeddings=embeddings_tensor,
            device=device,
            metadata_fields=metadata_fields,
            separator=separator,
            add_metadata_keys_to_text=add_metadata_keys_to_text,
        )

        return indexer
