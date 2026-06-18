import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from src.common.log import get_logger
from src.common.models import RetrievedSample
from src.common.torch_utils import get_autocast_context
from src.index.base import Indexer
from src.index.documents import Document, DocumentStore
from src.retriever.gemini import GeminiRetriever

logger = get_logger(__name__)


class GeminiIndexer(Indexer):
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
        separator: str = " ",
        device: str = "cpu",
    ) -> None:
        if metadata_fields is None:
            metadata_fields = []

        self.metadata_fields = metadata_fields
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

    def index(
        self,
        retriever: GeminiRetriever,
        documents: Optional[List[Document]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        force_reindex: bool = False,
        *args,
        **kwargs,
    ) -> "GeminiIndexer":

        if documents is None and self.documents is None:
            raise ValueError("Documents must be provided.")

        if self.embeddings is not None and not force_reindex and documents is None:
            logger.info(
                "Embeddings are already present and `force_reindex` is `False`. Skipping indexing."
            )
            return self

        if force_reindex:
            if documents is not None:
                self.documents.add_documents(documents)
            data = [k for k in self.get_passages()]

        else:
            if documents is not None:
                data = [k for k in self.get_passages(DocumentStore(documents))]
                # add the documents to the actual document store
                self.documents.add_documents(documents)
            else:
                if self.embeddings is None:
                    data = [k for k in self.get_passages()]
                else:
                    logger.info(
                        "Embeddings are already present and `force_reindex` is `False`. Skipping indexing."
                    )
                    return self

        if not data:
            logger.warning("No passages to index. Skipping embedding generation.")
            self.embeddings = None
            return self

        batch_size = min(batch_size, len(data))

        def batch_gen(iterable, n=1):
            length = len(iterable)
            for ndx in range(0, length, n):
                yield iterable[ndx : min(ndx + n, length)]

        batches = list(batch_gen(data, batch_size))
        passage_embeddings: List[torch.Tensor | None] = [None] * len(batches)

        if num_workers <= 1:
            for idx, batch in enumerate(tqdm(batches, total=len(batches))):
                passage_embeddings[idx] = retriever.encode(batch)
        else:
            # fan out work while capturing original batch order
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(retriever.encode, batch): idx
                    for idx, batch in enumerate(batches)
                }
                for future in tqdm(
                    as_completed(future_to_idx), total=len(future_to_idx)
                ):
                    idx = future_to_idx[future]
                    passage_embeddings[idx] = future.result()

        if any(embedding is None for embedding in passage_embeddings):
            raise RuntimeError("Failed to compute all embeddings during indexing.")

        passage_embeddings_tensors = [
            embedding for embedding in passage_embeddings if embedding is not None
        ]

        self.embeddings = torch.vstack(passage_embeddings_tensors)
        self.embeddings = self.embeddings.to(self.device_in_init)

        # free up memory from the unused variable
        del passage_embeddings

        return self

    @torch.inference_mode()
    @torch.no_grad()
    def search(self, query: Any, k: int = 1, *args, **kwargs) -> List:
        if self.embeddings is None:
            raise ValueError("No embeddings found. Please run `index()` first.")

        with get_autocast_context(self.embeddings.device, self.embeddings.dtype):
            # move query to the same device as embeddings
            query = query.to(self.embeddings.device)
            if query.dtype != self.embeddings.dtype:
                query = query.to(self.embeddings.dtype)
            similarity = torch.matmul(query, self.embeddings.T)
            # similarity = self.mm(query)
            # Retrieve the indices of the top k passage embeddings
            retriever_out: torch.return_types.topk = torch.topk(
                similarity, k=min(k, similarity.shape[-1]), dim=1
            )

        # get int values
        batch_top_k: List[List[int]] = retriever_out.indices.detach().cpu().tolist()
        # get float values
        batch_scores: List[List[float]] = retriever_out.values.detach().cpu().tolist()
        # Retrieve the passages corresponding to the indices
        batch_docs = [
            [self.get_document_from_index(i) for i in indices]
            for indices in batch_top_k
        ]
        # build the output object
        batch_retrieved_samples = [
            [
                RetrievedSample(document=doc, score=score)
                for doc, score in zip(docs, scores)
            ]
            for docs, scores in zip(batch_docs, batch_scores)
        ]
        return batch_retrieved_samples
