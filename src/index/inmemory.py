from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from typing import Any, Callable, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.log import get_logger
from src.common.models import RetrievedSample
from src.common.torch_utils import PRECISION_MAP, get_autocast_context
from src.data.dataset import BaseDataset
from src.index.base import SEARCH_METRIC_TO_FUNCTION, Indexer
from src.index.documents import Document, DocumentStore
from src.retriever.base import Retriever
from src.retriever.sentence_tr_retriever import SentenceTransformersRetriever

logger = get_logger(__name__)


class InMemoryIndexer(Indexer):
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
        device: str = "cpu",
        precision: str | int | torch.dtype = 32,
    ) -> None:
        super().__init__(
            documents=documents,
            embeddings=embeddings,
            metadata_fields=metadata_fields,
            add_metadata_keys_to_text=add_metadata_keys_to_text,
            separator=separator,
            device=device,
        )

        # convert the embeddings to the desired precision
        if precision is not None:
            if (
                self.embeddings is not None
                and str(device).split(":")[0] == "cpu"
                and PRECISION_MAP[precision] == PRECISION_MAP[16]
            ):
                logger.info(
                    f"Precision `{precision}` is not supported on CPU. "
                    f"Using `{PRECISION_MAP[32]}` instead."
                )
                precision = 32

            if (
                self.embeddings is not None
                and self.embeddings.dtype != PRECISION_MAP[precision]
            ):
                logger.info(
                    f"Index vectors are of type {self.embeddings.dtype}. "
                    f"Converting to {PRECISION_MAP[precision]}."
                )
                self.embeddings = self.embeddings.to(PRECISION_MAP[precision])
        else:
            # TODO: a bit redundant, fix this eventually
            if (
                device == "cpu"
                and self.embeddings is not None
                and self.embeddings.dtype != torch.float32
            ):
                logger.info(
                    f"Index vectors are of type {self.embeddings.dtype}. "
                    f"Converting to {PRECISION_MAP[32]}."
                )
                self.embeddings = self.embeddings.to(PRECISION_MAP[32])

        # move the embeddings to the desired device
        target_device = torch.device(device)
        if self.embeddings is not None and self.embeddings.device != target_device:
            self.embeddings = self.embeddings.to(target_device)

        # precision to be used for the embeddings
        self.precision = precision

    @torch.no_grad()
    @torch.inference_mode()
    def index(
        self,
        retriever: Retriever,
        documents: Optional[List[Document]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int | None = None,
        collate_fn: Callable | None = None,
        encoder_precision: str | int = 32,
        compute_on_cpu: bool = False,
        force_reindex: bool = False,
        dataloader: DataLoader | None = None,
        low_gpu_memory: bool = False,
        encode_concurrency: int | None = None,
        *args,
        **kwargs,
    ) -> "InMemoryIndexer":
        if documents is None and self.documents is None:
            raise ValueError(
                "Documents must be provided either during initialization or indexing."
            )

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

        if collate_fn is None:
            collate_fn = retriever.collate_fn

        batch_size = min(batch_size, len(data))
        dataloader = DataLoader(
            BaseDataset(name="passages", data=data),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            prefetch_factor=(
                max(1, 8 * batch_size // num_workers) if num_workers > 0 else None
            ),
        )

        encode_concurrency = encode_concurrency or num_workers
        use_concurrent_encode = (
            retriever.supports_concurrent_encode and encode_concurrency > 1
        )

        if use_concurrent_encode:
            passage_embeddings = self.api_encode(
                retriever,
                dataloader=dataloader,
                encode_concurrency=encode_concurrency,
                encoder_precision=encoder_precision,
                low_gpu_memory=low_gpu_memory,
            )
        else:
            passage_embeddings = self.torch_encode(
                retriever,
                dataloader=dataloader,
                encoder_precision=encoder_precision,
                compute_on_cpu=compute_on_cpu,
                low_gpu_memory=low_gpu_memory,
            )

        # move the passage embeddings to the gpu if needed
        if not self.device == "cpu":
            passage_embeddings = passage_embeddings.to(PRECISION_MAP[self.precision])
            if self.device != passage_embeddings.device:
                passage_embeddings = passage_embeddings.to(self.device)

        self.embeddings = passage_embeddings

        return self

    def torch_encode(
        self,
        retriever: Retriever,
        dataloader: DataLoader,
        encoder_precision: str | int,
        compute_on_cpu: bool,
        low_gpu_memory: bool,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode the passages using the retriever's `encode()` method in a non-concurrent manner,
        which is suitable for PyTorch-based retrievers.

        Returns:
            :obj:`torch.Tensor`: The embeddings of the passages.
        """
        # Create empty lists to store the passage embeddings and passage index
        passage_embeddings: List[torch.Tensor] = []

        retriever_device = "cpu" if compute_on_cpu else retriever.device
        with get_autocast_context(retriever_device, encoder_precision):
            # Iterate through each batch in the dataloader
            for batch in tqdm(dataloader, desc="Indexing"):
                # Move the batch to the device
                batch = batch.to(retriever_device)
                # Compute the passage embeddings
                if isinstance(retriever, SentenceTransformersRetriever):
                    batch.update(
                        {
                            "prompt_name": retriever.passage_prompt_name,
                        }
                    )
                passage_outs = retriever.encode(**batch)
                # Append the passage embeddings to the list
                if self.device == "cpu":
                    passage_embeddings.extend([c.detach().cpu() for c in passage_outs])
                else:
                    passage_embeddings.extend([c for c in passage_outs])

        # move the passage embeddings to the CPU if not already done
        # the move to cpu and then to gpu is needed to avoid OOM when using mixed precision
        # move them only if there is a low GPU memory environment
        if not self.device == "cpu" and low_gpu_memory:
            passage_embeddings = [c.detach().cpu() for c in passage_embeddings]

        passage_embeddings_tensor: torch.Tensor = torch.stack(passage_embeddings, dim=0)
        return passage_embeddings_tensor

    def api_encode(
        self,
        retriever: Retriever,
        dataloader: DataLoader,
        encode_concurrency: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode the passages using the retriever's `encode()` method in a concurrent manner,
        which is suitable for API-based retrievers.

        Returns:
            :obj:`torch.Tensor`: The embeddings of the passages.
        """
        logger.info(
            "Using concurrent encode with %s workers for `%s`.",
            encode_concurrency,
            retriever.__class__.__name__,
        )
        pending: dict[Future[torch.Tensor], int] = {}
        batch_outputs: dict[int, torch.Tensor] = {}
        batch_index = 0

        # Create empty lists to store the passage embeddings and passage index
        passage_embeddings: List[torch.Tensor] = []

        def consume_completed(
            futures_to_index: dict[Future[torch.Tensor], int],
            return_when: str,
        ) -> dict[Future[torch.Tensor], int]:
            if not futures_to_index:
                return futures_to_index
            done, not_done = wait(
                set(futures_to_index),
                return_when=return_when,
            )
            for future in done:
                idx = futures_to_index[future]
                batch_outputs[idx] = future.result()
            return {future: futures_to_index[future] for future in not_done}

        with ThreadPoolExecutor(max_workers=encode_concurrency) as executor:
            for batch in tqdm(dataloader, desc="Indexing"):
                texts = batch["text"]
                if isinstance(texts, str):
                    texts = [texts]
                pending[executor.submit(retriever.encode, texts)] = batch_index
                batch_index += 1

                if len(pending) >= encode_concurrency * 2:
                    pending = consume_completed(pending, FIRST_COMPLETED)

            while pending:
                pending = consume_completed(pending, ALL_COMPLETED)

        for idx in range(batch_index):
            passage_outs = batch_outputs[idx]
            passage_embeddings.extend([c for c in passage_outs])

        passage_embeddings_tensor: torch.Tensor = torch.stack(passage_embeddings, dim=0)
        return passage_embeddings_tensor

    @torch.inference_mode()
    @torch.no_grad()
    def search(
        self, query: Any, k: int = 1, metric: str = "matmul", *args, **kwargs
    ) -> List:
        if self.embeddings is None:
            raise ValueError("No embeddings found. Please run `index()` first.")

        with get_autocast_context(self.embeddings.device, self.embeddings.dtype):
            # move query to the same device as embeddings
            query = query.to(self.embeddings.device)
            if query.dtype != self.embeddings.dtype:
                query = query.to(self.embeddings.dtype)
            # similarity = torch.matmul(query, self.embeddings.T)
            similarity_fn = SEARCH_METRIC_TO_FUNCTION.get(metric, None)
            if similarity_fn is None:
                raise ValueError(
                    f"Unsupported similarity metric: {metric}."
                    f"Supported metrics are: {list(SEARCH_METRIC_TO_FUNCTION.keys())}"
                )

            similarity = similarity_fn(query, self.embeddings)

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
        batch_retrieved_samples = []
        for docs, scores in zip(batch_docs, batch_scores):
            retrieved_samples = []
            for doc, score in zip(docs, scores):
                if doc is None:
                    continue
                retrieved_samples.append(RetrievedSample(document=doc, score=score))
            batch_retrieved_samples.append(retrieved_samples)
        return batch_retrieved_samples
