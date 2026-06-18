from typing import List

import torch

from src.common.models import ModelInputs, RetrievedSample
from src.index.base import Indexer


class Retriever:
    """
    Base class for document retrievers.
    """

    supports_concurrent_encode: bool = False

    def __init__(
        self,
        indexer: str | Indexer | None = None,
        index_device: str | torch.device = "cpu",
        index_precision: str | int = 32,
        **kwargs,
    ) -> None:
        # indexer stuff
        index_device = torch.device(index_device)
        self.index_device = index_device
        self.index_precision = index_precision
        if indexer is not None and isinstance(indexer, str):
            indexer = Indexer.from_pretrained(
                indexer, device=index_device, precision=index_precision, **kwargs
            )
        self.indexer: Indexer | None = indexer

    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def retrieve(
        self,
        text: str | List[str] | None = None,
        text_pair: str | List[str] | None = None,
        k: int | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        num_workers: int = 4,
        progress_bar: bool = False,
        *args,
        **kwargs,
    ) -> List[List[RetrievedSample]]:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """
        The device of the model.
        """
        return torch.device("cpu")

    @staticmethod
    def collate_fn(*args, **kwargs) -> ModelInputs:
        """The collate function to use for the retrieval."""
        raise NotImplementedError
