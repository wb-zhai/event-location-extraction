from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
from typing import Any, Union

import torch

from src.common.lightning_utils import move_data_to_device
from src.common.log import get_logger
from src.index.documents import Document

logger = get_logger(__name__)


class ModelInputs(UserDict):
    """Model input dictionary wrapper."""

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError(f"`ModelInputs` has no attribute `{item}`")

    def __getitem__(self, item: str) -> Any:
        return self.data[item]

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    def keys(self):
        """A set-like object providing a view on D's keys."""
        return self.data.keys()

    def values(self):
        """An object providing a view on D's values."""
        return self.data.values()

    def items(self):
        """A set-like object providing a view on D's items."""
        return self.data.items()

    def to(self, device: Union[str, torch.device]) -> ModelInputs:
        """
        Send all tensors values to device.
        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            :class:`tokenizers.ModelInputs`: The same instance of :class:`~tokenizers.ModelInputs`
            after modification.
        """
        self.data = move_data_to_device(self.data, device)
        return self


@dataclass
class RetrievedSample(dict):
    """
    Dataclass for the output of the GoldenRetriever model.
    """

    score: float
    document: Document

    def __post_init__(self) -> None:
        self._sync_mapping()

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in {"score", "document"} and {
            "score",
            "document",
        }.issubset(self.__dict__):
            self._sync_mapping()

    def _sync_mapping(self) -> None:
        self.clear()
        self["score"] = float(self.score)
        self["document"] = (
            self.document.to_dict()
            if hasattr(self.document, "to_dict")
            else self.document
        )

    def to_dict(self) -> dict:
        return dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievedSample":
        document = d["document"]
        return cls(
            score=float(d["score"]),
            document=(
                Document.from_dict(document) if isinstance(document, dict) else document
            ),
        )
