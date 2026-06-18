import platform
from typing import Callable, List

import dotenv
import sentence_transformers as st
import torch
import transformers as tr
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from src.common.log import get_logger
from src.common.models import ModelInputs, RetrievedSample
from src.common.torch_utils import get_autocast_context
from src.data.dataset import BaseDataset
from src.index.base import Indexer
from src.retriever.base import Retriever

dotenv.load_dotenv()

logger = get_logger(__name__)


class HuggingFaceRetriever(Retriever):
    """
    Retriever that uses HuggingFace PreTrainedModel for document retrieval.
    """

    def __init__(
        self,
        model: str | tr.PreTrainedModel,
        indexer: Indexer | None = None,
        device: str | torch.device = "cpu",
        precision: str | int = 32,
        tokenizer: tr.PreTrainedTokenizer | None = None,
        index_device: str | torch.device | None = None,
        index_precision: str | int | None = None,
        model_kwargs: dict | None = None,
    ) -> None:

        index_device = index_device or device
        index_precision = index_precision or precision
        super().__init__(
            indexer=indexer, index_device=index_device, index_precision=index_precision
        )

        model_kwargs = model_kwargs or {}
        model_kwargs.update({"trust_remote_code": True})

        if isinstance(model, str):
            self.model_name: str = model
            self.model: tr.PreTrainedModel = tr.AutoModel.from_pretrained(
                model, **model_kwargs
            )
        else:
            self.model_name: str = model.config._name_or_path
            self.model: tr.PreTrainedModel = model

        # move the model to the specified device
        self.model = self.model.to(torch.device(device))

        # set the precision
        self.precision = precision

        # lazy load the tokenizer for inference
        self._tokenzer = tokenizer

    def encode(self, **kwargs) -> torch.Tensor:
        """
        Encode a list of texts into embeddings using an HuggingFace PreTrainedModel.

        Returns:
            :obj:`torch.Tensor`: The embeddings of the input texts.
        """
        attention_mask = kwargs.get("attention_mask", None)
        mean_pooling = kwargs.get("mean_pooling", True)

        model_outputs = self.model.forward(**kwargs)

        if attention_mask is None or not mean_pooling:
            pooler_output = model_outputs.pooler_output
        else:
            token_embeddings = model_outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            pooler_output = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return pooler_output

    def retrieve(
        self,
        text: str | List[str] | None = None,
        text_pair: str | List[str] | None = None,
        k: int = 100,
        precision: str | int | None = None,
        batch_size: int | None = None,
        num_workers: int = 4,
        progress_bar: bool = False,
        max_length: int | None = None,
        collate_fn: Callable | None = None,
        *args,
        **kwargs,
    ) -> List[List[RetrievedSample]]:
        """
        Retrieve the passages for the questions.

        Args:
            text (`Optional[Union[str, List[str]]]`):
                The questions to retrieve the passages for.
            text_pair (`Optional[Union[str, List[str]]]`):
                The questions to retrieve the passages for.
            k (`int`):
                The number of top passages to retrieve.
            precision (`Optional[Union[str, int]]`):
                The precision to use for the model.
            collate_fn (`Callable`):
                The collate function to use for the retrieval.
            batch_size (`int`):
                The batch size to use for the retrieval.
            num_workers (`int`):
                The number of workers to use for the retrieval.
            progress_bar (`bool`):
                Whether to show a progress bar.
            max_length (`Optional[int]`):
                The maximum length to use for the tokenizer. If `None`, the tokenizer's default max length will be used.
            collate_fn (`Callable`):
                The collate function to use for the retrieval. If `None`, the default collate
                function will be used, which simply tokenizes the input text using the model's tokenizer.

        Returns:
            `List[List[RetrievedSample]]`: The retrieved passages and their indices.
        """
        if self.indexer is None:
            raise ValueError(
                "The indexer must be indexed before it can be used within the retriever."
            )
        if text is None:
            raise ValueError("`text` must be provided for retrieval.")

        if isinstance(text, str):
            text = [text]
        if text_pair is not None:
            if isinstance(text_pair, str):
                text_pair = [text_pair]
        else:
            text_pair = [None] * len(text)

        if collate_fn is None:
            tokenizer = self.tokenizer
            collate_fn = partial(
                self.collate_fn, max_length=max_length, tokenizer=tokenizer
            )

        dataloader = DataLoader(
            BaseDataset(name="questions", data=list(zip(text, text_pair))),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        pbar = None
        if progress_bar:
            pbar = tqdm(total=len(text), desc="Retrieving passages", unit="sample")

        retrieved = []
        try:
            with get_autocast_context(self.device, precision or self.precision):
                for batch in dataloader:
                    batch = batch.to(self.device)
                    question_encodings = self.encode(**batch)
                    retrieved += self.indexer.search(question_encodings, k)
                    if pbar is not None:
                        pbar.update(len(batch["text"]))
        except AttributeError as e:
            # apparently num_workers > 0 gives some issue on MacOS as of now
            if "mac" in platform.platform().lower():
                raise ValueError(
                    "DataLoader with num_workers > 0 is not supported on MacOS. "
                    "Please set num_workers=0 or try to run on a different machine."
                ) from e
            else:
                raise e

        if pbar is not None:
            pbar.close()

        return retrieved

    def collate_fn(
        self,
        batch: tuple,
        tokenizer: tr.PreTrainedTokenizer | None = None,
        max_length: int | None = None,
    ) -> ModelInputs:

        if tokenizer is None:
            tokenizer = self.tokenizer

        # get text and text pair
        if not isinstance(batch, list):
            batch = [batch]
        if batch and isinstance(batch[0], (tuple, list)):
            _text = [sample[0] for sample in batch]
            _text_pair = [sample[1] for sample in batch]
            _text_pair = None if any(t is None for t in _text_pair) else _text_pair
        else:
            _text = [str(s) for s in batch]
            _text_pair = None
        return ModelInputs(
            tokenizer(
                _text,
                text_pair=_text_pair,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length or tokenizer.model_max_length,
            )
        )

    @property
    def tokenizer(self) -> tr.PreTrainedTokenizer:
        """
        The tokenizer of the model. Lazy loaded to avoid unnecessary loading during indexing.

        Returns:
            `tr.PreTrainedTokenizer`: The tokenizer of the model.
        """
        if self._tokenzer is not None:
            return self._tokenzer

        self._tokenzer = tr.AutoTokenizer.from_pretrained(
            self.model.config.name_or_path
        )
        return self._tokenzer
