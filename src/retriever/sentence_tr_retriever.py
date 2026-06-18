import platform
from typing import List

import sentence_transformers as st
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.models import ModelInputs, RetrievedSample
from src.data.dataset import BaseDataset
from src.index.base import Indexer
from src.retriever.base import Retriever


class SentenceTransformersRetriever(Retriever):
    """
    Retriever backed by SentenceTransformers.
    """

    def __init__(
        self,
        model: str | st.SentenceTransformer,
        indexer: Indexer | str | None = None,
        device: str | torch.device = "cpu",
        index_device: str | torch.device | None = None,
        index_precision: str | int = 32,
        model_kwargs: dict | None = None,
        normalize_embeddings: bool = False,
        query_prompt_name: str | None = None,
        passage_prompt_name: str | None = None,
        task: str | None = None,
    ) -> None:
        index_device = index_device or device
        super().__init__(
            indexer=indexer,
            index_device=index_device,
            index_precision=index_precision,
        )

        model_kwargs = model_kwargs or {}

        if isinstance(model, str):
            self.model_name = model
            self.model = st.SentenceTransformer(
                model_name_or_path=model,
                device=str(device),
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
        else:
            self.model = model
            self.model_name = (
                model.model_card_data.base_model
                if model.model_card_data
                else "sentence-transformers"
            )
            self.model = self.model.to(torch.device(device))

        self.normalize_embeddings = normalize_embeddings
        self.query_prompt_name = query_prompt_name
        self.passage_prompt_name = passage_prompt_name

        self.task = task

    def encode(
        self,
        text: str | List[str],
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
        prompt_name: str | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode text into sentence embeddings.
        """
        if isinstance(text, str):
            text = [text]

        if normalize_embeddings is None:
            normalize_embeddings = self.normalize_embeddings

        embeddings = self.model.encode(
            sentences=text,
            batch_size=batch_size or len(text),
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            prompt_name=prompt_name,
            task=self.task,
            *args,
            **kwargs,
        )

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        return embeddings

    def retrieve(
        self,
        text: str | List[str] | None = None,
        text_pair: str | List[str] | None = None,
        k: int = 100,
        batch_size: int | None = None,
        num_workers: int = 4,
        progress_bar: bool = False,
        normalize_embeddings: bool | None = None,
        *args,
        **kwargs,
    ) -> List[List[RetrievedSample]]:
        if self.indexer is None:
            raise ValueError(
                "The indexer must be indexed before it can be used within the retriever."
            )
        if text is None:
            raise ValueError("`text` must be provided for retrieval.")

        if isinstance(text, str):
            text = [text]

        if text_pair is not None and isinstance(text_pair, str):
            text_pair = [text_pair]

        if text_pair is None:
            payload = text
        else:
            payload = list(zip(text, text_pair))

        dataloader = DataLoader(
            BaseDataset(name="questions", data=payload),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

        pbar = None
        if progress_bar:
            pbar = tqdm(total=len(text), desc="Retrieving passages", unit="sample")

        retrieved = []
        try:
            for batch in dataloader:
                question_encodings = self.encode(
                    text=batch["text"],
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    prompt_name=self.query_prompt_name,
                    *args,
                    **kwargs,
                )
                retrieved += self.indexer.search(question_encodings, k)
                if pbar is not None:
                    pbar.update(len(batch["text"]))
        except AttributeError as e:
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

    @staticmethod
    def collate_fn(batch) -> ModelInputs:
        if not isinstance(batch, list):
            batch = [batch]

        if len(batch) == 0:
            return ModelInputs(text=[])

        if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2:
            text = [
                f"{sample[0]} {sample[1]}" if sample[1] else sample[0]
                for sample in batch
            ]
        else:
            text = batch

        return ModelInputs(text=text)

    @property
    def device(self) -> torch.device:
        return self.model.device
