import os
import platform
import random
import re
import time
from typing import List

import dotenv
import torch
from google import genai
from google.genai import types
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.log import get_logger
from src.common.models import ModelInputs, RetrievedSample
from src.data.dataset import BaseDataset
from src.index.base import Indexer
from src.retriever.base import Retriever

dotenv.load_dotenv()

logger = get_logger(__name__)


GCP_PROJECT: str | None = os.getenv("PROJECT_ID")
GCP_LOCATION: str | None = os.getenv("VERTEX_LOCATION")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")


class GeminiRetriever(Retriever):
    """
    Retriever that uses Gemini embeddings for document retrieval.
    """

    supports_concurrent_encode: bool = True
    default_max_retries: int = 5
    default_initial_backoff_seconds: float = 0.5
    default_backoff_multiplier: float = 2.0
    default_max_backoff_seconds: float = 8.0
    default_backoff_jitter_seconds: float = 0.2

    @staticmethod
    def _extract_retry_delay_seconds(error: Exception) -> float | None:
        """Extract a retry delay hint (in seconds) from Gemini/Google API errors."""
        error_str = str(error)
        patterns = [
            r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s",
            r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s['\"]",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        indexer: Indexer | str | None = None,
        index_device: str | torch.device = "cpu",
        index_precision: str | int = 32,
        task_type: str = "SEMANTIC_SIMILARITY",
        output_dimensionality: int = 1536,
    ) -> None:

        super().__init__(
            indexer=indexer, index_device=index_device, index_precision=index_precision
        )

        self.model = model
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        if GEMINI_API_KEY is not None:
            self.client = genai.Client()
        else:
            if GCP_PROJECT is None or GCP_LOCATION is None:
                raise ValueError(
                    "Either GEMINI_API_KEY or both PROJECT_ID and VERTEX_LOCATION must be set in environment variables."
                )
            self.client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT,
                location=GCP_LOCATION,
            )

    def _encode_batch(
        self,
        text: str | List[str],
        output_dimensionality: int | None = None,
        task_type: str | None = None,
        poll_interval_seconds: float = 30.0,
        display_name: str | None = None,
    ) -> torch.Tensor:
        """
        Encode texts using the Gemini Batch API with inline requests.

        This is an asynchronous, cost-effective alternative to :meth:`encode`
        that submits an embedding batch job and polls until completion.
        Batch API usage is priced at 50% of the standard interactive API cost.

        Args:
            text: The text or list of texts to encode.
            output_dimensionality: The dimensionality of the output embeddings.
                Defaults to the instance's ``output_dimensionality``.
            task_type: The task type for the embeddings.
                Defaults to the instance's ``task_type``.
            poll_interval_seconds: How often (in seconds) to poll for job
                completion. Defaults to ``30.0``.
            display_name: Optional human-readable name for the batch job.

        Returns:
            :obj:`torch.Tensor`: The embeddings of the input texts.

        Raises:
            RuntimeError: If the batch job fails, is cancelled, or expires.
            ValueError: If the batch job returns no embedding results.
        """
        if isinstance(text, str):
            text = [text]

        if output_dimensionality is None:
            output_dimensionality = self.output_dimensionality

        if task_type is None:
            task_type = self.task_type

        # Build inline EmbedContentBatch
        inlined_requests = types.EmbedContentBatch(
            contents=text,  # pyright: ignore[reportArgumentType]
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            ),
        )

        # Submit the batch embedding job
        batch_job = self.client.batches.create_embeddings(
            model=self.model,
            src=types.EmbeddingsBatchJobSource(inlined_requests=inlined_requests),
            config=types.CreateEmbeddingsBatchJobConfig(
                display_name=display_name or "gemini-retriever-encode-batch",
            ),
        )

        job_name = batch_job.name
        if job_name is None:
            raise RuntimeError("Batch job creation returned no job name.")
        logger.info("Created batch embedding job: %s", job_name)

        # Poll until completion
        _COMPLETED_STATES = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        while (
            batch_job.state is not None
            and batch_job.state.name not in _COMPLETED_STATES
        ):
            logger.info(
                "Batch job %s state: %s. Polling again in %.0fs.",
                job_name,
                batch_job.state.name,
                poll_interval_seconds,
            )
            time.sleep(poll_interval_seconds)
            batch_job = self.client.batches.get(name=job_name)

        if batch_job.state is None or batch_job.state.name != "JOB_STATE_SUCCEEDED":
            state_name = batch_job.state.name if batch_job.state else "UNKNOWN"
            raise RuntimeError(
                f"Batch embedding job {job_name} finished with state "
                f"{state_name}. Error: {getattr(batch_job, 'error', None)}"
            )

        logger.info("Batch embedding job %s succeeded.", job_name)

        # Extract embeddings from inline responses
        embed_responses = (
            batch_job.dest.inlined_embed_content_responses if batch_job.dest else None
        )
        if not embed_responses:
            raise ValueError(
                f"Batch job {job_name} succeeded but returned no embedding responses."
            )

        embeddings = []
        for resp in embed_responses:
            if resp.response is None or resp.response.embedding is None:
                raise ValueError(
                    f"Batch job {job_name}: one of the responses contains no embedding."
                )
            embeddings.append(resp.response.embedding.values)

        embeddings = torch.tensor(embeddings)
        if output_dimensionality != 3072:
            embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return embeddings

    def encode(
        self,
        text: str | List[str],
        output_dimensionality: int | None = None,
        task_type: str | None = None,
        max_retries: int | None = None,
        initial_backoff_seconds: float | None = None,
        backoff_multiplier: float | None = None,
        max_backoff_seconds: float | None = None,
        backoff_jitter_seconds: float | None = None,
        batch_api: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode a list of texts into embeddings using Gemini.
        Args:
            texts (:obj:`str` or :obj:`List[str]`):
                The text or list of texts to encode.
            batch_size (:obj:`int`, `optional`):
                The batch size for encoding. Defaults to 32.
            output_dimensionality (:obj:`int`, `optional`):
                The dimensionality of the output embeddings. Defaults to 1536.
        Returns:
            :obj:`torch.Tensor`: The embeddings of the input texts.
        """
        if batch_api:
            return self._encode_batch(
                text=text,
                output_dimensionality=output_dimensionality,
                task_type=task_type,
                **kwargs,
            )

        if isinstance(text, str):
            text = [text]

        if output_dimensionality is None:
            output_dimensionality = self.output_dimensionality

        if task_type is None:
            task_type = self.task_type

        if max_retries is None:
            max_retries = self.default_max_retries
        if initial_backoff_seconds is None:
            initial_backoff_seconds = self.default_initial_backoff_seconds
        if backoff_multiplier is None:
            backoff_multiplier = self.default_backoff_multiplier
        if max_backoff_seconds is None:
            max_backoff_seconds = self.default_max_backoff_seconds
        if backoff_jitter_seconds is None:
            backoff_jitter_seconds = self.default_backoff_jitter_seconds

        response = None
        attempt = 0
        delay_seconds = initial_backoff_seconds
        while attempt <= max_retries:
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=text,  # pyright: ignore[reportArgumentType]
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=output_dimensionality,
                    ),
                )
                if response.embeddings is None:
                    raise ValueError("Failed to get embeddings from Gemini.")
                break
            except Exception as error:
                if attempt == max_retries:
                    raise RuntimeError(
                        "Gemini embedding request failed after retries."
                    ) from error

                is_429_error = "429" in str(error)
                backoff_sleep_seconds = min(delay_seconds, max_backoff_seconds)

                if is_429_error:
                    retry_delay_seconds = self._extract_retry_delay_seconds(error)
                    if retry_delay_seconds is not None:
                        sleep_seconds = retry_delay_seconds
                    else:
                        sleep_seconds = backoff_sleep_seconds
                    logger.warning(
                        "Gemini returned 429 RESOURCE_EXHAUSTED (attempt %d/%d). Retrying in %.2fs. Error: %s",
                        attempt + 1,
                        max_retries + 1,
                        sleep_seconds,
                        error,
                    )
                else:
                    sleep_seconds = backoff_sleep_seconds
                    if backoff_jitter_seconds > 0:
                        sleep_seconds += random.uniform(0.0, backoff_jitter_seconds)
                    logger.warning(
                        "Gemini embedding request failed (attempt %d/%d). Retrying in %.2fs. Error: %s",
                        attempt + 1,
                        max_retries + 1,
                        sleep_seconds,
                        error,
                    )

                time.sleep(sleep_seconds)
                delay_seconds *= backoff_multiplier
                attempt += 1

        if response is None or response.embeddings is None:
            raise ValueError("Failed to get embeddings from Gemini.")

        embeddings = [emb.values for emb in response.embeddings]
        embeddings = torch.tensor(embeddings)
        if output_dimensionality != 3072:
            embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return embeddings

    def retrieve(
        self,
        text: str | List[str] | None = None,
        text_pair: str | List[str] | None = None,
        k: int = 100,
        precision: str | int | None = None,
        batch_size: int | None = None,
        num_workers: int = 4,
        progress_bar: bool = False,
        output_dimensionality: int | None = None,
        task_type: str | None = None,
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
            input_ids (`torch.Tensor`):
                The input ids of the questions.
            attention_mask (`torch.Tensor`):
                The attention mask of the questions.
            token_type_ids (`torch.Tensor`):
                The token type ids of the questions.
            k (`int`):
                The number of top passages to retrieve.
            max_length (`int | None`):
                The maximum length of the questions.
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

        Returns:
            `List[List[RetrievedSample]]`: The retrieved passages and their indices.
        """
        if self.indexer is None:
            raise ValueError(
                "The indexer must be indexed before it can be used within the retriever."
            )
        if text is None:
            raise ValueError("`text` must be provided for retrieval.")

        if task_type is None:
            task_type = self.task_type

        if output_dimensionality is None:
            output_dimensionality = self.output_dimensionality

        if isinstance(text, str):
            text = [text]

        if text_pair is not None:
            if isinstance(text_pair, str):
                text_pair = [text_pair]
            # combine text and text_pair into a single list of strings for encoding
            # Gemini embedding API only takes a single list of strings as input, so we need to combine text and text_pair
            text = [f"{t} {tp}" for t, tp in zip(text, text_pair)]

        dataloader = DataLoader(
            BaseDataset(name="questions", data=text),
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
                    batch["text"],
                    output_dimensionality=output_dimensionality,
                    task_type=task_type,
                )
                retrieved += self.indexer.search(question_encodings, k, metric="cosine")
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

    @staticmethod
    def collate_fn(batch) -> ModelInputs:
        return ModelInputs(text=batch)

    @property
    def tokenizer(self):
        """
        The tokenizer of the model. Gemini does not have a tokenizer, but we return a dummy tokenizer for compatibility with the base retriever class.

        Returns:
            A dummy tokenizer that return the input text as is.
        """

        class DummyTokenizer:
            def __call__(self, text, *args, **kwargs):
                return {"text": text}

        return DummyTokenizer()
