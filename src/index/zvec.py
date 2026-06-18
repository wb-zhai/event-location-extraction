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


class ZvecIndexer(Indexer):
    pass
