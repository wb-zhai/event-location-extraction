import importlib
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gliner.modeling.span_rep import SpanRepLayer
from gliner2 import GLiNER2
from gliner2.layers import CountLSTM, CountLSTMoE, CountLSTMv2, create_mlp
from gliner2.processor import PreprocessedBatch, SamplingConfig, SchemaTransformer
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


class EventArgumentExtractionEvaluatorGliNER2(GLiNER2):
    @classmethod
    def from_pretrained(cls, repo_or_dir: str, **kwargs):
        """
        Load model from Hugging Face Hub or local directory.

        Args:
            repo_or_dir: HuggingFace repo ID or local directory path.
            quantize: If True, convert model to fp16 after loading.
            compile: If True, torch.compile the encoder and span-rep
                with ``dynamic=True`` for fused GPU kernels.
            map_location: Device to load the model onto (e.g. "cpu", "cuda").
            **kwargs: Additional keyword arguments.

        To use a LoRA adapter:
            1. Load the base model first
            2. Then load the adapter using model.load_adapter()

        Example:
            model = Extractor.from_pretrained("base-model-name")
            model.load_adapter("path/to/adapter")
        """
        from huggingface_hub import hf_hub_download

        quantize = kwargs.pop("quantize", False)
        compile_model = kwargs.pop("compile", False)
        map_location = kwargs.pop("map_location", None)

        def download_or_local(repo, filename):
            if os.path.isdir(repo):
                return os.path.join(repo, filename)
            return hf_hub_download(repo, filename)

        config_path = download_or_local(repo_or_dir, "config.json")
        config = cls.config_class.from_pretrained(config_path)

        encoder_config_path = download_or_local(
            repo_or_dir, "encoder_config/config.json"
        )
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)

        tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)
        model = cls(config, encoder_config=encoder_config, tokenizer=tokenizer)

        # Load weights
        try:
            model_path = download_or_local(repo_or_dir, "model.safetensors")
            state_dict = load_file(model_path)
        except Exception:
            model_path = download_or_local(repo_or_dir, "pytorch_model.bin")
            state_dict = torch.load(model_path, map_location="cpu")

        # Handle embedding size mismatch
        try:
            saved_emb = state_dict["encoder.embeddings.word_embeddings.weight"]
            model_emb = model.encoder.embeddings.word_embeddings.weight
            if saved_emb.shape[0] != model_emb.shape[0]:
                extra = model_emb.shape[0] - saved_emb.shape[0]
                state_dict["encoder.embeddings.word_embeddings.weight"] = torch.cat(
                    [saved_emb, torch.randn(extra, saved_emb.shape[1]) * 0.02], dim=0
                )
        except KeyError:
            pass

        model.load_state_dict(state_dict)

        if map_location is not None:
            model = model.to(map_location)

        if quantize:
            model.quantize()

        if compile_model:
            model.compile()

        return model
