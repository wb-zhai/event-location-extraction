from dataclasses import dataclass
from typing import Dict, Any, Optional
from pydantic import BaseModel


@dataclass
class LLMResponse:
    """Response from LLM."""

    text: str
    metadata: Dict[str, Any] = None
    parsed: BaseModel | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMSettings:
    """Configuration settings for the LLM client."""

    # Model configuration
    model_name: str
    prompts_dir: Optional[str] = None
    project_id: str = "impactai-430615"
    location: str = "us-east1"

    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 8192

    # Reasoning and caching
    reasoning_effort: Optional[str] = None
    use_content_cache: bool = False

    # Request configuration
    timeout: int = 300  # seconds
    max_retries: int = 10
    retry_strategy: str = "exponential_backoff_retry"
    stream: bool = False
    verbose: bool = False

    # Gemini specific
    cache_display_name: str = "gemini-client-cache"
    cache_ttl: str = "3600s"

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary format for API calls."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any]) -> "LLMSettings":
        """Create settings instance from dictionary."""
        return cls(**settings_dict)