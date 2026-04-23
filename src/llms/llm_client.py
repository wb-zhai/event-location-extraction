import asyncio
import logging
import os
import pathlib
import threading
import traceback
from typing import Any, Dict, List, Optional

import jinja2
from anthropic import AnthropicVertex

# import litellm
from google import genai
from google.genai import types

# from litellm import acompletion
from openai import AsyncOpenAI, OpenAI
from openai._types import NOT_GIVEN
from openai.lib._parsing import parse_chat_completion
from pydantic import BaseModel, Field, create_model

from src.llms.settings import LLMSettings

REASONING_EFFORT = {"disable": 0, "low": 1024, "medium": 2048, "high": 4096}


logger = logging.getLogger(__name__)


def create_schema_from_dict(
    field_types: dict[str, type], name: str = "MyModel"
) -> BaseModel:
    """
    Create a Pydantic model from a dictionary of field names and types.

    Args:
        field_types (dict): Dictionary where keys are field names and values are Python types.
        name (str): Name of the model to be created.
    """
    # Build a model from field name and Python type
    fields = {name: (py_type, ...) for name, py_type in field_types.items()}
    return create_model(name, **fields)


class LLMResponse(BaseModel):
    """Response from LLM."""

    text: str
    metadata: Dict[str, Any] = {}
    parsed: BaseModel | None = None


class LLMClient:
    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        prompts_dir: str | os.PathLike | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        reasoning_effort: str | int | None = None,
        use_content_cache: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ):
        self.model_name = model_name

        # Initialize Jinja2 environment for loading prompts
        self.prompts_dir = prompts_dir
        self.prompt_env = self._setup_prompt_environment(prompts_dir)
        # Save the system prompt if provided
        self.system_prompt_name = system_prompt if system_prompt else None
        self.system_prompt = (
            self.render_prompt(system_prompt) if system_prompt else None
        )

        # Common settings for LLM generation
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        self.use_content_cache = use_content_cache
        self.verbose = verbose

    def generate(self, *args, **kwargs):
        """
        Generate a response using the configured LLM model.
        This method should be overridden by subclasses to implement specific generation logic.
        """
        raise NotImplementedError(
            "The `generate` method must be implemented by subclasses of LLMClient."
        )

    @staticmethod
    def parse_response_format(
        response_format: Any, add_cot_field: bool = True
    ) -> BaseModel:
        """
        Parse the response format into a Pydantic model.
        This method converts a dictionary or Pydantic model into a structured response format.

        Args:
            response_format: A dictionary or Pydantic BaseModel defining the response format.
            add_cot_field: Whether to add a 'chain_of_thoughts' field to the response format.

        Returns:
            BaseModel: A Pydantic model representing the response format.
        """
        if isinstance(response_format, dict):
            if add_cot_field:
                if "chain_of_thoughts" not in response_format:
                    response_format["chain_of_thoughts"] = str

                # put reasoning at the beginning of the response
                if "chain_of_thoughts" in response_format:
                    response_format = {
                        "chain_of_thoughts": response_format.pop("chain_of_thoughts"),
                        **response_format,
                    }
            response_format_model = create_schema_from_dict(
                response_format, name="ResponseFormat"
            )
        elif isinstance(response_format, BaseModel):
            response_format_model = response_format
        else:
            raise ValueError(
                "`response_format` must be a dictionary or a Pydantic `BaseModel` instance."
            )

        return response_format_model

    def render_prompt(self, template_name: str, **kwargs) -> str:
        """
        Render a prompt template with the provided variables.
        This method uses Jinja2 to load and render the prompt template.

        Args:
            template_name: Name of the prompt template to render.
            **kwargs: Variables to pass to the template for rendering.

        Returns:
            str: The rendered prompt text.
        """
        # Append .prompt if not already present
        if not template_name.endswith(".prompt"):
            template_name = f"{template_name}.prompt"

        template = self.prompt_env.get_template(template_name)
        rendered_prompt = template.render(**kwargs)
        return rendered_prompt

    @staticmethod
    def _setup_prompt_environment(
        prompts_dir: str | os.PathLike | None = None,
    ) -> jinja2.Environment:
        """
        Set up the Jinja2 environment for loading prompt templates.

        Args:
            prompts_dir: Directory containing prompt templates. If None, defaults to the config/prompts directory.

        Returns:
            jinja2.Environment: Configured Jinja2 environment for loading templates.
        """
        if prompts_dir is not None:
            base_dir = pathlib.Path(prompts_dir)
        else:
            # Determine the base path for prompt templates
            # Assuming prompts are stored in agent/config/prompts
            base_dir = (
                pathlib.Path(__file__).parent.parent.parent / "config" / "prompts"
            )

            if not base_dir.exists():
                logger.warning(
                    f"Prompt directory not found at {base_dir}. Using current directory."
                )
                base_dir = pathlib.Path(".")

        # Create a file system loader for the templates
        loader = jinja2.FileSystemLoader(base_dir)

        # Create the environment
        env = jinja2.Environment(
            loader=loader,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env


class GeminiLLMClient(LLMClient):
    """Client for interacting with Gemini APIs."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None,
        prompts_dir: str | os.PathLike | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        reasoning_effort: str | int | None = None,
        use_content_cache: bool = False,
        verbose: bool = False,
        cache_display_name: str = "gemini-client-cache",
        cache_ttl: str = "3600s",
        **kwargs: Any,
    ):
        """
        Initialize the Gemini LLM client.

        Args:
            model_name: Name of the Gemini model to use
            system_prompt: Optional system prompt to set the context
            prompts_dir: Directory containing prompt templates
            temperature: Sampling temperature for generation
            max_tokens: Maximum number of tokens to generate
            reasoning_effort: Level of reasoning effort (e.g., "low", "medium", "high")
            use_content_cache: Whether to use content caching for responses
            verbose: Whether to enable verbose logging
            cache_display_name: Display name for the content cache
            cache_ttl: Time-to-live for the content cache in seconds (default is 3600s)
        """

        super().__init__(
            model_name,
            system_prompt,
            prompts_dir,
            temperature,
            max_tokens,
            reasoning_effort,
            use_content_cache,
            verbose,
        )

        # Initialize the Google GenAI client
        if os.getenv("GEMINI_API_KEY") is None:
            raise ValueError(
                "`GEMINI_API_KEY` environment variable must be set to use GeminiLLMClient."
            )

        if "gemini" in self.model_name:
            self.client = genai.Client(
                vertexai=True, project=os.getenv("PROJECT_ID"), location="global"
            )
        elif (
            "deepseek-r1-0528-maas" in self.model_name
            or "deepseek-v3.1-maas" in self.model_name
        ):
            self.client = genai.Client(
                vertexai=True, project=os.getenv("PROJECT_ID"), location="us-central1"
            )
        elif "-maas" in self.model_name:
            self.client = genai.Client(
                vertexai=True, project=os.getenv("PROJECT_ID"), location="global"
            )

        # Setup explicit context cache
        self.cache = None
        if self.use_content_cache:
            if system_prompt is None:
                logger.warning(
                    "`system_prompt` is None, global content cache will not be created."
                )
            else:
                logger.info(
                    f"Setting up content cache with display name: {cache_display_name}"
                )
                # Create or get the cache
                self.cache = self.client.caches.create(
                    model=self.model_name,
                    config=types.CreateCachedContentConfig(
                        display_name=cache_display_name,  # used to identify the cache
                        system_instruction=self.system_prompt,
                        ttl=cache_ttl,
                    ),
                )

    def manage_content_cache(
        self,
        client: genai.Client,
        model_name: str,
        system_prompt: str,
        cache_display_name: str,
        cache_ttl: int,
    ):
        """
        Create or update a content cache for the specified model.

        Args:
            client: Instance of the genai.Client to interact with the Gemini API.
            model_name: Name of the model for which to create the cache.
            system_prompt: str: System prompt to set the context for the cache.
            cache_display_name: str: Display name for the cache, used to identify it.
            cache_ttl: str: Time-to-live for the cache in seconds (default is 3600s).

        Returns:
            types.CachedContent: The created or updated cache object.
        """
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=cache_display_name,  # used to identify the cache
                system_instruction=system_prompt,
                ttl=cache_ttl,
            ),
        )
        return cache

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        override_settings: Dict[str, Any] | None = None,
        response_format: Dict[str, Any] | BaseModel | None = None,
        add_cot_field: bool = True,
        conversation_history: List[Dict[str, str]] | None = None,
        reasoning_effort: str | int | None = None,
        streaming: bool = False,
        **kwargs,
    ):
        """
        Generate a structured response using the configured Gemini model.

        Args:
            prompt: Input prompt text
            system_prompt: Optional system prompt to set the context
            override_settings: Optional dictionary to override settings for this call
            response_format: Dict or Pydantic BaseModel defining the response format
            add_cot_field: Whether to add a chain of thought field to the response
            conversation_history: Optional list of previous messages in the conversation
            reasoning_effort: Optional reasoning effort level (e.g., "low", "medium", "high")
            streaming: Whether to stream the response
            **kwargs: Additional keyword arguments for the generation call

        Returns:
            LLMResponse: The generated response containing text, metadata, and parsed content.
        """

        if override_settings is None:
            override_settings = {}

        config = types.GenerateContentConfig(**override_settings)

        if reasoning_effort is None:
            reasoning_effort = 0
        elif isinstance(reasoning_effort, str):
            # check if it can be converted to an int
            try:
                reasoning_effort = int(reasoning_effort)
            except ValueError:
                # if it cannot be converted, check if it is a valid key in REASONING_EFFORT
                if reasoning_effort not in REASONING_EFFORT:
                    raise ValueError(
                        f"Invalid reasoning effort: {reasoning_effort}. "
                        f"Valid values are: {', '.join(REASONING_EFFORT.keys())}."
                    )
                # map to int
                reasoning_effort = REASONING_EFFORT[reasoning_effort]

        if "2.5" in self.model_name and "gemini" in self.model_name:
            # otherwise thinking budget is not supported
            thinking_config = types.ThinkingConfig(thinking_budget=reasoning_effort)
            config.thinking_config = thinking_config
        elif "3" in self.model_name and "gemini" in self.model_name:
            reasoning_effort = "low" if reasoning_effort <= 1024 else "high"
            thinking_config = types.ThinkingConfig(thinking_level=reasoning_effort)
            config.thinking_config = thinking_config
        elif "-maas" in self.model_name:
            thinking_config = types.ThinkingConfig(
                thinking_budget=reasoning_effort, include_thoughts=False
            )
            config.thinking_config = thinking_config

        response_format_model = None
        if response_format is not None:
            response_format_model = self.parse_response_format(
                response_format, add_cot_field
            )

            config.response_mime_type = "application/json"
            config.response_schema = response_format_model

        if system_prompt is not None:
            if system_prompt != self.system_prompt:
                # ignore cache
                config.system_instruction = system_prompt
        else:
            # create cache
            if self.cache is not None:
                cache = self.client.caches.get(name=self.cache.name)
                config.cached_content = cache.name
            else:
                config.system_instruction = self.system_prompt

        if self.verbose:
            if config.system_instruction is not None:
                logger.info(f"System Prompt: {config.system_instruction}")
            logger.info(f"User Prompt: {prompt}")

        if conversation_history is None:
            conversation_history = []

        contents = []
        # check if first message is a system message
        if (
            len(conversation_history) > 0
            and conversation_history[0].get("role") == "system"
        ):
            # if so, pop it from the conversation history
            # system message is handled via config.system_instruction
            conversation_history = conversation_history[1:]
        for conversation in conversation_history:
            if conversation.get("role") == "assistant":
                # we want to make sure to use the "model" role in case
                # we are using Gemini
                role = "model"
            else:
                role = conversation.get("role")
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=conversation.get("text"))],
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        try:
            if streaming:
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue[object] = asyncio.Queue()
                done_sentinel = object()

                def _stream_worker() -> None:
                    try:
                        for chunk in self.client.models.generate_content_stream(
                            model=self.model_name, config=config, contents=contents
                        ):
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
                        loop.call_soon_threadsafe(queue.put_nowait, done_sentinel)
                    except Exception as exc:  # pragma: no cover
                        loop.call_soon_threadsafe(queue.put_nowait, exc)
                        loop.call_soon_threadsafe(queue.put_nowait, done_sentinel)

                threading.Thread(target=_stream_worker, daemon=True).start()

                while True:
                    item = await queue.get()
                    if item is done_sentinel:
                        break
                    if isinstance(item, Exception):
                        raise item
                    chunk = item
                    logger.info(f"Received chunk: {chunk.text}")
                    # logger.info(f"Received chunk: {chunk}")
                    yield LLMResponse(
                        text=chunk.text,
                        metadata={
                            "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                            "completion_tokens": chunk.usage_metadata.candidates_token_count,
                            "cached_tokens": chunk.usage_metadata.cached_content_token_count
                            or 0,
                            "thoughts_token_count": chunk.usage_metadata.thoughts_token_count
                            or 0,
                        },
                        parsed=chunk.parsed if response_format_model else None,
                    )

            else:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    config=config,
                    contents=contents,
                )

                yield LLMResponse(
                    text=response.text,
                    metadata={
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "cached_tokens": response.usage_metadata.cached_content_token_count
                        or 0,
                        "thoughts_token_count": response.usage_metadata.thoughts_token_count
                        or 0,
                    },
                    parsed=response.parsed if response_format_model else None,
                )

        except Exception as e:
            logger.error(traceback.format_exc())
            raise Exception(f"Error generating structured response: {str(e)}")


class AnthropicVertexLLMClient(LLMClient):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        use_content_cache: bool = False,
        system_prompt: str | None = None,  # TODO
        prompts_dir: str | os.PathLike | None = None,
        reasoning_effort: str | int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            model_name,
            system_prompt,
            prompts_dir,
            temperature,
            max_tokens,
            reasoning_effort,
            use_content_cache,
            verbose,
        )

        self.project_id = os.getenv("PROJECT_ID", "")
        self.region = "us-east5"
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_content_cache = use_content_cache

        self.client = AnthropicVertex(
            project_id=self.project_id,
            region=self.region,
        )

        # Log configuration
        logger.info(f"Model Name: {self.model_name}")
        logger.info(f"Default Temperature: {self.temperature}")
        logger.info(f"Default Max Tokens: {self.max_tokens}")
        logger.info(f"Use Content Cache: {self.use_content_cache}")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        override_settings: Dict[str, Any] | None = None,
        response_format: Dict[str, Any] | BaseModel | None = None,
        add_cot_field: bool = True,
        conversation_history: List[Dict[str, str]] | None = None,
        reasoning_effort: str | int | None = None,
        streaming: bool = False,
        **kwargs,
    ):
        """
        Send a message to the model and return response.
        """
        temp = self.temperature
        tokens = self.max_tokens

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            system=system_prompt,
        )

        # breakpoint()

        response_format_model = None

        yield LLMResponse(
            text=response.content[0].text,
            metadata={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0 or 0,
                "thoughts_token_count": 0 or 0,
            },
            parsed=getattr(response, "parsed", None) if response_format_model else None,
        )


# class LiteLLMClient(LLMClient):
#     """LLM client class for handling prompts and generating responses using liteLLM."""

#     litellm.enable_json_schema_validation = True
#     litellm.drop_params = True

#     def __init__(
#         self,
#         model_name: str,
#         system_prompt: str | None = None,
#         prompts_dir: str | os.PathLike | None = None,
#         temperature: float = 0.1,
#         max_tokens: int = 8192,
#         reasoning_effort: str | int | None = None,
#         use_content_cache: bool = False,
#         verbose: bool = False,
#         **kwargs: Any,
#     ):
#         """
#         Initialize the LLM client.

#         Args:
#             model_name: Name of the LLM model to use
#             system_prompt: Optional system prompt to set the context
#             prompts_dir: Directory containing prompt templates
#             temperature: Sampling temperature for generation
#             max_tokens: Maximum number of tokens to generate
#             reasoning_effort: Level of reasoning effort (e.g., "low", "medium", "high")
#             use_content_cache: Whether to use content caching for responses
#             verbose: Whether to enable verbose logging
#         """
#         super().__init__(
#             model_name,
#             system_prompt,
#             prompts_dir,
#             temperature,
#             max_tokens,
#             reasoning_effort,
#             use_content_cache,
#             verbose,
#         )

#         # log a warning that content cache and gemini do not work well with LiteLLMClient for the moment
#         if self.use_content_cache and "gemini" in self.model_name:
#             logger.warning(
#                 f"`use_content_cache=True` with a Gemini model ({self.model_name}) is not fully supported in `LiteLLMClient`. "
#                 "Content cache integration for Gemini models is currently unstable and may lead to unexpected behavior."
#             )

#     async def generate(
#         self,
#         prompt: str,
#         system_prompt: str | None = None,
#         override_settings: Optional[Dict[str, Any]] = None,
#         response_format: Dict[str, Any] | BaseModel | None = None,
#         add_cot_field: bool = True,
#         conversation_history: Optional[List[Dict[str, str]]] = None,
#         reasoning_effort: str | int | None = None,
#         num_retries: int = 10,
#         retry_strategy: str = "exponential_backoff_retry",
#         **kwargs,
#     ) -> LLMResponse:
#         """
#         Generate a response using the configured LLM.

#         Args:
#             prompt: Input prompt text
#             system_prompt: Optional system prompt to set the context
#             override_settings: Optional dictionary to override settings for this call
#             response_format: Optional response format schema
#             add_cot_field: Whether to add a chain of thought field to the response
#             conversation_history: Optional list of previous messages in the conversation
#             reasoning_effort: Optional reasoning effort level (e.g., "low", "medium", "high")
#             num_retries: Number of retries for the generation
#             retry_strategy: Strategy for retrying failed requests
#             **kwargs: Additional keyword arguments for the generation call

#         Returns:
#             Generated response text
#         """

#         response_format_model = None
#         if response_format is not None:
#             response_format_model = self.parse_response_format(
#                 response_format, add_cot_field
#             )

#         system_prompt = system_prompt or self.system_prompt

#         messages = []
#         if system_prompt is not None:
#             if self.use_content_cache:
#                 # Use system prompt as a message if not using content cache
#                 messages.append(
#                     {
#                         "role": "system",
#                         "content": [
#                             {
#                                 "type": "text",
#                                 "text": system_prompt,
#                                 "cache_control": {"type": "ephemeral"},
#                             }
#                         ],
#                     }
#                 )
#             else:
#                 messages.append({"role": "system", "content": system_prompt})

#         if conversation_history is None:
#             conversation_history = []

#         for conversation in conversation_history:
#             if self.use_content_cache:
#                 messages.append(
#                     {
#                         "role": conversation.get("role"),
#                         "content": [
#                             {
#                                 "type": "text",
#                                 "text": conversation.get("message"),
#                                 "cache_control": {"type": "ephemeral"},
#                             }
#                         ],
#                     }
#                 )
#             else:
#                 messages.append(
#                     {"role": conversation.get("role"), "content": conversation}
#                 )

#         if self.use_content_cache:
#             messages.append(
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": prompt,
#                             "cache_control": {"type": "ephemeral"},
#                         }
#                     ],
#                 }
#             )
#         else:
#             messages.append({"role": "user", "content": prompt})

#         logger.info("Messages:")
#         logger.info(messages)

#         reasoning_kwargs = {}
#         if reasoning_effort is not None:
#             logger.info("Reasoning effort: %s", reasoning_effort)
#             reasoning_kwargs["reasoning_effort"] = reasoning_effort
#             reasoning_kwargs["thinking"] = {
#                 "type": "enabled",
#                 "budget_tokens": reasoning_effort,
#             }

#         try:
#             # Use acompletion with full message history
#             response = await acompletion(
#                 model=self.model_name,
#                 messages=messages,
#                 **{
#                     k: v
#                     for k, v in override_settings.items()
#                     # this is weird. drop_params=True doesn't seem to always work
#                     if k not in {"model_name", "prompt_name"}
#                 },
#                 drop_params=True,
#                 response_format=response_format_model,
#                 num_retries=num_retries,
#                 retry_strategy=retry_strategy,
#                 **reasoning_kwargs,
#             )

#             parsed_response = None
#             if response_format_model is not None:
#                 # spoofing to avoid parse_chat_completion error
#                 response.choices[0].message.refusal = None
#                 # use OpenAI's parse_chat_completion to parse the response
#                 parsed_response = parse_chat_completion(
#                     response_format=response_format_model,
#                     input_tools=NOT_GIVEN,
#                     chat_completion=response,
#                 )

#             yield LLMResponse(
#                 text=response.choices[0].message.content,
#                 metadata={
#                     "prompt_tokens": response.usage.prompt_tokens,
#                     "completion_tokens": response.usage.completion_tokens,
#                     "cached_tokens": response.usage.prompt_tokens_details.cached_tokens
#                     or 0,
#                 },
#                 parsed=(
#                     parsed_response.choices[0].message.parsed
#                     if parsed_response
#                     else None
#                 ),
#             )

#         except Exception as e:
#             raise Exception(f"Error generating response: {str(e)}")

#     async def generate_with_template(
#         self,
#         template_name: str,
#         system_template_name: str | None = None,
#         template_vars: dict[str, str] | None = None,
#         system_template_vars: dict[str, str] | None = None,
#         override_settings: Optional[Dict[str, Any]] = None,
#         response_format: Dict[str, Any] | BaseModel | None = None,
#         add_cot_field: bool = True,
#         conversation_history: Optional[List[Dict[str, str]]] = None,
#         reasoning_effort: Optional[str] = None,
#         num_retries: int = 10,
#         retry_strategy: str = "exponential_backoff_retry",
#     ) -> str:
#         """
#         Generate a response using a prompt template.

#         Args:
#             template_name: Name of the prompt template to use
#             system_template_name: Optional name of the system prompt template
#             template_vars: Variables to render in the prompt template
#             system_template_vars: Variables to render in the system prompt template
#             override_settings: Optional dictionary to override settings for this call
#             response_format: Optional response format schema
#             add_cot_field: Whether to add a chain of thought field to the response
#             conversation_history: Optional list of previous messages in the conversation
#             reasoning_effort: Optional reasoning effort level (e.g., "low", "medium", "high")
#             num_retries: Number of retries for the generation
#             retry_strategy: Strategy for retrying failed requests

#         Returns:
#             Generated response text
#         """
#         if template_vars is None:
#             template_vars = {}
#         prompt = self.render_prompt(template_name, **template_vars)

#         if system_template_name is not None:
#             if system_template_vars is None:
#                 system_template_vars = {}
#             system_prompt = self.render_prompt(
#                 system_template_name, **system_template_vars
#             )

#         return await self.generate(
#             prompt,
#             system_prompt,
#             override_settings,
#             response_format,
#             add_cot_field,
#             conversation_history,
#             reasoning_effort,
#             num_retries,
#             retry_strategy,
#         )

#     @classmethod
#     def from_env(
#         cls, prompts_dir: str | os.PathLike = "config/prompts"
#     ) -> "LiteLLMClient":
#         """
#         Create an LLMClient instance from environment variables.

#         Args:
#             prompts_dir: Directory containing prompt templates

#         Returns:
#             Configured LLMClient instance
#         """
#         settings = LLMSettings(
#             model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
#             max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
#             temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
#             prompts_dir=prompts_dir,
#         )
#         return cls(**settings.to_dict())


class Citation(BaseModel):
    title: str = Field(..., description="The title of the cited work.")
    url: str = Field(..., description="The URL of the cited work.")
    text: str = Field(..., description="Text excerpt used from the cited work.")


class OpenAIModel(BaseModel):
    text: str = Field(..., description="The generated text from the model.")
    citations: List[Citation] = Field(
        ..., description="List of citations used in the generated text."
    )


class OpenAILLMClient(LLMClient):
    """LLM client class for handling prompts and generating responses using OpenAI API."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        prompts_dir: str | os.PathLike | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        reasoning_effort: str | int | None = None,
        use_content_cache: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the LLM client.

        Args:
            model_name: Name of the LLM model to use
            system_prompt: Optional system prompt to set the context
            prompts_dir: Directory containing prompt templates
            temperature: Sampling temperature for generation
            max_tokens: Maximum number of tokens to generate
            reasoning_effort: Level of reasoning effort (e.g., "low", "medium", "high")
            use_content_cache: Whether to use content caching for responses
            verbose: Whether to enable verbose logging
        """
        super().__init__(
            model_name,
            system_prompt,
            prompts_dir,
            temperature,
            max_tokens,
            reasoning_effort,
            use_content_cache,
            verbose,
        )

        # log a warning that content cache and gemini do not work well with LiteLLMClient for the moment
        if self.use_content_cache and "gemini" in self.model_name:
            logger.warning(
                f"`use_content_cache=True` with a Gemini model ({self.model_name}) is not fully supported in `LiteLLMClient`. "
                "Content cache integration for Gemini models is currently unstable and may lead to unexpected behavior."
            )

        self.client = AsyncOpenAI()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        override_settings: Optional[Dict[str, Any]] = None,
        response_format: Dict[str, Any] | BaseModel | None = None,
        add_cot_field: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        reasoning_effort: str | int | None = None,
        num_retries: int = 10,
        retry_strategy: str = "exponential_backoff_retry",
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using the configured LLM.

        Args:
            prompt: Input prompt text
            system_prompt: Optional system prompt to set the context
            override_settings: Optional dictionary to override settings for this call
            response_format: Optional response format schema
            add_cot_field: Whether to add a chain of thought field to the response
            conversation_history: Optional list of previous messages in the conversation
            reasoning_effort: Optional reasoning effort level (e.g., "low", "medium", "high")
            num_retries: Number of retries for the generation
            retry_strategy: Strategy for retrying failed requests
            **kwargs: Additional keyword arguments for the generation call

        Returns:
            Generated response text
        """

        response_format_model = None
        if response_format is not None:
            response_format_model = self.parse_response_format(
                response_format, add_cot_field
            )

        system_prompt = system_prompt or self.system_prompt

        messages = []
        if system_prompt is not None:
            if self.use_content_cache:
                # Use system prompt as a message if not using content cache
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                messages.append({"role": "system", "content": system_prompt})

        if conversation_history is None:
            conversation_history = []

        for conversation in conversation_history:
            if self.use_content_cache:
                messages.append(
                    {
                        "role": conversation.get("role"),
                        "content": [
                            {
                                "type": "text",
                                "text": conversation.get("message"),
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                messages.append(
                    {"role": conversation.get("role"), "content": conversation}
                )

        if self.use_content_cache:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        logger.info("Messages:")
        logger.info(messages)

        reasoning_kwargs = {}
        if reasoning_effort is not None:
            logger.info("Reasoning effort: %s", reasoning_effort)
            reasoning_kwargs["reasoning_effort"] = reasoning_effort
            reasoning_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": reasoning_effort,
            }

        try:
            # Use acompletion with full message history
            response = await self.client.responses.parse(
                model=self.model_name,
                tools=[{"type": "web_search", "search_context_size": "low"}],
                input=messages,
                reasoning={"effort": "low"},
                text_format=OpenAIModel,
                # **{
                #     k: v
                #     for k, v in override_settings.items()
                #     # this is weird. drop_params=True doesn't seem to always work
                #     if k not in {"model_name", "prompt_name"}
                # },
                # drop_params=True,
                # response_format=response_format_model,
                # num_retries=num_retries,
                # retry_strategy=retry_strategy,
                # **reasoning_kwargs,
            )

            parsed_response = None
            # if response_format_model is not None:
            #     # spoofing to avoid parse_chat_completion error
            #     response.choices[0].message.refusal = None
            #     # use OpenAI's parse_chat_completion to parse the response
            #     parsed_response = parse_chat_completion(
            #         response_format=response_format_model,
            #         input_tools=NOT_GIVEN,
            #         chat_completion=response,
            #     )

            parsed = response.output_parsed.model_dump(mode="json")
            yield LLMResponse(
                # text=response.output_text,
                text=response.output_parsed.text,
                metadata={
                    "sources": parsed["citations"],
                    # "prompt_tokens": response.usage.prompt_tokens,
                    # "completion_tokens": response.usage.completion_tokens,
                    # "cached_tokens": response.usage.prompt_tokens_details.cached_tokens
                    # or 0,
                },
                # parsed=(
                #     parsed_response.choices[0].message.parsed
                #     if parsed_response
                #     else None
                # ),
            )

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
