import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union
from pydantic import BaseModel

from semantic_msa.agents.core.factory import LLMFactory, LLMProvider
from semantic_msa.utils.logger import logger


class BaseAgent(ABC):
    """
    Abstract base class for all pipeline agents.
    Encapsulates prompt retrieval, provider instantiation, thinking configuration,
    robust JSON validation, and automatic retries with exponential backoff.
    """

    def __init__(self, name: str, config: Dict[str, Any], factory: LLMFactory):
        self.name = name
        self.config = config
        self.factory = factory
        self.model = config.get("model", "gemini-2.5-flash")
        self.temperature = float(config.get("temperature", 0.2))
        self.grounding = bool(config.get("grounding", False))
        self.explainable = bool(config.get("explainable", False))
        self.thinking_budget = config.get("thinking_budget", None)
        self.thinking_level = config.get("thinking_level", None)
        self.max_retries = int(config.get("max_retries", 2))

        self.provider: LLMProvider = self.factory.create_provider(
            model_name=self.model,
            grounding=self.grounding,
        )

    def get_system_prompt(self) -> str:
        """Loads system prompt from semantic_msa/agents/prompts/."""
        suffix = "_explainable" if self.explainable else ""
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        if not prompts_dir.is_dir():
            prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        
        prompt_path = prompts_dir / f"{self.name}{suffix}.txt"
        if not prompt_path.is_file() and suffix:
            prompt_path = prompts_dir / f"{self.name}.txt"
        
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found for agent '{self.name}': {prompt_path}")

        return prompt_path.read_text(encoding="utf-8")

    def clean_json_response(self, text: str) -> str:
        """Strips markdown code fences and extraneous text surrounding the JSON response."""
        stripped = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return stripped

    def parse_response(self, raw_text: str, schema_class: Optional[Type[BaseModel]] = None) -> Any:
        """Parses raw text into JSON dictionary or Pydantic model."""
        clean = self.clean_json_response(raw_text)
        data = json.loads(clean)
        if schema_class:
            return schema_class.model_validate(data)
        return data

    def generate_and_parse(
        self,
        system_prompt: str,
        user_message: str,
        schema_class: Optional[Type[BaseModel]] = None,
        audio_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[str, Any]:
        """
        Executes generation and schema parsing with exponential backoff retries on
        transient API errors or JSON parse/validation failures.
        Returns: (raw_response_text, parsed_object_or_dict)
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                raw_response = self.provider.generate(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=self.temperature,
                    grounding=self.grounding,
                    response_schema=schema_class,
                    thinking_budget=self.thinking_budget,
                    thinking_level=self.thinking_level,
                    audio_path=audio_path,
                )
                parsed = self.parse_response(raw_response, schema_class)
                return raw_response, parsed
            except Exception as e:
                last_error = e
                if attempt <= self.max_retries:
                    err_str = str(e).lower()
                    is_rate_limit = (
                        "429" in err_str
                        or "resource_exhausted" in err_str
                        or "resourceexhausted" in err_str
                        or "rate limit" in err_str
                        or "quota" in err_str
                    )
                    is_server_unavailable = (
                        "503" in err_str
                        or "unavailable" in err_str
                        or "high demand" in err_str
                    )
                    if is_rate_limit:
                        retry_match = (
                            re.search(r"retry in ([0-9.]+)s", err_str, re.IGNORECASE)
                            or re.search(r"retrydelay['\"]?:\s*['\"]?([0-9.]+)s?", err_str, re.IGNORECASE)
                            or re.search(r"try again in ([0-9.]+)s", err_str, re.IGNORECASE)
                        )
                        if retry_match:
                            wait_time = float(retry_match.group(1)) + 1.5
                        else:
                            wait_time = max(35.0, 20.0 * attempt)
                        logger.warning(
                            f"[bold yellow]Agent '{self.name}' attempt {attempt}/{self.max_retries + 1} hit Rate Limit / Quota Exhaustion (429).[/bold yellow] "
                            f"Holding for {wait_time:.1f}s to allow quota window reset..."
                        )
                    elif is_server_unavailable:
                        wait_time = max(5.0, 5.0 * attempt)
                        logger.warning(
                            f"[bold yellow]Agent '{self.name}' attempt {attempt}/{self.max_retries + 1} hit Server Overload / High Demand (503).[/bold yellow] "
                            f"Backing off for {wait_time:.1f}s before retrying..."
                        )
                    else:
                        wait_time = 1.5 ** attempt
                        logger.warning(
                            f"[bold yellow]Agent '{self.name}' attempt {attempt}/{self.max_retries + 1} failed:[/bold yellow] {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"[bold red]Agent '{self.name}' failed after {self.max_retries + 1} attempts:[/bold red] {e}"
                    )
                    raise
        if last_error:
            raise last_error
        raise RuntimeError(f"Agent '{self.name}' failed to generate output.")

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Executes the agent's step in the pipeline."""
        pass
