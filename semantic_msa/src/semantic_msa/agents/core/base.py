import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

from semantic_msa.agents.core.factory import LLMFactory, LLMProvider


class BaseAgent(ABC):
    """
    Abstract base class for all pipeline agents.
    Encapsulates prompt retrieval, provider instantiation, and robust JSON output validation.
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

        self.provider: LLMProvider = self.factory.create_provider(
            model_name=self.model,
            grounding=self.grounding,
        )

    def get_system_prompt(self) -> str:
        """Loads system prompt from semantic_msa/agents/prompts/."""
        suffix = "_explainable" if self.explainable else ""
        # Look in core/prompts or fallback to parent agents/prompts
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        if not prompts_dir.is_dir():
            prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        
        # Primary candidate
        prompt_path = prompts_dir / f"{self.name}{suffix}.txt"
        if not prompt_path.is_file() and suffix:
            prompt_path = prompts_dir / f"{self.name}.txt"
        
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found for agent '{self.name}': {prompt_path}")

        return prompt_path.read_text(encoding="utf-8")

    def clean_json_response(self, text: str) -> str:
        """Strips markdown code fences and extraneous text surrounding the JSON response."""
        stripped = text.strip()
        # Look for markdown ```json ... ``` block
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

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Executes the agent's step in the pipeline."""
        pass
