import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type
from dotenv import load_dotenv

# Try importing google-genai
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# Try importing openai
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def find_and_load_env():
    """Searches for .env file in parent directories or common locations."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[3] / ".env",
        Path(__file__).resolve().parents[4] / ".env",
    ]
    for p in candidates:
        if p.is_file():
            load_dotenv(p)
            break


find_and_load_env()


class LLMProvider(ABC):
    """Abstract adapter seam for LLM inference backends."""

    @property
    @abstractmethod
    def supports_native_search(self) -> bool:
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        grounding: bool = False,
        response_schema: Optional[Type] = None,
        thinking_budget: Optional[int] = None,
    ) -> str:
        """Executes generation and returns raw text response."""
        pass


class GeminiNativeGroundedProvider(LLMProvider):
    """Client for Google GenAI SDK with support for native Google Search Grounding."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        if genai is None:
            raise ImportError("google-genai is not installed. Please run: uv pip install google-genai")
        # Normalize model name
        self.model_name = model_name if not model_name.startswith("models/") else model_name[7:]
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY / LLM_API_KEY missing from environment variables.")
        self.client = genai.Client(api_key=self.api_key)

    @property
    def supports_native_search(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        grounding: bool = False,
        response_schema: Optional[Type] = None,
        thinking_budget: Optional[int] = None,
    ) -> str:
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
        }

        if grounding:
            config_kwargs["tools"] = [{"google_search": {}}]

        if response_schema and not grounding:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        if thinking_budget is not None and thinking_budget >= 0:
            config_kwargs["thinking_config"] = {"thinking_budget": thinking_budget}

        config = genai_types.GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=config,
        )
        return response.text.strip() if response.text else "{}"


class OpenAICompatibleProvider(LLMProvider):
    """Client for OpenAI-compatible endpoints (OpenAI, Gemini OpenAI-compatibility, local vLLM)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if OpenAI is None:
            raise ImportError("openai is not installed. Please run: uv pip install openai")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("LLM_API_KEY / OPENAI_API_KEY missing from environment variables.")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120.0)

    @property
    def supports_native_search(self) -> bool:
        return False

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        grounding: bool = False,
        response_schema: Optional[Type] = None,
        thinking_budget: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"
        return content.strip()


class StubProvider(LLMProvider):
    """Deterministic offline mock provider for testing and validation without API calls."""

    def __init__(self, model_name: str = "mock-stub"):
        self.model_name = model_name

    @property
    def supports_native_search(self) -> bool:
        return False

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        grounding: bool = False,
        response_schema: Optional[Type] = None,
        thinking_budget: Optional[int] = None,
    ) -> str:
        prompt_lower = system_prompt.lower()
        if "blind profiler" in prompt_lower or "macro_profile" in prompt_lower:
            return json.dumps({
                "macro_profile": {
                    "inferred_genres": ["Electronic", "House"],
                    "energy_tier": "High",
                    "rationale": "High BPM and spectral flux indicate electronic dance structure.",
                },
                "vocal_profile": {
                    "presence_level": "Sparse",
                    "lyrics_validity": "Reliable",
                    "rationale": "Clear vocal cues detected with positive intensity.",
                },
                "dynamic_trajectory": {
                    "energy_flow": "Steady build up followed by driving rhythm drops.",
                    "rationale": "Token indices 1 through 10 demonstrate climbing ROE.",
                },
            })
        elif "cultural historian" in prompt_lower or "cultural_prior" in prompt_lower:
            return json.dumps({
                "cultural_prior": {
                    "source_identity": {
                        "identified_track_name": "Satisfaction",
                        "identified_original_artist": "Benny Benassi",
                        "original_cultural_genre": "Electro House",
                    },
                    "cultural_impact_and_history": {
                        "scale_of_impact": "Global Pop Phenomenon",
                        "zeitgeist_summary": "Pioneering electro house anthem released in 2002.",
                    },
                    "inferred_reality": {
                        "track_classification": "Studio Original",
                        "dominant_acoustic_style": "Electro House with buzzing sawtooth bassline.",
                        "target_audience": "Club and festival dancefloors.",
                    },
                    "researcher_notes": "Iconic electro house standard verified from lyrics and telemetry.",
                },
            })
        elif "acoustic captioner" in prompt_lower or "token_captions" in prompt_lower:
            # Extract number of tokens from user message if possible
            count = len(re.findall(r'"index":\s*(\d+)', user_message))
            if count == 0:
                count = 10
            captions = [
                {
                    "token_index": i,
                    "caption": f"Driving synth pulse with quantized electronic beat {i}.",
                    "rationale": f"High ROE and steady BTR observed in token {i}.",
                }
                for i in range(count)
            ]
            return json.dumps({"token_captions": captions})
        elif "macro-structural segmenter" in prompt_lower or "segments" in prompt_lower:
            return json.dumps({
                "segments": [
                    {
                        "segment_index": 0,
                        "segment_name": "Intro",
                        "token_range": [0, 2],
                        "macro_caption": "Minimal introductory beat establishing foundational tempo.",
                        "rationale": "Initial tokens feature low spectral flux before main bassline.",
                    },
                    {
                        "segment_index": 1,
                        "segment_name": "Main Drop",
                        "token_range": [3, 8],
                        "macro_caption": "Full energetic body of the track with driving bass.",
                        "rationale": "Significant jump in MSF and peak ROE across tokens 3 to 8.",
                    },
                    {
                        "segment_index": 2,
                        "segment_name": "Outro",
                        "token_range": [9, 10],
                        "macro_caption": "Gradual reduction in arrangement density leading to fade.",
                        "rationale": "Energy drops sharply back to sparse percussive elements.",
                    },
                ],
            })
        return json.dumps({"status": "ok"})


class LLMFactory:
    """Factory that instantiates the correct LLM provider adapter according to configuration."""

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")

    def create_provider(
        self,
        model_name: str,
        grounding: bool = False,
    ) -> LLMProvider:
        if self.mock_mode or model_name.startswith("mock"):
            return StubProvider(model_name)

        # If grounding is requested, we prefer Gemini native SDK
        # TODO will change in the future when we implement our grounding via duckduckgo
        if grounding:
            return GeminiNativeGroundedProvider(model_name=model_name, api_key=self.api_key)

        # Check if user specified an OpenAI endpoint / base URL
        if self.base_url and "openai" in self.base_url and not ("generativelanguage" in self.base_url):
            return OpenAICompatibleProvider(model_name=model_name, api_key=self.api_key, base_url=self.base_url)

        # Default to GeminiNative SDK if google-genai is available
        if genai is not None:
            return GeminiNativeGroundedProvider(model_name=model_name, api_key=self.api_key)
        elif OpenAI is not None:
            return OpenAICompatibleProvider(model_name=model_name, api_key=self.api_key, base_url=self.base_url)
        else:
            raise RuntimeError("Neither google-genai nor openai is installed.")
