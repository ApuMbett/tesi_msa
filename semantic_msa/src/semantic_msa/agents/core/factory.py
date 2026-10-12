import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from dotenv import load_dotenv

from semantic_msa.utils.logger import logger

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

# Try importing groq
try:
    from groq import Groq  # type: ignore
except ImportError:
    Groq = None


def find_and_load_env():
    """Searches for .env file in parent directories or common locations."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / "semantic_msa" / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[3] / ".env",
        Path(__file__).resolve().parents[4] / ".env",
        Path(__file__).resolve().parents[5] / ".env",
    ]
    for p in candidates:
        if p.is_file():
            load_dotenv(p)


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
        thinking_level: Optional[str] = None,
        audio_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Executes generation and returns raw text response."""
        pass


class GeminiNativeGroundedProvider(LLMProvider):
    """Client for Google GenAI SDK with support for native Google Search Grounding and multimodal audio."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        if genai is None:
            raise ImportError("google-genai is not installed. Please run: uv pip install google-genai")
        # Normalize model name
        self.model_name = model_name if not model_name.startswith("models/") else model_name[7:]
        raw_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.api_key = raw_key.split("#")[0].strip() if raw_key else ""
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
        thinking_level: Optional[str] = None,
        audio_path: Optional[Union[str, Path]] = None,
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

        thinking_kwargs: Dict[str, Any] = {}
        if thinking_budget is not None and thinking_budget >= 0:
            thinking_kwargs["thinking_budget"] = thinking_budget
        if thinking_level:
            thinking_kwargs["thinking_level"] = thinking_level.lower()

        if thinking_kwargs:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(**thinking_kwargs)

        config = genai_types.GenerateContentConfig(**config_kwargs)

        uploaded_file = None
        try:
            if audio_path:
                audio_file_path = Path(audio_path).resolve()
                if not audio_file_path.is_file():
                    raise FileNotFoundError(f"Audio file not found for Gemini inference: {audio_file_path}")
                logger.info(f"Uploading audio file to Gemini Files API: [bold cyan]{audio_file_path.name}[/bold cyan]")
                uploaded_file = self.client.files.upload(file=str(audio_file_path))
                # Ensure file is ACTIVE if state is PROCESSING
                if hasattr(uploaded_file, "state") and str(uploaded_file.state).upper().endswith("PROCESSING"):
                    for _ in range(30):
                        time.sleep(1)
                        uploaded_file = self.client.files.get(name=uploaded_file.name)
                        if str(uploaded_file.state).upper().endswith("ACTIVE"):
                            break
                        elif str(uploaded_file.state).upper().endswith("FAILED"):
                            raise RuntimeError(
                                f"Gemini Files API upload failed: {getattr(uploaded_file, 'error', 'unknown error')}"
                            )
                contents = [uploaded_file, user_message]
            else:
                contents = user_message

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            return response.text.strip() if response.text else "{}"
        finally:
            if uploaded_file is not None:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception as del_err:
                    logger.debug(f"Failed to delete uploaded Gemini file {uploaded_file.name}: {del_err}")


class OpenAICompatibleProvider(LLMProvider):
    """Client for OpenAI-compatible endpoints (OpenAI, OpenRouter, Gemini OpenAI-compatibility, local vLLM)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        if OpenAI is None:
            raise ImportError("openai is not installed. Please run: uv pip install openai")
        self.model_name = model_name
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError("LLM_API_KEY / OPENAI_API_KEY / HF_TOKEN missing from environment variables.")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

        headers = dict(default_headers or {})
        if "openrouter" in self.base_url.lower():
            headers.setdefault("HTTP-Referer", "https://github.com/ApuMbett/autodj")
            headers.setdefault("X-Title", "Semantic MSA")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=headers or None,
            timeout=120.0,
        )

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
        thinking_level: Optional[str] = None,
        audio_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if audio_path:
            logger.warning(f"Audio input was provided, but OpenAICompatibleProvider does not support audio files. Ignoring audio.")
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
        if "<think>" in content and "</think>" in content:
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content.strip()


class GroqProvider(LLMProvider):
    """Client for Groq API (e.g. qwen/qwen3.8-27b, llama models)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if Groq is None and OpenAI is None:
            raise ImportError("Neither groq nor openai is installed. Please run: uv pip install openai")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY / LLM_API_KEY missing from environment variables.")
        self.base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        if Groq is not None and not base_url:
            self.client = Groq(api_key=self.api_key)
        else:
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
        thinking_level: Optional[str] = None,
        audio_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if audio_path:
            logger.warning(f"Audio input was provided, but GroqProvider does not support audio files. Ignoring audio.")
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

        # Thinking / reasoning mode handling for Qwen/Groq models (e.g. qwen/qwen3.8-27b)
        # Groq docs: reasoning_effort can be "none", "low", "medium", "high", or "default"
        if thinking_level:
            level = thinking_level.lower()
            if level in ["none", "low", "medium", "high", "default"]:
                kwargs["reasoning_effort"] = level
            elif level == "off":
                kwargs["reasoning_effort"] = "none"

        # Hide reasoning tokens in response so JSON output is not corrupted
        kwargs["extra_body"] = {"reasoning_format": "hidden"}

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"

        # Defense-in-depth: strip any residual <think> tags if model leaks them
        if "<think>" in content and "</think>" in content:
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

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
        thinking_level: Optional[str] = None,
        audio_path: Optional[Union[str, Path]] = None,
    ) -> str:
        schema_name = getattr(response_schema, "__name__", "").lower() if response_schema else ""
        prompt_lower = system_prompt.lower()
        if "researcher" in schema_name or "cultural historian" in prompt_lower or "cultural_prior" in prompt_lower:
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
        elif "profiler" in schema_name or "profiler" in prompt_lower or "macro_profile" in prompt_lower:
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
        elif "captioner" in schema_name or "caption" in prompt_lower:
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
        elif (
            "directaudio" in schema_name
            or "direct_audio" in schema_name
            or "functional music structure analysis" in prompt_lower
            or "segment the entire song into contiguous structural sections" in prompt_lower
        ):
            dur_match = re.search(r"Duration:\s*([0-9.]+)\s*seconds", f"{user_message} {system_prompt}", re.IGNORECASE)
            duration = float(dur_match.group(1)) if dur_match else 191.30
            t0 = 0.00
            t1 = round(min(15.50, duration * 0.1), 2)
            t2 = round(min(60.00, duration * 0.4), 2)
            t3 = round(min(120.00, duration * 0.7), 2)
            t4 = round(duration, 2)
            return json.dumps({
                "segments": [
                    {
                        "start": t0,
                        "end": t1,
                        "label": "intro",
                        "caption": "Atmospheric introductory passage with soft instrumentation establishing the key.",
                    },
                    {
                        "start": t1,
                        "end": t2,
                        "label": "verse",
                        "caption": "Narrative vocal development over steady rhythmic accompaniment.",
                    },
                    {
                        "start": t2,
                        "end": t3,
                        "label": "chorus",
                        "caption": "High-energy recurring hook with dynamic climax and full arrangement.",
                    },
                    {
                        "start": t3,
                        "end": t4,
                        "label": "outro",
                        "caption": "Gradual dissipation of energy and instrumental resolution leading to silence.",
                    },
                ]
            })
        elif "macro-structural segmenter" in prompt_lower or "segments" in prompt_lower:
            return json.dumps({
                "segments": [
                    {
                        "segment_index": 0,
                        "standard_label": "intro",
                        "segment_name": "Intro",
                        "token_range": [0, 2],
                        "macro_caption": "Minimal introductory beat establishing foundational tempo.",
                        "rationale": "Initial tokens feature low spectral flux before main bassline.",
                    },
                    {
                        "segment_index": 1,
                        "standard_label": "chorus",
                        "segment_name": "Main Drop",
                        "token_range": [3, 8],
                        "macro_caption": "Full energetic body of the track with driving bass.",
                        "rationale": "Significant jump in MSF and peak ROE across tokens 3 to 8.",
                    },
                    {
                        "segment_index": 2,
                        "standard_label": "outro",
                        "segment_name": "Outro",
                        "token_range": [9, 10],
                        "macro_caption": "Gradual reduction in arrangement density leading to fade.",
                        "rationale": "Energy drops sharply back to sparse percussive elements.",
                    },
                ],
            })
        return json.dumps({"status": "ok"})


class RateLimiter:
    """
    Thread-safe sliding-interval rate limiter.
    Ensures consecutive calls do not start closer than 60.0 / rpm seconds apart.
    """

    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError(f"RPM must be positive, got {rpm}")
        self.rpm = rpm
        # Add 0.2s safety buffer to prevent boundary clock jitter with Google servers
        self.min_interval = (60.0 / float(rpm)) + 0.2
        self.last_call_time: float = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        Blocks until the rate limit allows dispatching the next request.
        Returns the number of seconds waited.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call_time
            wait_time = self.min_interval - elapsed
            if wait_time > 0.0:
                time.sleep(wait_time)
                now = time.time()
            self.last_call_time = now
            return max(0.0, wait_time)


class RateLimitedProvider(LLMProvider):
    """
    Adapter seam: wraps any underlying LLMProvider and enforces rate limiting
    via a shared RateLimiter instance before dispatching calls.
    """

    def __init__(self, provider: LLMProvider, rate_limiter: RateLimiter):
        self._provider = provider
        self.rate_limiter = rate_limiter

    @property
    def supports_native_search(self) -> bool:
        return self._provider.supports_native_search

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        grounding: bool = False,
        response_schema: Optional[Type] = None,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
        audio_path: Optional[Union[str, Path]] = None,
    ) -> str:
        waited = self.rate_limiter.acquire()
        if waited > 0.05:
            logger.info(
                f"[bold cyan]Rate Limiter:[/bold cyan] Pacing active ({self.rate_limiter.rpm} RPM ceiling). "
                f"Paused {waited:.2f}s before dispatch."
            )
        return self._provider.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            grounding=grounding,
            response_schema=response_schema,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            audio_path=audio_path,
        )


class LLMFactory:
    """Factory that instantiates the correct LLM provider adapter according to configuration."""

    def __init__(self, mock_mode: bool = False, rate_limit_rpm: Optional[int] = None):
        self.mock_mode = mock_mode
        self.rate_limit_rpm = rate_limit_rpm
        self.rate_limiter = (
            RateLimiter(rpm=rate_limit_rpm) if rate_limit_rpm and rate_limit_rpm > 0 else None
        )
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")

    def create_provider(
        self,
        model_name: str,
        grounding: bool = False,
    ) -> LLMProvider:
        if self.mock_mode or model_name.startswith("mock"):
            provider = StubProvider(model_name)
        elif (
            model_name.startswith("openrouter/")
            or model_name.endswith(":free")
            or (self.base_url and "openrouter" in self.base_url.lower())
        ):
            real_model = model_name[11:] if model_name.startswith("openrouter/") else model_name
            openrouter_key = os.getenv("OPENROUTER_API_KEY") or self.api_key
            openrouter_base = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
            provider = OpenAICompatibleProvider(model_name=real_model, api_key=openrouter_key, base_url=openrouter_base)
        elif (
            model_name.startswith("hf/")
            or model_name.startswith("huggingface/")
            or (self.base_url and "huggingface" in self.base_url.lower())
        ):
            if model_name.startswith("huggingface/"):
                real_model = model_name[12:]
            elif model_name.startswith("hf/"):
                real_model = model_name[3:]
            else:
                real_model = model_name
            hf_key = (
                os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_API_KEY")
                or os.getenv("HUGGING_FACE_HUB_TOKEN")
                or self.api_key
            )
            hf_base = os.getenv("HF_BASE_URL") or "https://router.huggingface.co/v1"
            provider = OpenAICompatibleProvider(model_name=real_model, api_key=hf_key, base_url=hf_base)
        elif (
            model_name.startswith("groq/")
            or model_name.startswith("qwen/")
            or (self.base_url and "groq" in self.base_url.lower())
        ):
            real_model = model_name[5:] if model_name.startswith("groq/") else model_name
            groq_key = os.getenv("GROQ_API_KEY") or self.api_key
            provider = GroqProvider(model_name=real_model, api_key=groq_key)
        elif grounding:
            # If grounding is requested, we prefer Gemini native SDK
            # TODO will change in the future when we implement our grounding via duckduckgo
            provider = GeminiNativeGroundedProvider(model_name=model_name, api_key=self.api_key)
        elif self.base_url and "openai" in self.base_url and not ("generativelanguage" in self.base_url):
            # Check if user specified an OpenAI endpoint / base URL
            provider = OpenAICompatibleProvider(model_name=model_name, api_key=self.api_key, base_url=self.base_url)
        elif genai is not None and not model_name.startswith(("gpt", "claude")):
            # Default to GeminiNative SDK if google-genai is available and model is not third-party
            provider = GeminiNativeGroundedProvider(model_name=model_name, api_key=self.api_key)
        elif OpenAI is not None:
            provider = OpenAICompatibleProvider(model_name=model_name, api_key=self.api_key, base_url=self.base_url)
        else:
            raise RuntimeError("Neither google-genai nor openai is installed.")

        if self.rate_limiter is not None:
            return RateLimitedProvider(provider=provider, rate_limiter=self.rate_limiter)
        return provider
