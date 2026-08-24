import os
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv

# Guarded imports for the native Google Gen AI library
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

class LLMProvider(ABC):
    """
    Abstract Base Class establishing a uniform contract and capability mapping
    for all downstream language model inference backends.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("LLM_API_KEY missing from environment variables.")

    @property
    @abstractmethod
    def supports_native_search(self) -> bool:
        """Read-only capability flag checking if the client runs native search tooling."""
        pass

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        """Uniform generation execution method contract."""
        pass


class OpenAICompatibleProvider(LLMProvider):
    """
    Handles standard text-only backends (OpenAI, local servers, cluster execution nodes).
    Requires external context injection because it cannot search natively.
    """
    def __init__(self):
        super().__init__()
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    @property
    def supports_native_search(self) -> bool:
        return False  # Standard text completion endpoints cannot run native web lookups

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()


class GeminiNativeGroundedProvider(LLMProvider):
    """
    Connects to Gemini Pro using the official Google Gen AI SDK
    to activate native live Google Search Grounding.
    """
    def __init__(self):
        super().__init__()
        if genai is None:
            raise ImportError("Please run: pip install google-genai")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gemini-1.5-pro")
        self.client = genai.Client(api_key=self.api_key)

    @property
    def supports_native_search(self) -> bool:
        return True  # This provider activates Google Search Grounding at the API level

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.1) -> str:
        # Bind Google's live web indexes directly to the generation config
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=temperature
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=config
        )
        return response.text.strip()