import os
import logging
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging for reproducibility and debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MSAPipeline")


class LLMProvider:
    """
    Global configuration singleton that handles API authentication,
    token lifecycle, and unified model inference requests.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMProvider, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load local .env file containing environment configurations
        load_dotenv()
        
        # Pull API keys and custom base URLs (useful if running local servers)
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")
        self.default_model = os.getenv("LLM_MODEL_NAME")

        if not self.api_key:
            logger.warning("LLM_API_KEY not found in environment variables. Ensure your .env file is set.")

        # Initialize the underlying HTTP communication client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"LLM Provider successfully initialized using model: {self.default_model}")

    def generate_text(
        self, 
        system_prompt: str, 
        user_message: str, 
        temperature: float = 0.2,
        json_mode: bool = False
    ) -> str:
        """
        Unified text generation interface wrapper.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        kwargs: Dict[str, Any] = {
            "model": self.default_model,
            "messages": messages,
            "temperature": temperature
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty response from the language model.")
            return content.strip()
        except Exception as e:
            logger.error(f"API Call Failed: {str(e)}")
            raise e