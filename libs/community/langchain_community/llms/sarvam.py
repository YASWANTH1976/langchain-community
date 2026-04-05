from typing import Any, Dict, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import requests

class SarvamAI(LLM):
    """LangChain wrapper for Sarvam AI's language model API.

    Sarvam AI provides state-of-the-art models for Indian languages.
    Get your API key at: https://www.sarvam.ai/apis

    Example:
        .. code-block:: python

            from langchain_community.llms import SarvamAI
            llm = SarvamAI(api_key="your-key", model="saaras:v1")
            result = llm.invoke("नमस्ते, आप कैसे हैं?")
    """

    api_key: str
    model: str = "saaras:v1"
    base_url: str = "https://api.sarvam.ai/v1"
    temperature: float = 0.7
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "sarvam-ai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url}
