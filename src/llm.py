import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

class LLM:
    """
    Thin provider-agnostic wrapper. Default: OpenAI Chat Completions API (SDK v1).
    You can swap provider by overriding `_chat_openai` or adding new provider methods.
    """
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "openai":
            self._init_openai()

    # ---------- OpenAI ----------
    def _init_openai(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI SDK not installed. `pip install openai>=1.52.0`"
            ) from e
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY.")
        self._openai_client = OpenAI(api_key=api_key)

    def _chat_openai(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self._openai_client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content or ""

    # ---------- public ----------
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional(float) = None,  # type: ignore
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        model = model or self.model
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.provider == "openai":
            return self._chat_openai(
                messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
            )
        raise NotImplementedError(f"Provider '{self.provider}' is not implemented.")
