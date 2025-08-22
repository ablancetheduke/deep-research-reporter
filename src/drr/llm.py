# src/drr/llm.py
# -*- coding: utf-8 -*-

"""
Unified LLM wrapper with multi-provider support (OpenAI, Gemini, DeepSeek, ChatGLM, MCP).

Goals:
- Single `LLM.chat(messages, ...) -> str` interface across providers.
- Lazy SDK imports so you only need deps for the provider you use.
- Minimal but robust error handling and small retry w/ backoff.
- Conservative defaults (low temperature, bounded max tokens).

Environment variables (by provider):
- OpenAI : OPENAI_API_KEY
- Gemini : GEMINI_API_KEY
- DeepSeek: DEEPSEEK_API_KEY  (optional DEEPSEEK_API_BASE, default https://api.deepseek.com)
- ChatGLM: CHATGLM_API_KEY
- MCP    : MCP_SERVER_URL, MCP_API_KEY (optional)

Message format (same as OpenAI):
    messages = [
        {"role": "system"|"user"|"assistant", "content": "text"},
        ...
    ]
"""

from __future__ import annotations

import os
import time
import json
import requests
from typing import Any, Dict, List, Optional


# ---------------------------
# Exceptions
# ---------------------------

class LLMError(RuntimeError):
    """Provider or request level error."""


# ---------------------------
# Utilities
# ---------------------------

_ALLOWED_ROLES = {"system", "user", "assistant"}


def _sanitize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    sane: List[Dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        if role not in _ALLOWED_ROLES:
            role = "user"
        content = str(m.get("content") or "")
        sane.append({"role": role, "content": content})
    return sane


def _join_as_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten chat messages for single-pass providers (e.g., Gemini fallback)."""
    lines: List[str] = []
    for m in messages:
        prefix = "USER"
        if m["role"] == "system":
            prefix = "SYSTEM"
        elif m["role"] == "assistant":
            prefix = "ASSISTANT"
        lines.append(f"[{prefix}] {m['content']}")
    return "\n\n".join(lines)


def _retry_call(fn, *, retries: int = 2, backoff: float = 1.5, first_delay: float = 0.5):
    """Tiny retry helper without extra deps."""
    delay = first_delay
    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt >= retries:
                break
            time.sleep(delay)
            delay *= backoff
    raise LLMError(str(last_exc)) from last_exc


# ---------------------------
# LLM Wrapper
# ---------------------------

class LLM:
    """
    Unified wrapper for multiple providers.

    Example:
        llm = LLM(provider="gemini", model="gemini-1.5-flash")
        text = llm.chat([{"role":"user","content":"Outline quantum computing."}])
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        request_timeout: float = 60.0,
        retries: int = 2,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.request_timeout = float(request_timeout)
        self.retries = int(retries)

        # lazy clients (initialized on first use)
        self._openai_client = None
        self._deepseek_client = None
        self._gemini = None
        self._chatglm_client = None
        self._mcp_session = None

        # quick provider validation
        if self.provider not in {"openai", "gemini", "deepseek", "chatglm", "mcp"}:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # -----------------------
    # Public API
    # -----------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat completion request and return plain text.
        `messages` uses OpenAI-like structure.
        """
        messages = _sanitize_messages(messages)
        model = model or self.model
        temperature = self.temperature if temperature is None else float(temperature)
        max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)

        if self.provider == "openai":
            return self._chat_openai(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if self.provider == "deepseek":
            return self._chat_deepseek(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if self.provider == "gemini":
            return self._chat_gemini(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if self.provider == "chatglm":
            return self._chat_chatglm(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if self.provider == "mcp":
            return self._chat_mcp(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)

        raise ValueError(f"Unsupported provider: {self.provider}")

    # -----------------------
    # Provider: OpenAI
    # -----------------------

    def _init_openai(self) -> None:
        if self._openai_client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise LLMError("OpenAI SDK not installed. `pip install openai>=1.52.0`") from e
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("Missing OPENAI_API_KEY.")
        # Base URL stays default for OpenAI
        self._openai_client = OpenAI(api_key=api_key)

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **_: Any,
    ) -> str:
        self._init_openai()
        def _call() -> str:
            resp = self._openai_client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=messages, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.request_timeout,
            )
            choice = resp.choices[0].message
            text = (choice.content or "").strip()
            return text
        return _retry_call(_call, retries=self.retries)

    # -----------------------
    # Provider: DeepSeek (OpenAI-compatible API)
    # -----------------------

    def _init_deepseek(self) -> None:
        if self._deepseek_client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise LLMError("OpenAI SDK is required for DeepSeek. `pip install openai>=1.52.0`") from e
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMError("Missing DEEPSEEK_API_KEY.")
        base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self._deepseek_client = OpenAI(api_key=api_key, base_url=base_url)

    def _chat_deepseek(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **_: Any,
    ) -> str:
        self._init_deepseek()
        def _call() -> str:
            resp = self._deepseek_client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=messages, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.request_timeout,
            )
            msg = resp.choices[0].message
            # DeepSeek (reasoner) may include reasoning_content; prepend if present.
            reasoning = getattr(msg, "reasoning_content", None)
            content = (msg.content or "").strip()
            if reasoning:
                return (str(reasoning).strip() + "\n\n" + content).strip()
            return content
        return _retry_call(_call, retries=self.retries)

    # -----------------------
    # Provider: Gemini
    # -----------------------

    def _init_gemini(self) -> None:
        if self._gemini is not None:
            return
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise LLMError("Gemini SDK not installed. `pip install google-generativeai>=0.7.2`") from e
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("Missing GEMINI_API_KEY.")
        genai.configure(api_key=api_key) # type: ignore
        self._gemini = genai

    def _chat_gemini(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **_: Any,
    ) -> str:
        self._init_gemini()

        # Separate system prompts for Gemini system_instruction if available.
        system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system").strip()
        user_and_assistant = [m for m in messages if m["role"] != "system"]
        prompt_text = _join_as_prompt(user_and_assistant)

        genai = self._gemini
        # Some versions accept plain dict for generation_config; types.GenerationConfig is optional.
        try:
            from google.generativeai.types import GenerationConfig  # type: ignore
            gen_cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        except Exception:  # noqa: BLE001
            gen_cfg = {"temperature": temperature, "max_output_tokens": max_tokens}

        def _call() -> str:
            model_obj = genai.GenerativeModel(model, system_instruction=system_text or None)  # type: ignore[attr-defined]
            resp = model_obj.generate_content(
                prompt_text,
                generation_config=gen_cfg, # type: ignore
                request_options={"timeout": self.request_timeout},
            )
            # Prefer .text; fallback to candidates/parts
            txt = getattr(resp, "text", None)
            if txt:
                return str(txt).strip()
            # Fallback extraction
            try:
                parts = []
                for cand in getattr(resp, "candidates", []) or []:
                    for part in getattr(cand, "content", {}).get("parts", []):
                        if isinstance(part, dict) and "text" in part:
                            parts.append(part["text"])
                        elif hasattr(part, "text"):
                            parts.append(getattr(part, "text"))
                return "\n".join(p for p in parts if p).strip()
            except Exception:  # noqa: BLE001
                pass
            raise LLMError("Gemini returned no text.")
        return _retry_call(_call, retries=self.retries)

    # -----------------------
    # Provider: ChatGLM (ZhipuAI)
    # -----------------------

    def _init_chatglm(self) -> None:
        if self._chatglm_client is not None:
            return
        try:
            from zhipuai import ZhipuAI  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise LLMError("ZhipuAI SDK not installed. `pip install zhipuai>=2.1.0`") from e
        api_key = os.getenv("CHATGLM_API_KEY")
        if not api_key:
            raise LLMError("Missing CHATGLM_API_KEY.")
        self._chatglm_client = ZhipuAI(api_key=api_key)

    def _chat_chatglm(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **_: Any,
    ) -> str:
        self._init_chatglm()
        # ZhipuAI chat.completions API is OpenAI-like; some models also support responses API.
        def _call() -> str:
            resp = self._chatglm_client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Try OpenAI-like extraction
            try:
                msg = resp.choices[0].message # type: ignore
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if isinstance(content, list):
                    # sometimes it's a list of segments
                    texts = []
                    for seg in content:
                        if isinstance(seg, dict) and "text" in seg:
                            texts.append(seg["text"])
                        elif isinstance(seg, str):
                            texts.append(seg)
                    return "\n".join(texts).strip()
                if content:
                    return str(content).strip()
            except Exception:
                pass
            # Fallback for responses API style
            text = getattr(resp, "output_text", None)
            if text:
                return str(text).strip()
            raise LLMError("ChatGLM returned no text.")
        return _retry_call(_call, retries=self.retries)

    # -----------------------
    # Provider: MCP
    # -----------------------

    def _init_mcp(self) -> None:
        if self._mcp_session is not None:
            return
        
        server_url = os.getenv("MCP_SERVER_URL")
        if not server_url:
            raise LLMError("Missing MCP_SERVER_URL environment variable.")
        
        api_key = os.getenv("MCP_API_KEY")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self._mcp_session = {
            "server_url": server_url.rstrip("/"),
            "headers": headers
        }

    def _chat_mcp(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **_: Any,
    ) -> str:
        self._init_mcp()
        
        def _call() -> str:
            # 构建MCP请求
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "mcp_server_stdin",
                    "arguments": {
                        "messages": messages,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
            }
            
            response = requests.post(
                f"{self._mcp_session['server_url']}/tools/call",
                headers=self._mcp_session["headers"],
                json=payload,
                timeout=self.request_timeout
            )
            
            if response.status_code != 200:
                raise LLMError(f"MCP request failed: {response.status_code} - {response.text}")
            
            result = response.json()
            if "error" in result:
                raise LLMError(f"MCP error: {result['error']}")
            
            # 提取响应内容
            content = result.get("result", {}).get("content", "")
            if not content:
                raise LLMError("MCP returned no content")
            
            return str(content).strip()
        
        return _retry_call(_call, retries=self.retries)
