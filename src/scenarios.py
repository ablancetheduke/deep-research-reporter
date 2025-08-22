"""Scenario prediction and trend analysis module."""
from __future__ import annotations

from .llm import LLM
from . import prompts


def generate_scenarios(llm: LLM, topic: str, horizon: str = "5 years") -> str:
    """Return scenario analysis for a topic and time horizon."""
    payload = f"Topic: {topic}\nTime horizon: {horizon}"
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.SCENARIO_PROMPT},
        {"role": "user", "content": payload},
    ]
    return llm.chat(messages, max_tokens=1200)
