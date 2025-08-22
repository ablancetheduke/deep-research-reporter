"""Simple claim-evidence fact checking using an LLM."""
from __future__ import annotations

import json
from typing import Dict

from .llm import LLM
from . import prompts


def check_fact(llm: LLM, claim: str, evidence: str) -> Dict[str, str]:
    """Return a verdict on whether evidence supports a claim.

    The LLM is asked to respond with a JSON object
    {"verdict": "SUPPORTED|REFUTED|INSUFFICIENT", "rationale": str}.
    """
    payload = f"Claim: {claim}\n\nEvidence:\n{evidence}"
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.FACTCHECK_PROMPT},
        {"role": "user", "content": payload},
    ]
    raw = llm.chat(messages, max_tokens=600, temperature=0.0)
    try:
        data = json.loads(raw)
        verdict = str(data.get("verdict", "INSUFFICIENT")).upper()
        rationale = str(data.get("rationale", "")).strip()
        if verdict not in {"SUPPORTED", "REFUTED", "INSUFFICIENT"}:
            verdict = "INSUFFICIENT"
        return {"verdict": verdict, "rationale": rationale}
    except Exception:
        return {"verdict": "INSUFFICIENT", "rationale": raw.strip()}
