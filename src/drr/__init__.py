"""
Deep Research Reporter (drr) package.

This package provides a modular pipeline for generating structured
analysis reports from a topic prompt using large language models (LLMs).

Modules:
- llm.py       : Thin wrapper for LLM API calls
- prompts.py   : Centralized prompt templates
- pipeline.py  : Core pipeline (intent parsing → outline → writing → critique → compose)
- cli.py       : Command-line entry point
"""

__all__ = ["llm", "prompts", "pipeline"]
