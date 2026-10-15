# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

from .json_parser import _extract_json

# Source: experiments/run_agent_v2_electricity.py:296-302
def _list_from(text, key):
    data = _extract_json(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and key in data:
        return data[key]
    raise ValueError(f"expected JSON object with a '{key}' list")

def _call_llm(system, user, stage, model):
    """Runtime must attach the frozen experimental transport before an LLM call."""
    raise RuntimeError('No experiment LLM transport attached; configure unified/runtime.py')

def _default_model():
    raise RuntimeError('No experiment LLM model attached; configure unified/runtime.py')
