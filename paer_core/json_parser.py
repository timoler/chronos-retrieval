# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# Source: priors/schema.py:72-108
def _close_unbalanced(text: str):
    """Close unbalanced brackets in a JSON fragment, or ``None`` if truncated mid-string.

    Handles the common LLM truncation where a JSON object/array is cut off just
    before its final closing brace(s). Returns ``None`` (unrepairable) when the
    text ends inside an unterminated string, since closing brackets then would not
    produce a trustworthy value.
    """
    pairs = {"{": "}", "[": "]"}
    stack = []
    in_str = False
    esc = False
    for c in text:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack and pairs[stack[-1]] == c:
                    stack.pop()
    if in_str:
        return None
    if not stack:
        return text
    suffix = "".join(pairs[o] for o in reversed(stack))
    candidate = text.rstrip()
    if candidate.endswith(","):
        candidate = candidate[:-1].rstrip()
    return candidate + suffix

# Source: priors/schema.py:111-165
def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object/list from an LLM response."""
    text = (text or "").strip()

    # strip markdown code fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # fallback: locate the outermost balanced JSON object or array anywhere in text
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        break

    # fallback: repair truncation by closing unbalanced brackets (only at a
    # bracket/whitespace boundary; mid-string truncation is left to the caller)
    repaired = _close_unbalanced(text)
    if repaired is not None:
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            pass

    raise ValueError("could not extract JSON from LLM response: " + repr(text[:800]))
