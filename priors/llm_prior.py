"""LLM interface for generating retrieval priors, with on-disk caching.

The provider is NOT hardcoded: the client is built from environment variables
(``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` / ``ANTHROPIC_API_KEY``).
No API key is ever written into code.

Cache directory: ``cache/llm_priors/``. A call is keyed by (dataset,
prompt_type, metadata hash, expert_hint hash, model, prompt_version), so the
same inputs are never re-billed and results are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompts import (
    PROMPT_VERSION,
    build_prompt,
    build_prompt_blind,
    build_prompt_v2,
    build_prompt_v21,
)
from .schema import Prior, parse_priors, parse_priors_v2


def _default_model() -> str:
    return (
        os.environ.get("ANTHROPIC_MODEL")
        or os.environ.get("LLM_MODEL")
        or "deepseek-v4-pro[1m]"
    )


def _get_client():
    import anthropic

    kwargs: Dict[str, str] = {}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if base_url:
        kwargs["base_url"] = base_url
    if auth_token:
        kwargs["auth_token"] = auth_token
    elif api_key:
        kwargs["api_key"] = api_key
    else:
        raise RuntimeError(
            "no LLM credential: set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY"
        )
    return anthropic.Anthropic(**kwargs)


def _hash(value: Any) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_for(prompt_type: str) -> str:
    if prompt_type.startswith("blind_"):
        return prompt_type
    if "v21" in prompt_type:
        return "expert_llm" if "expert" in prompt_type else "metadata_llm"
    return "expert+llm" if "expert" in prompt_type else "llm"


def _parse(raw_response, source, prompt_version):
    if prompt_version == "v2_blind_1":
        from .schema import parse_blind_priors

        return parse_blind_priors(raw_response, source=source)
    if prompt_version == "v21":
        from .executable import parse_executable

        return parse_executable(raw_response, source=source)
    if prompt_version == "v2":
        return parse_priors_v2(raw_response, source=source)
    return parse_priors(raw_response, source=source)


def generate_priors(
    metadata: Dict[str, Any],
    expert_hint: Optional[str] = None,
    model: Optional[str] = None,
    prompt_version: str = PROMPT_VERSION,
    prompt_type: str = "llm",
    cache_dir: str = "cache/llm_priors",
) -> Dict[str, Any]:
    """Generate candidate retrieval priors for one (metadata, expert_hint) input.

    Returns a dict with keys: ``priors`` (List[Prior]), ``model``,
    ``prompt_version``, ``prompt_type``, ``prompt`` (system/user/full),
    ``raw_response``, ``from_cache``, ``cache_path``.
    """
    model = model or _default_model()
    dataset = str(metadata.get("dataset_name", "unknown"))

    metadata_hash = _hash(metadata)
    expert_hint_hash = _hash(expert_hint)

    key_obj = {
        "dataset": dataset,
        "prompt_type": prompt_type,
        "metadata_hash": metadata_hash,
        "expert_hint_hash": expert_hint_hash,
        "model": model,
        "prompt_version": prompt_version,
    }
    cache_key = hashlib.sha256(
        json.dumps(key_obj, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"

    # ---- cache hit: load + re-verify hashes (guards against stale/edited files) ----
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("metadata_hash") == metadata_hash
                and cached.get("expert_hint_hash") == expert_hint_hash
                and cached.get("model") == model
                and cached.get("prompt_version") == prompt_version
            ):
                source = _source_for(prompt_type)
                priors = _parse(cached["raw_response"], source, prompt_version)
                return {
                    "priors": priors,
                    "model": model,
                    "prompt_version": prompt_version,
                    "prompt_type": prompt_type,
                    "prompt": cached.get("prompt"),
                    "raw_response": cached["raw_response"],
                    "from_cache": True,
                    "cache_path": str(cache_path),
                }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # fall through and regenerate on a corrupt cache

    # ---- cache miss: call the LLM ----
    if prompt_version == "v2_blind_1":
        prompt = build_prompt_blind(metadata, expert_hint=expert_hint, prompt_version=prompt_version)
    elif prompt_version == "v21":
        prompt = build_prompt_v21(metadata, expert_hint=expert_hint, prompt_version=prompt_version)
    elif prompt_version == "v2":
        prompt = build_prompt_v2(metadata, expert_hint=expert_hint, prompt_version=prompt_version)
    else:
        prompt = build_prompt(metadata, expert_hint=expert_hint, prompt_version=prompt_version)
    client = _get_client()
    resp = client.messages.create(
        model=model,
        max_tokens=20000,
        system=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}],
    )

    content = resp.content
    if isinstance(content, str):
        raw_response = content
    else:
        # keep only text blocks; skip reasoning-model 'thinking' blocks
        raw_response = "".join(
            getattr(block, "text", "")
            for block in content
            if getattr(block, "type", "text") == "text"
        )
    if not raw_response.strip():
        raise RuntimeError(
            f"LLM returned no text (stop_reason={getattr(resp, 'stop_reason', None)}; "
            f"blocks={[getattr(b, 'type', '?') for b in content]})"
        )

    source = _source_for(prompt_type)
    priors = _parse(raw_response, source, prompt_version)

    cache_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "prompt_type": prompt_type,
                "metadata_hash": metadata_hash,
                "expert_hint_hash": expert_hint_hash,
                "model": model,
                "prompt_version": prompt_version,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed_priors": [p.to_dict() for p in priors],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "priors": priors,
        "model": model,
        "prompt_version": prompt_version,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "raw_response": raw_response,
        "from_cache": False,
        "cache_path": str(cache_path),
    }
