"""Serving interface — the ONE boundary between the control plane and the GPU plane.

The control plane deliberately does NOT know the GPU type, cloud provider, or serving
precision. It reaches serving over HTTP (config.ML_SERVICE_URL / INFERENCE_SERVICE_URL)
and identifies models only by `model_ref`. Swapping L40S<->H100, AWS<->neocloud, 4-bit<->FP8,
or one model for another is a CONFIG change (the env below), never a code change here or in any
product feature — features import `serving`, not a hardware assumption.

Two config-driven things live here:
  * the reachable serving profile (backend path + endpoints — delegated to config), and
  * the MODEL REGISTRY: the product's model tiers (e.g. Fast/Balanced/Best) mapped to the
    `model_ref` + precision each is served at. Entries are PLACEHOLDERS until the cloud/box
    decision lands; ops fills them via MODEL_REGISTRY_JSON with zero code changes.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

from . import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTier:
    """A product-facing model choice (the onboarding menu's Fast/Balanced/Best) mapped to the
    concrete weights the GPU plane serves. `model_ref` is what a cart is stamped/bound to
    (cartridges.model_binding); an empty `model_ref` marks a placeholder tier the UI renders as
    'coming soon' until a model + box is chosen. precision/context are informational (the review
    step). No field names a GPU or cloud — that lives entirely in the deploy config."""
    id: str             # stable id: "fast" | "balanced" | "best" | ...
    label: str          # UI label, e.g. "Balanced"
    description: str     # one line for the menu
    model_ref: str       # HF id / weights id stamped into carts ("" = not chosen yet)
    precision: str       # "fp8" | "w4a4" | "bf16" | "" — informational
    context_tokens: int  # max context, informational
    enabled: bool        # selectable in the menu (needs a live serving engine)

    @property
    def available(self) -> bool:
        """Selectable right now: a real model behind a live engine, not a placeholder."""
        return self.enabled and bool(self.model_ref)


# Default registry: PLACEHOLDERS. No model or hardware is committed yet (cloud-credit decision
# pending), so every tier ships disabled with an empty model_ref and the UI shows it as
# "coming soon". When a box is chosen, set MODEL_REGISTRY_JSON (below) — or flip a tier's
# model_ref + enabled — with no code change. The descriptions name PLAN.md candidates only as a
# hint; nothing here assumes any of them runs on a particular GPU or cloud.
_DEFAULT_TIERS: tuple[ModelTier, ...] = (
    ModelTier("fast", "Fast", "Lowest latency, highest document density.",
              model_ref="", precision="", context_tokens=0, enabled=False),
    ModelTier("balanced", "Balanced", "Best grounded accuracy per cost. The default.",
              model_ref="", precision="", context_tokens=0, enabled=False),
    ModelTier("best", "Best", "Highest grounded fact-retrieval accuracy.",
              model_ref="", precision="", context_tokens=0, enabled=False),
)


def _load_registry() -> list[ModelTier]:
    """MODEL_REGISTRY_JSON (env) overrides the placeholder tiers. Shape: a JSON list of objects
    with the ModelTier fields (`id` required; the rest default). This is the seam that wires a
    real model + precision the moment the cloud/box is chosen, with no code change. Malformed
    JSON is logged and ignored (fall back to placeholders) so a bad env var can't take the API
    down."""
    raw = os.environ.get("MODEL_REGISTRY_JSON", "").strip()
    if not raw:
        return list(_DEFAULT_TIERS)
    try:
        items = json.loads(raw)
        tiers = [
            ModelTier(
                id=str(it["id"]),
                label=str(it.get("label", str(it["id"]).title())),
                description=str(it.get("description", "")),
                model_ref=str(it.get("model_ref", "")),
                precision=str(it.get("precision", "")),
                context_tokens=int(it.get("context_tokens", 0)),
                enabled=bool(it.get("enabled", False)),
            )
            for it in items
        ]
        return tiers or list(_DEFAULT_TIERS)
    except (ValueError, KeyError, TypeError) as e:  # noqa: BLE001-adjacent: narrow + logged
        logger.warning("MODEL_REGISTRY_JSON ignored (%s); using placeholder tiers", e)
        return list(_DEFAULT_TIERS)


_TIERS: list[ModelTier] = _load_registry()
DEFAULT_TIER_ID: str = os.environ.get("DEFAULT_MODEL_TIER", "balanced")


def tiers() -> list[ModelTier]:
    """All product model tiers (for the onboarding menu), placeholders included."""
    return list(_TIERS)


def tier(tier_id: str) -> ModelTier | None:
    return next((t for t in _TIERS if t.id == tier_id), None)


def available_tiers() -> list[ModelTier]:
    """Tiers a user can actually pick right now (enabled + a real model_ref)."""
    return [t for t in _TIERS if t.available]


def model_ref_for_tier(tier_id: str) -> str:
    """The weights id a chosen tier onboards/serves against (stamped into carts). '' if the tier
    is a placeholder — callers must treat that as 'no serving engine yet'."""
    t = tier(tier_id)
    return t.model_ref if t else ""


def backend() -> str:
    """The active inference backend path ('hf' | 'vllm'). The hardware/provider is NOT modeled —
    only the path shape; the endpoints (config.*_SERVICE_URL) are the swap point."""
    return config.INFERENCE_BACKEND


def profile() -> dict:
    """The serving profile as a plain dict (ops / debugging): backend + endpoints + the tier
    registry. This is the whole 'where + what' the control plane knows; 'on what hardware / which
    cloud' is intentionally absent."""
    return {
        "backend": backend(),
        "ml_service_url": config.ML_SERVICE_URL,
        "inference_service_url": config.INFERENCE_SERVICE_URL,
        "default_tier": DEFAULT_TIER_ID,
        "tiers": [asdict(t) for t in _TIERS],
    }
