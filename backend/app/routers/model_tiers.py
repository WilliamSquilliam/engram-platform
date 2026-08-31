"""Model tiers for the onboarding flow's 'choose model' step. Reads the config-driven serving
registry (serving.py) so the menu never hardcodes a model, precision, GPU, or cloud — tiers
render as 'coming soon' until a serving engine is wired. Route lives at /models."""
from fastapi import APIRouter, Depends

from .. import serving
from ..deps import get_current_user
from ..models import User

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_model_tiers(user: User = Depends(get_current_user)) -> dict:
    """The model tiers shown in the onboarding 'choose model' step. `available` tiers are
    selectable now; the rest render as 'coming soon' until a serving engine is wired. Does not
    expose the internal `model_ref` — the client sends a tier `id` at onboarding and the backend
    maps it to weights (serving.model_ref_for_tier)."""
    return {
        "default_tier": serving.DEFAULT_TIER_ID,
        "tiers": [
            {
                "id": t.id,
                "label": t.label,
                "description": t.description,
                "precision": t.precision,
                "context_tokens": t.context_tokens,
                "available": t.available,
            }
            for t in serving.tiers()
        ],
    }
