"""Source connectors for the 'connect a source' menu. Reads the config-driven connector
registry (connectors/registry.py) so the menu never hardcodes availability — `filesystem` is
always available; `google_drive` and `sharepoint` render as 'coming soon' until their OAuth
creds are configured. Route lives at /connectors. Mirrors routers/model_tiers.py."""
from fastapi import APIRouter, Depends

from ..connectors import connectors as list_connectors
from ..deps import get_current_user
from ..models import User

router = APIRouter(prefix="/connectors", tags=["connectors"])


@router.get("")
def list_source_connectors(user: User = Depends(get_current_user)) -> dict:
    """The ingestion sources shown in the 'connect a source' menu. `available` connectors are
    selectable now; the rest render as 'coming soon' until their OAuth app credentials are
    configured (config.*_ENABLED). No connector's secrets are exposed — only the availability
    flag derived from them."""
    return {
        "connectors": [
            {
                "id": c.id,
                "label": c.label,
                "available": c.available,
                "description": c.description,
            }
            for c in list_connectors()
        ]
    }
