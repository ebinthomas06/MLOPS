import hashlib
import hmac
import os

from fastapi import APIRouter, Header, HTTPException, Request

router = APIRouter(prefix="/webhook", tags=["Webhook"])


def verify_github_signature(
    payload: bytes,
    signature: str | None,
    secret: str,
) -> bool:
    if not signature:
        return False

    expected_signature = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature)


@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
):
    payload = await request.body()

    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")

    if not webhook_secret:
        raise HTTPException(
            status_code=500,
            detail="GITHUB_WEBHOOK_SECRET is not configured",
        )

    if not verify_github_signature(
        payload,
        x_hub_signature_256,
        webhook_secret,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid GitHub webhook signature",
        )

    return {
        "status": "received",
        "event": x_github_event,
    }
