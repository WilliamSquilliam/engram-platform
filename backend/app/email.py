"""Transactional email — gated the same way serving/connectors are.

EMAIL_BACKEND (config) selects the provider:
  none (default): nothing is sent. The caller keeps the link and returns it in the
                  API response (and we log it) so every flow — invites, password
                  reset, access-request approval — is fully usable and testable with
                  NO email provider configured.
  ses:            AWS SES via boto3 (already a dependency).
  smtp:           stdlib smtplib.

The single seam is `send_email(...) -> bool`: True when a real provider accepted the
message, False when EMAIL_BACKEND=none (so routers know to leak the link). Never
raises on a provider error — email is best-effort and must not 500 the request; a
failure is logged and reported as not-sent, and the router falls back to returning
the link.

Security: when a real backend is configured the link is emailed and NEVER returned
in the response body or logged, so tokens don't leak into logs/clients in prod.
"""
from __future__ import annotations

import logging

from . import config

logger = logging.getLogger(__name__)


def email_enabled() -> bool:
    """True when a real provider is configured (ses/smtp). Read at call time so a
    test can flip EMAIL_BACKEND via monkeypatch without a module reload."""
    return config.EMAIL_BACKEND in ("ses", "smtp")


def _send_ses(to: str, subject: str, body: str, html: str | None = None) -> bool:
    import boto3  # imported lazily so `none`/`smtp` don't need AWS creds

    ses_body: dict = {"Text": {"Data": body}}
    if html:
        ses_body["Html"] = {"Data": html}
    client = boto3.client("ses", region_name=config.AWS_REGION)
    client.send_email(
        Source=config.EMAIL_FROM,
        Destination={"ToAddresses": [to]},
        Message={
            "Subject": {"Data": subject},
            "Body": ses_body,
        },
    )
    return True


def _send_smtp(to: str, subject: str, body: str, html: str | None = None) -> bool:
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = config.EMAIL_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as smtp:
        if config.SMTP_STARTTLS:
            smtp.starttls()
        if config.SMTP_USER:
            smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
        smtp.send_message(msg)
    return True


def send_email(to: str, subject: str, body: str, html: str | None = None) -> bool:
    """Send to `to`: `body` is the plain-text part, `html` the optional styled part
    (see email_templates.py). Returns True if a real provider accepted it, False when
    EMAIL_BACKEND=none (caller then returns the link in the response). Never raises."""
    backend = config.EMAIL_BACKEND
    if backend == "none":
        # No provider: the link travels in the API response instead. Log it so a
        # local operator can complete the flow from the server logs too.
        logger.info("EMAIL(none) to=%s subject=%r body=%s", to, subject, body)
        return False
    try:
        if backend == "ses":
            return _send_ses(to, subject, body, html)
        if backend == "smtp":
            return _send_smtp(to, subject, body, html)
        logger.error("Unknown EMAIL_BACKEND=%r; not sending", backend)
        return False
    except Exception:  # noqa: BLE001 — email is best-effort, must not 500 the request
        logger.exception("Failed to send email to %s via %s", to, backend)
        return False
