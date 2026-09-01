"""Branded transactional-email templates (invite / access-approved / password-reset).

Each builder returns (subject, text, html): a plain-text part for clients that want it
and an inline-styled HTML card for everyone else (email clients ignore <style> blocks,
so every style is inline — that's the norm for email, not a shortcut). The brand mark is
the landing site's hosted PNG lockup (sig-logo.png — PNG because Gmail strips SVG), with
the name as alt text so clients that block remote images still show the brand.
"""
from __future__ import annotations

import html as _html

_INDIGO = "#5b54ee"          # the ribbons-gradient anchor color
_BG = "#f4f5fb"
_TEXT = "#1e2130"
_MUTED = "#6b7085"


def _layout(heading: str, intro: str, button_label: str, button_url: str, note: str) -> str:
    """One centered card: wordmark, heading, one paragraph, one button, small print.
    Table-based layout + inline styles — the only markup that renders consistently
    across Outlook/Gmail/Apple Mail."""
    h = _html.escape
    return f"""\
<!doctype html>
<html>
<body style="margin:0;padding:0;background:{_BG};">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_BG};padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
             style="max-width:480px;background:#ffffff;border-radius:12px;padding:36px 40px;
                    font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
        <tr><td align="center" style="padding-bottom:24px;">
          <img src="https://engramdynamics.org/sig-logo.png" alt="Engram Dynamics" width="104"
               style="display:block;width:104px;height:auto;border:0;" />
        </td></tr>
        <tr><td style="font-size:21px;font-weight:600;color:{_TEXT};padding-bottom:12px;">{h(heading)}</td></tr>
        <tr><td style="font-size:15px;line-height:1.6;color:{_TEXT};padding-bottom:28px;">{h(intro)}</td></tr>
        <tr><td align="center" style="padding-bottom:28px;">
          <a href="{h(button_url)}"
             style="display:inline-block;background:{_INDIGO};color:#ffffff;text-decoration:none;
                    font-size:15px;font-weight:600;padding:12px 32px;border-radius:8px;">{h(button_label)}</a>
        </td></tr>
        <tr><td style="font-size:12px;line-height:1.6;color:{_MUTED};border-top:1px solid #ececf4;padding-top:20px;">
          {h(note)}<br>
          If the button doesn't work, paste this link into your browser:<br>
          <a href="{h(button_url)}" style="color:{_INDIGO};word-break:break-all;">{h(button_url)}</a>
        </td></tr>
      </table>
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:480px;">
        <tr><td align="center" style="font-size:12px;color:{_MUTED};padding-top:16px;
                 font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
          Engram Dynamics · engramdynamics.org
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def _text(heading: str, intro: str, button_label: str, button_url: str, note: str) -> str:
    return (f"{heading}\n\n{intro}\n\n{button_label}: {button_url}\n\n{note}\n\n"
            f"— Engram Dynamics · engramdynamics.org")


def invite_email(workspace: str, link: str) -> tuple[str, str, str]:
    """A teammate invite into an existing workspace."""
    heading = f"Join {workspace} on Engram Dynamics"
    intro = (f"{workspace} uses Engram Dynamics to chat with its document library. "
             "You've been invited to join the team — set a password to get started.")
    note = ("This link is yours alone and expires in 7 days. "
            "If you weren't expecting this invitation, you can safely ignore this email.")
    return (f"You're invited to {workspace} on Engram Dynamics",
            _text(heading, intro, "Accept invitation", link, note),
            _layout(heading, intro, "Accept invitation", link, note))


def access_approved_email(workspace: str, link: str) -> tuple[str, str, str]:
    """A waitlist access request was approved — their new workspace is ready."""
    heading = "Your workspace is ready"
    intro = (f"Your access request was approved — {workspace} is set up and waiting. "
             "Set a password to sign in and connect your first documents.")
    note = ("This link is yours alone and expires in 7 days. "
            "If you didn't request access, you can safely ignore this email.")
    return ("Your Engram Dynamics access is ready",
            _text(heading, intro, "Set up my account", link, note),
            _layout(heading, intro, "Set up my account", link, note))


def password_reset_email(link: str) -> tuple[str, str, str]:
    heading = "Reset your password"
    intro = "Choose a new password for your Engram Dynamics account."
    note = ("This link expires in 1 hour and can be used once. "
            "If you didn't request a reset, you can safely ignore this email — "
            "your password is unchanged.")
    return ("Reset your Engram Dynamics password",
            _text(heading, intro, "Reset password", link, note),
            _layout(heading, intro, "Reset password", link, note))
