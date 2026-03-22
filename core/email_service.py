"""
core/email_service.py — Email sending via SMTP
Free: Gmail (500/day) · Outlook · Zoho

Gmail setup (2 minutes):
1. Google Account → Security → 2-Step Verification → Enable
2. Security → App Passwords → Select app: Mail → Generate
3. Copy the 16-char password → paste as EMAIL_HOST_PASSWORD in .env
"""

import os
import asyncio
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText

log = logging.getLogger(__name__)

EMAIL_HOST      = os.getenv("EMAIL_HOST",          "smtp.gmail.com")
EMAIL_PORT      = int(os.getenv("EMAIL_PORT",      "587"))
EMAIL_USER      = os.getenv("EMAIL_HOST_USER",     "")
EMAIL_PASSWORD  = os.getenv("EMAIL_HOST_PASSWORD", "")
EMAIL_FROM      = os.getenv("EMAIL_FROM",          EMAIL_USER)
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME",     "RecoMind")
APP_URL         = os.getenv("APP_URL",             "http://localhost:8000")


def is_configured() -> bool:
    return bool(EMAIL_USER and EMAIL_PASSWORD)


def _send_smtp(to: str, subject: str, html: str, text: str) -> bool:
    if not is_configured():
        log.warning(
            "Email not configured — set EMAIL_HOST_USER + EMAIL_HOST_PASSWORD in .env\n"
            f"  Would send to {to}: {subject}"
        )
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"{EMAIL_FROM_NAME} <{EMAIL_FROM}>"
    msg["To"]      = to
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=15) as s:
            s.ehlo(); s.starttls(); s.login(EMAIL_USER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_FROM, to, msg.as_string())
        log.info(f"✉ Email sent → {to} | {subject}")
        return True
    except Exception as exc:
        log.error(f"Email failed → {to}: {exc}")
        return False


async def send_email(to: str, subject: str, html: str, text: str = "") -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _send_smtp, to, subject, html, text or subject)


# ── Base template ─────────────────────────────────────────────────────────────
def _wrap(body: str) -> str:
    return f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#0a0c10;font-family:Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:40px 20px">
<table width="100%" style="max-width:520px;background:#0f1219;border:1px solid #1e2530;border-radius:16px;overflow:hidden">
<tr><td style="background:#0d1018;padding:20px 32px;border-bottom:1px solid #1e2530">
  <span style="background:#00e5a0;padding:4px 10px;border-radius:6px;font-family:monospace;font-size:13px;font-weight:700;color:#000">R</span>
  <span style="font-family:monospace;font-size:13px;color:#c8d4e3;margin-left:8px">RecoMind</span>
</td></tr>
<tr><td style="padding:32px">{body}</td></tr>
<tr><td style="padding:16px 32px;border-top:1px solid #1e2530;font-size:11px;color:#4a5a70;text-align:center">
  © 2026 RecoMind · Built in India 🇮🇳<br/>If you didn't request this, ignore it safely.
</td></tr>
</table></td></tr></table></body></html>"""


# ── Email templates ───────────────────────────────────────────────────────────

async def send_verification_email(email: str, token: str) -> bool:
    url  = f"{APP_URL}/auth/verify?token={token}"
    body = f"""
<h2 style="color:#c8d4e3;font-size:22px;margin:0 0 12px">Verify your email ✉</h2>
<p style="color:#8a9ab0;font-size:14px;line-height:1.7;margin:0 0 28px">
  One click to activate your RecoMind account:
</p>
<div style="text-align:center;margin:0 0 24px">
  <a href="{url}" style="display:inline-block;padding:14px 36px;background:#00e5a0;
     color:#000;font-weight:700;font-size:15px;border-radius:10px;text-decoration:none">
    Verify my email →
  </a>
</div>
<p style="color:#5a6b82;font-size:11px;text-align:center">
  Or copy: <span style="color:#00e5a0;font-family:monospace;word-break:break-all">{url}</span><br/>
  <em>Link expires in 24 hours.</em>
</p>"""
    return await send_email(email, "Verify your RecoMind account", _wrap(body),
                            f"Verify your account: {url}")


async def send_password_reset_email(email: str, token: str) -> bool:
    url  = f"{APP_URL}/reset-password?token={token}"
    body = f"""
<h2 style="color:#c8d4e3;font-size:22px;margin:0 0 12px">Reset your password 🔑</h2>
<p style="color:#8a9ab0;font-size:14px;line-height:1.7;margin:0 0 28px">
  Click below to choose a new password for your RecoMind account.
</p>
<div style="text-align:center;margin:0 0 24px">
  <a href="{url}" style="display:inline-block;padding:14px 36px;background:#00e5a0;
     color:#000;font-weight:700;font-size:15px;border-radius:10px;text-decoration:none">
    Reset password →
  </a>
</div>
<p style="color:#5a6b82;font-size:11px;text-align:center">
  Or copy: <span style="color:#00e5a0;font-family:monospace;word-break:break-all">{url}</span><br/>
  <em>Expires in 1 hour. If you didn't request this, ignore safely.</em>
</p>"""
    return await send_email(email, "Reset your RecoMind password", _wrap(body),
                            f"Reset password: {url}")


async def send_welcome_email(email: str, name: str = "") -> bool:
    first = (name.split()[0] if name else "there")
    body  = f"""
<h2 style="color:#c8d4e3;font-size:22px;margin:0 0 12px">Welcome to RecoMind, {first}! 🎉</h2>
<p style="color:#8a9ab0;font-size:14px;line-height:1.7;margin:0 0 20px">Your account is ready. Here's how to get started in 3 steps:</p>
<table width="100%" style="margin:0 0 24px">
  <tr><td style="padding:8px 0;border-bottom:1px solid #1e2530;color:#c8d4e3;font-size:13px">
    <span style="color:#00e5a0;font-family:monospace">01</span>&nbsp;&nbsp;Upload your documents (PDF, DOCX, CSV, Markdown)</td></tr>
  <tr><td style="padding:8px 0;border-bottom:1px solid #1e2530;color:#c8d4e3;font-size:13px">
    <span style="color:#00e5a0;font-family:monospace">02</span>&nbsp;&nbsp;Ask questions in plain English — get instant answers</td></tr>
  <tr><td style="padding:8px 0;color:#c8d4e3;font-size:13px">
    <span style="color:#00e5a0;font-family:monospace">03</span>&nbsp;&nbsp;Switch to Recommend mode for structured recommendation cards</td></tr>
</table>
<div style="text-align:center">
  <a href="{APP_URL}/app" style="display:inline-block;padding:12px 32px;background:#00e5a0;
     color:#000;font-weight:700;font-size:14px;border-radius:10px;text-decoration:none">
    Open RecoMind →
  </a>
</div>"""
    return await send_email(email, "Welcome to RecoMind 🚀", _wrap(body),
                            f"Welcome! Open the app: {APP_URL}/app")