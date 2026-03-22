"""
main.py — FastAPI RAG + Auth Application
Run: uvicorn main:app --reload --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# ── Silence noisy loggers ─────────────────────────────────────────────────────
for _lib in ("httpx", "httpcore", "urllib3", "sentence_transformers",
             "transformers", "huggingface_hub", "filelock", "hpack",
             "passlib", "bcrypt"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")
APP_URL = os.getenv("APP_URL", "http://localhost:8000")

# ── Core imports ──────────────────────────────────────────────────────────────
from core.vector_store import (
    ensure_index, get_stats, insert_documents,
    semantic_search, warm_up_embedder, get_collection, close_connection,
)
from core.rag      import generate_answer, get_backend_info, docker_warmup
from core.reranker import rerank_and_explain, warm_up_reranker, RERANK_CANDIDATES
from core.ingestor  import file_to_document, load_folder, SUPPORTED_EXTENSIONS
from core.feedback import (
    save_feedback, apply_feedback_boost, get_feedback_stats
)
from core.razorpay_billing import (
    create_subscription  as rz_create_subscription,
    get_subscription     as rz_get_subscription,
    cancel_subscription  as rz_cancel_subscription,
    verify_webhook       as rz_verify_webhook,
    get_config           as rz_get_config,
)
from core.billing   import (
    create_subscription_link, get_subscription_details,
    cancel_subscription, verify_webhook,
    get_plan_info, plan_key_from_paypal_plan_id,
    PAYPAL_PLAN_PRO, PAYPAL_PLAN_TEAM,
)

# ── Auth imports ──────────────────────────────────────────────────────────────
from core.auth import (
    fastapi_users, auth_backend,
    current_active_user, current_optional_user,
    init_db, close_db,
    User, PLAN_LIMITS,
    UserRead, UserCreate, UserUpdate,
)

# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up …")

    # Init Beanie (async MongoDB ODM for users)
    await init_db()
    log.info("Auth DB ready ✓")

    # Warm up ML models
    warm_up_embedder()
    log.info("Warming up re-ranker …")
    warm_up_reranker()

    # Warm up Docker LLM
    if os.getenv("LLM_BACKEND", "docker") == "docker":
        log.info("Warming up Docker LLM …")
        ok = docker_warmup()
        log.info("Docker LLM ready ✓" if ok else "Docker LLM warmup skipped")

    # Ensure vector index exists
    get_collection()
    ensure_index()

    # Auto-ingest ./documents folder (shared / demo data, no user_id)
    docs_folder = os.getenv("DOCS_FOLDER", "documents")
    if Path(docs_folder).is_dir():
        docs = load_folder(docs_folder)
        if docs:
            n = insert_documents(docs)
            log.info(f"Auto-ingested {n} chunks from '{docs_folder}'")

    yield
    close_connection()
    await close_db()
    log.info("Shutdown complete.")


app = FastAPI(title="RecoMind API", version="4.0", lifespan=lifespan)

# ── Static files ──────────────────────────────────────────────────────────────
Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Auth routes ───────────────────────────────────────────────────────────────
# POST /auth/register   — create account
# POST /auth/login      — returns JWT token
# POST /auth/logout
# POST /auth/forgot-password
# POST /auth/reset-password
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth", tags=["auth"],
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth", tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth", tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth", tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users", tags=["users"],
)

# ── Request schemas ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:           str
    backend:         Optional[str]  = None
    stream:          bool           = True
    limit:           int            = 5
    score_threshold: float          = 0.68
    metadata_filter: Optional[Dict] = None
    mode:            str            = "qa"    # 'qa' | 'recommend'

class SearchRequest(BaseModel):
    query:           str
    limit:           int            = 5
    score_threshold: float          = 0.68
    metadata_filter: Optional[Dict] = None


# ════════════════════════════════════════════════════════════════════════════
# HELPER — usage guard
# ════════════════════════════════════════════════════════════════════════════

async def _check_query_limit(user: User) -> None:
    """Raise 429 if the user has exceeded their monthly query limit."""
    # Reset counter if it's a new month
    now = datetime.utcnow()
    if (now.year  != user.usage_reset_at.year or
        now.month != user.usage_reset_at.month):
        user.queries_this_month = 0
        user.usage_reset_at     = now
        await user.save()

    if not user.within_limits("queries"):
        limit = user.plan_limit("queries")
        raise HTTPException(
            status_code=429,
            detail={
                "error":   "Query limit reached",
                "plan":    user.plan,
                "limit":   limit,
                "used":    user.queries_this_month,
                "upgrade": "Visit /pricing to upgrade your plan.",
            }
        )


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    """Landing page for visitors, dashboard for logged-in users."""
    p = Path("static/landing.html")
    return p.read_text(encoding="utf-8") if p.exists() else HTMLResponse(
        "<h2>Place landing.html in the static/ folder.</h2>")

@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    p = Path("static/index.html")
    return p.read_text(encoding="utf-8") if p.exists() else HTMLResponse(
        "<h2>Place index.html in the static/ folder.</h2>")


@app.get("/auth/verify", response_class=HTMLResponse)
async def verify_email(token: str):
    """Handle email verification link click."""
    from fastapi.responses import RedirectResponse
    from core.auth.setup   import get_user_manager, get_user_db
    from fastapi_users.exceptions import InvalidVerifyToken, UserAlreadyVerified

    try:
        async for user_db in get_user_db():
            async for manager in get_user_manager(user_db):
                await manager.verify(token, None)
        return RedirectResponse(url="/app?verified=true", status_code=302)
    except UserAlreadyVerified:
        return RedirectResponse(url="/app?verified=already", status_code=302)
    except Exception as exc:
        log.error(f"Verification failed: {exc}")
        return RedirectResponse(url="/login?verify_error=true", status_code=302)


@app.post("/api/auth/resend-verification")
async def resend_verification(user: User = Depends(current_active_user)):
    """Resend verification email to the current user."""
    if user.is_verified:
        raise HTTPException(400, "Email already verified")
    try:
        from core.auth.setup   import get_user_manager, get_user_db
        async for user_db in get_user_db():
            async for manager in get_user_manager(user_db):
                await manager.request_verify(user, None)
        return {"message": "Verification email sent"}
    except Exception as exc:
        log.error(f"Resend verification error: {exc}")
        raise HTTPException(500, str(exc))


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(token: str = ""):
    p = Path("static/reset_password.html")
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Inline fallback
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Reset Password</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  body{{background:#0a0c10;color:#c8d4e3;font-family:'DM Sans',sans-serif;
    display:flex;align-items:center;justify-content:center;min-height:100vh}}
  .card{{background:#0f1219;border:1px solid #2a3344;border-radius:16px;
    padding:36px;width:100%;max-width:400px}}
  input{{width:100%;padding:10px 14px;background:#161b24;border:1.5px solid #2a3344;
    border-radius:8px;color:#c8d4e3;font-size:14px;outline:none;margin-bottom:14px;
    font-family:inherit}}
  input:focus{{border-color:#00e5a0}}
  button{{width:100%;padding:12px;background:#00e5a0;border:none;border-radius:8px;
    color:#000;font-size:14px;font-weight:600;cursor:pointer}}
  h2{{color:#00e5a0;font-size:20px;margin-bottom:20px;font-family:monospace}}
  .msg{{font-size:13px;margin-bottom:12px;padding:8px 12px;border-radius:6px}}
  .err{{background:#ff5f6d12;color:#ff5f6d;border:1px solid #ff5f6d30}}
  .ok {{background:#00e5a012;color:#00e5a0;border:1px solid #00e5a030}}
</style></head><body>
<div class="card">
  <h2>Reset Password</h2>
  <div id="msg"></div>
  <input type="password" id="pw" placeholder="New password (min 8 chars)" minlength="8"/>
  <input type="password" id="pw2" placeholder="Confirm new password"/>
  <button onclick="submit()">Set new password</button>
</div>
<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '{token}';
async function submit() {{
  const pw = document.getElementById('pw').value;
  const pw2= document.getElementById('pw2').value;
  const msg= document.getElementById('msg');
  if (pw !== pw2) {{ msg.className='msg err'; msg.textContent='Passwords do not match'; return; }}
  if (pw.length < 8) {{ msg.className='msg err'; msg.textContent='Min 8 characters'; return; }}
  const r = await fetch('/auth/reset-password', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{token: TOKEN, password: pw}})
  }});
  if (r.ok) {{
    msg.className='msg ok'; msg.textContent='Password updated! Redirecting…';
    setTimeout(()=>window.location.href='/login', 1500);
  }} else {{
    const d = await r.json();
    msg.className='msg err'; msg.textContent = d.detail || 'Reset failed';
  }}
}}
</script></body></html>""")


# ════════════════════════════════════════════════════════════════════════════
# EMAIL VERIFICATION
# ════════════════════════════════════════════════════════════════════════════

@app.get("/auth/verify", response_class=HTMLResponse)
async def verify_email_page(token: str):
    """Handle verification link click from email."""
    from fastapi_users.exceptions import InvalidVerifyToken, UserAlreadyVerified
    from core.auth.setup import get_user_manager, get_user_db
    from fastapi_users.db import BeanieUserDatabase

    try:
        async for user_db in get_user_db():
            async for manager in get_user_manager(user_db):
                user = await manager.verify(token, None)
                log.info(f"Email verified: {user.email}")
                return HTMLResponse("""
<html><head><meta http-equiv="refresh" content="3;url=/app"/></head>
<body style="background:#0a0c10;color:#c8d4e3;font-family:Arial;text-align:center;padding:80px 20px">
  <div style="font-size:48px;margin-bottom:16px">✅</div>
  <h2 style="color:#00e5a0">Email verified!</h2>
  <p style="color:#8a9ab0">Redirecting to your dashboard…</p>
</body></html>""")
    except Exception as exc:
        log.warning(f"Verification failed: {exc}")
        return HTMLResponse("""
<html><head><meta http-equiv="refresh" content="4;url=/login"/></head>
<body style="background:#0a0c10;color:#c8d4e3;font-family:Arial;text-align:center;padding:80px 20px">
  <div style="font-size:48px;margin-bottom:16px">❌</div>
  <h2 style="color:#ff5f6d">Verification failed</h2>
  <p style="color:#8a9ab0">Token may be expired. Request a new one from your profile.</p>
</body></html>""", status_code=400)


@app.post("/api/auth/resend-verification")
async def resend_verification(user: User = Depends(current_active_user)):
    """Resend the verification email."""
    if user.is_verified:
        raise HTTPException(400, "Email is already verified")

    from core.auth.setup import get_user_manager, get_user_db
    try:
        async for user_db in get_user_db():
            async for manager in get_user_manager(user_db):
                await manager.request_verify(user, None)
        return {"status": "sent", "message": f"Verification email sent to {user.email}"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/auth/change-password")
async def change_password(
    request: Request,
    user: User = Depends(current_active_user),
):
    """Change password for logged-in user — verifies old password first."""
    import json
    from passlib.context import CryptContext
    from core.auth.setup import get_user_manager, get_user_db

    body    = await request.json()
    old_pw  = body.get("old_password", "")
    new_pw  = body.get("new_password", "")

    if not old_pw or not new_pw:
        raise HTTPException(400, "old_password and new_password are required")
    if len(new_pw) < 8:
        raise HTTPException(400, "New password must be at least 8 characters")

    # Verify old password using fastapi-users password helper
    try:
        async for user_db in get_user_db():
            async for manager in get_user_manager(user_db):
                verified, _ = manager.password_helper.verify_and_update(
                    old_pw, user.hashed_password
                )
                if not verified:
                    raise HTTPException(400, "Current password is incorrect")

                # Hash and save new password
                new_hash = manager.password_helper.hash(new_pw)
                user.hashed_password = new_hash
                await user.save()
                log.info(f"Password changed for {user.email}")
                return {"status": "ok", "message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"Password change error: {exc}")
        raise HTTPException(500, str(exc))


@app.get("/profile", response_class=HTMLResponse)
async def serve_profile():
    p = Path("static/profile.html")
    return p.read_text(encoding="utf-8") if p.exists() else HTMLResponse(
        "<h2>Place profile.html in the static/ folder.</h2>")

@app.get("/pricing", response_class=HTMLResponse)
async def serve_pricing():
    p = Path("static/pricing.html")
    return p.read_text(encoding="utf-8") if p.exists() else HTMLResponse("<h2>Pricing page not found.</h2>")

@app.get("/reset-password", response_class=HTMLResponse)
async def serve_reset_password():
    """Serves login.html — it auto-detects ?token= and shows the reset form."""
    p = Path("static/login.html")
    return p.read_text(encoding="utf-8") if p.exists() else HTMLResponse(
        "<h2>Place login.html in the static/ folder.</h2>")

@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    p = Path("static/login.html")
    return p.read_text(encoding="utf-8") if p.exists() else HTMLResponse(
        "<h2>Place login.html in the static/ folder.</h2>")


# ════════════════════════════════════════════════════════════════════════════
# API — PUBLIC
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/stats")
async def stats(user: Optional[User] = Depends(current_optional_user)):
    uid = str(user.id) if user else None
    try:
        data = get_stats(user_id=uid)
        if user:
            data["plan"]          = user.plan
            data["queries_used"]  = user.queries_this_month
            data["queries_limit"] = user.plan_limit("queries")
            data["docs_limit"]    = user.plan_limit("docs")
        return data
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/backends")
async def backends():
    return get_backend_info()


@app.get("/api/me")
async def me(user: User = Depends(current_active_user)):
    return {
        "id":                 str(user.id),
        "email":              user.email,
        "full_name":          user.full_name,
        "plan":               user.plan,
        "queries_this_month": user.queries_this_month,
        "queries_limit":      user.plan_limit("queries"),
        "docs_limit":         user.plan_limit("docs"),
        "is_verified":        user.is_verified,
    }


# ════════════════════════════════════════════════════════════════════════════
# API — SEARCH
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/search")
async def search(req: SearchRequest,
                 user: Optional[User] = Depends(current_optional_user)):
    uid     = str(user.id) if user else None
    results = semantic_search(
        req.query, req.limit, req.score_threshold,
        req.metadata_filter, user_id=uid,
    )
    return {"query": req.query, "results": results, "count": len(results)}


# ════════════════════════════════════════════════════════════════════════════
# API — CHAT / RECOMMEND  (requires login)
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(req: ChatRequest,
               user: User = Depends(current_active_user)):
    """RAG + Re-rank + Recommendation endpoint — requires auth."""

    # Enforce plan limits
    await _check_query_limit(user)

    uid = str(user.id)

    # Step 1 — vector search scoped to this user
    candidates = semantic_search(
        req.query,
        limit           = RERANK_CANDIDATES,
        score_threshold = req.score_threshold,
        metadata_filter = req.metadata_filter,
        user_id         = uid,
    )

    # Fallback: if user has no tagged docs yet (legacy data), search unscoped
    if not candidates:
        candidates = semantic_search(
            req.query,
            limit           = RERANK_CANDIDATES,
            score_threshold = req.score_threshold,
            metadata_filter = req.metadata_filter,
        )

    # Step 2 — re-rank
    ranked   = rerank_and_explain(req.query, candidates)
    chunks   = ranked["chunks"]
    metadata = {
        "confidence": ranked.get("confidence", "Low"),
        "best_score": ranked.get("best_score", 0.0),
        "score_gap":  ranked.get("score_gap",  0.0),
        "mode":       req.mode,
    }

    # Apply feedback boost from past interactions
    chunks = apply_feedback_boost(uid, req.query, chunks)

    # Step 3 — increment usage counter (fire-and-forget, don't block response)
    user.queries_this_month += 1
    await user.save()

    if req.stream:
        def token_stream():
            import json
            sources_payload = json.dumps({
                "type":    "sources",
                "sources": [{"text":         c["text"][:200],
                             "score":        c["score"],
                             "vector_score": c.get("vector_score", c["score"]),
                             "source":       c.get("metadata", {}).get("source", "?")}
                            for c in chunks],
                **metadata,
            })
            yield f"data: {sources_payload}\n\n"
            for token in generate_answer(req.query, chunks, req.backend,
                                         stream=True, mode=req.mode):
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"
            yield 'data: {"type": "done"}\n\n'

        return StreamingResponse(token_stream(), media_type="text/event-stream")
    else:
        answer = "".join(generate_answer(req.query, chunks, req.backend,
                                          stream=False, mode=req.mode))
        return {
            "query":   req.query,
            "answer":  answer,
            "mode":    req.mode,
            **metadata,
            "sources": [{"text":   c["text"][:300],
                         "score":  c["score"],
                         "source": c.get("metadata", {}).get("source", "?")}
                        for c in chunks],
        }


# ════════════════════════════════════════════════════════════════════════════
# API — UPLOAD  (requires login)
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# BILLING — PayPal Subscriptions
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/billing/plans")
async def billing_plans():
    """Return plan info and PayPal configuration status."""
    return get_plan_info()


@app.post("/api/billing/subscribe/{plan_key}")
async def subscribe(plan_key: str,
                    user: User = Depends(current_active_user)):
    """
    Create a PayPal subscription and return the approval URL.
    Frontend redirects user to approve_url to complete payment on PayPal.
    """
    if plan_key not in ("pro", "team"):
        raise HTTPException(400, "plan_key must be 'pro' or 'team'")
    if user.plan == plan_key:
        raise HTTPException(400, f"You are already on the {plan_key} plan")

    try:
        result = create_subscription_link(
            plan_key   = plan_key,
            user_id    = str(user.id),
            return_url = f"{APP_URL}/billing/success?plan={plan_key}",
            cancel_url = f"{APP_URL}/pricing",
        )
        return result
    except Exception as exc:
        import traceback
        log.error(f"PayPal subscribe error: {exc}")
        log.error(traceback.format_exc())
        raise HTTPException(500, str(exc))


@app.get("/billing/success")
async def billing_success(
    subscription_id: Optional[str] = None,
    plan:            Optional[str] = None,
    token:           Optional[str] = None,
    ba_token:        Optional[str] = None,
):
    """
    PayPal redirects here after approval — no auth needed (browser redirect).
    The webhook handles the actual plan upgrade asynchronously.
    Just redirect to dashboard with a success flag.
    """
    from fastapi.responses import RedirectResponse
    # Webhook will fire and upgrade the plan — just send user to dashboard
    return RedirectResponse(url="/?upgraded=true", status_code=302)


@app.post("/api/billing/cancel")
async def cancel_plan(user: User = Depends(current_active_user)):
    """Cancel the user's active PayPal subscription."""
    if user.plan == "free":
        raise HTTPException(400, "No active paid plan to cancel")

    sub_id = getattr(user, "paypal_sub_id", None)
    if sub_id:
        cancel_subscription(sub_id, reason="User requested cancellation")

    user.plan          = "free"
    user.paypal_sub_id = None
    await user.save()
    return {"message": "Subscription cancelled. Plan reverted to free."}


@app.post("/api/billing/webhook")
async def paypal_webhook(request: Request):
    """
    PayPal calls this on payment events (BILLING.SUBSCRIPTION.ACTIVATED,
    BILLING.SUBSCRIPTION.CANCELLED, PAYMENT.SALE.COMPLETED, etc.)
    Add this URL in: PayPal Dashboard → Webhooks
    """
    import json
    body    = await request.body()
    headers = dict(request.headers)

    webhook_id = os.getenv("PAYPAL_WEBHOOK_ID", "")
    # Skip verification in sandbox mode or if webhook_id not set yet
    if webhook_id and os.getenv("PAYPAL_MODE", "sandbox") == "live":
        valid = verify_webhook(headers, body, webhook_id)
        if not valid:
            log.warning("PayPal webhook signature verification failed")
            raise HTTPException(400, "Invalid webhook signature")
    elif not webhook_id:
        log.info("Webhook received (no PAYPAL_WEBHOOK_ID set — skipping verification in sandbox)")

    try:
        event      = json.loads(body)
        event_type = event.get("event_type", "")
        resource   = event.get("resource", {})
        log.info(f"PayPal webhook: {event_type}")

        # Subscription activated / payment completed → upgrade plan
        if event_type in ("BILLING.SUBSCRIPTION.ACTIVATED",
                          "BILLING.SUBSCRIPTION.RE-ACTIVATED"):
            sub_id   = resource.get("id", "")
            plan_id  = resource.get("plan_id", "")
            user_id  = resource.get("custom_id", "")
            plan_key = plan_key_from_paypal_plan_id(plan_id)

            if plan_key and user_id:
                user = await User.get(user_id)
                if user:
                    user.plan          = plan_key
                    user.paypal_sub_id = sub_id
                    await user.save()
                    log.info(f"Webhook: upgraded {user.email} → {plan_key}")

        # Subscription cancelled → downgrade to free
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            sub_id = resource.get("id", "")
            users  = await User.find(User.paypal_sub_id == sub_id).to_list()
            for u in users:
                u.plan          = "free"
                u.paypal_sub_id = None
                await u.save()
                log.info(f"Webhook: downgraded {u.email} → free")

        return {"status": "ok"}
    except Exception as exc:
        log.error(f"Webhook processing error: {exc}")
        raise HTTPException(500, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# BILLING — Razorpay (India)
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/razorpay/config")
async def razorpay_config():
    return rz_get_config()


@app.post("/api/razorpay/subscribe/{plan_key}")
async def razorpay_subscribe(plan_key: str,
                              user: User = Depends(current_active_user)):
    if plan_key not in ("pro", "team"):
        raise HTTPException(400, "plan_key must be 'pro' or 'team'")
    try:
        result = rz_create_subscription(
            plan_key   = plan_key,
            user_id    = str(user.id),
            user_email = user.email,
            user_name  = user.full_name or "",
        )
        return result
    except Exception as exc:
        log.error(f"Razorpay subscribe error: {exc}")
        raise HTTPException(500, str(exc))



@app.post("/api/razorpay/verify")
async def razorpay_verify(
    request: Request,
    user: User = Depends(current_active_user),
):
    """
    Called immediately after Razorpay payment succeeds in the browser.
    Verifies the payment and upgrades the plan right away —
    no webhook needed.
    """
    import json, hmac, hashlib
    body = await request.json()

    payment_id      = body.get("razorpay_payment_id", "")
    subscription_id = body.get("razorpay_subscription_id", "")
    signature       = body.get("razorpay_signature", "")
    plan_key        = body.get("plan_key", "pro")

    # Verify signature (HMAC-SHA256)
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").encode()
    message    = f"{payment_id}|{subscription_id}".encode()
    expected   = hmac.new(key_secret, message, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")

    # Upgrade plan immediately
    user.plan          = plan_key
    user.paypal_sub_id = subscription_id
    await user.save()
    log.info(f"Razorpay verified: upgraded {user.email} → {plan_key}")

    return {
        "success": True,
        "plan":    plan_key,
        "message": f"Plan upgraded to {plan_key}",
    }


@app.post("/api/razorpay/webhook")
async def razorpay_webhook(request: Request):
    """Razorpay calls this on payment events."""
    import json
    body      = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")

    if not rz_verify_webhook(body, signature):
        raise HTTPException(400, "Invalid Razorpay webhook signature")

    try:
        event      = json.loads(body)
        event_type = event.get("event", "")
        log.info(f"Razorpay webhook: {event_type}")

        # Subscription activated → upgrade plan
        if event_type == "subscription.activated":
            sub     = event["payload"]["subscription"]["entity"]
            user_id = sub.get("notes", {}).get("user_id")
            plan_key= sub.get("notes", {}).get("plan_key", "pro")
            sub_id  = sub.get("id")

            if user_id:
                from beanie import PydanticObjectId
                user = await User.get(PydanticObjectId(user_id))
                if user:
                    user.plan          = plan_key
                    user.paypal_sub_id = sub_id   # reuse field for sub ID
                    await user.save()
                    log.info(f"Razorpay: upgraded {user.email} → {plan_key}")

        # Subscription cancelled → downgrade
        elif event_type in ("subscription.cancelled", "subscription.completed"):
            sub     = event["payload"]["subscription"]["entity"]
            sub_id  = sub.get("id")
            users   = await User.find(User.paypal_sub_id == sub_id).to_list()
            for u in users:
                u.plan          = "free"
                u.paypal_sub_id = None
                await u.save()
                log.info(f"Razorpay: downgraded {u.email} → free")

        return {"status": "ok"}
    except Exception as exc:
        log.error(f"Razorpay webhook error: {exc}")
        raise HTTPException(500, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# FEEDBACK — 👍 / 👎
# ════════════════════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    query:      str
    chunk_text: str
    source:     str
    rating:     int          # +1 thumbs up, -1 thumbs down
    answer:     str  = ""
    mode:       str  = "qa"


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest,
                          user: User = Depends(current_active_user)):
    """Save 👍/👎 feedback for a query/chunk pair."""
    ok = save_feedback(
        user_id    = str(user.id),
        query      = req.query,
        chunk_text = req.chunk_text,
        source     = req.source,
        rating     = req.rating,
        answer     = req.answer,
        mode       = req.mode,
    )
    if not ok:
        raise HTTPException(500, "Failed to save feedback")
    return {"status": "saved", "rating": req.rating}


@app.get("/api/feedback/stats")
async def feedback_stats(user: User = Depends(current_active_user)):
    """Return feedback stats for the current user."""
    return get_feedback_stats(str(user.id))


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...),
                 user: User = Depends(current_active_user)):
    uid     = str(user.id)
    results = []
    docs    = []

    # Check doc limit
    col     = get_collection()
    current_sources = len(col.distinct("metadata.source", {"user_id": uid}))
    doc_limit       = user.plan_limit("docs")
    remaining       = doc_limit - current_sources

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({"file": f.filename, "status": "unsupported"})
            continue
        if remaining <= 0:
            results.append({
                "file":   f.filename,
                "status": "limit_reached",
                "detail": f"Plan '{user.plan}' allows {doc_limit} documents. Upgrade to add more.",
            })
            continue
        try:
            data = await f.read()
            doc  = file_to_document(data, f.filename)
            if doc:
                docs.append(doc)
                results.append({"file": f.filename, "status": "queued"})
                remaining -= 1
            else:
                results.append({"file": f.filename, "status": "empty"})
        except Exception as exc:
            results.append({"file": f.filename, "status": "error", "detail": str(exc)})

    inserted = insert_documents(docs, user_id=uid) if docs else 0
    return {"inserted_chunks": inserted, "files": results}


@app.delete("/api/documents")
async def delete_source(source: str = Query(...),
                        user: User = Depends(current_active_user)):
    col    = get_collection()
    result = col.delete_many({
        "metadata.source": source,
        "user_id":         str(user.id),
    })
    return {"deleted_chunks": result.deleted_count, "source": source}