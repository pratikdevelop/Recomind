"""
start.py — Production startup with graceful fallbacks for cloud environments
Run on Render: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import sys
import logging

log = logging.getLogger("startup")

def check_environment():
    """Validate required env vars before starting."""
    required = ["MONGODB_URI", "JWT_SECRET"]
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"❌ Missing required env vars: {', '.join(missing)}")
        print("   Set them in Render Dashboard → Environment")
        sys.exit(1)

    # Warn about optional but important vars
    optional = {
        "GROQ_API_KEY":      "LLM will not work (set LLM_BACKEND=groq + GROQ_API_KEY)",
        "APP_URL":           "Email links will use localhost (set to your Render URL)",
        "EMAIL_HOST_USER":   "Emails won't send (set Gmail credentials for verification)",
    }
    for k, msg in optional.items():
        if not os.getenv(k):
            print(f"⚠  {k} not set — {msg}")

    print("✓ Environment check passed")

if __name__ == "__main__":
    check_environment()
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")