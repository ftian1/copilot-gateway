"""
OAuth device-code authentication for GitHub Copilot.

Based on the OpenCode reference:
  packages/opencode/src/plugin/github-copilot/copilot.ts

Flow:
  1. POST /auth/device  → initiate device code flow
  2. User visits verification_uri, enters user_code
  3. POST /auth/token   → poll for access token
  4. GET  /auth/status   → check token validity
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from config import CLIENT_ID, USER_AGENT, Config

logger = logging.getLogger(__name__)

# Polling safety margin (ms), from copilot.ts
OAUTH_POLLING_SAFETY_MARGIN_SECS = 3.0


@dataclass
class TokenInfo:
    """Stored token information."""
    access_token: str
    refresh_token: str = ""
    obtained_at: float = 0.0
    expires_at: float = 0.0  # 0 = no expiry known
    enterprise_domain: str = ""

    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at - 60  # 60s grace


class AuthManager:
    """Manages GitHub Copilot OAuth tokens.

    Thread-safe. Provides token for proxy requests.
    Supports optional file-based persistence.
    """

    def __init__(self, config: Config):
        self._config = config
        self._token: Optional[TokenInfo] = None
        self._lock = asyncio.Lock()
        self._pending_device_code: Optional[str] = None
        self._pending_interval: int = 5
        self._pending_domain: str = ""

        # Try to load persisted token
        if config.token_file:
            self._load_token()

    # ── public API ──────────────────────────────────────────────

    @property
    def has_token(self) -> bool:
        return self._token is not None and not self._token.is_expired()

    def get_token(self) -> Optional[str]:
        """Return the bearer token for upstream requests, or None."""
        if self._token and not self._token.is_expired():
            return self._token.access_token
        return None

    def get_auth_headers(self) -> dict[str, str]:
        """Return Authorization header dict for upstream requests."""
        token = self.get_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    # ── device code flow ────────────────────────────────────────

    async def initiate_device_code(self, deployment_type: str = "github.com",
                                   enterprise_url: str = "") -> dict:
        """Step 1: Initiate the OAuth device code flow.

        Returns dict with verification_uri, user_code, device_code, interval
        for the caller to display to the user.
        """
        domain = "github.com"
        if deployment_type == "enterprise" and enterprise_url:
            domain = self._normalize_domain(enterprise_url)

        device_code_url = f"https://{domain}/login/device/code"
        access_token_url = f"https://{domain}/login/oauth/access_token"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                device_code_url,
                json={
                    "client_id": CLIENT_ID,
                    "scope": "read:user",
                },
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                },
                timeout=30,
            )

            if not resp.is_success:
                text = resp.text[:500]
                logger.error(f"Device code initiation failed: {resp.status_code} {text}")
                raise ValueError(f"Failed to initiate device authorization: {resp.status_code}")

            data = resp.json()

        async with self._lock:
            self._pending_device_code = data["device_code"]
            self._pending_interval = data.get("interval", 5)
            self._pending_domain = domain

        return {
            "verification_uri": data["verification_uri"],
            "user_code": data["user_code"],
            "device_code": data["device_code"],
            "interval": data.get("interval", 5),
        }

    async def poll_token(self, device_code: str) -> dict:
        """Step 2: Poll for the access token.

        Blocks until the user completes authorization or the flow fails.
        Returns {"status": "success", "access_token": "..."} or {"status": "failed", "error": "..."}.
        """
        domain = "github.com"
        async with self._lock:
            domain = self._pending_domain or "github.com"
            interval = self._pending_interval

        access_token_url = f"https://{domain}/login/oauth/access_token"

        async with httpx.AsyncClient() as client:
            while True:
                resp = await client.post(
                    access_token_url,
                    json={
                        "client_id": CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "User-Agent": USER_AGENT,
                    },
                    timeout=30,
                )

                if not resp.is_success:
                    return {"status": "failed", "error": f"HTTP {resp.status_code}"}

                data = resp.json()

                if "access_token" in data and data["access_token"]:
                    token = TokenInfo(
                        access_token=data["access_token"],
                        refresh_token=data.get("refresh_token", data["access_token"]),
                        obtained_at=time.time(),
                        expires_at=data.get("expires_in", 0) and time.time() + data.get("expires_in", 0),
                        enterprise_domain=domain if domain != "github.com" else "",
                    )

                    async with self._lock:
                        self._token = token
                        self._pending_device_code = None

                    self._save_token()
                    logger.info("OAuth token obtained successfully")
                    return {"status": "success", "access_token": token.access_token}

                error = data.get("error", "")
                if error == "authorization_pending":
                    await asyncio.sleep(interval + OAUTH_POLLING_SAFETY_MARGIN_SECS)
                    continue
                elif error == "slow_down":
                    new_interval = data.get("interval", interval + 5)
                    await asyncio.sleep(new_interval + OAUTH_POLLING_SAFETY_MARGIN_SECS)
                    continue
                elif error:
                    return {"status": "failed", "error": error}

                # No error, no token — wait and retry
                await asyncio.sleep(interval + OAUTH_POLLING_SAFETY_MARGIN_SECS)

    async def get_status(self) -> dict:
        """Return current authentication status."""
        if self._token is None:
            return {"authenticated": False, "reason": "no_token"}
        if self._token.is_expired():
            return {"authenticated": False, "reason": "expired"}
        return {
            "authenticated": True,
            "obtained_at": self._token.obtained_at,
            "enterprise_domain": self._token.enterprise_domain or "github.com",
        }

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _normalize_domain(url: str) -> str:
        """Strip protocol and trailing slash from a URL/domain."""
        return url.replace("https://", "").replace("http://", "").rstrip("/")

    def _save_token(self) -> None:
        """Persist token to file (persisted by default to ~/.copilot-gateway/token.json)."""
        if not self._config.token_file or not self._token:
            return
        try:
            # Ensure parent directory exists
            token_path = os.path.dirname(self._config.token_file)
            if token_path:
                os.makedirs(token_path, exist_ok=True)

            data = {
                "access_token": self._token.access_token,
                "refresh_token": self._token.refresh_token,
                "obtained_at": self._token.obtained_at,
                "expires_at": self._token.expires_at,
                "enterprise_domain": self._token.enterprise_domain,
            }
            with open(self._config.token_file, "w") as f:
                json.dump(data, f)
            logger.info(f"Token persisted to {self._config.token_file}")
        except OSError as e:
            logger.warning(f"Failed to persist token: {e}")

    def _load_token(self) -> None:
        """Load persisted token from file."""
        if not self._config.token_file:
            return
        try:
            if not os.path.exists(self._config.token_file):
                logger.info(f"No persisted token found at {self._config.token_file}")
                return
            with open(self._config.token_file) as f:
                data = json.load(f)
            self._token = TokenInfo(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", ""),
                obtained_at=data.get("obtained_at", 0),
                expires_at=data.get("expires_at", 0),
                enterprise_domain=data.get("enterprise_domain", ""),
            )
            logger.info(f"Loaded persisted token from {self._config.token_file}")
        except (OSError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load persisted token: {e}")
            self._token = None
