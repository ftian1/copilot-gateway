"""
Configuration for the GitHub Copilot Gateway.

Loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Constants from the OpenCode reference implementation
CLIENT_ID = "Ov23li8tweQw6odWQebz"
API_VERSION = "2026-06-01"
USER_AGENT = "copilot-gateway/1.0"

# Base URLs
PUBLIC_API_BASE = "https://api.githubcopilot.com"
GITHUB_DOMAIN = "github.com"

# OAuth endpoints (relative to github.com or enterprise domain)
DEVICE_CODE_PATH = "/login/device/code"
ACCESS_TOKEN_PATH = "/login/oauth/access_token"

# Model refresh interval in seconds
DEFAULT_MODEL_REFRESH_SECS = 300  # 5 minutes

# Default token file for persistence (in user's home directory)
DEFAULT_TOKEN_DIR = Path.home() / ".copilot-gateway"
DEFAULT_TOKEN_FILE = str(DEFAULT_TOKEN_DIR / "token.json")


@dataclass
class Config:
    """Gateway configuration loaded from environment variables."""

    port: int = 9992
    host: str = "0.0.0.0"
    enterprise_domain: str = ""  # empty = public GitHub
    token_file: str = ""  # optional JSON file for token persistence
    model_refresh_secs: int = DEFAULT_MODEL_REFRESH_SECS
    verbose: bool = False  # dump raw Copilot /models response at startup

    @property
    def api_base_url(self) -> str:
        """Return the GitHub Copilot API base URL."""
        if self.enterprise_domain:
            # Normalize domain: strip protocol and trailing slash
            domain = self.enterprise_domain
            domain = domain.replace("https://", "").replace("http://", "").rstrip("/")
            return f"https://copilot-api.{domain}"
        return PUBLIC_API_BASE

    @property
    def github_auth_base(self) -> str:
        """Return the base URL for GitHub OAuth endpoints."""
        if self.enterprise_domain:
            domain = self.enterprise_domain
            domain = domain.replace("https://", "").replace("http://", "").rstrip("/")
            return f"https://{domain}"
        return f"https://{GITHUB_DOMAIN}"

    @property
    def device_code_url(self) -> str:
        return f"{self.github_auth_base}{DEVICE_CODE_PATH}"

    @property
    def access_token_url(self) -> str:
        return f"{self.github_auth_base}{ACCESS_TOKEN_PATH}"


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        port=int(os.getenv("GATEWAY_PORT", "9992")),
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        enterprise_domain=os.getenv("GATEWAY_ENTERPRISE_DOMAIN", ""),
        token_file=os.getenv("GATEWAY_TOKEN_FILE", DEFAULT_TOKEN_FILE),
        model_refresh_secs=int(
            os.getenv("GATEWAY_MODEL_REFRESH_SECS", str(DEFAULT_MODEL_REFRESH_SECS))
        ),
    )
