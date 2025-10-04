from __future__ import annotations

# =========================
# mcp_team_server/config.py
# =========================
# - Loads environment variables (.env)
# - Builds a dev-friendly Azure credential chain
# - Initializes:
#     * Azure OpenAI GPT-4.1 client (Entra ID auth)
#     * Azure AI Inference client (DeepSeek deployment)
#     * Azure AI Projects client (endpoint-based; supports Agents API)
# - Exposes: SETTINGS (Settings), AzureClients (holder of initialized SDK clients)

import os
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# ---- Azure Identity & helpers ----
from azure.identity import (
    DefaultAzureCredential,
    AzureCliCredential,
    InteractiveBrowserCredential,
    ChainedTokenCredential,
    get_bearer_token_provider,
)

# ---- Azure OpenAI (for GPT-4.1, O-series, etc.) ----
from openai import AzureOpenAI

# ---- Azure AI Inference (DeepSeek via /inference) ----
from azure.ai.inference import ChatCompletionsClient

# ---- Azure AI Projects (Foundry Projects / Agents / Threads) ----
from azure.ai.projects import AIProjectClient

# -----------------------------
# Environment & base config
# -----------------------------
load_dotenv(override=True)

logger = logging.getLogger("actor_critic_mcp.config")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# -----------------------------
# Settings (centralized env)
# -----------------------------
@dataclass
class Settings:
    # Which model-provider the team/agents should use by default
    #   choices: "gpt41" | "deepseek" | "aiproject"
    model_provider: str = os.getenv("MODEL_PROVIDER", "gpt41")

    # Scratchpad DB path for the orchestrator
    scratchpad_db: str = os.getenv("SCRATCHPAD_DB", "./scratchpad.db")

    # Azure OpenAI GPT-4.1
    gpt41_endpoint: str = os.getenv("GPT41_ENDPOINT", "")
    gpt41_deployment: str = os.getenv("GPT41_DEPLOYMENT", "")

    # DeepSeek via Azure AI Inference
    deepseek_endpoint: str = os.getenv("ENDPOINT", "")          # e.g., https://<your>.ai.azure.com
    deepseek_deployment: str = os.getenv("MODEL_NAME", "")      # e.g., DeepSeek-R1-0528

    # Azure AI Projects (Foundry) — prefer endpoint; keep conn str for fallback
    ai_projects_endpoint: str = os.getenv("AZURE_PROJECT_ENDPOINT", "")
    ai_projects_connection_string: str = os.getenv("AZURE_PROJECT_CONNECTION_STRING", "")
    ai_projects_agent_id: str = os.getenv("AZURE_AGENT_ID", "")

    # Optional: set these to prefer a specific tenant in interactive login
    azure_tenant_id: Optional[str] = os.getenv("AZURE_TENANT_ID") or None

    # Allow preferring CLI for local dev auth (default on)
    use_cli_credential: bool = os.getenv("AZURE_USE_CLI", "1") == "1"


SETTINGS = Settings()


# -----------------------------
# Helpers
# -----------------------------
def build_credential() -> ChainedTokenCredential:
    """
    Dev-friendly credential chain:

      1) Azure CLI           (az login)           ← default enabled for local dev
      2) Interactive Browser (respects AZURE_TENANT_ID if set)
      3) DefaultAzureCredential (Service Principal, Managed Identity, VSCode, etc.)

    Toggle CLI via env: AZURE_USE_CLI=0 to skip AzureCliCredential.
    """
    chain = []
    if SETTINGS.use_cli_credential:
        chain.append(AzureCliCredential())
    chain.append(
        InteractiveBrowserCredential(tenant_id=SETTINGS.azure_tenant_id)
        if SETTINGS.azure_tenant_id
        else InteractiveBrowserCredential()
    )
    # Avoid double CLI in DefaultAzureCredential if we already added it
    chain.append(DefaultAzureCredential(exclude_cli_credential=True))
    return ChainedTokenCredential(*chain)


def _endpoint_from_conn_str(conn_str: str) -> str | None:
    """
    Back-compat helper to extract a project endpoint from older connection strings like:
      'endpoint=...;subscriptionId=...;resourceGroupName=...;projectName=...'
    """
    try:
        parts = dict(
            item.split("=", 1) for item in conn_str.split(";") if "=" in item
        )
        for key in ("endpoint", "project_endpoint", "Endpoint", "PROJECT_ENDPOINT", "host"):
            if parts.get(key):
                return parts[key]
    except Exception:
        pass
    return None


# -----------------------------
# Azure Clients bootstrap
# -----------------------------
class AzureClients:
    """
    Initializes and verifies all SDK clients used by the team:
      - self.gpt4_client (Azure OpenAI)
      - self.deepseek_client (Azure AI Inference)
      - self.project_client (Azure AI Projects)
    Stores verified deployment/agent IDs back on self for convenience.
    """

    def __init__(self):
        cred = build_credential()

        # ---- GPT-4.1 (Azure OpenAI) ----
        logger.info("Initializing GPT-4.1 client with Entra ID authentication...")
        if not (SETTINGS.gpt41_endpoint and SETTINGS.gpt41_deployment):
            raise EnvironmentError("GPT41_ENDPOINT or GPT41_DEPLOYMENT missing from environment.")

        token_provider = get_bearer_token_provider(
            cred, "https://cognitiveservices.azure.com/.default"
        )
        self.gpt4_client = AzureOpenAI(
            azure_endpoint=SETTINGS.gpt41_endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2025-01-01-preview",  # keep aligned with your earlier code
        )
        self.gpt41_deployment = SETTINGS.gpt41_deployment

        logger.info("Verifying GPT-4.1 deployment '%s'...", self.gpt41_deployment)
        try:
            self.gpt4_client.chat.completions.create(
                model=self.gpt41_deployment,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
            )
            logger.info("✅ GPT-4.1 client verification successful.")
        except Exception as e:
            logger.critical("❌ GPT-4.1 client test FAILED. Check GPT41_* env vars. Error: %s", e)
            raise

        # ---- DeepSeek (Azure AI Inference) ----
        logger.info("Initializing DeepSeek client with Entra ID authentication...")
        if not (SETTINGS.deepseek_endpoint and SETTINGS.deepseek_deployment):
            raise EnvironmentError("DeepSeek ENDPOINT or MODEL_NAME missing from environment.")

        self.deepseek_client = ChatCompletionsClient(
            endpoint=SETTINGS.deepseek_endpoint,
            credential=cred,
            credential_scopes=["https://cognitiveservices.azure.com/.default"],
        )
        self.deepseek_deployment = SETTINGS.deepseek_deployment

        logger.info("Verifying DeepSeek deployment '%s'...", self.deepseek_deployment)
        try:
            self.deepseek_client.complete(
                model=self.deepseek_deployment,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
            )
            logger.info("✅ DeepSeek client verification successful.")
        except Exception as e:
            logger.critical("❌ DeepSeek client test FAILED. Check ENDPOINT/MODEL_NAME. Error: %s", e)
            raise
