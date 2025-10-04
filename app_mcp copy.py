import os
import json
import time
import re
import base64
from pathlib import Path
from dotenv import load_dotenv
import requests

import uuid
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple, Callable
import logging

# Document Processing
import fitz  # PyMuPDF
import pymupdf4llm
import docx

# Azure Cosmos DB
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# Azure OpenAI Client (used for Vision and Chat)
from openai import AzureOpenAI

import streamlit as st


# Azure Blob & Identity
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, AzureCliCredential, get_bearer_token_provider

# Azure AI Inference (Client not strictly needed here but kept for completeness)
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

# Audio Processing Imports
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment

load_dotenv()

# Set up logging for debugging tool issues
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# ====================================================================
# === CORE TOOL IMPLEMENTATION (MANUALLY DEFINED TO BYPASS MCP ISSUE)
# ====================================================================

# --- Math Tools ---
def calc(expression: str) -> str:
    """Safely evaluates a mathematical expression and returns the result as a string."""
    allowed = "0123456789+-*/().% "
    if not all(c in allowed for c in expression.replace("**","").replace("//","")):
        return "error: disallowed character"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"

def lbs_to_kg(pounds: float) -> float:
    """Converts pounds to kilograms."""
    return pounds * 0.45359237

def kg_to_lbs(kg: float) -> float:
    """Converts kilograms to pounds."""
    return kg / 0.45359237

# --- JSON Tools ---
def json_parse(text: str) -> dict[str, Any]:
    """Parses a JSON string into a Python dictionary."""
    try:
        return {"ok": True, "data": json.loads(text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Note: json_validate logic is complex and will be skipped for a simple demo
# to keep the file size manageable, but the function signature is retained.
def json_validate(data_json: str, schema_json: str) -> dict[str, Any]:
    """Validates a JSON object against a basic schema."""
    # Simplified validation for demo purposes.
    try:
        json.loads(data_json)
        json.loads(schema_json)
        return {"ok": True, "errors": []}
    except Exception as e:
        return {"ok": False, "error": f"json parse error: {e}"}

# --- Table Tools ---
def to_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Converts a list of dictionaries (rows) into a markdown table string."""
    if not rows: return ""
    headers = list(rows[0].keys())
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"

# --- Schema Tools ---
def schema_sketch(json_list_text: str) -> dict[str, Any]:
    """Given JSON list of objects, infers a field->types map."""
    try:
        rows = json.loads(json_list_text)
        if not isinstance(rows, list): return {"error":"expect list"}
    except Exception as e:
        return {"error": f"parse: {e}"}

    sig: Dict[str, set] = {}
    for r in rows:
        if not isinstance(r, dict): continue
        for k, v in r.items():
            t = type(v).__name__
            sig.setdefault(k, set()).add(t)
    return {k: sorted(list(v)) for k, v in sig.items()}

def schema_diff(a_text: str, b_text: str) -> dict[str, Any]:
    """Diff two field->types maps produced by schema_sketch."""
    try:
        a = json.loads(a_text); b = json.loads(b_text)
    except Exception as e:
        return {"error": f"parse: {e}"}
    added = [k for k in b.keys() if k not in a]
    removed = [k for k in a.keys() if k not in b]
    changed = [k for k in a.keys() if k in b and a[k] != b[k]]
    return {"added": added, "removed": removed, "changed": changed}

# Map of tool names to their corresponding functions
TOOL_FUNCTIONS: Dict[str, Callable] = {
    "calc": calc,
    "lbs_to_kg": lbs_to_kg,
    "kg_to_lbs": kg_to_lbs,
    "json_parse": json_parse,
    "json_validate": json_validate,
    "to_markdown_table": to_markdown_table,
    "schema_sketch": schema_sketch,
    "schema_diff": schema_diff,
}
# ====================================================================

# --------------------------- Page Config ---------------------------
st.set_page_config(
    page_title="WBI AI",
    page_icon="🤖",
    layout="wide"
)


# =========================== UTILITIES ===========================
def b64_file(path: str | Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def inject_brand_and_theme(circle_b64: str | None, word_b64: str | None):
    """Respect Chrome's light/dark preference and inject logos visibly."""
    st.markdown(
        """
        <style>
        /* ---------- LIGHT THEME ---------- */
        @media (prefers-color-scheme: light) {
            html, body, .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"],
            .block-container {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            section[data-testid="stSidebar"] {
                background: #f7f7f9 !important;
                color: #000 !important;
            }
            /* invert only the wordmark in light mode */
            .brand-word img { filter: invert(1) !important; }
            .brand-circle img { filter: none !important; }
        }

        /* ---------- DARK THEME ---------- */
        @media (prefers-color-scheme: dark) {
            html, body, .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"],
            .block-container {
                background-color: #060a18 !important;
                color: #f3f6ff !important;
            }
            section[data-testid="stSidebar"] {
                background: #0a122a !important;
                color: #f3f6ff !important;
            }
            /* native logos in dark mode */
            .brand-word img { filter: none !important; }
            .brand-circle img { filter: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render both logos (if present) at top of the sidebar
    parts = ["<div style='text-align:center;margin:10px 0;'>"]
    if circle_b64:
        parts.append(
            f"<div class='brand-circle'><img src='data:image/png;base64,{circle_b64}' "
            f"style='max-width:80px;display:block;margin:0 auto 8px auto;' alt='Logo mark'/></div>"
        )
    if word_b64:
        parts.append(
            f"<div class='brand-word'><img src='data:image/png;base64,{word_b64}' "
            f"style='max-width:180px;display:block;margin:0 auto;' alt='Logo wordmark'/></div>"
        )
    parts.append("</div>")
    st.sidebar.markdown("".join(parts), unsafe_allow_html=True)

    # Minimal diagnostics so you can see why nothing shows
    if not circle_b64 or not word_b64:
        with st.sidebar.expander("Logo diagnostics", expanded=False):
            st.write({
                "circle_b64_present": bool(circle_b64),
                "word_b64_present": bool(word_b64),
            })



def guess_emoji(name: str) -> str:
    n = name.lower()
    mapping = {
        "pirate": "🏴‍☠️",
        "finance": "💸",
        "boe": "📊",
        "manager": "🧑‍💼",
        "project": "📁",
        "tech": "🧪",
        "engineer": "🛠️",
        "specialist": "🎯",
        "acquisition": "🛒",
        "transfer": "🔁",
        "commercialization": "💼",
        "case": "📚",
        "research": "🔬",
        "ai": "🤖",
    }
    for k, v in mapping.items():
        if k in n:
            return v
    return "🧠"


# =========================== AUTH HELPERS ===========================
def get_user_claims_from_headers(headers: dict) -> dict | None:
    encoded = headers.get("x-ms-client-principal")
    if not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
        principal = json.loads(decoded)
        claims = {c["typ"]: c["val"] for c in principal.get("claims", [])}
        claims["auth_typ"] = principal.get("auth_typ")
        claims["name_typ"] = principal.get("name_typ")
        claims["role_typ"] = principal.get("role_typ")
        return claims
    except Exception as e:
        st.sidebar.error(f"Error parsing principal header: {e}")
        return None


def _fallback_dot_auth_me():
    path = "/.auth/me"
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            upn = data[0].get("user_id")
            return {"oid": upn, "email": upn, "name": data[0].get("user_details", upn)}
    except Exception:
        pass
    return None


def get_current_user():
    fallback = {"id": "local_dev", "email": "local_dev_user@example.com", "name": "Local Dev"}
    try:
        headers = getattr(st.context, "headers", {})
        headers = {k.lower(): v for k, v in headers.items()} if headers else {}
    except Exception:
        headers = {}

    claims = get_user_claims_from_headers(headers)
    if claims:
        oid = (
            claims.get("http://schemas.microsoft.com/identity/claims/objectidentifier")
            or headers.get("x-ms-client-principal-id")
            or claims.get("sub")
        )
        email = (
            claims.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress")
            or headers.get("x-ms-client-principal-name")
        )
        name = (
            claims.get("name")
            or claims.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name")
            or email
        )
        if oid:
            return {"id": oid, "email": email or "", "name": name or email or oid}

    me = _fallback_dot_auth_me()
    if me:
        return {"id": me["oid"], "email": me["email"], "name": me["name"]}
    return fallback


# =========================== AZURE / STATE ===========================
# --- Load Credentials ---
if "credentials_loaded" not in st.session_state:
    # GPT-4.1 for Vision & Query Generation
    st.session_state.GPT41_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    st.session_state.GPT41_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
    st.session_state.GPT41_DEPLOYMENT = os.getenv("GPT41_DEPLOYMENT")

    # O3 for Chat Synthesis
    st.session_state.O3_DEPLOYMENT = os.getenv("O3_DEPLOYMENT_NAME")

    # Speech-to-Text Credentials
    st.session_state.SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
    st.session_state.SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

    st.session_state.credentials_loaded = True

# --- Check for Missing Env Vars ---
missing = []
if not st.session_state.GPT41_ENDPOINT: missing.append("AZURE_AI_SEARCH_ENDPOINT")
if not st.session_state.GPT41_API_KEY: missing.append("AZURE_AI_SEARCH_API_KEY")
if not st.session_state.GPT41_DEPLOYMENT: missing.append("GPT41_DEPLOYMENT")
if not st.session_state.O3_DEPLOYMENT: missing.append("O3_DEPLOYMENT_NAME")
if not st.session_state.SPEECH_KEY: missing.append("AZURE_SPEECH_KEY")
if not st.session_state.SPEECH_REGION: missing.append("AZURE_SPEECH_REGION")


if missing:
    st.error(f"🚨 Missing required environment variables: **{', '.join(missing)}**.")
    st.stop()

# --- Startup Verification Block ---
with st.expander("Startup Credential Verification", expanded=False):
    st.write("Verifying connections to Azure AI services...")
    all_verified = True

    # 1. Verify GPT-4.1 (for Query Generation & Vision)
    st.markdown("--- \n**Checking GPT-4.1 Connection...**")
    try:
        st.info(f"Initializing client for GPT-4.1 at: {st.session_state.GPT41_ENDPOINT}")
        verify_gpt_client = AzureOpenAI(
            azure_endpoint=st.session_state.GPT41_ENDPOINT,
            api_key=st.session_state.GPT41_API_KEY,
            api_version="2024-05-01-preview"
        )
        st.info(f"Verifying deployment '{st.session_state.GPT41_DEPLOYMENT}'...")
        response = verify_gpt_client.chat.completions.create(
            model=st.session_state.GPT41_DEPLOYMENT,
            messages=[{"role": "user", "content": "Test connection"}]
        )
        st.success(f"✅ GPT-4.1 connection successful. Test response: '{response.choices[0].message.content.strip()}'")
    except Exception as e:
        st.error(f"❌ GPT-4.1 connection FAILED. Please check your `AZURE_AI_SEARCH_ENDPOINT`, `AZURE_AI_SEARCH_API_KEY`, and `GPT41_DEPLOYMENT` values. Error: {e}")
        all_verified = False

    # 2. Verify O3 (as the primary synthesis model)
    st.markdown("--- \n**Checking Synthesis Model (O3) Connection...**")
    try:
        st.info(f"Initializing O3 client for endpoint: {st.session_state.GPT41_ENDPOINT}")
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
        verify_o3_client = AzureOpenAI(
            azure_endpoint=st.session_state.GPT41_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview"
        )
        st.info(f"Verifying deployment '{st.session_state.O3_DEPLOYMENT}'...")
        response = verify_o3_client.chat.completions.create(
            model=st.session_state.O3_DEPLOYMENT,
            messages=[{"role": "user", "content": "Test connection"}],
        )
        st.success(f"✅ O3 synthesis model connection successful. Test response: '{response.choices[0].message.content.strip()}'")
    except Exception as e:
        st.error(f"❌ O3 synthesis model connection FAILED. Please check your `O3_DEPLOYMENT_NAME` value and ensure Entra ID permissions are set. Error: {e}")
        all_verified = False

    # 3. Verify Speech Service
    st.markdown("--- \n**Checking Speech Service Connection...**")
    try:
        st.info(f"Initializing Speech client for region: {st.session_state.SPEECH_REGION}")
        speech_config = speechsdk.SpeechConfig(
            subscription=st.session_state.SPEECH_KEY,
            region=st.session_state.SPEECH_REGION
        )
        st.success("✅ Azure Speech Service configured successfully.")
    except Exception as e:
        st.error(f"❌ Azure Speech Service configuration FAILED. Please check your `SPEECH_KEY` and `SPEECH_REGION` values. Error: {e}")
        all_verified = False

    # 4. Final Check
    if not all_verified:
        st.error("One or more primary AI service connections failed. The application cannot continue.")
        st.stop()
    else:
        st.success("All primary AI services connected successfully.")


STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
if not STORAGE_ACCOUNT_URL or not CONTAINER_NAME:
    st.error("🚨 STORAGE_ACCOUNT_URL and CONTAINER_NAME must be set.")
    st.stop()


@st.cache_resource
def get_blob_service_client():
    current_user_id = get_current_user().get("id")
    if current_user_id == "local_dev":
        credential = AzureCliCredential()
    else:
        credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)


def save_user_data(user_id, data):
    if not user_id:
        return
    try:
        blob = get_blob_service_client().get_blob_client(CONTAINER_NAME, f"{user_id}.json")
        blob.upload_blob(json.dumps(data, indent=2), overwrite=True)
    except Exception as e:
        st.error(f"Failed to save user data: {e}")


def load_user_data(user_id):
    default_personas = {
        "General Assistant": {
            "prompt": (
                "You are a helpful assistant. "
                "Do your best to reflect on user questions, research best approaches to answer the question or gather needed information, "
                "and present the results in an appropriate manner."
            ),
            "type": "simple",
            "params": {"temperature": 0.7}
        },
        "Multi-Agent Team": {
            "prompt": "You are a team of specialist AI agents collaborating to solve the user's request.",
            "type": "agentic", # This new type will trigger the agentic workflow
            "params": {"temperature": 0.5}
        },
        "AFRL Tech Manager": {
            "prompt": (
                "You are an expert Technology Manager at the Air Force Research Laboratory (AFRL). "
                "Your focus is on identifying, developing, and transitioning cutting-edge science and technology (S&T). "
                "You have expertise in Technology Readiness Levels (TRLs), innovation, and aligning research with Air Force and Space Force needs. "
                "Provide insightful, forward-thinking analysis based on this persona."
            ),
            "type": "simple",
            "params": {"temperature": 0.7}
        },
        "AFLCMC Program Manager": {
            "prompt": (
                "You are a seasoned Program Manager at the Air Force Life Cycle Management Center (AFLCMC) at Wright-Patterson AFB. "
                "You excel in cradle-to-grave acquisition and sustainment of Air Force weapon systems, managing cost, schedule, performance, and risk mitigation. "
                "Provide practical, execution-focused advice based on this persona."
            ),
            "type": "simple",
            "params": {"temperature": 0.7}
        },
        "Finance & BOE Specialist": {
            "prompt": (
                "You are a specialist in finance and Basis of Estimate (BOE) development for government proposals, specifically for the Department of the Air Force. "
                "Rely heavily on provided Case History for your estimates, labor category mapping, and rationale. "
                "If the case history is empty, clearly state you need data to provide an accurate estimate. "
                "Always thoroughly break down your reasoning."
            ),
            "type": "rag",
            "case_history": "",
            "params": {"temperature": 0.5}
        },
        "Defense Innovation Facilitator": {
            "prompt": (
                "You are an innovation facilitator at the Wright Brothers Institute (WBI). "
                "Your mission is to accelerate technology transfer, transition, and commercialization between AFRL, industry partners, and academia. "
                "Provide clear, strategic guidance for fostering collaboration and solving complex challenges."
            ),
            "type": "simple",
            "params": {"temperature": 0.8}
        },
        "Small Business Acquisition Advisor": {
            "prompt": (
                "You are a specialist in helping small businesses understand and successfully navigate the U.S. Air Force acquisition system. "
                "Provide practical, clear explanations and actionable advice tailored for small businesses aiming to collaborate with defense entities."
            ),
            "type": "simple",
            "params": {"temperature": 0.6}
        },
        "Tech Transfer & Commercialization Expert": {
            "prompt": (
                "You specialize in technology transfer, commercialization, and dual-use innovations at the Wright Brothers Institute. "
                "Offer commercialization strategies, partnership facilitation, and insights on bridging commercial innovations with military applications."
            ),
            "type": "rag",
            "case_history": "",
            "params": {"temperature": 0.7}
        },
        "Collaboration Accelerator Lead": {
            "prompt": (
                "You lead cross-disciplinary innovation programs (e.g., TECH-ARTS, Collaboration Accelerator). "
                "Integrate artistic creativity, technical expertise, and entrepreneurial mindsets to foster novel problem-solving."
            ),
            "type": "simple",
            "params": {"temperature": 0.85}
        },
        "Regional Economic Development Strategist": {
            "prompt": (
                "You are a strategist focused on enhancing the Dayton region's economic development through aerospace and defense innovation. "
                "Provide strategic insights on fostering economic resilience and technology-driven job creation."
            ),
            "type": "simple",
            "params": {"temperature": 0.75}
        },
        "DoD Workforce Development Advisor": {
            "prompt": (
                "You specialize in workforce development for the DoD, emphasizing STEM talent cultivation and acquisition workforce enhancement. "
                "Provide strategic advice on workforce planning, talent acquisition, and training programs."
            ),
            "type": "simple",
            "params": {"temperature": 0.7}
        },
        "Inter-Service Collaboration Manager": {
            "prompt": (
                "You manage inter-service collaboration initiatives between Air Force, Navy, and other DoD branches. "
                "Focus on identifying collaborative opportunities and overcoming organizational barriers."
            ),
            "type": "simple",
            "params": {"temperature": 0.8}
        },
        "Crazy Pirate": {
            "prompt": (
                "You are a crazy pirate who has completely lost his mind. Somehow this has caused you to make everything very funny."
                "Finish with a one liner joke."
            ),
            "type": "simple",
            "params": {"temperature": 0.9}
        },
        "Boring Agent": {
            "prompt": (
                "You are a very boring agent. Your responses should be dry , short, and apathetic. "
            ),
            "type": "simple",
            "params": {"temperature": 0.8}
        }
    }

    default_data = {"conversations": {}, "active_conversation_id": None, "personas": default_personas}

    if not user_id:
        return default_data

    try:
        blob = get_blob_service_client().get_blob_client(CONTAINER_NAME, f"{user_id}.json")
        data = json.loads(blob.download_blob(encoding="UTF-8").readall())
        if "personas" not in data:
            data["personas"] = default_personas
        else:
            for k, v in default_personas.items():
                if k not in data["personas"]:
                    data["personas"][k] = v
                if "params" not in data["personas"][k]:
                    data["personas"][k]["params"] = v.get("params", {"temperature": 0.7})
            data["personas"] = {**default_personas, **data["personas"]} # Merge default with loaded, giving loaded priority
        return data
    except ResourceNotFoundError:
        return default_data
    except Exception as e:
        st.error(f"Failed to load/parse user data: {e}")
        return default_data


def create_new_chat(user_id, user_data, persona_name):
    chat_id = f"chat_{int(time.time())}"
    persona = user_data["personas"][persona_name]
    sys_prompt = persona["prompt"]
    if persona.get("type") == "rag" and persona.get("case_history"):
        sys_prompt += f"\n\n--- CASE HISTORY ---\n{persona['case_history']}"
    user_data["conversations"][chat_id] = [
        {"role": "system", "content": sys_prompt, "persona_name": persona_name}
    ]
    user_data["active_conversation_id"] = chat_id
    save_user_data(user_id, user_data)
    return user_data


# =========================== COSMOS DB UPLOADER ===========================
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "DefianceDB")

@st.cache_data(ttl=600)
def get_available_containers():
    """
    Retrieves a list of all containers from all databases, formatted as 'db/container'.
    Excludes internal or system databases.
    """
    all_container_paths = []
    try:
        client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        excluded_dbs = {'_self', '_rid', '_ts', '_etag'}
        database_proxies = client.list_databases()
        db_ids = [db['id'] for db in database_proxies if db['id'] not in excluded_dbs]
        for db_id in db_ids:
            database = client.get_database_client(db_id)
            containers_properties = database.list_containers()
            for container_props in containers_properties:
                all_container_paths.append(f"{db_id}/{container_props['id']}")
        return sorted(all_container_paths)
    except Exception as e:
        st.sidebar.error(f"Failed to list all containers: {e}")
        return ["DefianceDB/Documents"]

def create_container_if_not_exists(db_name: str, container_name: str, partition_key: str = "/id"):
    """Creates a new container in Cosmos DB if it doesn't already exist."""
    try:
        client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database = client.get_database_client(db_name)
        database.create_container_if_not_exists(id=container_name, partition_key=PartitionKey(path=partition_key))
        st.toast(f"Container '{container_name}' is ready.")
        get_available_containers.clear()
        return True
    except exceptions.CosmosHttpResponseError as e:
        st.sidebar.error(f"Failed to create container: {e}")
        return False

def save_verified_fact(question: str, answer: str):
    """Saves a question-answer pair to the VerifiedFacts container in DefianceDB."""
    try:
        fact_uploader = get_cosmos_uploader("DefianceDB", "VerifiedFacts")
        if fact_uploader:
            fact_document = {
                "id": f"fact_{uuid.uuid4()}",
                "question": question,
                "answer": answer,
                "verified_by": get_current_user().get("email"),
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            fact_uploader.upload_chunks([fact_document])
    except Exception as e:
        st.error(f"Failed to save verified fact: {e}")

class CosmosUploader:
    """Handles connection and document upsert operations for a specific Cosmos DB container."""
    def __init__(self, database_name: str, container_name: str):
        if not all([COSMOS_ENDPOINT, COSMOS_KEY]):
            raise ValueError("Cosmos DB credentials not found in environment variables.")
        self.client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)
        if container_name == "VerifiedFacts":
            self.partition_key_field = "/question"
        elif container_name == "ProjectSummaries":
            self.partition_key_field = "/projectName"
        else:
            self.partition_key_field = "/id"

    def upload_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[int, int]:
        success_count, failure_count = 0, 0
        for chunk in chunks:
            try:
                self.container.upsert_item(body=chunk)
                success_count += 1
            except exceptions.CosmosHttpResponseError as e:
                st.error(f"Failed to upsert chunk id '{chunk.get('id')}': {e.reason}")
                failure_count += 1
            except Exception as e:
                st.error(f"Unexpected error during upsert: {e}")
                failure_count += 1
        return success_count, failure_count

    def execute_query(self, query_string: str) -> List[Dict[str, Any]]:
        """Executes a raw SQL query string against the container."""
        try:
            items = list(self.container.query_items(query=query_string, enable_cross_partition_query=True))
            for item in items:
                item['_source_container'] = f"{self.database.id}/{self.container.id}"
            return items
        except exceptions.CosmosHttpResponseError as e:
            st.warning(f"Cosmos DB query failed for {self.container.id}: {e.reason}")
            return [{"error": str(e)}]
        except Exception as e:
            st.error(f"General query error for {self.container.id}: {e}")
            return [{"error": str(e)}]

@st.cache_resource
def get_cosmos_uploader(database_name: str, container_name: str):
    if not database_name or not container_name:
        st.error("🚨 Invalid database or container name.")
        return None
    try:
        return CosmosUploader(database_name, container_name)
    except ValueError as e:
        st.error(f"🚨 {e}")
        return None

# =========================== FILE PROCESSING & RAG ===========================
def process_text_file(file_bytes: bytes, filename: str) -> List[str]:
    """Processes a plain text (.txt, .md) file by decoding and chunking it."""
    try:
        # Decode the bytes into a string, assuming UTF-8 encoding.
        full_text = file_bytes.decode('utf-8')
        st.info(f"Read {len(full_text):,} characters from '{filename}'. Chunking text...")
        # Use the existing helper function to split the transcript into manageable chunks
        return chunk_text(full_text)
    except UnicodeDecodeError:
        st.error(f"Failed to decode '{filename}'. The file may not be UTF-8 encoded.")
        return []
    except Exception as e:
        st.error(f"An error occurred while processing the text file '{filename}': {e}")
        return []

def call_vision_model(base64_image: str, prompt_text: str) -> str:
    """Calls the GPT-4.1 vision model and returns the text content."""
    try:
        client = AzureOpenAI(
            azure_endpoint=st.session_state.GPT41_ENDPOINT,
            api_key=st.session_state.GPT41_API_KEY,
            api_version="2024-05-01-preview"
        )

        response = client.chat.completions.create(
            model=st.session_state.GPT41_DEPLOYMENT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Azure Vision API Error: {e}")
        return f"[VISION_PROCESSING_ERROR: {e}]"

def extract_structured_data(full_text: str, filename: str) -> dict:
    """
    Uses a powerful AI model to read the full text of a document and extract
    key project details into a structured JSON format.
    """
    st.info("Extracting structured data from document...")

    # Define the data structure we want the AI to fill
    extraction_schema = {
        "projectName": "string",
        "doc_type": "ProjectSummary",
        "sourceDocument": "string",
        "summary": "string",
        "timeline": {
            "value": "string (e.g., '6 months', 'Q4 2025')",
            "startDate": "string (ISO 8601 format, e.g., '2025-08-01')"
        },
        "budget": {
            "amount": "number",
            "currency": "string (e.g., 'USD')"
        },
        "risks": ["list of strings"],
        "optionalExtensions": ["list of objects with 'name' and 'details' properties"]
    }

    # Create a powerful prompt that instructs the AI to act as a data extractor
    system_prompt = f"""You are an expert data extraction AI. Your task is to read the provided document text and extract key project details into a structured JSON format.
    Analyze the text and populate all fields of the following JSON schema.
    - For dates, infer the full date if possible (e.g., "August 2025" becomes "2025-08-01").
    - If a specific piece of information is not present in the text, use `null` for the value.
    - The `sourceDocument` should be the filename provided.

    JSON Schema to populate:
    {json.dumps(extraction_schema, indent=2)}

    You must respond with only the populated JSON object.
    """

    try:
        extraction_client = AzureOpenAI(
            azure_endpoint=st.session_state.GPT41_ENDPOINT,
            api_key=st.session_state.GPT41_API_KEY,
            api_version="2024-05-01-preview"
        )

        context = f"FILENAME: '{filename}'\n\nDOCUMENT TEXT:\n---\n{full_text}"

        response = extraction_client.chat.completions.create(
            model=st.session_state.GPT41_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        extracted_data = json.loads(response.choices[0].message.content)
        # Add a unique ID to the extracted data
        extracted_data['id'] = f"proj_{uuid.uuid4()}"

        st.success("Successfully extracted structured data.")
        return extracted_data

    except Exception as e:
        st.error(f"Failed to extract structured data: {e}")
        return None

def process_pdf_with_vision(file_bytes: bytes, filename: str) -> List[str]:
    """Processes a PDF using pymupdf4llm for text and GPT-4.1 Vision for refinement."""
    page_contents = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        progress_bar = st.progress(0, text=f"Analyzing {filename}...")

        for i, page in enumerate(doc):
            progress_text = f"Processing page {i + 1}/{len(doc)} of {filename}..."
            progress_bar.progress((i + 1) / len(doc), text=progress_text)

            md_text = pymupdf4llm.to_markdown(doc, pages=[i], write_images=False)

            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            vision_prompt = f"""You are an expert document analysis AI. Review the initial text extracted from a document page and the corresponding page image. Your task is to produce a final, corrected version of the page's content in markdown format.

- Ensure all text from the image is accurately transcribed.
- Correct any OCR errors or formatting issues from the initial extraction.
- Preserve the original structure, including paragraphs, lists, and tables.
- Do NOT add any summary or commentary. Return ONLY the corrected full text from the page.

--- INITIAL EXTRACTED TEXT ---
{md_text}
--- END INITIAL TEXT ---

Now, provide the final, corrected markdown based on the attached image."""

            final_content = call_vision_model(base64_image, vision_prompt)
            page_contents.append(final_content)

        progress_bar.empty()
        doc.close()
        return page_contents
    except Exception as e:
        st.error(f"Failed to process PDF '{filename}': {e}")
        return []


def process_docx(file_bytes: bytes, paragraphs_per_chunk: int = 15) -> List[str]:
    """Processes a DOCX file using python-docx, chunking by paragraph count."""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        chunks = []
        current_chunk_text = []
        para_count = 0

        for para in doc.paragraphs:
            if para.text.strip():
                current_chunk_text.append(para.text.strip())
                para_count += 1
            if para_count >= paragraphs_per_chunk:
                chunks.append("\n\n".join(current_chunk_text))
                current_chunk_text = []
                para_count = 0

        if current_chunk_text:
            chunks.append("\n\n".join(current_chunk_text))

        return chunks
    except Exception as e:
        st.error(f"Failed to process DOCX file: {e}")
        return []

def process_uploaded_file(uploaded_file):
    """Dispatcher to process uploaded files based on their type."""
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    # Use splitext to reliably get the file extension
    file_ext = os.path.splitext(filename.lower())[1]

    if file_ext == ".pdf":
        return process_pdf_with_vision(file_bytes, filename)
    elif file_ext == ".docx":
        return process_docx(file_bytes)
    elif file_ext == ".m4a":
        return process_m4a(file_bytes, filename)
    elif file_ext in [".txt", ".md"]:
        return process_text_file(file_bytes, filename)
    else:
        st.warning(f"Unsupported file type: {filename}. Please upload a .pdf, .docx, .m4a, .txt, or .md file.")
        return []


def prepare_chunks_for_cosmos(chunks: List[str], original_filename: str) -> List[Dict[str, Any]]:
    """Formats text chunks into JSON for Cosmos DB ingestion."""
    parent_doc_id = f"doc_{uuid.uuid4()}"
    output_chunks = []

    for i, chunk_content in enumerate(chunks):
        output_chunks.append({
            "id": f"{parent_doc_id}_chunk_{i}",
            "parent_document_id": parent_doc_id,
            "content": chunk_content,
            "metadata": {
                "original_filename": original_filename,
                "chunk_index": i,
            },
            "doc_type": "chunk"
        })
    return output_chunks

def chunk_text(full_text: str, sentences_per_chunk: int = 10) -> List[str]:
    """Splits a long text into smaller chunks of a specified number of sentences."""
    # Split the text into sentences using regex that looks for sentence-ending punctuation.
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    chunks = []

    # Group sentences into chunks
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk]).strip()
        if chunk:  # Ensure the chunk is not empty
            chunks.append(chunk)

    return chunks

# --- NEW: M4A Audio Processing Function ---
def process_m4a(file_bytes: bytes, filename: str) -> List[str]:
    """Transcribes an M4A audio file using the Azure Speech fast transcription REST API."""
    try:
        # 1. Convert M4A to a compatible WAV format in-memory. This remains a best practice.
        st.info(f"Converting '{filename}' to a compatible audio format...")
        audio = AudioSegment.from_file(BytesIO(file_bytes), format="m4a")
        wav_io = BytesIO()
        audio.export(wav_io, format="wav", codec="pcm_s16le", parameters=["-ac", "1", "-ar", "16000"])
        wav_io.seek(0) # Reset buffer to the beginning before reading
        st.success("Audio format conversion complete.")

        # 2. Prepare the REST API request
        speech_key = st.session_state.SPEECH_KEY
        speech_region = st.session_state.SPEECH_REGION
        if not all([speech_key, speech_region]):
            st.error("Azure Speech Service credentials are not configured.")
            return []

        endpoint = f"https://{speech_region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"

        headers = {
            'Ocp-Apim-Subscription-Key': speech_key,
        }

        # 3. Construct the multipart/form-data payload
        # The 'definition' part contains the transcription configuration as a JSON string.
        definition = {"locales": ["en-US"]}

        # The 'requests' library will automatically handle the multipart/form-data encoding.
        files = {
            'audio': (filename, wav_io, 'audio/wav'),
            'definition': (None, json.dumps(definition), 'application/json')
        }

        st.info(f"Transcribing '{filename}' via fast transcription API... this may take a moment.")

        # 4. Make the synchronous POST request
        response = requests.post(endpoint, headers=headers, files=files)
        response.raise_for_status()  # This will raise an exception for HTTP error codes (4xx or 5xx)

        # 5. Extract the transcript from the JSON response
        response_json = response.json()

        # The full transcript is in the 'text' field of the first item in 'combinedPhrases'
        combined_phrases = response_json.get('combinedPhrases', [])
        if not combined_phrases:
            st.warning("No speech could be recognized. The 'combinedPhrases' field was empty.")
            return []

        full_transcript = combined_phrases[0].get('text', '')

        if full_transcript:
            st.success("Transcription successful.")
            st.info("Chunking transcript for knowledge base...")
            # Use the existing helper function to split the transcript into manageable chunks
            return chunk_text(full_transcript)
        else:
            st.warning("No speech could be recognized from the audio.")
            return []

    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred during transcription: {http_err}")
        st.error(f"Response body: {response.text}")
        return []
    except Exception as e:
        st.error(f"An error occurred while processing the M4A file: {e}")
        st.error("This may be due to a missing `ffmpeg` installation or an issue with the audio file itself.")
        return []


def transcribe_m4a_audio(file_bytes: bytes) -> str:
    """
    Transcribes an in-memory M4A file using the Azure AI Speech batch transcription REST API.
    This is an asynchronous process involving polling.
    """
    import requests

    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    if not all([speech_key, speech_region]):
        st.error("Azure Speech Service credentials not found in environment variables.")
        return ""

    # 1. --- Start Transcription Job ---
    endpoint = f"https://{speech_region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"
    headers = {
        'Ocp-Apim-Subscription-Key': speech_key,
        'Content-Type': 'application/json'
    }
    # Using a SAS URL is recommended for production, but for this self-contained app,
    # we will use the batch API's ability to take direct audio content, which is simpler.
    # Note: The direct content upload method is not explicitly shown in the basic REST docs
    # but is a feature of the underlying service. Let's re-implement with the SDK's
    # long-running recognition method which is the Python equivalent of the batch REST API.

    import azure.cognitiveservices.speech as speechsdk
    st.info("Starting long-running audio transcription... This may take a few minutes for large files.")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "en-US"
    speech_config.request_word_level_timestamps() # Optional: for more detailed data

    audio_stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Write the file bytes to the stream that the SDK will manage
    audio_stream.write(file_bytes)
    audio_stream.close()

    done = False
    full_transcript = []

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon session stopped event"""
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True

    def recognized_cb(evt):
        """callback that collects the recognized text"""
        full_transcript.append(evt.result.text)

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    progress_bar = st.progress(0, text="Transcription in progress...")
    while not done:
        # This loop provides UI feedback while the SDK works in the background
        progress_bar.progress(time.time() % 1.0) # Just an indicator that it's working
        time.sleep(1)

    progress_bar.empty()
    final_text = " ".join(full_transcript)

    if final_text:
        st.success("Transcription successful.")
        return final_text
    else:
        st.warning("No speech could be recognized from the audio.")
        return ""

# =========================== MULTI-AGENT FRAMEWORK ===========================
MAX_LOOPS = 20

TOOL_DEFINITIONS = f"""
AVAILABLE TOOLS:
- search_knowledge_base(keywords: list[str], semantic_query_text: str, rank_limit: int): Executes a **broad keyword search** against the knowledge base (RAG). **Keywords are combined with OR logic.** Use this to find documents or data points.
- calc(expression: str): Safely evaluates a mathematical expression (e.g., "12 * (3 + 5)").
- lbs_to_kg(pounds: float): Converts pounds to kilograms.
- kg_to_lbs(kg: float): Converts kilograms to pounds.
- currency_converter(amount: float, from_currency: str, to_currency: str): Converts an amount between USD, EUR, and CAD.
- json_parse(text: str): Parses a JSON string and returns a structured object or error.
- to_markdown_table(rows: list[dict]): Converts a Python list of dictionaries into a clean Markdown table.
- schema_sketch(json_list_text: str): Infers field names and data types from a JSON list of objects.

OUTPUT FORMAT:
The Agent MUST respond with ONLY a JSON object representing either a tool call, a structured plan, a draft, or a final response.
"""

# (Around line 1787)
AGENT_PERSONAS = {
        "Orchestrator": """You are an expert project manager and orchestrator. Your role is to understand the user's goal and guide your team of specialist agents to achieve it.

    Your team consists of:
    - Tool Agent: Specialized for all data retrieval, processing, conversion, and validation tasks using tools.
    - Engineer: Best for tasks involving code, technical analysis, or debugging.
    - Writer: Best for composing, formatting, and refining the final answer to the user.
    - Validator: Best for reviewing the work of other agents, checking for factual accuracy, and ensuring the plan is on track.

    CRITICAL INSTRUCTIONS:
    1.  Review the scratchpad, which contains the user's goal and the history of steps taken.
    2.  If the scratchpad is empty or only contains the user's goal, your first step is to delegate the task of creating a clear, step-by-step plan to the Writer.
    3.  On every subsequent turn, review the plan and the completed steps. DO NOT repeat tasks.
    4.  Decide the single next logical action to move the project forward.
    5.  Delegate the task to the most appropriate agent. Your instructions must be clear and specific.
    6.  You MUST respond in a JSON object with two keys: "agent" and "task".
    7.  If all steps in the plan are complete and the user's request is fully answered, delegate to the "FINISH" agent. **The final answer given in the "task" field MUST BE STRICTLY GROUNDED IN THE SCRATCHPAD CONTENT. DO NOT ADD SPECULATION OR UNGROUNDED KNOWLEDGE.**
    """,

        "Tool Agent": f"""You are an expert data agent responsible for executing tools.

    CRITICAL INSTRUCTIONS:
    1. Analyze the given task from another agent (e.g., "Calculate 2+2").
    2. Determine the single best tool to use to answer the task.
    3. You MUST respond with ONLY a JSON object that strictly follows this format: `{{"tool_use": {{"name": "tool_name", "params": {{"arg1": "value1", ...}}}}}}`

    EXAMPLE:
    If the task is "Convert 100 lbs to kg", your response MUST be:
    `{{"tool_use": {{"name": "lbs_to_kg", "params": {{"pounds": 100}}}}}}`

    {TOOL_DEFINITIONS}""", # Keep TOOL_DEFINITIONS for reference

        "Engineer": """You are a senior software engineer and technical analyst.
        
    CRITICAL INSTRUCTIONS:
    1.  Review your assigned task and the information in the scratchpad.
    2.  If you need external data, calculations, or conversion, delegate the task to the `Tool Agent` (e.g., {"agent": "Tool Agent", "task": "Convert 100 lbs to kg"}).
    3.  If you have enough information, perform your analysis and provide the complete technical answer. **The response MUST BE STRICTLY LIMITED TO THE FACTS/DATA IN THE SCRATCHPAD. DO NOT SPECULATE OR USE EXTERNAL KNOWLEDGE.** Respond ONLY with a JSON object containing your answer: {"response": "The analysis shows the memory leak is caused by..."}""",
        
        "Writer": """You are a professional technical writer.

    CRITICAL INSTRUCTIONS:
    1.  Review your assigned task and the information in the scratchpad.
    2.  If you need to format data, perform a calculation, or retrieve information, delegate the task to the `Tool Agent` (e.g., {"agent": "Tool Agent", "task": "Take this JSON data and format it into a markdown table"}).
    3.  If you have sufficient information, write the full report. **THE REPORT MUST BE STRICTLY LIMITED TO THE FACTS/DATA IN THE SCRATCHPAD. DO NOT ADD SPECULATION, DISCLAIMERS ABOUT MISSING INFORMATION, OR UNGROUNDED KNOWLEDGE (e.g., "SFIS might also mean..."). ONLY REPORT WHAT WAS FOUND.** Respond ONLY with a JSON object containing your final, well-formatted report: {"response": "# SFIS Project Report..."}""",

        "Validator": """You are a meticulous validator and quality assurance specialist.

    CRITICAL INSTRUCTIONS:
    1.  Review the plan, the scratchpad, and your assigned task (e.g., "Validate the calculated budget against original documents").
    2.  **Crucially, check the content of any drafts/responses for grounding.** If a draft includes speculation, information that was not in a tool observation/context note, or attempts to use general external knowledge, you MUST mark it for revision.
    3.  If you need to cross-reference specific data or perform a validation check, delegate the task to the `Tool Agent` (e.g., {"agent": "Tool Agent", "task": "Use the schema_diff tool on the provided Schema A and Schema B to check for changes"}).
    4.  If you have enough information, provide your validation and recommend the next step. Respond ONLY with a JSON object containing your assessment: {"response": "Validation complete. The data is accurate. The next step should be to have the Writer compile the final report."}""",
        
        "FINISH_NOW": """You are the final summarizer. The agent team has reached its operational limit and must now provide the best possible answer based on the work done so far.

    CRITICAL INSTRUCTIONS:
    1.  Review the user's original goal and the entire scratchpad history.
    2.  Synthesize all the information, tool outputs, and drafts into a coherent final answer.
    3.  **Acknowledge that the process was stopped before it could be fully completed.**
    4.  Present the best possible answer based on the gathered information. **DO NOT add speculation or information not present in the scratchpad.**
    5.  Your response should be a complete and well-formatted final answer to the user."""
    }

# Replacement for execute_tool_call function (around line 1184)

def execute_tool_call(tool_name: str, params: Dict[str, Any]) -> str:
    """Executes a tool function based on the requested name and parameters."""
    # This check ensures the tool is recognized, including our custom search tool.
    if tool_name not in TOOL_FUNCTIONS and tool_name != "search_knowledge_base":
        return f"ToolExecutionError: Tool '{tool_name}' is not registered."

    # Define single quote constant (ASCII 39) to safely handle SQL string formatting errors
    SINGLE_QUOTE = chr(39)

    if tool_name == "search_knowledge_base":

        # The agent provides: keywords, semantic_query_text, rank_limit
        keywords = params.get("keywords", [])
        rank_limit = params.get("rank_limit", 10)

        if not keywords:
            return "ToolExecutionError: search_knowledge_base requires 'keywords' (list[str])."

        # 1. Prepare terms for safe Full Text Search (FTS) using CONTAINS
        contains_clauses = []
        for k in keywords:
            # Escape single quotes (' -> '') for SQL safety
            safe_keyword = k.replace(SINGLE_QUOTE, SINGLE_QUOTE * 2)

            # We use CONTAINS(c.content, 'keyword', true) for case-insensitive FTS
            contains_clauses.append(f"CONTAINS(c.content, '{safe_keyword}', true)")

        # 2. CRITICAL FIX: Combine with OR logic for broader relevance
        where_clause = " OR ".join(contains_clauses)

        # 3. Final Query Assembly
        sql_query = (
            f"SELECT TOP {rank_limit} c.content, c.metadata "
            f"FROM c "
            f"WHERE {where_clause}"
        )

        return f"Tool Observation: Constructed Safe Query:\n{sql_query}"

    # --- Standard Tool Execution (calc, json_parse, etc.) ---
    tool_func = TOOL_FUNCTIONS[tool_name]
    import inspect
    sig = inspect.signature(tool_func)

    try:
        # Bind and execute arguments
        bound_args = sig.bind(**params)
        result = tool_func(*bound_args.args, **bound_args.kwargs)

        if isinstance(result, (dict, list)):
            return f"Tool Observation: {json.dumps(result, indent=2)}"
        return f"Tool Observation: {result}"

    except TypeError as e:
        return f"ToolExecutionError: Invalid parameters for '{tool_name}': {e}. Required signature: {sig}"
    except Exception as e:
        return f"ToolExecutionError: An unexpected error occurred during tool execution: {e}"
# Replacement for run_agentic_workflow function (around line 1242)
def run_agentic_workflow(user_prompt: str, client: AzureOpenAI, log_placeholder, final_answer_placeholder, query_expander):
    """
    Manages the multi-agent collaboration loop.
    """
    scratchpad = f"User's Goal: {user_prompt}\n\n"
    # NOTE: All log_placeholder.markdown calls must now use unsafe_allow_html=True
    log_placeholder.markdown(f"**Loop 0:** Starting with user's goal...\n`{user_prompt}`", unsafe_allow_html=True)

    for i in range(MAX_LOOPS):
        loop_num = i + 1
        st.toast(f"Agent Loop {loop_num}/{MAX_LOOPS}")

        # 1. Orchestrator: Plan and delegate the next step
        with st.spinner(f"Loop {loop_num}: Orchestrator is planning..."):
            orchestrator_messages = [
                {"role": "system", "content": AGENT_PERSONAS["Orchestrator"]},
                {"role": "user", "content": f"Here is the current scratchpad with the history of our work. Based on this, what is the single next action to take?\n\n<scratchpad>\n{scratchpad}\n</scratchpad>"}
            ]
            response = client.chat.completions.create(model=st.session_state.O3_DEPLOYMENT, messages=orchestrator_messages, response_format={"type": "json_object"})
            decision_json = json.loads(response.choices[0].message.content)

        next_agent = decision_json.get("agent")
        task = decision_json.get("task")

        # *** NEW: Check if this is the last loop ***
        if i == MAX_LOOPS - 1:
            next_agent = "FINISH_NOW"
            task = "The maximum number of loops has been reached. Please review the scratchpad and provide the best possible final answer based on the information gathered so far. Acknowledge that the process was stopped before natural completion."

        scratchpad += f"\n**Loop {loop_num}:**\n- **Thought:** The orchestrator decided the next step is to delegate to **{next_agent}** with the task: '{task}'.\n"
        log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

        if next_agent == "FINISH":
            final_answer_placeholder.markdown(task)
            scratchpad += "- **Action:** FINISHED."
            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
            return task
        
        # *** NEW: Handle the FINISH_NOW agent ***
        if next_agent == "FINISH_NOW":
            with st.spinner("Loop limit reached. Generating final summary..."):
                finish_now_messages = [
                    {"role": "system", "content": AGENT_PERSONAS["FINISH_NOW"]},
                    {"role": "user", "content": f"Here is the scratchpad with the history of our work. Your task is: {task}\n\n<scratchpad>\n{scratchpad}\n</scratchpad>"}
                ]
                response = client.chat.completions.create(model=st.session_state.O3_DEPLOYMENT, messages=finish_now_messages)
                final_answer = response.choices[0].message.content
                final_answer_placeholder.markdown(final_answer)
                scratchpad += f"- **Action:** Forced Finish. The agent team provided the best possible answer within the loop limit."
                log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                return final_answer


        # 2. Execute Task with Specialist Agent
        observation = ""
        agent_output = {}
        tool_call_to_execute = None
        
        try:
            with st.spinner(f"Loop {loop_num}: {next_agent} is working on the task..."):
                if next_agent not in AGENT_PERSONAS:
                    observation = f"Error: Unknown agent '{next_agent}'. Please choose from the available agents."
                else:
                    agent_messages = [
                        {"role": "system", "content": AGENT_PERSONAS[next_agent]},
                        {"role": "user", "content": f"Here is the history of our work:\n<scratchpad>\n{scratchpad}\n</scratchpad>\n\nYour current task is: {task}"}
                    ]
                    response = client.chat.completions.create(model=st.session_state.O3_DEPLOYMENT, messages=agent_messages, response_format={"type": "json_object"})
                    agent_output = json.loads(response.choices[0].message.content)

                    # --- Handle Delegation to another Agent ---
                    if agent_output.get("agent"):
                        sub_agent = agent_output["agent"]
                        sub_task = agent_output["task"]
                        scratchpad += f"- **Delegation:** {next_agent} is delegating to {sub_agent}: '{sub_task}'\n"
                        log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                        
                        if sub_agent == "Tool Agent":
                            # CRITICAL FIX: Treat Tool Agent delegation as a direct, executable request
                            query_agent_messages = [{"role": "system", "content": AGENT_PERSONAS["Tool Agent"]}, {"role": "user", "content": sub_task}]
                            response = client.chat.completions.create(model=st.session_state.O3_DEPLOYMENT, messages=query_agent_messages, response_format={"type": "json_object"})
                            agent_output = json.loads(response.choices[0].message.content)
                        else:
                             # Defer task execution to next Orchestrator loop
                             observation = f"The task was delegated to {sub_agent}, but the execution is deferred to the next Orchestrator loop. The task is: {sub_task}"
                             scratchpad += f"- **Action Result:** {observation}\n"
                             continue 

                    # --- Tool Output Format Detection ---
                    tool_name = None
                    params = {}

                    if "tool" in agent_output:
                        tool_name = agent_output["tool"]
                        if "kwargs" in agent_output:
                            params = agent_output["kwargs"]
                        elif "args" in agent_output:
                            params = agent_output["args"]
                    
                    elif "tool_use" in agent_output:
                        tool = agent_output["tool_use"]
                        tool_name = tool.get("name")
                        params = tool.get("params", {})

                    if tool_name:
                        # Execution is scheduled to happen here
                        tool_call_to_execute = (tool_name, params)
                    elif "response" in agent_output:
                        observation = agent_output["response"]
                    
                    # --- FAILURE DETECTION: Check if Tool Agent failed to produce a tool call ---
                    elif next_agent == "Tool Agent" and tool_name is None:
                         observation = f"Error: Tool Agent failed to produce a valid tool call. The response was: {json.dumps(agent_output)}. Returning to Orchestrator for correction."
                         next_agent = "Orchestrator" 
                         task = f"ERROR: The Tool Agent failed to execute its assigned task ({task}) because its output was not a valid tool call/response. Review the preceding scratchpad content and assign a corrective action to the appropriate agent to get back on track or FINISH."
                         scratchpad += f"- **Action Result:** {observation}\n"
                         continue

                    else:
                        observation = f"The agent returned an unexpected response: {json.dumps(agent_output)}"


        except Exception as e:
            observation = f"Error executing task for {next_agent}: {e}"
            tool_call_to_execute = None # Cancel execution if the agent call itself failed.
        
        # 3. System Execution Layer (simulated)
        if tool_call_to_execute:
            tool_name, params = tool_call_to_execute
            
            # --- Display green tool call (AESTHETIC) ---
            colored_tool_call = f"<span style='color:green;'>Executing tool `{tool_name}` with parameters: {params}</span>"
            scratchpad += f"- **Tool Call:** {colored_tool_call}\n"
            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
            # --- END FIX ---

            try:
                tool_result_observation = execute_tool_call(tool_name, params)
                
                # Handle search_knowledge_base execution
                if tool_name == "search_knowledge_base" and tool_result_observation.startswith("Tool Observation: Constructed Safe Query:"):
                    
                    sql_query = tool_result_observation.split("Tool Observation: Constructed Safe Query:\n", 1)[1]
                    search_results = []
                    selected_kbs = st.session_state.get('selected_containers', [])
                    
                    # Explicitly show the query in the expander
                    with query_expander: st.write("**Standard Search Query:**"); st.code(sql_query, language="sql")
                    
                    if not selected_kbs:
                        observation = "Observation: No knowledge bases selected to search."
                    else:
                        with st.spinner(f"Loop {loop_num}: Executing Standard Search against selected KBs..."):
                            for kb_path in selected_kbs:
                                db_name, cont_name = kb_path.split('/')
                                uploader = get_cosmos_uploader(db_name, cont_name)
                                if uploader:
                                    search_results.extend(uploader.execute_query(sql_query)) 
                    
                    observation = f"Found {len(search_results)} item(s) using Standard Search. Content: {json.dumps(search_results, indent=2)}"
                    tool_result_observation = f"Tool Observation (Standard Search): Query executed successfully. {observation}"

                    # ADDED: Explicitly show final search results in the expander
                    with query_expander: 
                        if search_results:
                            st.write("**Final Search Results:**")
                            st.json(search_results)
                        else:
                            st.write("**Final Search Results:** No items found.")

                # --- FIX: Display raw data if it looks like the search result payload before formatting ---
                if tool_name == "to_markdown_table" and params.get('rows') and isinstance(params['rows'], list):
                    with query_expander:
                        st.write("**Raw Search Data Found (Pre-Formatting):**")
                        st.json(params['rows'])
                # --- END FIX ---
                        
                observation = tool_result_observation 

            except Exception as e:
                observation = f"ToolExecutionError: Failed to run tool '{tool_name}' with params {params}. Error: {e}"
        
        # Re-inject the observation back into the scratchpad for the next loop
        
        # --- UPDATED: Smarter summarization for long observations ---
        if len(observation) > 20000:
            with st.spinner(f"Loop {loop_num}: Summarizing {next_agent}'s findings..."):
                text_to_summarize = observation
                
                if observation.strip().startswith("Found") and "Content:" in observation:
                    try:
                        json_str_start = observation.find('[')
                        json_str = observation[json_str_start:]
                        search_results = json.loads(json_str)
                        
                        content_to_summarize = "\n\n---\n\n".join(
                            [str(item.get("content", "")) for item in search_results if item.get("content")]
                        )
                        
                        if content_to_summarize:
                            text_to_summarize = content_to_summarize
                        else:
                            text_to_summarize = "The search returned results, but they contained no readable content."

                    except (json.JSONDecodeError, IndexError):
                        text_to_summarize = observation 

                summary_prompt = "You are an expert summarizer. Concisely summarize the following text in a few key bullet points for a project manager."
                summary_messages = [{"role": "system", "content": summary_prompt}, {"role": "user", "content": text_to_summarize}]
                
                response = client.chat.completions.create(
                    model=st.session_state.O3_DEPLOYMENT,  
                    messages=summary_messages,  
                )
                summary = response.choices[0].message.content
                
                if summary:
                    observation = f"The agent's action resulted in a long output. Here is a summary:\n{summary}"
                else:
                    observation = "The agent's action resulted in a long output, which could not be summarized."
        
        scratchpad += f"- **Action Result:** {observation}\n"
        log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

    final_answer = "The agent team could not complete the request within the allowed number of steps."
    final_answer_placeholder.markdown(final_answer)
    return final_answer

# =========================== INIT STATE & BRAND ===========================
user = get_current_user()
user_id = user["id"]

if "user_data" not in st.session_state or st.session_state.get("user_id") != user_id:
    st.session_state.user_id = user_id
    st.session_state.user_data = load_user_data(user_id)
    # keep whatever was stored; new chat will be forced below

# ---- force GA new chat on first page load this session ----
DEFAULT_PERSONA = "General Assistant"
if "bootstrapped" not in st.session_state:
    st.session_state.bootstrapped = True
    target = DEFAULT_PERSONA if DEFAULT_PERSONA in st.session_state.user_data["personas"] \
        else list(st.session_state.user_data["personas"].keys())[0]
    st.session_state.user_data = create_new_chat(user_id, st.session_state.user_data, target)
    st.session_state.last_persona_selected = target
else:
    # ensure we still have a last_persona_selected for later widgets
    if "last_persona_selected" not in st.session_state:
        st.session_state.last_persona_selected = DEFAULT_PERSONA if DEFAULT_PERSONA in st.session_state.user_data["personas"] \
            else list(st.session_state.user_data["personas"].keys())[0]

# helper flags
for k in ("persona_nonce", "editing_persona", "creating_persona"):
    if k not in st.session_state:
        st.session_state[k] = 0 if k == "persona_nonce" else False

# Logos
BASE_DIR = Path(__file__).resolve().parent


def first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


circle_path = first_existing([
    os.getenv("WBI_CIRCLE_LOGO", "").strip(),
    "/home/site/wwwroot/assets/wbi_circle.png",
    str(BASE_DIR / "assets" / "wbi_circle.png"),
])
word_path = first_existing([
    os.getenv("WBI_WORD_LOGO", "").strip(),
    "/home/site/wwwroot/assets/wbi_word.png",
    str(BASE_DIR / "assets" / "wbi_word.png"),
])

#DEBUGGING FOR THE LOGO
#st.sidebar.caption(f"CIRCLE: {circle_path or 'not found'}")
#st.sidebar.caption(f"WORD: {word_path or 'not found'}")

circle_b64 = b64_file(circle_path) if circle_path else None
word_b64 = b64_file(word_path) if word_path else None

inject_brand_and_theme(circle_b64, word_b64)


# =========================== SIDEBAR UI ===========================
def on_persona_change(widget_key):
    if st.session_state.editing_persona or st.session_state.creating_persona:
        return
    sel = st.session_state[widget_key]
    st.session_state.user_data = create_new_chat(
        st.session_state.user_id, st.session_state.user_data, sel
    )
    st.session_state.last_persona_selected = sel

with st.sidebar:
    st.markdown('<div class="mini-header">User Account</div>', unsafe_allow_html=True)
    MODEL_DISPLAY = "O3" # Changed from DeepSeekR1
    if user_id == "local_dev":
        st.warning("Running in local mode.")
    st.markdown(f"""
    <div class='user-card'>
        <div class='u-name'>{user.get('name', '')}</div>
        <div class='u-email'><a href='mailto:{user.get('email', '')}'>{user.get('email', '')}</a></div>
        <div class='u-model'>AI Model: {MODEL_DISPLAY}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="mini-header">Upload & Ingest</div>', unsafe_allow_html=True)
    all_container_paths = get_available_containers()
    upload_options = [path for path in all_container_paths if 'VerifiedFacts' not in path and 'ProjectSummaries' not in path]

    # --- Popover UI to Create or Select a Container for Upload ---
    if 'upload_target' not in st.session_state or st.session_state.upload_target not in upload_options:
        st.session_state.upload_target = upload_options[0] if upload_options else None

    popover_label = f"Upload to: {st.session_state.upload_target}"
    with st.popover(popover_label, use_container_width=True):
        st.markdown("##### Choose a destination")

        new_container_name = st.text_input("Create new container in DefianceDB:", placeholder="e.g., Customer_Engineering_proj")
        if st.button("Create and Select"):
            if new_container_name.strip():
                target_db = COSMOS_DATABASE
                target_cont = new_container_name.strip().replace(" ", "_")
                full_path = f"{target_db}/{target_cont}"
                if create_container_if_not_exists(target_db, target_cont):
                    st.session_state.upload_target = full_path
                    st.rerun()
            else:
                st.warning("Please enter a name for the new container.")

        st.divider()
        st.markdown("...or select an existing one:")

        chosen_container = st.radio(
            "Existing containers",
            options=upload_options,
            index=upload_options.index(st.session_state.upload_target) if st.session_state.upload_target in upload_options else 0,
            label_visibility="collapsed"
        )
        if chosen_container != st.session_state.upload_target:
            st.session_state.upload_target = chosen_container
            st.rerun()

    # --- UPDATED: Allow multiple file uploads including .txt and .md ---
    uploaded_files = st.file_uploader(
        "Select one or more documents...",
        type=["pdf", "docx", "m4a", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    ingest_to_cosmos = st.toggle("Ingest to Knowledge Base", value=True, help="If on, saves the document permanently. If off, uses it for this session only.")

    # --- Process a list of files ---
    if uploaded_files:
        button_label = f"Process {len(uploaded_files)} File{'s' if len(uploaded_files) > 1 else ''}"
        if st.button(button_label, use_container_width=True, type="primary"):
            all_chunks = []
            all_statuses = []
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    chunks = process_uploaded_file(uploaded_file)

                if chunks:
                    all_chunks.extend(chunks)
                    status = {"filename": uploaded_file.name, "ingested_to_cosmos": ingest_to_cosmos}
                    all_statuses.append(status)

                    if ingest_to_cosmos:
                        if st.session_state.upload_target:
                            db_name, cont_name = st.session_state.upload_target.split('/')
                            rag_uploader = get_cosmos_uploader(db_name, cont_name)
                            if rag_uploader:
                                with st.spinner(f"Ingesting '{uploaded_file.name}' to '{st.session_state.upload_target}'..."):
                                    cosmos_chunks = prepare_chunks_for_cosmos(chunks, uploaded_file.name)
                                    s, f = rag_uploader.upload_chunks(cosmos_chunks)
                                    if f > 0: status["ingestion_error"] = "Some chunks failed to ingest."

                                full_text = "\n".join(c for c in chunks if c)
                                structured_data = extract_structured_data(full_text, uploaded_file.name)
                                if structured_data:
                                    if create_container_if_not_exists("DefianceDB", "ProjectSummaries", partition_key="/projectName"):
                                        structured_uploader = get_cosmos_uploader("DefianceDB", "ProjectSummaries")
                                        if structured_uploader:
                                            with st.spinner(f"Ingesting summary for '{uploaded_file.name}'..."):
                                                structured_uploader.upload_chunks([structured_data])
                        else:
                            st.warning("No upload container selected. Ingestion skipped.")
                else:
                    st.error(f"Document processing failed for '{uploaded_file.name}'.")

            # After the loop, update the session state with aggregated results
            if all_chunks:
                st.session_state.session_rag_context = "\n\n---\n\n".join(all_chunks)
                st.session_state.rag_file_status = all_statuses
            st.rerun()

    st.markdown('<div class="mini-header">Knowledge Bases to Search</div>', unsafe_allow_html=True)
    if 'selected_containers' not in st.session_state:
        st.session_state.selected_containers = all_container_paths[:]
    num_selected = len(st.session_state.selected_containers)
    total_containers = len(all_container_paths)
    popover_label = f"Searching {num_selected} of {total_containers} KBs"
    with st.popover(popover_label, use_container_width=True):
        st.markdown("##### Select Knowledge Bases")
        is_all_selected = (num_selected == total_containers)
        select_all_clicked = st.checkbox("Select / Deselect All", value=is_all_selected, key="select_all_kbs")
        if select_all_clicked != is_all_selected:
            st.session_state.selected_containers = all_container_paths[:] if select_all_clicked else []
            st.rerun()
        st.divider()
        for container in all_container_paths:
            is_checked = container in st.session_state.selected_containers
            if st.checkbox(container, value=is_checked, key=f"kb_{container}") != is_checked:
                if is_checked: st.session_state.selected_containers.remove(container)
                else: st.session_state.selected_containers.append(container)
                st.rerun()

    st.markdown('<div class="mini-header">Chat Persona</div>', unsafe_allow_html=True)
    personas = list(st.session_state.user_data["personas"].keys())
    widget_key = f"persona_sel_{st.session_state.persona_nonce}"
    current_idx = personas.index(st.session_state.last_persona_selected) if st.session_state.last_persona_selected in personas else 0
    col_sel, col_del = st.columns([0.8, 0.2])
    with col_sel:
        selected_persona = st.selectbox(label="Chat Persona", label_visibility="collapsed", options=personas, index=current_idx, key=widget_key, disabled=st.session_state.editing_persona, on_change=on_persona_change, args=(widget_key,))
    with col_del:
        if st.button("🗑️", key="delete_persona_btn", help="Delete persona", use_container_width=True):
            if len(personas) > 1:
                del st.session_state.user_data["personas"][selected_persona]
                save_user_data(st.session_state.user_id, st.session_state.user_data)
                st.session_state.last_persona_selected = next(iter(st.session_state.user_data["personas"]))
                st.session_state.persona_nonce += 1; st.rerun()
            else:
                st.warning("Cannot delete the last persona.")

    e_col, n_col = st.columns(2)
    with e_col:
        if st.button("✏️ Edit", use_container_width=True):
            st.session_state.persona_to_edit = selected_persona
            st.session_state.editing_persona = True; st.rerun()
    with n_col:
        if st.button("➕ New", use_container_width=True):
            st.session_state.creating_persona = True; st.rerun()

    if st.button("Start New Chat with Persona", use_container_width=True):
        st.session_state.user_data = create_new_chat(st.session_state.user_id, st.session_state.user_data, selected_persona); st.rerun()

    if 'chats_to_delete' not in st.session_state:
        st.session_state.chats_to_delete = []
    with st.expander("📂 Previous Chats", expanded=False):
        convs = st.session_state.user_data.get("conversations", {})
        if not convs:
            st.caption("No chat history.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                is_all_chats_selected = (len(st.session_state.chats_to_delete) == len(convs))
                if st.checkbox("Select All", value=is_all_chats_selected, key="select_all_chats_cb"):
                    if not is_all_chats_selected:
                        st.session_state.chats_to_delete = list(convs.keys()); st.rerun()
                elif is_all_chats_selected:
                    st.session_state.chats_to_delete = []; st.rerun()
            with col2:
                if st.button("Delete Selected", use_container_width=True) and st.session_state.chats_to_delete:
                    for chat_id in st.session_state.chats_to_delete:
                        if chat_id in st.session_state.user_data["conversations"]:
                            del st.session_state.user_data["conversations"][chat_id]
                        if st.session_state.user_data["active_conversation_id"] == chat_id:
                            st.session_state.user_data["active_conversation_id"] = next(iter(st.session_state.user_data.get("conversations", {})), None)
                    st.session_state.chats_to_delete = []
                    save_user_data(st.session_state.user_id, st.session_state.user_data)
                    st.rerun()
            st.divider()
            active_id = st.session_state.user_data.get("active_conversation_id")
            for chat_id, msgs in reversed(list(convs.items())):
                is_checked = chat_id in st.session_state.chats_to_delete
                c1, c2 = st.columns([0.15, 0.85])
                with c1:
                    new_check_state = st.checkbox(" ", value=is_checked, key=f"del_{chat_id}", label_visibility="collapsed")
                    if new_check_state != is_checked:
                        if new_check_state: st.session_state.chats_to_delete.append(chat_id)
                        else: st.session_state.chats_to_delete.remove(chat_id)
                        st.rerun()
                with c2:
                    persona_name = msgs[0].get("persona_name", "Custom")
                    title = f"({persona_name}) "
                    title_content = msgs[1]['content'] if len(msgs) > 1 else "New Chat"
                    title += title_content[:25] + "..." if len(title_content) > 25 else title_content
                    is_active = chat_id == active_id
                    label = f"▶ {title}" if is_active else title
                    if st.button(label, key=f"chat_{chat_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                        st.session_state.user_data["active_conversation_id"] = chat_id; st.rerun()


# =========================== MAIN CHAT ===========================
active_chat_id = st.session_state.user_data.get("active_conversation_id")
if active_chat_id and active_chat_id in st.session_state.user_data["conversations"]:
    active_persona = st.session_state.user_data["conversations"][active_chat_id][0].get("persona_name", "Persona")
else:
    active_persona = st.session_state.last_persona_selected

title_emoji = guess_emoji(active_persona)
st.markdown(f"<h1 style='margin-top:0;'>{title_emoji} {active_persona}</h1>", unsafe_allow_html=True)

if "session_rag_context" not in st.session_state: st.session_state.session_rag_context = ""
if "rag_file_status" not in st.session_state: st.session_state.rag_file_status = None

if st.session_state.rag_file_status:
    for status in st.session_state.rag_file_status:
        filename = status.get("filename")
        ingested = status.get("ingested_to_cosmos")
        error = status.get("ingestion_error")
        if error:
            st.error(f"❌ Processing failed for **{filename}**: {error}")
        elif ingested:
            st.success(f"✅ Context from **{filename}** loaded and ingested into the knowledge base.")
        else:
            st.info(f"✅ Context from **{filename}** loaded for this session only (temporary).")

    if st.button("Clear Temporary Context"):
        st.session_state.session_rag_context = ""
        st.session_state.rag_file_status = None
        st.rerun()

if not active_chat_id:
    st.info("Welcome! Start a new chat or select one from the sidebar."); st.stop()

messages = st.session_state.user_data["conversations"][active_chat_id]

for i, m in enumerate(messages):
    if m["role"] == "system": continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and i == len(messages) - 1 and "Would you like me to save this" not in m["content"]:
            if st.button("✅ Save as Verified Fact", key=f"save_fact_{active_chat_id}"):
                last_user_question = messages[i-1]["content"] if i > 0 else ""
                if last_user_question:
                    save_verified_fact(question=last_user_question, answer=m["content"])
                else:
                    st.warning("Could not find a preceding user question to save.")

if prompt := st.chat_input("Ask anything..."):
    messages.append({"role": "user", "content": prompt})
    save_user_data(st.session_state.user_id, st.session_state.user_data)
    st.rerun()

if messages and messages[-1]["role"] == "user":
    user_prompt = messages[-1]["content"]

    with st.chat_message("assistant"):
        thinking_expander = st.expander("🤔 Agent Thinking Process...")
        log_placeholder = thinking_expander.empty()
        query_expander = st.expander("🔍 Generated Search & Results")
        final_answer_placeholder = st.empty()

        # Get details of the current persona to decide which workflow to run
        persona_details = st.session_state.user_data["personas"].get(active_persona, {})
        persona_type = persona_details.get("type", "simple")

        # --- CHOOSE WORKFLOW BASED ON PERSONA TYPE ---
        if persona_type == "agentic":
            try:
                # agentic workflow ...
                ...
            except Exception as e:
                st.error(f"An error occurred in the agentic workflow: {e}")

        elif persona_type == "simple":  # General Assistant QUICK PATH (one call)
            try:
                with st.spinner("Synthesizing quick answer..."):
                    system_prompt = persona_details.get("prompt", "You are a helpful assistant.")
                    user_context = user_prompt
                    if st.session_state.session_rag_context:
                        user_context += "\n\nContext from uploaded files:\n" + st.session_state.session_rag_context

                    query_agent_client = AzureOpenAI(
                        azure_endpoint=st.session_state.GPT41_ENDPOINT,
                        api_key=st.session_state.GPT41_API_KEY,
                        api_version="2024-05-01-preview"
                    )

                    response = query_agent_client.chat.completions.create(
                        model=st.session_state.GPT41_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_context}
                        ],
                        # optional: keep it light
                        temperature=persona_details.get("params", {}).get("temperature", 0.7),
                        max_tokens=700
                    )

                    final_answer = response.choices[0].message.content.strip()
                    final_answer_placeholder.markdown(final_answer)
                    messages.append({"role": "assistant", "content": final_answer})
                    save_user_data(st.session_state.user_id, st.session_state.user_data)

            except Exception as e:
                st.error(f"Quick mode failed: {e}")

        else: # Run the original RAG or simple conversation logic
            try:
                agent_log = []
                query_agent_client = AzureOpenAI(azure_endpoint=st.session_state.GPT41_ENDPOINT, api_key=st.session_state.GPT41_API_KEY, api_version="2024-05-01-preview")

                with st.spinner("Step 0: Classifying user intent..."):
                    router_prompt = f"""You are an expert intent routing agent. Classify the user's latest message into one of three intents: `knowledge_base_query`, `fact_correction`, or `general_conversation`. USER'S MESSAGE: "{user_prompt}" Respond ONLY with a JSON object of the format: {{"intent": "YOUR_CLASSIFICATION"}}"""
                    response = query_agent_client.chat.completions.create(model=st.session_state.GPT41_DEPLOYMENT, messages=[{"role": "system", "content": router_prompt}], response_format={"type": "json_object"})
                    intent = json.loads(response.choices[0].message.content).get("intent")
                agent_log.append(f"✅ Step 0: Intent classified as **{intent}**")
                log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))

                persona_prompt_text = persona_details.get("prompt", "You are a helpful assistant.")
                synthesis_system_prompt = ""
                context_for_synthesis = ""

                if intent == "knowledge_base_query":
                    all_verified_facts = []
                    all_document_chunks = []
                    selected_kbs = st.session_state.get('selected_containers', [])
                    if not selected_kbs:
                        st.warning("No knowledge bases selected. Please select at least one from the sidebar.")
                        st.stop()

                    with st.spinner("Step 1: Generating comprehensive search query..."):
                        initial_system_prompt = f"""You are an AI that creates a broad Cosmos DB SQL query. Search across `c.content`, `c.metadata.original_filename`, and `c.id` using `OR` and `CONTAINS(field, "keyword", true)`. Respond with a JSON object: {{"query_string": "YOUR_QUERY"}}"""
                        response = query_agent_client.chat.completions.create(model=st.session_state.GPT41_DEPLOYMENT, messages=[{"role": "system", "content": initial_system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"})
                        broad_query = json.loads(response.choices[0].message.content).get("query_string", "SELECT * FROM c")
                    agent_log.append("✅ Step 1: Comprehensive query generated.")
                    log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))
                    with query_expander: st.write("**Comprehensive Query for Document Stores:**"); st.code(broad_query, language="sql")

                    for i, kb_path in enumerate(selected_kbs):
                        db_name, cont_name = kb_path.split('/')
                        with st.spinner(f"Step 2.{i}: Searching in `{kb_path}`..."):
                            uploader = get_cosmos_uploader(db_name, cont_name)
                            if uploader:
                                if cont_name == "VerifiedFacts":
                                    fact_query = f"SELECT TOP 1 * FROM c WHERE CONTAINS(c.question, '{user_prompt.split(' ')[0]}', true) ORDER BY c.verified_at DESC"
                                    all_verified_facts.extend(uploader.execute_query(fact_query))
                                else:
                                    all_document_chunks.extend(uploader.execute_query(broad_query))
                    agent_log.append("✅ Step 2: All selected knowledge bases searched.")
                    log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))

                    with st.spinner("Step 3: Distilling all retrieved data..."):
                        distillation_input = {"verified_facts": all_verified_facts, "document_chunks": all_document_chunks}
                        distillation_prompt = f"""You are an expert AI data analyst. Distill the following JSON data into key facts relevant to the user's question. **RULE 1: `verified_facts` are the absolute source of truth and OVERRIDE any conflicting information from `document_chunks`.** USER'S QUESTION: "{user_prompt}" SEARCH RESULTS: {json.dumps(distillation_input)}"""
                        response = query_agent_client.chat.completions.create(model=st.session_state.GPT41_DEPLOYMENT, messages=[{"role": "system", "content": distillation_prompt}])
                        distilled_context = response.choices[0].message.content
                    agent_log.append("✅ Step 3: Key facts distilled.")
                    log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))
                    with query_expander: st.write("**Distilled Key Facts:**"); st.markdown(distilled_context if distilled_context.strip() else "No relevant facts were distilled.")
                    if st.session_state.session_rag_context:
                        distilled_context += "\n\n**From Current Session Upload:**\n" + st.session_state.session_rag_context

                    synthesis_system_prompt = f"{persona_prompt_text} First, think step-by-step in a `<think>` block. Then, synthesize a clear answer based *only* on the provided key facts."
                    context_for_synthesis = f"My question was: '{user_prompt}'\n\nHere are the distilled key facts:\n{distilled_context}"

                elif intent == 'fact_correction':
                    with st.spinner("Interpreting new fact for confirmation..."):
                        fact_extraction_prompt = f"""You are an AI assistant. The user provided a new fact. Rephrase it into a clear Question and Answer pair. You MUST respond with a JSON object: {{"question": "YOUR_QUESTION", "answer": "YOUR_ANSWER"}}. User statement: "{user_prompt}" """
                        response = query_agent_client.chat.completions.create(model=st.session_state.GPT41_DEPLOYMENT, messages=[{"role": "system", "content": fact_extraction_prompt}], response_format={"type": "json_object"})
                        qa_pair = json.loads(response.choices[0].message.content)
                        question = qa_pair.get("question"); answer = qa_pair.get("answer")
                    agent_log.append("✅ Fact interpreted for confirmation."); log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))
                    if question and answer:
                        with final_answer_placeholder.container():
                            st.info("To improve the knowledge base, please review and save this fact:")
                            st.markdown(f"**Question:** `{question}`"); st.markdown(f"**Answer:** `{answer}`")
                            if st.button("Confirm and Save Fact", key=f"confirm_save_{active_chat_id}"):
                                save_verified_fact(question, answer)
                                messages.append({"role": "assistant", "content": "Fact saved successfully!"})
                                save_user_data(st.session_state.user_id, st.session_state.user_data); st.rerun()
                        st.stop()

                else: # general_conversation
                    agent_log.append("✅ Formulating conversational response."); log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))
                    history = [{"role": m["role"], "content": m["content"]} for m in messages]
                    synthesis_system_prompt = f"{persona_prompt_text} First, think step-by-step in a `<think>` block. Then, respond to the user's latest message based on the conversation history."
                    context_for_synthesis = json.dumps(history)

                with st.spinner("Synthesizing final answer with O3..."):
                    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
                    streaming_client = AzureOpenAI(
                        azure_endpoint=st.session_state.GPT41_ENDPOINT,
                        azure_ad_token_provider=token_provider,
                        api_version="2024-12-01-preview"
                    )
                    messages_for_o3 = [{"role": "system", "content": synthesis_system_prompt}, {"role": "user", "content": context_for_synthesis}]
                    stream = streaming_client.chat.completions.create(
                        model=st.session_state.O3_DEPLOYMENT,
                        messages=messages_for_o3,
                        stream=True,
                    )

                    response_parts = []
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            response_parts.append(token)
                            final_answer_placeholder.markdown("".join(response_parts) + " ▌")
                    full_response = "".join(response_parts)

                thinking_content = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
                final_answer = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                final_answer_placeholder.markdown(final_answer)
                if thinking_content:
                    agent_log.append("✅ Final answer synthesized.")
                    log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))
                    thinking_expander.info(thinking_content.group(1).strip())

                messages.append({"role": "assistant", "content": full_response})
                save_user_data(st.session_state.user_id, st.session_state.user_data)
                st.session_state.session_rag_context = ""; st.session_state.rag_file_status = None

            except Exception as e:
                st.error(f"An error occurred in the retrieval process: {e}")