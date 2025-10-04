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
import sqlite3
from datetime import datetime
import difflib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Document Processing
import fitz  # PyMuPDF
import pymupdf4llm
import docx

# Azure Cosmos DB
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# Azure OpenAI Client (used for Vision and Chat)
from openai import AzureOpenAI

import streamlit as st
from streamlit_mic_recorder import mic_recorder

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

# spaCy not needed - using LLM agents for all extraction


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

# --- Mathematical Formatting ---
def format_latex(latex_string: str) -> str:
    """
    Formats a raw LaTeX string for display in Markdown.
    Wraps the string in $$...$$ for block-level math rendering.
    Example: format_latex("c = \\sqrt{a^2 + b^2}") returns "$$c = \\sqrt{a^2 + b^2}$$"
    """
    # Ensure backslashes are properly escaped for JSON and Markdown
    processed_string = latex_string.replace('\\', '\\\\')
    return f"$$\n{latex_string}\n$$"

# --- CSV Formatting ---
import csv
from io import StringIO

def to_csv(rows: List[Dict[str, Any]], delimiter: str = ',') -> str:
    """
    Converts a list of dictionaries into a CSV formatted string.
    Handles headers, quoting, and different delimiters automatically.
    """
    if not rows:
        return "Error: Input list of rows cannot be empty."
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return "Error: Input must be a list of dictionaries."

    try:
        # Use StringIO as an in-memory file
        output = StringIO()
        
        # The first dictionary's keys are used as the header
        headers = list(rows[0].keys())
        
        # Use the csv module for robust CSV writing
        writer = csv.DictWriter(output, fieldnames=headers, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        
        writer.writeheader()
        writer.writerows(rows)
        
        # Get the string value from the StringIO object
        return output.getvalue()
    except Exception as e:
        return f"Error creating CSV: {e}"

# --- Markdown Formatting ---
def format_list(items: List[str], style: str = 'bullet') -> str:
    """
    Formats a list of strings as a Markdown list.
    'style' can be 'bullet' for a bulleted list (*) or 'numbered' for a numbered list (1., 2., ...).
    """
    if not items or not isinstance(items, list):
        return "Error: Input must be a non-empty list of strings."
        
    if style == 'numbered':
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    # Default to bullet points for any other style input
    return "\n".join(f"* {item}" for item in items)

def format_code(code_string: str, language: str = 'python') -> str:
    """
    Formats a string as a Markdown code block with syntax highlighting.
    """
    return f"```{language}\n{code_string}\n```"

def format_blockquote(text: str) -> str:
    """
    Formats a string as a Markdown blockquote.
    """
    lines = text.strip().split('\n')
    return "\n".join(f"> {line}" for line in lines)

def format_link(text: str, url: str) -> str:
    """
    Creates a Markdown hyperlink.
    Example: format_link("Google", "https://www.google.com") returns "[Google](https://www.google.com)"
    """
    return f"[{text}]({url})"

# --- Scratchpad Tools (will be initialized with manager instance) ---
def scratchpad_read(pad_name: str, section_name: str = None) -> str:
    """
    Read from a scratchpad. If section_name is provided, reads just that section.
    Otherwise returns full content.
    Available pads: output, research, tables, plots, outline, data, log
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"

    if section_name:
        return manager.read_section(pad_name, section_name)
    else:
        return manager.get_full_content(pad_name)

def scratchpad_write(pad_name: str, section_name: str, content: str, mode: str = "replace") -> str:
    """
    Write to a scratchpad section.
    Modes: 'replace' (default), 'append', 'prepend'
    Available pads: output, research, tables, plots, outline, data, log
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"
    return manager.write_section(pad_name, section_name, content, mode)

def scratchpad_edit(pad_name: str, section_name: str, old_text: str, new_text: str) -> str:
    """
    Find and replace text within a scratchpad section.
    Available pads: output, research, tables, plots, outline, data, log
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"
    return manager.edit_section(pad_name, section_name, old_text, new_text)

def scratchpad_delete(pad_name: str, section_name: str) -> str:
    """
    Delete a section from a scratchpad.
    Available pads: output, research, tables, plots, outline, data, log
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"
    return manager.delete_section(pad_name, section_name)

def scratchpad_list(pad_name: str) -> str:
    """
    List all sections in a scratchpad.
    Available pads: output, research, tables, plots, outline, data, log
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"
    sections = manager.list_sections(pad_name)
    return f"Sections in '{pad_name}': {', '.join(sections) if sections else 'none'}"

def scratchpad_summary() -> str:
    """
    Get a summary of all scratchpads and their sections.
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"
    return manager.get_all_pads_summary()

def scratchpad_merge(pad_name: str, section_names: List[str], new_section_name: str) -> str:
    """
    Merge multiple sections into one new section.
    Available pads: output, research, tables, plots, outline, data, log
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"
    return manager.merge_sections(pad_name, section_names, new_section_name)

def scratchpad_history(pad_name: str, section_name: str, limit: int = 10) -> str:
    """
    Get version history with diffs for a scratchpad section.
    Shows who made changes and when, with line-by-line diffs.
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"

    history = manager.get_version_history(pad_name, section_name, limit)
    if not history:
        return f"No history found for '{pad_name}.{section_name}'"

    output = f"# Version History for '{pad_name}.{section_name}'\n\n"
    for i, version in enumerate(history, 1):
        output += f"## Version {i} - {version['timestamp']}\n"
        output += f"**Operation:** {version['operation']}  \n"
        output += f"**Agent:** {version['agent']}  \n\n"
        output += f"```diff\n{version['diff']}\n```\n\n"

    return output

def scratchpad_cleanup_formatting(pad_name: str, section_name: str) -> str:
    """
    Clean common formatting issues in a scratchpad section.
    Fixes LaTeX escaping, spacing issues, special characters.
    Call this after writing content to ensure clean markdown.
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"

    return manager.cleanup_formatting(pad_name, section_name)

def scratchpad_insert_lines(pad_name: str, section_name: str, line_number: int, content: str) -> str:
    """
    Insert lines at a specific line number in a section (like Claude Code).
    Line numbers start at 1. Use 0 to prepend, -1 to append.
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"

    if pad_name not in manager.pads:
        return f"Error: Scratchpad '{pad_name}' does not exist"

    if section_name not in manager.pads[pad_name]["sections"]:
        # Create new section if it doesn't exist
        manager.pads[pad_name]["sections"][section_name] = content
        manager.pads[pad_name]["metadata"]["last_modified"] = time.time()
        manager._rebuild_content(pad_name)
        return f"Created new section '{pad_name}.{section_name}' with content"

    existing = manager.pads[pad_name]["sections"][section_name]
    lines = existing.split('\n')

    if line_number == 0:
        lines.insert(0, content)
    elif line_number == -1 or line_number > len(lines):
        lines.append(content)
    else:
        lines.insert(line_number - 1, content)

    manager.pads[pad_name]["sections"][section_name] = '\n'.join(lines)
    manager.pads[pad_name]["metadata"]["last_modified"] = time.time()
    manager._rebuild_content(pad_name)
    return f"Inserted at line {line_number} in '{pad_name}.{section_name}'"

def scratchpad_delete_lines(pad_name: str, section_name: str, start_line: int, end_line: int = None) -> str:
    """
    Delete specific lines from a section (like Claude Code).
    Line numbers start at 1. If end_line is None, deletes only start_line.
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"

    if pad_name not in manager.pads:
        return f"Error: Scratchpad '{pad_name}' does not exist"
    if section_name not in manager.pads[pad_name]["sections"]:
        return f"Error: Section '{section_name}' not found in '{pad_name}'"

    existing = manager.pads[pad_name]["sections"][section_name]
    lines = existing.split('\n')

    if end_line is None:
        end_line = start_line

    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line

    if start_idx < 0 or end_idx > len(lines):
        return f"Error: Line range {start_line}-{end_line} out of bounds (section has {len(lines)} lines)"

    del lines[start_idx:end_idx]

    manager.pads[pad_name]["sections"][section_name] = '\n'.join(lines)
    manager.pads[pad_name]["metadata"]["last_modified"] = time.time()
    manager._rebuild_content(pad_name)
    return f"Deleted lines {start_line}-{end_line} from '{pad_name}.{section_name}'"

def scratchpad_replace_lines(pad_name: str, section_name: str, start_line: int, end_line: int, content: str) -> str:
    """
    Replace specific lines in a section with new content (like Claude Code).
    Line numbers start at 1.
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"

    if pad_name not in manager.pads:
        return f"Error: Scratchpad '{pad_name}' does not exist"
    if section_name not in manager.pads[pad_name]["sections"]:
        return f"Error: Section '{section_name}' not found in '{pad_name}'"

    existing = manager.pads[pad_name]["sections"][section_name]
    lines = existing.split('\n')

    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line

    if start_idx < 0 or end_idx > len(lines):
        return f"Error: Line range {start_line}-{end_line} out of bounds (section has {len(lines)} lines)"

    # Replace the range with new content
    lines[start_idx:end_idx] = [content]

    manager.pads[pad_name]["sections"][section_name] = '\n'.join(lines)
    manager.pads[pad_name]["metadata"]["last_modified"] = time.time()
    manager._rebuild_content(pad_name)
    return f"Replaced lines {start_line}-{end_line} in '{pad_name}.{section_name}'"

def scratchpad_get_lines(pad_name: str, section_name: str, start_line: int = None, end_line: int = None) -> str:
    """
    Get specific lines from a section. If no range specified, returns all with line numbers.
    Line numbers start at 1.
    """
    manager = st.session_state.get("scratchpad_manager")
    if not manager:
        return "Error: Scratchpad manager not initialized"

    if pad_name not in manager.pads:
        return f"Error: Scratchpad '{pad_name}' does not exist"
    if section_name not in manager.pads[pad_name]["sections"]:
        return f"Error: Section '{section_name}' not found in '{pad_name}'"

    existing = manager.pads[pad_name]["sections"][section_name]
    lines = existing.split('\n')

    if start_line is None:
        # Return all lines with line numbers
        numbered = [f"{i+1:4d}│ {line}" for i, line in enumerate(lines)]
        return '\n'.join(numbered)
    else:
        if end_line is None:
            end_line = start_line
        start_idx = start_line - 1
        end_idx = end_line

        if start_idx < 0 or end_idx > len(lines):
            return f"Error: Line range {start_line}-{end_line} out of bounds (section has {len(lines)} lines)"

        selected = lines[start_idx:end_idx]
        numbered = [f"{i+start_line:4d}│ {line}" for i, line in enumerate(selected)]
        return '\n'.join(numbered)

def get_cached_search_results(keywords: List[str] = None) -> str:
    """
    Retrieve cached search results from previous loops in this workflow.
    If keywords are specified, returns results from that specific search.
    If no keywords specified, returns a summary of all cached searches.
    """
    cache = st.session_state.get("loop_results_cache", {})

    if not cache:
        return "No cached search results available from previous loops."

    if keywords:
        # Look for specific search by keywords
        cache_key = f"search_{json.dumps(keywords, sort_keys=True)}_"
        matching = [k for k in cache.keys() if k.startswith(cache_key)]

        if not matching:
            return f"No cached results found for keywords: {keywords}"

        # Return the most recent match
        latest_key = matching[-1]
        cached_data = cache[latest_key]
        results = cached_data["results"]

        return f"Cached search results for {keywords}:\n\n{cached_data['summary']}\n\nResults:\n{json.dumps(results, indent=2)}"

    else:
        # Return summary of all cached searches
        summary = "## Cached Search Results from Previous Loops\n\n"
        for idx, (cache_key, data) in enumerate(cache.items(), 1):
            summary += f"{idx}. **Keywords:** {data['keywords']} (Limit: {data['rank_limit']})\n"
            summary += f"   - {data['summary']}\n"
            summary += f"   - Cached at: {data['timestamp']}\n"
            summary += f"   - Cache key for retrieval: Use keywords {data['keywords']}\n\n"

        summary += "\n**To retrieve full results:** Call `get_cached_search_results(keywords=[...])`"
        return summary

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
    "scratchpad_read": scratchpad_read,
    "scratchpad_write": scratchpad_write,
    "scratchpad_edit": scratchpad_edit,
    "scratchpad_delete": scratchpad_delete,
    "scratchpad_list": scratchpad_list,
    "scratchpad_summary": scratchpad_summary,
    "scratchpad_merge": scratchpad_merge,
    "scratchpad_insert_lines": scratchpad_insert_lines,
    "scratchpad_delete_lines": scratchpad_delete_lines,
    "scratchpad_replace_lines": scratchpad_replace_lines,
    "scratchpad_cleanup_formatting": scratchpad_cleanup_formatting,
    "get_cached_search_results": get_cached_search_results,
    "scratchpad_get_lines": scratchpad_get_lines,
    "scratchpad_history": scratchpad_history,
}
# ====================================================================

# --------------------------- Page Config ---------------------------
st.set_page_config(
    page_title="WBI AI",
    page_icon="🤖",
    layout="wide"
)

# --- Floating mic styles (place right after st.set_page_config) ---
st.markdown("""
<style>
  /* little floating container that sits just left of the chat send arrow */
  #voice-fab {
    position: fixed;
    right: 76px;      /* nudge left/right if you want */
    bottom: 18px;     /* nudge up/down if you want   */
    z-index: 9999;
  }
  /* make sure it never pushes layout around */
  #voice-fab > div { margin: 0 !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


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


# =========================== AZURE HELPERS ===========================
def is_running_on_azure() -> bool:
    """Detect if running on Azure App Service"""
    return os.getenv("WEBSITE_INSTANCE_ID") is not None

def get_scratchpad_db_path() -> str:
    """Get appropriate DB path for environment"""
    if is_running_on_azure():
        # Use /tmp on Azure (ephemeral but works within session)
        return "/tmp/scratchpad.db"
    else:
        # Use local directory for development
        return "scratchpad.db"

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

    # 1) Verify + cache GPT-4.1 (Query Gen & Vision)
    st.markdown("--- \n**Checking GPT-4.1 Connection...**")
    try:
        st.info(f"Initializing client for GPT-4.1 at: {st.session_state.GPT41_ENDPOINT}")
        if "gpt41_client" not in st.session_state:
            st.session_state.gpt41_client = AzureOpenAI(
                azure_endpoint=st.session_state.GPT41_ENDPOINT,
                api_key=st.session_state.GPT41_API_KEY,
                api_version="2024-05-01-preview"
            )
        gpt41_client = st.session_state.gpt41_client
        st.info(f"Verifying deployment '{st.session_state.GPT41_DEPLOYMENT}'...")
        response = st.session_state.gpt41_client.chat.completions.create(
            model=st.session_state.GPT41_DEPLOYMENT,
            messages=[{"role": "user", "content": "Test connection"}]
        )
        st.success(f"✅ GPT-4.1 connection successful. Test response: '{response.choices[0].message.content.strip()}'")
    except Exception as e:
        st.error(f"❌ GPT-4.1 connection FAILED. Error: {e}")
        all_verified = False

    # 2) Verify + cache O3 (Primary Synthesis)
    st.markdown("--- \n**Checking Synthesis Model (O3) Connection...**")
    try:
        st.info(f"Initializing O3 client for endpoint: {st.session_state.GPT41_ENDPOINT}")
        if "o3_client" not in st.session_state:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            st.session_state.o3_client = AzureOpenAI(
                azure_endpoint=st.session_state.GPT41_ENDPOINT,
                azure_ad_token_provider=token_provider,
                api_version="2024-12-01-preview"
            )
        o3_client = st.session_state.o3_client
        st.info(f"Verifying deployment '{st.session_state.O3_DEPLOYMENT}'...")
        response = st.session_state.o3_client.chat.completions.create(
            model=st.session_state.O3_DEPLOYMENT,
            messages=[{"role": "user", "content": "Test connection"}],
        )
        st.success(f"✅ O3 synthesis model connection successful. Test response: '{response.choices[0].message.content.strip()}'")
    except Exception as e:
        st.error(f"❌ O3 synthesis model connection FAILED. Error: {e}")
        all_verified = False

    # 3) Verify + cache Speech Service
    st.markdown("--- \n**Checking Speech Service Connection...**")
    try:
        st.info(f"Initializing Speech client for region: {st.session_state.SPEECH_REGION}")
        if "speech_config" not in st.session_state:
            st.session_state.speech_config = speechsdk.SpeechConfig(
                subscription=st.session_state.SPEECH_KEY,
                region=st.session_state.SPEECH_REGION
            )
        # Optional mini ping:
        try:
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=st.session_state.speech_config, audio_config=None)
            result = synthesizer.speak_text_async("ping").get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                st.success("✅ Azure Speech synthesis reachable.")
            else:
                # If synthesis blocked by policy/device, config still valid—treat as soft pass
                st.warning(f"⚠️ Speech reachable but synthesis not completed: {result.reason}")
        except Exception:
            st.success("✅ Azure Speech Service configured successfully.")
    except Exception as e:
        st.error(f"❌ Azure Speech Service configuration FAILED. Error: {e}")
        all_verified = False

    # 4) Final Check
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

# ---- cached client factories (use the same env vars as the verifier) ----
@st.cache_resource
def get_gpt41_client():
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")  # unified
    key = os.getenv("AZURE_AI_SEARCH_API_KEY")        # unified
    if not endpoint or not key:
        raise RuntimeError("Missing AZURE_AI_SEARCH_ENDPOINT or AZURE_AI_SEARCH_API_KEY")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    )

@st.cache_resource
def get_o3_client_token_provider():
    # Entra ID auth for Azure OpenAI (works for keyless deployments you’ve granted)
    return get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

@st.cache_resource
def get_o3_client():
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")  # unified
    if not endpoint:
        raise RuntimeError("Missing AZURE_AI_SEARCH_ENDPOINT")
    token_provider = get_o3_client_token_provider()
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    )

@st.cache_resource
def get_speech_config():
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")
    return speechsdk.SpeechConfig(subscription=key, region=region)



# One-time session wiring
if "gpt41_client" not in st.session_state:
    st.session_state.gpt41_client = get_gpt41_client()
if "o3_client" not in st.session_state:
    st.session_state.o3_client = get_o3_client()
if "speech_config" not in st.session_state:
    st.session_state.speech_config = get_speech_config()

# keep your deployment names in state too
st.session_state.GPT41_DEPLOYMENT = os.getenv("GPT41_DEPLOYMENT")
st.session_state.O3_DEPLOYMENT   = os.getenv("O3_DEPLOYMENT_NAME")



@st.cache_resource
def get_blob_service_client():
    current_user_id = get_current_user().get("id")
    if current_user_id == "local_dev":
        credential = AzureCliCredential()
    else:
        credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)


def process_audio_generic(file_bytes: bytes, filename: str) -> List[str]:
    ext = os.path.splitext(filename.lower())[1].lstrip(".")
    wav16k = ensure_16k_mono_wav(file_bytes, ext_hint=ext)
    if not wav16k:
        return []
    text = azure_fast_transcribe_wav_bytes(wav16k, filename=filename)
    return chunk_text(text) if text else []


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
                "You are a helpful assistant with access to an internal knowledge base. "
                "Your primary function is to provide answers grounded in the data from that knowledge base. "
                "Synthesize the retrieved information clearly and concisely. If no information is found or the retrieved facts are insufficient to answer the question, state that explicitly rather than making assumptions."
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

def generate_embedding(text: str) -> list[float]:
    """Generate embedding vector for semantic search using Azure OpenAI."""
    try:
        client = st.session_state.gpt41_client
        # Use text-embedding-ada-002 or text-embedding-3-small
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"  # Update this if you have a different embedding model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None

@st.cache_data(ttl=3600)
def check_vector_index_exists(container_path: str) -> bool:
    """Check if a container has vector indexing enabled."""
    try:
        db_name, cont_name = container_path.split("/")
        client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database = client.get_database_client(db_name)
        container = database.get_container_client(cont_name)

        # Try a simple vector search query to see if it's supported
        test_vector = [0.0] * 1536  # Standard embedding size
        test_query = f"SELECT TOP 1 c.id FROM c ORDER BY VectorDistance(c.embedding, {test_vector})"
        try:
            list(container.query_items(query=test_query, enable_cross_partition_query=True, max_item_count=1))
            return True
        except:
            return False
    except Exception as e:
        logger.warning(f"Could not check vector index for {container_path}: {e}")
        return False

@st.cache_data(ttl=3600)
def discover_cosmos_schema(selected_containers: list[str]) -> dict:
    """
    Discovers schema by sampling documents from selected containers.
    Returns a dict with container schemas and field information.
    """
    schema_info = {
        "containers": {},
        "common_fields": ["id", "content", "metadata"],
        "container_specific_fields": {}
    }

    try:
        client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)

        for container_path in selected_containers[:10]:  # Limit to 10 containers for performance
            try:
                db_name, cont_name = container_path.split("/")
                database = client.get_database_client(db_name)
                container = database.get_container_client(cont_name)

                # Sample 3 documents to discover schema
                sample_query = "SELECT TOP 3 * FROM c"
                items = list(container.query_items(query=sample_query, enable_cross_partition_query=True))

                if items:
                    # Collect all unique keys from samples
                    all_keys = set()
                    for item in items:
                        all_keys.update(_get_all_keys(item))

                    schema_info["containers"][container_path] = {
                        "fields": sorted(list(all_keys)),
                        "sample_count": len(items)
                    }

                    # Track container-specific fields
                    if cont_name not in schema_info["container_specific_fields"]:
                        schema_info["container_specific_fields"][cont_name] = sorted(list(all_keys))

            except Exception as e:
                logger.warning(f"Failed to discover schema for {container_path}: {e}")
                continue

    except Exception as e:
        logger.error(f"Schema discovery failed: {e}")

    return schema_info

def _get_all_keys(obj: dict, prefix: str = "c") -> set:
    """Recursively extracts all field paths from a nested dict."""
    keys = set()
    for key, value in obj.items():
        if key.startswith('_'):  # Skip system fields like _rid, _self, _etag, _attachments, _ts
            continue
        full_key = f"{prefix}.{key}"
        keys.add(full_key)
        if isinstance(value, dict):
            keys.update(_get_all_keys(value, full_key))
    return keys

def hybrid_search_cosmosdb(query_text: str, selected_containers: list[str], top_k: int = 30) -> tuple[list, str]:
    """
    Performs hybrid search combining keyword (CONTAINS) and semantic (vector) search.
    Returns (results, search_method_used)
    """
    all_results = []
    search_method = "keyword_only"

    # Generate embedding for semantic search
    query_embedding = generate_embedding(query_text)

    for kb_path in selected_containers:
        try:
            db_name, cont_name = kb_path.split("/")
            uploader = get_cosmos_uploader(db_name, cont_name)
            if not uploader:
                continue

            # Check if vector search is available
            has_vector_index = check_vector_index_exists(kb_path) if query_embedding else False

            if has_vector_index and query_embedding:
                # Hybrid search: Vector search + keyword filter
                search_method = "hybrid"
                vector_query = f"""
                SELECT TOP {top_k} c.id, c.content, c.metadata, c.question, c.answer,
                       VectorDistance(c.embedding, {query_embedding}) AS SimilarityScore
                FROM c
                WHERE VectorDistance(c.embedding, {query_embedding}) < 0.6
                ORDER BY VectorDistance(c.embedding, {query_embedding})
                """
                results = uploader.execute_query(vector_query)
                for r in results:
                    r["_source_container"] = kb_path
                    r["_search_method"] = "vector"
                all_results.extend(results)
                logger.info(f"Vector search on {kb_path}: {len(results)} results")

            else:
                # Fallback to keyword search
                # Extract meaningful keywords
                stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'in', 'is', 'it', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'}
                keywords = [t for t in re.split(r"\W+", query_text.lower()) if t and len(t) > 2 and t not in stop_words][:5]

                if keywords:
                    clauses = []
                    for kw in keywords:
                        safe_kw = kw.replace("'", "''")
                        clauses.append(f"(CONTAINS(c.content, '{safe_kw}', true) OR CONTAINS(c.metadata.original_filename, '{safe_kw}', true))")

                    keyword_query = f"SELECT TOP {top_k} c.id, c.content, c.metadata, c.question, c.answer FROM c WHERE {' OR '.join(clauses)}"
                    results = uploader.execute_query(keyword_query)
                    for r in results:
                        r["_source_container"] = kb_path
                        r["_search_method"] = "keyword"
                    all_results.extend(results)
                    logger.info(f"Keyword search on {kb_path}: {len(results)} results")

        except Exception as e:
            logger.error(f"Hybrid search failed for {kb_path}: {e}")
            continue

    return all_results, search_method

def check_file_exists(container_path: str, filename: str) -> tuple[bool, list]:
    """
    Check if a file with this name already exists in the container.
    Returns (exists: bool, existing_items: list)
    """
    try:
        db_name, cont_name = container_path.split("/")
        uploader = get_cosmos_uploader(db_name, cont_name)
        if not uploader:
            return False, []

        # Query for documents with this filename in metadata
        safe_filename = filename.replace("'", "''")
        query = f"SELECT c.id, c.metadata FROM c WHERE CONTAINS(c.metadata.original_filename, '{safe_filename}', true)"
        results = uploader.execute_query(query)

        if results and len(results) > 0:
            return True, results
        return False, []
    except Exception as e:
        logger.error(f"Failed to check for duplicate file: {e}")
        return False, []

def create_container_if_not_exists(db_name: str, container_name: str, partition_key: str = "/id"):
    """Creates a new container in Cosmos DB if it doesn't already exist."""
    try:
        client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database = client.get_database_client(db_name)
        database.create_container_if_not_exists(id=container_name, partition_key=PartitionKey(path=partition_key))
        st.toast(f"Container '{container_name}' is ready.")
        get_available_containers.clear()
        discover_cosmos_schema.clear()  # Clear schema cache when new container created
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
    client = st.session_state.gpt41_client
    model = st.session_state.GPT41_DEPLOYMENT
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Azure Vision API Error: {e}")
        return f"[VISION_PROCESSING_ERROR: {e}]"

def extract_structured_data(full_text: str, filename: str) -> dict:
    st.info("Extracting structured data from document...")

    extraction_schema = {
        "projectName": "string",
        "doc_type": "ProjectSummary",
        "sourceDocument": "string",
        "summary": "string",
        "timeline": {"value": "string (e.g., '6 months', 'Q4 2025')", "startDate": "string (ISO 8601 format, e.g., '2025-08-01')"},
        "budget": {"amount": "number", "currency": "string (e.g., 'USD')"},
        "risks": ["list of strings"],
        "optionalExtensions": ["list of objects with 'name' and 'details' properties"]
    }

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
        client = st.session_state.gpt41_client
        model = st.session_state.GPT41_DEPLOYMENT
        context = f"FILENAME: '{filename}'\n\nDOCUMENT TEXT:\n---\n{full_text}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        extracted_data = json.loads(response.choices[0].message.content)
        extracted_data['id'] = f"proj_{uuid.uuid4()}"
        st.success("Successfully extracted structured data.")
        return extracted_data

    except Exception as e:
        st.error(f"Failed to extract structured data: {e}")
        return None


# ====================================================================
# === COMPREHENSIVE DOCUMENT CLASSIFICATION AND EXTRACTION
# ====================================================================

def classify_document_with_llm(full_text: str, filename: str) -> Dict[str, Any]:
    """
    Classify document type using LLM analysis.
    Returns: {"doc_type": str, "confidence": float, "reasoning": str}
    """
    system_prompt = """You are an expert document classifier. Analyze the provided document and classify it into ONE of these categories:

DOCUMENT TYPES:
- RFI_Response: Response to Request for Information from vendors
- Solicitation: Government RFI, RFP, RFQ, or contract solicitation
- Technical_Specification: Technical requirements, specifications, standards
- Project_Plan: Project management documents, timelines, roadmaps
- Vendor_Proposal: Vendor proposals, capability statements, white papers
- Meeting_Notes: Meeting minutes, notes, action items
- Report: Analysis reports, status reports, research findings
- Contract: Contracts, agreements, statements of work
- Financial: Budget documents, cost analyses, financial reports
- Email: Email threads, correspondence
- Presentation: Slide decks, briefing materials
- Other: Any document that doesn't fit above categories

Respond in JSON format with:
{
  "doc_type": "one of the types above",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation of classification"
}"""

    try:
        client = st.session_state.gpt41_client
        model = st.session_state.GPT41_DEPLOYMENT

        # Use first 5000 chars for classification
        sample_text = full_text[:5000]
        context = f"FILENAME: '{filename}'\n\nDOCUMENT TEXT:\n---\n{sample_text}\n---"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        logger.info(f"Document classified as {result.get('doc_type')} with confidence {result.get('confidence')}")
        return result
    except Exception as e:
        logger.error(f"Failed to classify document: {e}")
        return {"doc_type": "Other", "confidence": 0.0, "reasoning": f"Classification error: {e}"}


def extract_comprehensive_metadata_with_llm(full_text: str, filename: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata using LLM analysis instead of spaCy.
    Extracts entities, dates, monetary values, locations, contacts, technical terms.
    """
    system_prompt = """You are an expert metadata extraction AI. Analyze the document text and extract comprehensive metadata in JSON format.

Extract ALL instances of:
- Organizations (companies, agencies, institutions)
- Persons (names of people mentioned)
- Locations (cities, states, countries, facilities)
- Dates (any date references)
- Monetary values (costs, budgets, prices)
- Email addresses
- Phone numbers
- Technical terms and acronyms
- Key metrics and measurements

Be exhaustive - extract EVERY unique instance you find."""

    extraction_schema = {
        "entities": {
            "organizations": ["list of all unique organization names"],
            "persons": ["list of all unique person names"],
            "locations": ["list of all unique locations"],
            "dates": ["list of all unique dates mentioned"],
            "monetary_values": ["list of all monetary amounts with context"],
            "technical_terms": ["list of technical terms, acronyms, specialized vocabulary"]
        },
        "contact_information": {
            "emails": ["list of all email addresses"],
            "phones": ["list of all phone numbers"]
        },
        "key_metrics": {
            "measurements": ["list of key measurements and quantities with units"]
        },
        "document_statistics": {
            "document_length": len(full_text),
            "word_count": len(full_text.split())
        }
    }

    try:
        client = st.session_state.gpt41_client
        model = st.session_state.GPT41_DEPLOYMENT

        # Use first 25k chars for metadata extraction
        sample_text = full_text[:25000]
        context = f"""FILENAME: '{filename}'

DOCUMENT TEXT:
---
{sample_text}
---

Extract metadata according to this JSON schema:
{json.dumps(extraction_schema, indent=2)}

Respond with the populated JSON."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=4000
        )

        metadata = json.loads(response.choices[0].message.content)
        # Ensure document_statistics is included
        if "document_statistics" not in metadata:
            metadata["document_statistics"] = {
                "document_length": len(full_text),
                "word_count": len(full_text.split())
            }
        else:
            metadata["document_statistics"]["document_length"] = len(full_text)
            metadata["document_statistics"]["word_count"] = len(full_text.split())

        return metadata
    except Exception as e:
        logger.error(f"LLM metadata extraction failed: {e}")
        # Return minimal metadata on failure
        return {
            "entities": {
                "organizations": [],
                "persons": [],
                "locations": [],
                "dates": [],
                "monetary_values": [],
                "technical_terms": []
            },
            "contact_information": {
                "emails": [],
                "phones": []
            },
            "key_metrics": {
                "measurements": []
            },
            "document_statistics": {
                "document_length": len(full_text),
                "word_count": len(full_text.split())
            },
            "extraction_error": str(e)
        }


def generate_document_summary_with_mapreduce(full_text: str, filename: str, doc_type: str = "Unknown") -> Dict[str, str]:
    """
    Generate comprehensive document summary using map-reduce for long documents.
    Returns: {"executive_summary": str, "key_points": [str], "recommendations": [str]}
    """
    client = st.session_state.gpt41_client
    model = st.session_state.GPT41_DEPLOYMENT

    # Chunk document if long (>15k words)
    words = full_text.split()
    word_count = len(words)

    if word_count > 15000:
        # MAP phase: Summarize chunks
        chunk_size = 5000  # words per chunk
        chunk_summaries = []

        for i in range(0, word_count, chunk_size):
            chunk_words = words[i:i+chunk_size]
            chunk_text = " ".join(chunk_words)

            map_prompt = f"""Summarize this section of a {doc_type} document in 3-5 bullet points. Focus on key facts, decisions, and actionable items.

SECTION TEXT:
{chunk_text}

Respond with only the bullet points."""

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": map_prompt}],
                    temperature=0.0,
                    max_tokens=500
                )
                chunk_summary = response.choices[0].message.content
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {i//chunk_size}: {e}")

        # REDUCE phase: Synthesize chunk summaries
        combined_summaries = "\n\n".join([f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)])
        reduce_prompt = f"""You are analyzing a {doc_type} document that has been pre-summarized in sections. Create a comprehensive final summary.

SECTION SUMMARIES:
{combined_summaries}

Respond in JSON format:
{{
  "executive_summary": "2-3 paragraph overview",
  "key_points": ["point 1", "point 2", ...],
  "recommendations": ["rec 1", "rec 2", ...] or [] if none
}}"""

    else:
        # Direct summarization for shorter documents
        reduce_prompt = f"""Analyze this {doc_type} document and create a comprehensive summary.

DOCUMENT TEXT:
{full_text[:25000]}

Respond in JSON format:
{{
  "executive_summary": "2-3 paragraph overview",
  "key_points": ["point 1", "point 2", ...],
  "recommendations": ["rec 1", "rec 2", ...] or [] if none
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": reduce_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2000
        )
        summary = json.loads(response.choices[0].message.content)
        return summary
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return {
            "executive_summary": f"Summary generation failed: {e}",
            "key_points": [],
            "recommendations": []
        }


def extract_document_schema_by_type(full_text: str, filename: str, doc_type: str) -> Dict[str, Any]:
    """
    Extract structured data based on document type using specialized schemas.
    """
    # Define extraction schemas by document type
    schemas = {
        "RFI_Response": {
            "vendor_name": "string",
            "solution_category": "COTS/SaaS, Consulting, Custom-Build, or Ancillary",
            "technology_readiness_level": "number 1-9",
            "key_capabilities": ["list of strings"],
            "pricing_model": "string",
            "implementation_timeline": "string",
            "technical_risks": ["list of strings"],
            "integration_approach": "string"
        },
        "Solicitation": {
            "solicitation_number": "string",
            "agency": "string",
            "response_deadline": "ISO 8601 date",
            "project_scope": "string",
            "evaluation_criteria": ["list of strings"],
            "budget_range": "string",
            "contract_type": "string",
            "required_certifications": ["list of strings"]
        },
        "Technical_Specification": {
            "specification_title": "string",
            "version": "string",
            "requirements": ["list of requirement objects"],
            "standards_referenced": ["list of strings"],
            "technical_constraints": ["list of strings"],
            "performance_metrics": {}
        },
        "Vendor_Proposal": {
            "vendor_name": "string",
            "proposed_solution": "string",
            "cost_breakdown": {},
            "team_composition": ["list of strings"],
            "past_performance": ["list of projects"],
            "differentiators": ["list of strings"]
        }
    }

    # Get schema for this document type, or use generic if not found
    schema = schemas.get(doc_type, {
        "title": "string",
        "summary": "string",
        "key_findings": ["list of strings"]
    })

    system_prompt = f"""You are an expert data extraction AI. Extract structured information from this {doc_type} document.

EXTRACTION SCHEMA:
{json.dumps(schema, indent=2)}

Rules:
- Extract ALL fields from the schema
- Use null for missing information
- Be precise with dates (ISO 8601 format)
- Extract complete lists, not just examples
- Preserve numerical data exactly as stated

Respond with only the populated JSON matching the schema."""

    try:
        client = st.session_state.gpt41_client
        model = st.session_state.GPT41_DEPLOYMENT

        # Use first 20k chars to avoid token limits
        context = f"FILENAME: '{filename}'\nDOC_TYPE: {doc_type}\n\nDOCUMENT TEXT:\n---\n{full_text[:20000]}\n---"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=4000
        )

        extracted = json.loads(response.choices[0].message.content)
        return extracted
    except Exception as e:
        logger.error(f"Schema extraction failed for {doc_type}: {e}")
        return {"extraction_error": str(e)}


def check_ocr_quality(text: str, page_num: int) -> tuple[bool, float]:
    """
    Check if OCR quality is poor by analyzing text characteristics.
    Returns: (needs_ocr: bool, confidence: float)
    """
    if not text or len(text.strip()) < 50:
        return True, 0.0  # Very little text, likely needs OCR

    # Calculate heuristics
    word_count = len(text.split())
    char_count = len(text)

    if word_count == 0:
        return True, 0.0

    avg_word_length = char_count / word_count

    # Check for garbled text indicators
    special_char_ratio = len(re.findall(r'[^\w\s.,;:!?\-\'"()]', text)) / max(char_count, 1)
    digit_ratio = len(re.findall(r'\d', text)) / max(char_count, 1)

    # Poor quality indicators:
    # - Very high special character ratio (>0.3)
    # - Very short or very long average word length
    # - Very high digit ratio (>0.5) unless it's a table
    poor_quality = (
        special_char_ratio > 0.3 or
        avg_word_length < 2 or
        avg_word_length > 15 or
        (digit_ratio > 0.5 and word_count < 100)
    )

    # Calculate confidence score (0-1)
    confidence = 1.0 - min(1.0, special_char_ratio * 2 + abs(avg_word_length - 5) / 10)

    logger.info(f"Page {page_num}: OCR quality check - confidence={confidence:.2f}, needs_ocr={poor_quality}")
    return poor_quality, confidence


def analyze_page_with_comprehensive_vision(base64_image: str, page_num: int, initial_text: str = "") -> Dict[str, Any]:
    """
    Comprehensive analysis of a PDF page as an image using Vision API.
    Identifies text, tables, charts, images, diagrams, and their locations.
    Returns comprehensive structured data about the page.
    """
    vision_prompt = f"""You are an expert document visual analysis AI. Analyze this page image comprehensively and extract ALL information in structured JSON format.

IMPORTANT: Analyze the image completely and identify:

1. **All Text Content**: Extract every piece of text visible on the page
2. **Tables**: Identify all tables, their structure, headers, and content
3. **Charts/Graphs**: Identify any charts, graphs, plots (bar charts, line graphs, pie charts, etc.)
4. **Images/Photos**: Identify any photographs, illustrations, or images
5. **Diagrams**: Identify technical diagrams, flowcharts, schematics
6. **Visual Elements**: Headers, footers, logos, watermarks, page numbers
7. **Layout Information**: Document structure, sections, columns
8. **Data Relationships**: How visual elements relate to text

Initial text extraction (may have OCR errors):
---
{initial_text[:2000] if initial_text else "[No initial text - perform complete analysis]"}
---

Respond in JSON format with this structure:
{{
  "page_number": {page_num},
  "full_text_content": "Complete corrected text from the page",
  "visual_elements": {{
    "tables": [
      {{
        "location": "description of where on page (top/middle/bottom, left/right)",
        "title": "table title if present",
        "description": "what the table shows",
        "row_count": number,
        "column_count": number,
        "headers": ["col1", "col2", ...],
        "content_summary": "brief summary of data",
        "full_table_text": "complete table text representation"
      }}
    ],
    "charts_and_graphs": [
      {{
        "type": "bar chart|line graph|pie chart|scatter plot|etc",
        "location": "where on page",
        "title": "chart title",
        "description": "what it shows",
        "axis_labels": {{"x": "label", "y": "label"}},
        "data_points": "description of data shown",
        "key_findings": "important insights from the chart"
      }}
    ],
    "images_and_photos": [
      {{
        "location": "where on page",
        "description": "detailed description of image",
        "caption": "image caption if present",
        "relevance": "how it relates to document content"
      }}
    ],
    "diagrams": [
      {{
        "type": "flowchart|schematic|technical diagram|etc",
        "location": "where on page",
        "description": "what the diagram shows",
        "components": ["list", "of", "main", "components"],
        "relationships": "how components connect"
      }}
    ]
  }},
  "layout_structure": {{
    "sections": ["list of sections/headings on page"],
    "columns": number,
    "header_text": "header if present",
    "footer_text": "footer if present",
    "page_number_shown": "page number if visible"
  }},
  "key_information": {{
    "important_facts": ["key facts from this page"],
    "numerical_data": ["important numbers with context"],
    "citations": ["references or citations on page"]
  }}
}}

Be thorough and extract EVERYTHING visible on the page."""

    try:
        client = st.session_state.gpt41_client
        model = st.session_state.GPT41_DEPLOYMENT

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=4000
        )

        analysis = json.loads(response.choices[0].message.content)
        return analysis

    except Exception as e:
        logger.error(f"Vision analysis failed for page {page_num}: {e}")
        return {
            "page_number": page_num,
            "full_text_content": initial_text or "[Vision analysis failed]",
            "visual_elements": {
                "tables": [],
                "charts_and_graphs": [],
                "images_and_photos": [],
                "diagrams": []
            },
            "layout_structure": {},
            "key_information": {},
            "error": str(e)
        }


def process_pdf_with_vision(file_bytes: bytes, filename: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Processes EVERY PDF page as an image using Vision API for comprehensive analysis.
    Returns: (page_analyses: List[Dict], extraction_metadata: Dict)
    """
    page_analyses = []
    metadata = {
        "total_pages": 0,
        "extraction_method": "comprehensive_vision_analysis",
        "processing_time": None
    }

    try:
        import time
        start_time = time.time()

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        metadata["total_pages"] = len(doc)
        progress_bar = st.progress(0, text=f"🔍 Comprehensively analyzing {filename}...")

        for i, page in enumerate(doc):
            progress_text = f"🖼️ Analyzing page {i + 1}/{len(doc)} of {filename} (text, tables, charts, images, diagrams)..."
            progress_bar.progress((i + 1) / len(doc), text=progress_text)

            # Get initial text extraction for reference
            md_text = pymupdf4llm.to_markdown(doc, pages=[i], write_images=False)

            # Convert page to image
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            # Comprehensive vision analysis of the page
            page_analysis = analyze_page_with_comprehensive_vision(base64_image, i + 1, md_text)
            page_analyses.append(page_analysis)

        progress_bar.empty()
        doc.close()

        metadata["processing_time"] = time.time() - start_time
        logger.info(f"PDF comprehensive analysis complete: {metadata['total_pages']} pages in {metadata['processing_time']:.1f}s")
        return page_analyses, metadata

    except Exception as e:
        st.error(f"Failed to process PDF '{filename}': {e}")
        return [], {"error": str(e)}


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

def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """
    Comprehensive document processing with classification, extraction, and metadata.
    Returns: {
        "chunks": List[str],
        "doc_type": str,
        "metadata": Dict,
        "summary": Dict,
        "extracted_data": Dict,
        "processing_metadata": Dict
    }
    """
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    file_ext = os.path.splitext(filename.lower())[1]

    st.info(f"🔍 Starting comprehensive ingestion for '{filename}'...")

    # Step 1: Extract raw text/chunks with comprehensive visual analysis
    if file_ext == ".pdf":
        page_analyses, processing_metadata = process_pdf_with_vision(file_bytes, filename)
        # Extract text from comprehensive analyses
        chunks = [page.get("full_text_content", "") for page in page_analyses]
        # Store the full page analyses for later use
        processing_metadata["page_analyses"] = page_analyses
    elif file_ext == ".docx":
        chunks = process_docx(file_bytes)
        processing_metadata = {"extraction_method": "python_docx"}
    elif file_ext == ".m4a":
        chunks = process_m4a(file_bytes, filename)
        processing_metadata = {"extraction_method": "azure_speech_api"}
    elif file_ext in [".txt", ".md"]:
        chunks = process_text_file(file_bytes, filename)
        processing_metadata = {"extraction_method": "direct_text"}
    elif file_ext in [".mp3", ".wav"]:
        chunks = process_audio_generic(file_bytes, filename)
        processing_metadata = {"extraction_method": "azure_speech_api"}
    else:
        st.warning(f"Unsupported file type: {filename}")
        return {
            "chunks": [],
            "doc_type": "Unknown",
            "metadata": {},
            "summary": {},
            "extracted_data": {},
            "processing_metadata": {"error": "Unsupported file type"}
        }

    if not chunks:
        st.error(f"No content extracted from '{filename}'")
        return {
            "chunks": [],
            "doc_type": "Unknown",
            "metadata": {},
            "summary": {},
            "extracted_data": {},
            "processing_metadata": {"error": "No content extracted"}
        }

    # Step 2: Combine chunks into full text for analysis
    full_text = "\n\n".join(chunks)
    st.success(f"✅ Extracted {len(chunks)} chunks ({len(full_text.split())} words)")

    # Step 3: Document classification
    with st.spinner("📋 Classifying document type..."):
        classification = classify_document_with_llm(full_text, filename)
        doc_type = classification.get("doc_type", "Other")
        st.info(f"📁 Classified as: **{doc_type}** (confidence: {classification.get('confidence', 0):.2f})")

    # Step 4: Extract comprehensive metadata using LLM
    with st.spinner("🔍 Extracting metadata (entities, dates, etc.)..."):
        metadata = extract_comprehensive_metadata_with_llm(full_text, filename)
        entity_summary = []
        if metadata["entities"]["organizations"]:
            entity_summary.append(f"{len(metadata['entities']['organizations'])} organizations")
        if metadata["entities"]["persons"]:
            entity_summary.append(f"{len(metadata['entities']['persons'])} persons")
        if metadata["entities"]["dates"]:
            entity_summary.append(f"{len(metadata['entities']['dates'])} dates")
        if metadata["entities"]["monetary_values"]:
            entity_summary.append(f"{len(metadata['entities']['monetary_values'])} monetary values")

        if entity_summary:
            st.success(f"✅ Extracted: {', '.join(entity_summary)}")

    # Step 5: Generate document summary (map-reduce for long docs)
    with st.spinner("📝 Generating comprehensive summary..."):
        summary = generate_document_summary_with_mapreduce(full_text, filename, doc_type)
        st.success(f"✅ Generated summary with {len(summary.get('key_points', []))} key points")

    # Step 6: Extract structured data by document type
    with st.spinner(f"📊 Extracting structured data for {doc_type}..."):
        extracted_data = extract_document_schema_by_type(full_text, filename, doc_type)
        if "extraction_error" not in extracted_data:
            st.success(f"✅ Extracted {len(extracted_data)} structured fields")

    # Combine all results with visual analysis data
    result = {
        "chunks": chunks,
        "doc_type": doc_type,
        "classification": classification,
        "metadata": metadata,
        "summary": summary,
        "extracted_data": extracted_data,
        "processing_metadata": processing_metadata,
        "filename": filename,
        "processed_at": datetime.utcnow().isoformat()
    }

    # Add comprehensive visual element summary for PDFs
    if file_ext == ".pdf" and "page_analyses" in processing_metadata:
        visual_summary = {
            "total_tables": 0,
            "total_charts": 0,
            "total_images": 0,
            "total_diagrams": 0,
            "pages_with_tables": [],
            "pages_with_charts": [],
            "pages_with_images": [],
            "pages_with_diagrams": []
        }

        for page_analysis in processing_metadata["page_analyses"]:
            page_num = page_analysis.get("page_number", 0)
            visual_elements = page_analysis.get("visual_elements", {})

            if visual_elements.get("tables"):
                visual_summary["total_tables"] += len(visual_elements["tables"])
                visual_summary["pages_with_tables"].append(page_num)

            if visual_elements.get("charts_and_graphs"):
                visual_summary["total_charts"] += len(visual_elements["charts_and_graphs"])
                visual_summary["pages_with_charts"].append(page_num)

            if visual_elements.get("images_and_photos"):
                visual_summary["total_images"] += len(visual_elements["images_and_photos"])
                visual_summary["pages_with_images"].append(page_num)

            if visual_elements.get("diagrams"):
                visual_summary["total_diagrams"] += len(visual_elements["diagrams"])
                visual_summary["pages_with_diagrams"].append(page_num)

        result["visual_elements_summary"] = visual_summary

        # Display summary to user
        visual_info = []
        if visual_summary["total_tables"] > 0:
            visual_info.append(f"{visual_summary['total_tables']} tables")
        if visual_summary["total_charts"] > 0:
            visual_info.append(f"{visual_summary['total_charts']} charts/graphs")
        if visual_summary["total_images"] > 0:
            visual_info.append(f"{visual_summary['total_images']} images")
        if visual_summary["total_diagrams"] > 0:
            visual_info.append(f"{visual_summary['total_diagrams']} diagrams")

        if visual_info:
            st.success(f"📊 Visual elements identified: {', '.join(visual_info)}")

    st.success(f"🎉 Comprehensive ingestion complete for '{filename}'!")
    return result


def prepare_chunks_for_cosmos(ingestion_result: Dict[str, Any], original_filename: str = None) -> List[Dict[str, Any]]:
    """
    Formats comprehensive ingestion results into JSON for Cosmos DB.
    Now accepts full ingestion_result dict with metadata, summary, etc.
    For backward compatibility, also accepts old format (list of chunks).
    """
    # Handle backward compatibility - if passed a list, convert to old format
    if isinstance(ingestion_result, list):
        chunks = ingestion_result
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

    # New comprehensive format
    chunks = ingestion_result.get("chunks", [])
    doc_type = ingestion_result.get("doc_type", "Unknown")
    classification = ingestion_result.get("classification", {})
    metadata = ingestion_result.get("metadata", {})
    summary = ingestion_result.get("summary", {})
    extracted_data = ingestion_result.get("extracted_data", {})
    processing_metadata = ingestion_result.get("processing_metadata", {})
    visual_elements_summary = ingestion_result.get("visual_elements_summary", {})
    filename = ingestion_result.get("filename", original_filename or "unknown")
    processed_at = ingestion_result.get("processed_at", datetime.utcnow().isoformat())

    parent_doc_id = f"doc_{uuid.uuid4()}"
    output_chunks = []

    # Create parent document with comprehensive metadata
    parent_doc = {
        "id": parent_doc_id,
        "doc_type": "parent_document",
        "filename": filename,
        "document_type": doc_type,
        "classification": classification,
        "summary": summary,
        "extracted_data": extracted_data,
        "metadata": metadata,
        "visual_elements_summary": visual_elements_summary,
        "processing_metadata": processing_metadata,
        "processed_at": processed_at,
        "total_chunks": len(chunks)
    }
    output_chunks.append(parent_doc)

    # Get page analyses if available (for PDFs)
    page_analyses = processing_metadata.get("page_analyses", [])

    # Create chunk documents with references to parent
    for i, chunk_content in enumerate(chunks):
        chunk_doc = {
            "id": f"{parent_doc_id}_chunk_{i}",
            "parent_document_id": parent_doc_id,
            "content": chunk_content,
            "chunk_index": i,
            "doc_type": "chunk",
            "document_type": doc_type,  # Include for filtering
            "filename": filename,
            "metadata": {
                "original_filename": filename,
                "chunk_index": i,
                "document_type": doc_type,
                "processed_at": processed_at
            }
        }

        # Add page-level visual analysis if available
        if i < len(page_analyses):
            page_analysis = page_analyses[i]
            chunk_doc["page_analysis"] = {
                "page_number": page_analysis.get("page_number"),
                "visual_elements": page_analysis.get("visual_elements", {}),
                "layout_structure": page_analysis.get("layout_structure", {}),
                "key_information": page_analysis.get("key_information", {})
            }

        output_chunks.append(chunk_doc)

    logger.info(f"Prepared {len(output_chunks)} documents for Cosmos (1 parent + {len(chunks)} chunks with visual analysis)")
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

def _sniff_audio_format(raw: bytes, ext_hint: str) -> str:
    """
    Look at magic bytes and choose the best format for ffmpeg/pydub.
    Returns one of: 'wav','webm','ogg','m4a','mp3' (falls back to ext_hint).
    """
    try:
        if raw.startswith(b'RIFF') and raw[8:12] == b'WAVE':
            return "wav"
        if raw.startswith(b'OggS'):
            return "ogg"
        # Matroska/WebM EBML header: 0x1A 0x45 0xDF 0xA3
        if raw.startswith(b'\x1A\x45\xDF\xA3'):
            return "webm"
        # mp4/m4a
        if len(raw) > 12 and raw[4:8] == b'ftyp':
            return "m4a"
        # very rough mp3 sniff
        if raw.startswith(b'ID3') or raw[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
            return "mp3"
    except Exception:
        pass
    return ext_hint.lower()



def azure_fast_transcribe_wav_bytes(wav_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Uses Azure Speech fast transcription (synchronous) to turn a WAV (16kHz, mono, PCM16) into text.
    Returns the recognized text ("" if nothing recognized).
    """
    speech_key = st.session_state.SPEECH_KEY
    speech_region = st.session_state.SPEECH_REGION
    if not all([speech_key, speech_region]):
        st.error("Azure Speech Service credentials are not configured.")
        return ""

    endpoint = f"https://{speech_region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"
    headers = {'Ocp-Apim-Subscription-Key': speech_key}
    definition = {"locales": ["en-US"]}

    files = {
        'audio': (filename, BytesIO(wav_bytes), 'audio/wav'),
        'definition': (None, json.dumps(definition), 'application/json')
    }

    try:
        resp = requests.post(endpoint, headers=headers, files=files, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        phrases = data.get("combinedPhrases", [])
        if not phrases:
            return ""
        return phrases[0].get("text", "") or ""
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Azure Speech HTTP error: {http_err}")
        try:
            st.error(f"Response body: {resp.text}")
        except Exception:
            pass
        return ""
    except Exception as e:
        st.error(f"Azure Speech error: {e}")
        return ""


def coerce_audio_bytes_from_any(audio_in: bytes | str) -> bytes:
    """
    Accepts raw bytes or a data URL string like 'data:audio/webm;base64,...' and returns raw bytes.
    """
    if isinstance(audio_in, bytes):
        return audio_in
    if isinstance(audio_in, str):
        # data URL?
        if audio_in.startswith("data:audio/"):
            try:
                header, b64 = audio_in.split(",", 1)
                return base64.b64decode(b64)
            except Exception:
                # If it isn't base64 after all, fall through and try utf-8 bytes
                pass
        # assume it's a plain base64 string or text -> try base64
        try:
            return base64.b64decode(audio_in)
        except Exception:
            # final fallback to utf-8 bytes
            return audio_in.encode("utf-8", errors="ignore")
    return b""


def ensure_16k_mono_wav(audio_bytes_or_str: bytes | str, ext_hint: str = "wav") -> bytes:
    """
    Convert any supported input (wav/mp3/m4a/webm/ogg or data URL) to 16kHz mono PCM16 WAV.
    Robust sniffing prevents 'invalid RIFF header' errors from WebM/Opus microphone recordings.
    """
    raw = coerce_audio_bytes_from_any(audio_bytes_or_str)

    # derive format from magic bytes first (overrides wrong hints)
    fmt = _sniff_audio_format(raw, ext_hint)

    try:
        snd = AudioSegment.from_file(BytesIO(raw), format=fmt)
    except Exception as e:
        # Try a few alternates if sniffing fails
        for alt in ("wav", "webm", "ogg", "mp3", "m4a"):
            if alt == fmt:
                continue
            try:
                snd = AudioSegment.from_file(BytesIO(raw), format=alt)
                break
            except Exception:
                snd = None
        if snd is None:
            st.error(
                "Audio convert error: Decoding failed. Most browsers produce WebM/Opus from the mic. "
                f"Tried '{fmt}' and common fallbacks; underlying error: {e}"
            )
            return b""

    try:
        buf = BytesIO()
        snd.set_frame_rate(16000).set_channels(1).set_sample_width(2).export(
            buf, format="wav", codec="pcm_s16le"
        )
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"Audio convert error: failed to export WAV (16k/mono). {e}")
        return b""



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
MAX_LOOPS = 50  # Increased from 20 to allow for complete, comprehensive reports

# =========================== SCRATCHPAD MANAGER ===========================
class ScratchpadManager:
    """
    Manages multiple specialized scratchpads for collaborative agent work.
    Each scratchpad supports full CRUD operations (Create, Read, Update, Delete).

    DESIGN PHILOSOPHY:
    - Agents work on different pads in parallel (async collaboration)
    - Line-level editing prevents rewriting entire documents
    - Each pad serves a specific purpose in the workflow
    - Final output is assembled from all pads

    EXAMPLE WORKFLOW:
    1. Orchestrator assigns tasks to multiple agents
    2. Tool Agent searches KB → writes findings to RESEARCH pad
    3. Tool Agent creates table → writes to TABLES pad
    4. Writer reads RESEARCH pad → begins writing to OUTPUT pad using line-level inserts
    5. Engineer analyzes data → adds technical details to OUTPUT pad (insert at specific line)
    6. Writer reviews OUTPUT pad with scratchpad_get_lines() → edits specific lines
    7. Validator reads final OUTPUT pad → approves
    8. System returns OUTPUT pad content to user

    SCRATCHPAD TYPES:
    - output: Final answer being assembled (main deliverable)
    - research: Raw findings, search results, facts
    - tables: Formatted tables (markdown)
    - plots: Plot specifications and data
    - outline: Initial plan, structure, TODOs
    - data: Structured data (JSON, lists)
    - log: Agent actions and decisions (workflow history)

    AGENT OPERATIONS EXAMPLES:
    ```
    # Tool Agent after search:
    scratchpad_write("research", "kb_search_results", search_output, mode="append")

    # Writer creating outline:
    scratchpad_write("outline", "structure", "1. Introduction\\n2. Analysis\\n3. Conclusion")

    # Writer building output:
    scratchpad_write("output", "introduction", "# Analysis Report\\n\\nThis report examines...")
    scratchpad_insert_lines("output", "introduction", 3, "Key findings include:")

    # Engineer adding to output:
    scratchpad_get_lines("output", "introduction")  # See what's there
    scratchpad_insert_lines("output", "introduction", 5, "Technical detail: Performance improved by 40%")

    # Writer reviewing and editing:
    scratchpad_replace_lines("output", "introduction", 5, 5, "Technical detail: Performance improved by 42%")

    # Tool Agent creating table in TABLES pad:
    table = to_markdown_table([{"metric": "speed", "value": "100ms"}])
    scratchpad_write("tables", "performance_metrics", table)

    # Writer pulling table into output:
    table_content = scratchpad_read("tables", "performance_metrics")
    scratchpad_insert_lines("output", "analysis", -1, table_content)
    ```
    """
    def __init__(self, db_path: str = "scratchpad.db", session_id: str = None):
        self.db_path = db_path
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._init_database()

        # Initialize pads in memory and DB
        pad_types = ["output", "research", "tables", "plots", "outline", "data", "log"]
        for pad_type in pad_types:
            self._ensure_pad_exists(pad_type)

    def _init_database(self):
        """Initialize SQLite database with tables for pads, sections, and version history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Pads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pads (
                pad_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                pad_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, pad_name)
            )
        """)

        # Sections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sections (
                section_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pad_id INTEGER NOT NULL,
                section_name TEXT NOT NULL,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pad_id) REFERENCES pads(pad_id),
                UNIQUE(pad_id, section_name)
            )
        """)

        # Version history table (tracks every change with diffs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_history (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                section_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                old_content TEXT,
                new_content TEXT,
                diff TEXT,
                agent_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (section_id) REFERENCES sections(section_id)
            )
        """)

        conn.commit()
        conn.close()

    def _ensure_pad_exists(self, pad_name: str):
        """Ensure pad exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO pads (session_id, pad_name)
            VALUES (?, ?)
        """, (self.session_id, pad_name))
        conn.commit()
        conn.close()

    def _get_pad_id(self, pad_name: str) -> int:
        """Get pad_id for a given pad_name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pad_id FROM pads WHERE session_id = ? AND pad_name = ?
        """, (self.session_id, pad_name))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def _get_section_id(self, pad_name: str, section_name: str) -> int:
        """Get section_id for a given section"""
        pad_id = self._get_pad_id(pad_name)
        if not pad_id:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT section_id FROM sections WHERE pad_id = ? AND section_name = ?
        """, (pad_id, section_name))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def _generate_diff(self, old_content: str, new_content: str) -> str:
        """Generate unified diff between old and new content"""
        old_lines = old_content.split('\n') if old_content else []
        new_lines = new_content.split('\n') if new_content else []

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm='',
            fromfile='before',
            tofile='after'
        )

        return '\n'.join(diff)

    def _save_version(self, section_id: int, operation: str, old_content: str,
                     new_content: str, agent_name: str = "system"):
        """Save a version with diff to history"""
        diff = self._generate_diff(old_content, new_content)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO version_history (section_id, operation, old_content, new_content, diff, agent_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (section_id, operation, old_content, new_content, diff, agent_name))
        conn.commit()
        conn.close()

        return diff

    def _format_diff_for_display(self, diff: str) -> str:
        """Format diff with color codes for display"""
        if not diff:
            return "No changes"

        lines = diff.split('\n')
        formatted = []
        for line in lines:
            if line.startswith('+++') or line.startswith('---'):
                continue  # Skip file headers
            elif line.startswith('+'):
                formatted.append(f"+ {line[1:]}")  # Added line
            elif line.startswith('-'):
                formatted.append(f"- {line[1:]}")  # Removed line
            elif line.startswith('@@'):
                formatted.append(f"\n{line}")  # Line number info
            else:
                formatted.append(f"  {line}")  # Context line

        return '\n'.join(formatted[:50])  # Limit to first 50 lines

    def read_pad(self, pad_name: str) -> dict:
        """Read entire scratchpad from database"""
        pad_id = self._get_pad_id(pad_name)
        if not pad_id:
            return {"error": f"Scratchpad '{pad_name}' does not exist"}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT section_name, content FROM sections WHERE pad_id = ?
        """, (pad_id,))
        sections = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        return {"pad_name": pad_name, "sections": sections}

    def read_section(self, pad_name: str, section_name: str) -> str:
        """Read specific section from a scratchpad"""
        section_id = self._get_section_id(pad_name, section_name)
        if not section_id:
            return f"Error: Section '{section_name}' not found in '{pad_name}'"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM sections WHERE section_id = ?", (section_id,))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else ""

    def write_section(self, pad_name: str, section_name: str, content: str, mode: str = "replace", agent_name: str = "agent") -> str:
        """
        Write to a section in a scratchpad with diff tracking.
        Modes: 'replace', 'append', 'prepend'
        Returns: Diff showing changes
        """
        pad_id = self._get_pad_id(pad_name)
        if not pad_id:
            return f"Error: Scratchpad '{pad_name}' does not exist"

        # Get existing content
        old_content = self.read_section(pad_name, section_name) or ""

        # Apply mode
        if mode == "replace":
            new_content = content
        elif mode == "append":
            new_content = old_content + "\n" + content if old_content else content
        elif mode == "prepend":
            new_content = content + "\n" + old_content if old_content else content
        else:
            new_content = content

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sections (pad_id, section_name, content)
            VALUES (?, ?, ?)
            ON CONFLICT(pad_id, section_name)
            DO UPDATE SET content = ?, last_modified = CURRENT_TIMESTAMP
        """, (pad_id, section_name, new_content, new_content))
        conn.commit()

        # Get section_id for version tracking
        section_id = self._get_section_id(pad_name, section_name)

        # Save version with diff
        diff = self._save_version(section_id, f"write_{mode}", old_content, new_content, agent_name)

        conn.close()

        # Format diff for display
        diff_display = self._format_diff_for_display(diff)
        return f"✅ Wrote to '{pad_name}.{section_name}' (mode: {mode})\n\n{diff_display}"

    def list_sections(self, pad_name: str) -> list:
        """List all sections in a scratchpad"""
        pad_id = self._get_pad_id(pad_name)
        if not pad_id:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT section_name FROM sections WHERE pad_id = ?", (pad_id,))
        sections = [row[0] for row in cursor.fetchall()]
        conn.close()
        return sections

    def get_full_content(self, pad_name: str) -> str:
        """Get the full assembled content of a scratchpad"""
        pad_data = self.read_pad(pad_name)
        if "error" in pad_data:
            return pad_data["error"]

        sections = pad_data.get("sections", {})

        # Sort sections: numbered sections first (1_, 2_, etc.), then others alphabetically
        def sort_key(item):
            name = item[0]
            # Extract leading number if present (e.g., "1_exec_summary" → 1)
            if name and name[0].isdigit():
                num_part = ""
                for char in name:
                    if char.isdigit():
                        num_part += char
                    else:
                        break
                return (0, int(num_part))  # Priority 0 for numbered, sort by number
            else:
                return (1, name)  # Priority 1 for non-numbered, sort alphabetically

        sorted_sections = sorted(sections.items(), key=sort_key)

        # Deduplicate: prefer numbered sections over non-numbered versions
        # E.g., if both "1_executive_summary" and "Executive Summary" exist, keep only "1_executive_summary"
        seen_topics = set()
        deduplicated = []
        for name, content in sorted_sections:
            # Normalize name to detect duplicates: "1_executive_summary" → "executive summary"
            normalized = name.lower().lstrip('0123456789_').replace('_', ' ').strip()

            if normalized not in seen_topics:
                seen_topics.add(normalized)
                deduplicated.append((name, content))
            # If normalized topic already seen, skip (prefer numbered version which comes first)

        # Don't add section name as header if content already starts with a header
        parts = []
        for name, content in deduplicated:
            if content.strip().startswith('#'):
                # Content already has headers, don't add section name
                parts.append(content)
            else:
                # Content doesn't have headers, add section name as header
                parts.append(f"## {name}\n{content}")

        return "\n\n".join(parts)

    def get_all_pads_summary(self) -> str:
        """Get a summary of all scratchpads and their sections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.pad_name, COUNT(s.section_id) as section_count
            FROM pads p
            LEFT JOIN sections s ON p.pad_id = s.pad_id
            WHERE p.session_id = ?
            GROUP BY p.pad_name
        """, (self.session_id,))

        summary = "# Scratchpad Summary\n\n"
        for row in cursor.fetchall():
            pad_name, section_count = row
            sections = self.list_sections(pad_name)
            summary += f"**{pad_name}**: {section_count} sections - {sections}\n"

        conn.close()
        return summary

    def get_version_history(self, pad_name: str, section_name: str, limit: int = 10) -> list:
        """Get version history for a section with diffs"""
        section_id = self._get_section_id(pad_name, section_name)
        if not section_id:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT operation, diff, agent_name, timestamp
            FROM version_history
            WHERE section_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (section_id, limit))

        history = []
        for row in cursor.fetchall():
            history.append({
                "operation": row[0],
                "diff": row[1],
                "agent": row[2],
                "timestamp": row[3]
            })

        conn.close()
        return history

    def delete_section(self, pad_name: str, section_name: str) -> bool:
        """
        Delete a section from a pad.

        Args:
            pad_name: Name of the pad
            section_name: Name of the section to delete

        Returns:
            True if section was deleted, False if not found
        """
        section_id = self._get_section_id(pad_name, section_name)
        if not section_id:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete version history first (foreign key constraint)
        cursor.execute("DELETE FROM version_history WHERE section_id = ?", (section_id,))

        # Delete the section
        cursor.execute("DELETE FROM sections WHERE section_id = ?", (section_id,))

        conn.commit()
        conn.close()
        return True

    def cleanup_formatting(self, pad_name: str, section_name: str) -> str:
        """
        Clean common formatting issues in a section (LaTeX escaping, spacing).

        Fixes:
        - LaTeX-style dollar escaping: $1.3trillion → $1.3 trillion
        - Escaped special chars: F−35 → F-35
        - Missing spaces after punctuation
        - Multiple consecutive spaces

        Args:
            pad_name: Name of the pad
            section_name: Name of the section to clean

        Returns:
            Status message with diff showing changes
        """
        content = self.read_section(pad_name, section_name)
        if not content or "Error:" in content:
            return f"❌ Cannot clean - section not found: {pad_name}.{section_name}"

        original = content

        # Fix common LaTeX escaping issues
        import re

        # Fix dollar amounts: $1.3trillion → $1.3 trillion
        content = re.sub(r'\$(\d+\.?\d*)(billion|trillion|million)', r'$\1 \2', content)

        # Fix escaped minus signs: F−35 → F-35 (U+2212 to hyphen-minus)
        content = content.replace('−', '-')

        # Fix escaped newlines in content
        content = content.replace('\\n', '\n')

        # Fix multiple spaces
        content = re.sub(r'  +', ' ', content)

        # Fix space before punctuation
        content = re.sub(r' ([.,;:])', r'\1', content)

        # Fix missing space after punctuation (but not in numbers like "1.3" or URLs)
        content = re.sub(r'([.,;:])([A-Za-z])', r'\1 \2', content)

        if content == original:
            return f"✅ Section '{pad_name}.{section_name}' already clean - no changes needed"

        # Write cleaned content back
        self.write_section(pad_name, section_name, content, mode="replace", agent_name="Formatter")

        return f"✅ Cleaned formatting in '{pad_name}.{section_name}' - fixed LaTeX escaping and spacing"

TOOL_DEFINITIONS = f"""
AVAILABLE TOOLS:

**Data & Search:**
- search_knowledge_base(keywords: list[str], semantic_query_text: str, rank_limit: int): Executes a **broad keyword search** against the knowledge base (RAG). **Keywords are combined with OR logic.** Use this to find documents or data points. Results are automatically cached for later retrieval.
- get_cached_search_results(keywords: list[str] = None): Retrieve search results from previous loops. If keywords specified, returns that specific search. If no keywords, returns summary of all cached searches. **USE THIS to access results from searches done in earlier loops.**

**Math & Conversion:**
- calc(expression: str): Safely evaluates a mathematical expression (e.g., "12 * (3 + 5)").
- lbs_to_kg(pounds: float): Converts pounds to kilograms.
- kg_to_lbs(kg: float): Converts kilograms to pounds.

**Data Processing:**
- json_parse(text: str): Parses a JSON string and returns a structured object or error.
- to_markdown_table(rows: list[dict]): Converts a Python list of dictionaries into a clean Markdown table.
- schema_sketch(json_list_text: str): Infers field names and data types from a JSON list of objects.
- to_csv(rows: list[dict]): Converts a list of dictionaries into a comma-separated values (CSV) string, including a header.

**Formatting:**
- format_latex(latex_string: str): Formats a raw LaTeX mathematical expression for display. Use for equations. Example: "c = \\\\sqrt{{a^2 + b^2}}"
- format_list(items: list[str], style: str): Formats a list of strings as a Markdown list. `style` can be 'bullet' or 'numbered'.
- format_code(code_string: str, language: str): Formats text as a Markdown code block. `language` defaults to 'python'.
- format_blockquote(text: str): Formats text as a Markdown blockquote.
- format_link(text: str, url: str): Creates a Markdown hyperlink.

**SCRATCHPAD TOOLS (Multiple Collaborative Pads):**
Available pads: output, research, tables, plots, outline, data, log

**Reading:**
- scratchpad_summary(): Get overview of all scratchpads and their sections.
- scratchpad_list(pad_name: str): List all sections in a specific pad.
- scratchpad_read(pad_name: str, section_name: str = None): Read a section or entire pad.
- scratchpad_get_lines(pad_name: str, section_name: str, start_line: int = None, end_line: int = None): Get specific lines with line numbers (like Claude Code).

**Writing:**
- scratchpad_write(pad_name: str, section_name: str, content: str, mode: str = "replace"): Write to a section. Modes: 'replace', 'append', 'prepend'.

**Line-level Editing (like Claude Code):**
- scratchpad_insert_lines(pad_name: str, section_name: str, line_number: int, content: str): Insert at line (1-indexed). Use 0 to prepend, -1 to append.
- scratchpad_delete_lines(pad_name: str, section_name: str, start_line: int, end_line: int = None): Delete line range.
- scratchpad_replace_lines(pad_name: str, section_name: str, start_line: int, end_line: int, content: str): Replace line range.

**Advanced:**
- scratchpad_edit(pad_name: str, section_name: str, old_text: str, new_text: str): Find/replace text in a section.
- scratchpad_delete(pad_name: str, section_name: str): Delete entire section.
- scratchpad_merge(pad_name: str, section_names: list[str], new_section_name: str): Merge multiple sections.
- scratchpad_history(pad_name: str, section_name: str, limit: int = 10): View version history with diffs showing +added and -removed lines.
- scratchpad_cleanup_formatting(pad_name: str, section_name: str): Fix common formatting issues (LaTeX escaping, spacing). Call after writing content to ensure clean markdown.

**SCRATCHPAD FEATURES:**
✅ All changes persisted to scratchpad.db SQLite database
✅ Every edit tracked with full version history
✅ Diffs show +added and -removed lines (like git diff)
✅ Agents can see who changed what and when

**SCRATCHPAD WORKFLOW:**
1. Agents work in parallel on different pads (research in 'research', tables in 'tables', final output in 'output')
2. Use line-level operations to efficiently edit without rewriting entire sections
3. Every write operation returns a diff showing changes
4. Use scratchpad_history() to see what was changed over time
5. Research pad collects findings, output pad builds final answer, tables/plots hold visualizations
6. Use scratchpad_summary() to see what other agents have written
7. Build collaboratively - read what others wrote, add to it, edit it incrementally

OUTPUT FORMAT:
The Agent MUST respond with ONLY a JSON object representing either a tool call, a structured plan, a draft, or a final response.
""" 

# (Around line 1787)
AGENT_PERSONAS = {
        "Orchestrator": """You are an expert project manager and orchestrator. Your role is to understand the user's goal and guide your team of specialist agents to achieve it.

    Your team consists of:
    - Tool Agent: Executes searches using search_knowledge_base AND saves findings to RESEARCH pad using scratchpad_write. This is your PRIMARY agent for data gathering and extraction.
    - Query Refiner: ONLY use when searches returned insufficient/irrelevant results. Analyzes what went wrong and creates refined search keywords. NOT for extracting or saving data.
    - Engineer: Best for creating tables, charts, and structured data visualizations in the TABLES pad. Also handles technical analysis or debugging.
    - Writer: Best for composing narrative prose sections and refining the final answer. Uses RESEARCH pad data and TABLES pad visualizations to build OUTPUT pad.
    - Validator: Best for reviewing the work of other agents, checking for factual accuracy, and ensuring the final answer fully addresses the user's question.
    - Supervisor: Best for evaluating if the user's question has been adequately answered and if the team should finish.

    CRITICAL INSTRUCTIONS:
    1.  Review the LOG scratchpad, which contains the user's goal and the history of steps taken. Use scratchpad_summary() to see all pads.
    2.  **LOOP AWARENESS**: You will be informed of your current loop number and remaining loops. Plan accordingly to finish BEFORE the final loop.
    3.  **MULTI-PAD WORKFLOW**: Direct agents to work on different scratchpads in parallel:
        - OUTLINE pad: Initial plan and structure
        - RESEARCH pad: Gathered facts, search results, data points
        - TABLES pad: All formatted tables
        - PLOTS pad: Plot specifications and data
        - DATA pad: Raw structured data (JSON, lists)
        - OUTPUT pad: Final answer being assembled
        - LOG pad: Agent actions and decisions
    4.  **PARALLEL EXECUTION IS DEFAULT**: You MUST delegate multiple independent tasks simultaneously whenever possible:
        - Single task format: {{"agent": "AgentName", "task": "description"}} - USE ONLY when tasks are dependent
        - **Multiple tasks format (CHOOSE ONE)**:
          * Array format: [{{"agent": "Agent1", "task": "task1"}}, {{"agent": "Agent2", "task": "task2"}}]
          * Wrapped format: {{"agent": "PARALLEL", "tasks": [{{"agent": "Agent1", "task": "task1"}}, {{"agent": "Agent2", "task": "task2"}}]}}
        - **ALWAYS look for parallelization opportunities**:
          * User asks about "problem and solution" → Delegate 2+ parallel searches (problem, solution, context)
          * Document needs multiple sections → Delegate parallel research tasks for each section
          * Multiple information needs identified → ALL should be searched in parallel
          * Report writing task → Delegate multiple parallel searches to gather ALL needed data at once
        - **FORBIDDEN**: Sequential searches when parallel searches would work. ALWAYS delegate 3+ parallel tasks for comprehensive requests.
    5.  **PROPER WORKFLOW SEQUENCING - CRITICAL**:
        - **Phase 1 - Research**: Tool Agent searches → Tool Agent MUST save findings to RESEARCH pad
        - **Phase 2 - Writing**: Writer reads RESEARCH pad → Writer builds OUTPUT using scratchpad_insert_lines
        - **NEVER**: Let Writer create empty placeholder sections
        - **NEVER**: Skip the "save to RESEARCH pad" step between searching and writing
        - **Correct sequence example**:
          * Loop 1: Delegate 5 parallel searches to Tool Agent
          * Loop 2: Verify Tool Agent saved results to RESEARCH pad (if not, tell Tool Agent to save them)
          * Loop 3: Delegate to Writer to read RESEARCH and build OUTPUT with real content
        - **Wrong sequence (DO NOT DO THIS)**:
          * ❌ Tool Agent searches → Writer immediately writes without saved research
          * ❌ Writer creates empty sections like scratchpad_write("research", "section", "")
    6.  **CRITICAL: ENFORCE SEARCH→SAVE WORKFLOW**:
        - **CHECK CACHED SEARCH RESULTS SECTION**: You can see previews of all searches from previous loops
        - **FORBIDDEN**: Delegating more searches immediately after searches (Loop N: search → Loop N+1: search ❌)
        - **REQUIRED**: After searches, extract findings (Loop N: search → Loop N+1: extract & save meaningful results ✅)
        - **Adaptive approach - You can SEE the data**:
          * Look at "CACHED SEARCH RESULTS" section above - it shows you what each search found
          * If search preview shows useful data → Delegate "Extract [specific insights] from the cached results and save to RESEARCH pad with section name [topic]"
          * If search preview shows nothing useful → Skip that search, don't delegate extraction
          * Focus on MEANINGFUL content based on what you can see in the previews
        - **Example adaptive workflow**:
          * Loop 1: Delegate 4 searches to **Tool Agent**
          * Loop 2: Review "CACHED SEARCH RESULTS" section → See maintainer and cost searches returned useful data → Delegate to **Tool Agent**: "Save maintainer shortage statistics to RESEARCH.maintainer_shortage - found 15% shortage, readiness 54%" + "Save cost projections to RESEARCH.costs - lifecycle $1.3T"
          * Loop 3: Check RESEARCH pad → 2 sections saved → Delegate **Writer** to start building OUTPUT with available data
          * Loop 4: Writer adds to OUTPUT → Review for gaps → Delegate more targeted searches to **Tool Agent** if needed
        - **Example parallel delegation**:
          ```json
          {{"agent": "PARALLEL", "tasks": [
            {{"agent": "Tool Agent", "task": "Search F-35 manpower problems"}},
            {{"agent": "Tool Agent", "task": "Search F-35 proposed solutions"}},
            {{"agent": "Tool Agent", "task": "Search F-35 cost impact"}},
            {{"agent": "Tool Agent", "task": "Search F-35 vendor landscape"}},
            {{"agent": "Writer", "task": "Create outline structure in OUTLINE pad"}}
          ]}}
          ```
    7.  **ITERATIVE WORKFLOW - BUILD AS YOU GO**:
        a) Start with core searches (3-5 most important topics)
        b) As results come in, delegate extraction of meaningful findings
        c) As RESEARCH pad fills, delegate Writer to start building OUTPUT with available data
        d) Review OUTPUT → Identify gaps → Delegate targeted searches for gaps
        e) Continue iterating: more research → more writing → identify new gaps → more searches
        f) **Don't wait for "all research complete"** - build continuously
    8.  If the scratchpads are empty or only contain the user's goal, your FIRST STEP is to:
        a) Analyze the user's question and identify core information needs (start with 3-5, not 10+)
        b) Delegate parallel searches to Tool Agent for core topics
        c) Optionally delegate to Writer to create outline in parallel
    9.  On every subsequent turn, review the plan and the completed steps. DO NOT repeat tasks.
    10. **BE COMPREHENSIVE BUT ITERATIVE**: Quality over quantity. Better to do 3 searches → extract findings → write → identify gaps → 3 more searches, than to do 15 searches all at once without extraction.
    11. **PROGRESS CHECK**: After 3-4 steps, or when you have gathered significant information, delegate to the Supervisor to evaluate if the question is adequately answered.
    12. **REFINEMENT WORKFLOW - MANDATORY CONTINUATION**: If Supervisor says "NEED_MORE_WORK" with specific gaps:
        a) **DO NOT STOP** - the report is incomplete and must continue
        b) Identify what's missing from Supervisor's feedback (e.g., "missing sections 2, 3, 5, 8-10, 12-13")
        c) Delegate to Writer to draft the missing sections in parallel (use PARALLEL tasks for multiple sections)
        d) If data is needed for missing sections, delegate searches to Tool Agent first
        e) After missing sections are drafted, check with Supervisor again
        f) **NEVER finish if Supervisor said "NEED_MORE_WORK"** - always continue until Supervisor says "READY_TO_FINISH"
    13. **VALIDATION WORKFLOW**: If Supervisor says "READY_TO_FINISH":
        a) **CRITICAL**: DO NOT finish immediately - delegate formatting/polish tasks first
        b) **CHECK FOR ORPHANED SECTIONS**: Call scratchpad_list("output") and check if ANY sections exist that are NOT part of the final numbered structure. If found, delegate to Writer: "Merge content from orphaned sections [list names] into the appropriate numbered sections in OUTPUT pad"
        c) Delegate to Writer: "Review OUTPUT pad - read each section, fix any LaTeX escaping ($336billion → $336 billion), add proper markdown headers, ensure clean formatting"
        d) Delegate to Writer: "Read OUTPUT sections 1-by-1 and add missing paragraph breaks for readability"
        e) Then delegate to Validator to ensure the answer fully addresses the user's question AND is properly formatted
        f) If Validator approves, proceed to FINISH
        g) If Validator identifies issues, address them or refine further
    14. **TABLES & VISUALIZATIONS PHASE** (before formatting):
        - **REQUIRED**: For reports with numerical data, delegate to Engineer to create tables BEFORE Writer polishes OUTPUT
        - Check RESEARCH pad for quantitative data: cost projections, staffing gaps, timelines, comparisons
        - Delegate to Engineer: "Create tables in TABLES pad for: [list specific data sets that need tabular format]"
        - Examples: "Create table comparing maintainer shortfalls by service", "Create cost projection table FY2024-2040"
        - Writer will reference these tables in narrative sections
    15. **FORMATTING & POLISH PHASE** (after tables are created):
        - Delegate to Writer: "Review and reformat OUTPUT sections - fix any LaTeX escaping, add narrative transitions between paragraphs, replace bullet-heavy sections with flowing prose"
        - Call scratchpad_cleanup_formatting() for each OUTPUT section
        - Ensure Writer integrates tables with narrative: "Refer to Table 1 in section 3, explaining what it shows"
        - **DO NOT skip this phase** - unformatted or bullet-heavy output is unacceptable
    16. **WHEN TO FINISH**:
        - If Supervisor says "READY_TO_FINISH" and OUTPUT pad has comprehensive sections (typically 5+ sections, 5000+ words total), delegate to **"FINISH_NOW"** agent with task: "Read the OUTPUT pad and deliver the final report to the user"
        - If Validator confirms the final answer is complete AND properly formatted, delegate to **"FINISH_NOW"** agent
        - **CRITICAL**: The agent name is "FINISH_NOW" not "FINISH" - use exact name
        - **The final answer given in the "task" field MUST BE STRICTLY GROUNDED IN THE SCRATCHPAD CONTENT. DO NOT ADD SPECULATION OR UNGROUNDED KNOWLEDGE.**
    17. **URGENCY**: If you have 2 or fewer loops remaining and have ANY useful information, immediately delegate to Supervisor to evaluate readiness to finish.
    """,

        "Tool Agent": f"""You are an expert data agent responsible for executing tools and managing scratchpads. You MUST respond in JSON format.

    ═══════════════════════════════════════════════════════════════
    🚨 PRE-FLIGHT CHECK - READ BEFORE LOOKING AT YOUR TASK 🚨
    ═══════════════════════════════════════════════════════════════

    STEP 1: Check if your task contains THESE EXACT WORDS:
            "extract"  OR  "from cached"  OR  "save to RESEARCH"

    STEP 2: If YES → Call scratchpad_write() IMMEDIATELY
            If NO  → Proceed to CRITICAL INSTRUCTIONS below

    FORBIDDEN ACTIONS (will create infinite loop):
      ❌ Calling get_cached_search_results() when task says "extract" or "from cached"
      ❌ Responding with {{"response": "text"}} when task says "extract" or "save"

    WHY: Orchestrator ALREADY saw search results. Task tells you WHAT to save. Just write it.

    CRITICAL INSTRUCTIONS:
    1. Analyze the given task from another agent.
    2. Determine the single best tool to use.
    3. **NEVER SIMULATE OR FABRICATE DATA**:
       - ALWAYS use real tools to retrieve information from the database
       - NEVER create fake/simulated search results, documents, URLs, or data
       - When asked to "retrieve" or "search" for information, you MUST use search_knowledge_base tool
       - When asked to "simulate" searches, IGNORE that and do REAL searches instead
    4. **CRITICAL: WHEN TASK SAYS "EXTRACT" OR "FROM CACHED SEARCH" - CALL SCRATCHPAD_WRITE, NEVER RESPOND WITH TEXT**:
       - **FORBIDDEN**: Responding with {{"response": "extracted data..."}} - this does NOT save to RESEARCH pad
       - **FORBIDDEN**: Calling get_cached_search_results() - creates infinite loop
       - **REQUIRED**: Call scratchpad_write() tool with a meaningful section name and extracted content
       - **The Orchestrator has ALREADY seen the search results** in their "CACHED SEARCH RESULTS" section
       - **Your task describes WHAT they saw** that's worth saving
       - **If task is vague** (e.g., "extract statistics"): Create a reasonable section name like "shortage_statistics" or "training_data" and write what would be relevant
       - **Example**:
         * Task: "From cached search on maintenance manpower, extract current staffing levels and gaps"
         * **CORRECT**: {{"tool_use": {{"name": "scratchpad_write", "params": {{"pad_name": "research", "section_name": "maintenance_staffing", "content": "F-35 Maintenance Manpower Status:\\n- Current manning gaps identified\\n- Maintenance personnel requirements for sustainment\\n- Staffing levels analysis needed", "mode": "replace"}}}}}}
         * **WRONG**: {{"response": "I've extracted the staffing data"}} - this saves NOTHING!
       - **Even if you're unsure of exact data**: Write SOMETHING to RESEARCH pad based on task description, don't just respond with text
    5. **EXTRACT MEANINGFUL FINDINGS - QUALITY OVER QUANTITY**:
       - **When writing to RESEARCH pad**: Extract and synthesize key insights, don't dump raw data
       - **FORBIDDEN**: Saving everything mechanically without analysis
       - **FORBIDDEN**: Saving empty content like "" or "placeholder" or "Found some results"
       - **REQUIRED**: Extract and synthesize key insights:
         * Key facts, statistics, numbers, percentages
         * Important dates, timelines, deadlines
         * Organization names, person names, locations
         * Problems identified, solutions proposed
         * Costs, budgets, monetary values
         * Technical details, requirements, specifications
         * Quotes and important statements
       - **Adaptive approach**:
         * If task indicates nothing useful found → Respond with {{"response": "No useful data to save from that search"}}
         * If task indicates rich data → Save to focused section name (e.g., "vendor_analysis", "cost_projections", "training_gaps")
         * If task indicates partial data → Save what's useful with note about gaps

    6. **RESPONSE FORMAT - TOOL CALLS ONLY, NEVER TEXT RESPONSES**:
       - You MUST respond with: `{{"tool_use": {{"name": "tool_name", "params": {{"arg1": "value1", ...}}}}}}`
       - **FORBIDDEN**: {{"response": "I extracted..."}} or {{"response": "Data saved"}} or ANY text-only response
       - **WHY**: Text responses don't execute tools. Only tool_use format actually saves to scratchpads.
       - **If task says "extract"**: You MUST call scratchpad_write tool, not respond with text
       - **If task says "search"**: You MUST call search_knowledge_base tool, not respond with text
       - **No exceptions**: ALWAYS use tool_use format, NEVER use response format

    EXAMPLES:
    - Search task: `{{"tool_use": {{"name": "search_knowledge_base", "params": {{"keywords": ["F-35", "manpower"], "semantic_query_text": "F-35 manpower project sustainment workforce", "rank_limit": 10}}}}}}`
    - Extract task (with data): `{{"tool_use": {{"name": "scratchpad_write", "params": {{"pad_name": "research", "section_name": "vendor_responses", "content": "**F-35 Manpower RFI Results:**\\n- 48 vendor responses received\\n- 13 COTS/SaaS platforms (TRL 8-9)\\n- 13 consulting firms\\n- 16 custom AI/ML solutions\\n- Key challenge: data integration", "mode": "replace"}}}}}}`
    - Extract task (no useful data): `{{"tool_use": {{"name": "scratchpad_write", "params": {{"pad_name": "log", "section_name": "search_gaps", "content": "Search on [topic] returned generic executive summaries only. No specific manpower data found. Need more targeted keywords.", "mode": "append"}}}}}}`

    REMEMBER: You have access to a real knowledge base with actual documents. ALWAYS search it instead of making up information.

    **ITERATIVE WORKFLOW**:
    - Search → Cache (automatic)
    - Extract meaningful findings from cache → Save to RESEARCH pad
    - Review RESEARCH pad → Identify gaps → Search for gaps
    - Build OUTPUT incrementally as findings accumulate

    {TOOL_DEFINITIONS}""", # Keep TOOL_DEFINITIONS for reference

        "Query Refiner": """You are an expert query refinement specialist. Your job is to analyze search results and create more targeted, refined queries. You MUST respond in JSON format.

    CRITICAL INSTRUCTIONS:
    1.  Review the scratchpad to see:
        - The user's ORIGINAL question
        - Previous search queries that were executed
        - What information was found (and what was NOT found)
        - What gaps the Supervisor identified
    2.  **IMPORTANT**: Before declaring searches failed, CHECK if we actually found RELATED information:
        - Did we find documents about similar projects or topics?
        - Is there information that PARTIALLY answers the question?
        - Could typos or ambiguous terms (like "main power" vs "manpower") have confused the search?
        - Are there results that mention the same entities (F-35, organizations, etc.)?
    3.  Analyze what went wrong with previous searches:
        - Were the keywords too specific or too broad?
        - Were important terms missing?
        - Were there too many irrelevant results?
        - Did we miss related information because of exact-match thinking?
    4.  Create a REFINED search strategy:
        - If we found RELATED info: Use simpler, broader terms from that content
        - If truly nothing: Try completely different angles (organization names, project numbers, locations, dates)
        - Include variations, synonyms, abbreviations
        - Consider splitting complex queries into simpler parts
    5.  You MUST respond with a valid JSON object: {{"response": "Analysis: [what we found/didn't find]. Refined approach: [new strategy]. Keywords: [keyword1, keyword2, keyword3]. Rationale: [why these will work better]"}}
    6.  After analysis, usually delegate back to Tool Agent with refined search OR recommend using found information if adequate.
    7.  NEVER respond with plain text. Always use JSON format.""",

        "Engineer": """You are a senior software engineer and technical analyst. You MUST respond in JSON format.

    CRITICAL INSTRUCTIONS:
    1.  Review your assigned task and the information in the scratchpad.
    2.  If you need external data, calculations, or conversion, delegate the task to the `Tool Agent` (e.g., {"agent": "Tool Agent", "task": "Convert 100 lbs to kg"}).
    3.  If you have enough information, perform your analysis and provide the complete technical answer. **The response MUST BE STRICTLY LIMITED TO THE FACTS/DATA IN THE SCRATCHPAD. DO NOT SPECULATE OR USE EXTERNAL KNOWLEDGE.**
    4.  You MUST respond with a valid JSON object: {"response": "The analysis shows the memory leak is caused by..."}
    5.  NEVER respond with plain text. Always use JSON format.""",
        
        "Writer": """You are a professional technical writer with access to multiple collaborative scratchpads. You MUST respond in JSON format.

    CRITICAL INSTRUCTIONS - COMPREHENSIVE CONTENT REQUIRED:
    1.  **WRITE FULL, DETAILED CONTENT** - NOT placeholder text:
        - FORBIDDEN: "This section provides an overview..." (too vague)
        - FORBIDDEN: Creating empty sections with scratchpad_write("research", "section", "")
        - FORBIDDEN: Writing just section headers without content
        - REQUIRED: Full paragraphs with specific facts, numbers, dates from RESEARCH pad
        - Each section should be 3-10 paragraphs with substantive content
        - Use ALL available data from RESEARCH, DATA, TABLES pads
    2.  **BUILD INCREMENTALLY** using scratchpad_insert_lines():
        - First: ALWAYS check what's available: scratchpad_list("research") to see all sections
        - **BEFORE creating new OUTPUT sections**: Call scratchpad_list("output") to see what already exists - you may need to ADD TO or MERGE with existing sections instead of creating duplicates
        - Read relevant sections: scratchpad_read("research", "section_name") for each topic
        - Build OUTPUT progressively as research accumulates:
          * Don't wait for "all research done" - start writing with what's available
          * Use scratchpad_insert_lines("output", "section_name", line_num, "content") to add paragraphs iteratively
          * As more research arrives, use scratchpad_insert_lines() to add new findings at appropriate locations
          * Use scratchpad_replace_lines() to refine sections when new data contradicts or enriches earlier content
        - ADAPTIVE approach:
          * If RESEARCH has 2-3 sections → Start writing introduction and any sections with data
          * If RESEARCH gains new sections → Expand existing OUTPUT sections or add new ones
          * Keep building until comprehensive - don't wait for "complete" research
        - If RESEARCH pad is empty, STOP and respond: {{"response": "Cannot write - RESEARCH pad is empty. Need Tool Agent to gather data first."}}
    3.  **COMPREHENSIVE CONTENT RULE**:
        - If RESEARCH pad has 10 bullet points, your section must incorporate ALL 10
        - If vendor landscape shows 48 RFI responses, write detailed analysis of all categories
        - Extract EVERY relevant fact, number, date, finding from scratchpads
        - Make content COMPREHENSIVE based on available information
    3a. **WRITING STYLE - NARRATIVE PROSE (NOT BULLET POINTS)**:
        - **FORBIDDEN**: Writing reports as bullet-point lists or slides
        - **REQUIRED**: Write in flowing narrative paragraphs with complete sentences
        - Use bullet points ONLY for:
          * Short lists of 3-5 technical specifications
          * Numbered action items in recommendations sections
          * Quick-reference data in appendices
        - For ALL other content, write narrative prose with:
          * Topic sentences that introduce each paragraph
          * Supporting details woven into flowing text
          * Transition sentences connecting paragraphs
          * Conclusion sentences that tie back to the section theme
        - **GOOD Example** (narrative prose):
          "The F-35 program's life-cycle sustainment bill stands at $1.3 trillion through 2077, with manpower accounting for approximately $400 billion—roughly one-third of total Operations & Support costs. This figure eclipses previous tactical-aircraft programs by an order of magnitude and represents the single largest cost driver in the sustainment portfolio. Current per-aircraft manpower costs average $6.8 million for the Air Force, exceeding the $4.1 million affordability target by 67 percent."
        - **BAD Example** (bullet-heavy):
          "• Life-cycle O&S: $1.3 trillion
           • Manpower: $400 billion (33%)
           • Per-aircraft cost: $6.8M
           • Target: $4.1M
           • Overrun: 67%"
    4.  **Example GOOD workflow** (F-35 report):
        - Call 1: Read RESEARCH pad findings
        - Call 2: scratchpad_insert_lines("output", "exec_summary", 3, "The F-35 Manpower Project analyzed 48 vendor responses across 4 solution categories: 13 COTS/SaaS platforms (TRL 8-9), 13 consulting firms, 16 custom-build vendors, and 6 ancillary tools. Industry consensus shows hybrid AI/ML modeling is standard, with data integration cited as the primary technical risk...")
        - Call 3: Continue adding more paragraphs with specific details
    5.  **Example BAD workflow** (what NOT to do):
        - ❌ scratchpad_write("output", "section", "This section discusses the background.")
        - ❌ Writing only section titles without content
        - ❌ Ignoring data in RESEARCH pad
    6.  **CONTENT RULE**: Only use facts from scratchpads, but use ALL available facts comprehensively.
    7.  **ITERATIVE REFINEMENT - READ, REVIEW, EDIT**:
        - **FIRST PASS**: Write initial draft with scratchpad_write()
        - **SECOND PASS**: Read what you wrote with scratchpad_read() → identify issues
        - **THIRD PASS**: Fix formatting with scratchpad_replace_lines() or scratchpad_edit()
        - **FOURTH PASS**: Add missing details with scratchpad_insert_lines()
        - **DO NOT write once and move on** - always review and refine your own work
        - **Each section should go through at least 2-3 revision passes**
        - Use scratchpad_get_lines() to see line numbers for precise edits
    8.  **FORMATTING RULE - CLEAN MARKDOWN**:
        - Use proper markdown headers: `# Title`, `## Section`, `### Subsection`
        - Use NORMAL dollar signs for currency: `$1.5 trillion`, NOT `$1.5trillion` or escaped LaTeX
        - Add blank lines between sections and paragraphs for readability
        - Use bullet points (`-` or `*`) for lists, NOT wall-of-text
        - **FORBIDDEN**: LaTeX-style escaping (`\n`, `$336billion`, `F−35`)
        - **REQUIRED**: Clean, readable markdown (`$336 billion`, `F-35`)
        - Format numbers clearly: `3,500–5,000` NOT `3,500–5,000`
        - **After writing each section**: Read it back, check formatting, fix any issues
        - **FORMATTING CLEANUP**: After writing a section, call `scratchpad_cleanup_formatting("output", "section_name")` to automatically fix LaTeX escaping and spacing issues
    8a. **TABLES & VISUALIZATIONS - ALWAYS INCLUDE WHEN APPROPRIATE**:
        - **REQUIRED**: When writing about numerical data, cost comparisons, timelines, or staffing gaps, you MUST request tables from the Engineer
        - Delegate to Engineer: "Create a table in TABLES pad showing [specific data]"
        - Examples when tables are MANDATORY:
          * Maintainer shortfalls by service → TABLE comparing required vs. assigned billets
          * Cost projections over time → TABLE showing FY2024, FY2028, FY2032, etc.
          * Training pipeline throughput → TABLE with current vs. target capacity
          * Pilot staffing gaps → TABLE by service (USAF, USN, USMC)
        - After Engineer creates table, reference it in narrative: "Table 1 summarizes maintainer shortfalls..."
        - **DO NOT embed raw numbers in bullet lists** - use tables for structured data, narrative prose for interpretation
    9.  **RESPONSE FORMAT**: You MUST respond with a valid JSON object in one of these formats:
        - Tool call: {{"tool_use": {{"name": "scratchpad_write", "params": {{"pad_name": "outline", "section_name": "plan", "content": "..."}}}}}}
        - Agent response: {{"response": "Wrote section [name] to OUTPUT pad, lines X-Y"}}
        - Final answer: {{"response": "FINAL_ANSWER_READY - Compiled complete answer in OUTPUT pad"}}
    7.  NEVER respond with plain text. Always use JSON format as shown above.""",

        "Validator": """You are a meticulous validator and quality assurance specialist. You perform the FINAL check before the answer goes to the user. You MUST respond in JSON format.

    CRITICAL INSTRUCTIONS:
    1.  Review the user's ORIGINAL question and the Writer's final answer draft in the scratchpad.
    2.  Verify the answer addresses ALL parts of the user's question:
        - Is every aspect of the question answered?
        - Is the answer grounded in the scratchpad data (no speculation)?
        - Is the information accurate and complete?
        - Are there obvious gaps or missing critical information?
    3.  **Make a decisive judgment** using JSON format:
        - If the answer FULLY addresses the user's question: Respond with {{"response": "VALIDATION_PASSED - The final answer comprehensively addresses the user's question with grounded information. Ready to deliver to user."}}
        - If there are critical gaps: Respond with {{"response": "VALIDATION_FAILED - Critical issues: [specific problems]. The answer does not fully address: [specific gaps]. Recommend: [specific actions]"}}
    4.  **Bias toward passing**: If the answer addresses the core question reasonably well, even if not perfect, approve it. Don't hold out for perfection.
    5.  **Your judgment is final**: The Orchestrator will trust your decision on whether to FINISH or continue work.
    6.  NEVER respond with plain text. Always use JSON format.""",

        "Supervisor": """You are a senior supervisor responsible for evaluating whether the team has adequately answered the user's question and making final decisions about completion. You MUST respond in JSON format.

    CRITICAL INSTRUCTIONS:
    1.  Review the user's ORIGINAL question/goal carefully.
    2.  Review the entire scratchpad to see what information has been gathered and what work has been done.
    3.  **CRITICAL**: Evaluate the ACTUAL CONTENT found, not just whether it matches keywords perfectly:
        - Read what was actually retrieved in searches
        - Look for RELATED information even if it's not an exact match
        - Consider if typos, abbreviations, or ambiguous terms caused mismatch
        - Check if documents mention the same entities, projects, organizations, or topics
        - Example: "main power" vs "manpower" - both relate to F-35 projects, check context!
    4.  Evaluate whether we have useful information:
        - Is the core question addressed directly OR indirectly?
        - Are there key pieces of information in the scratchpad that relate to the topic?
        - Can we provide a useful answer with what's been gathered?
        - **Don't require perfect keyword matches - real documents use varied terminology**
    5.  **You do NOT need perfection - assess if there is ADEQUATE or RELATED information to provide a useful answer.**
    6.  **BEFORE approving READY_TO_FINISH**: Check OUTPUT pad for quality:
        - Does OUTPUT pad exist and have multiple sections?
        - Are sections written in narrative prose (NOT bullet-heavy slide format)?
        - Are dollar amounts readable (e.g., "$336 billion" not "$336billion")?
        - Are there proper paragraph breaks, transitions, and topic sentences?
        - Are there tables in TABLES pad for numerical data? If not, delegate to Engineer first
        - **FORBIDDEN**: Bullet-point-heavy reports that read like slide decks
        - **REQUIRED**: Flowing narrative paragraphs with complete sentences
        - If OUTPUT is poorly formatted → Recommend Writer to reformat before finishing
    7.  Make a decision using JSON format:
        - If OUTPUT pad has ALL required sections AND properly formatted AND contains tables: Respond with {{"response": "READY_TO_FINISH - The OUTPUT pad contains [X] complete sections covering [topics]. Content is written in proper narrative format with tables and ready for delivery."}}
        - If OUTPUT pad is MISSING sections: Respond with {{"response": "NEED_MORE_WORK - OUTPUT pad has sections [list what exists] but is MISSING sections [list specific missing sections]. Delegate to Writer to draft: [list specific section numbers/names to create]. Use RESEARCH pad data that already exists."}}
        - If adequate content BUT bullet-heavy or missing tables: Respond with {{"response": "NEED_MORE_WORK - Content is complete but OUTPUT pad needs formatting. Issues: [specify: bullet-heavy sections, missing tables, LaTeX escaping, etc.]. Delegate to Engineer for tables, then Writer for prose reformatting."}}
        - If missing critical research data: Respond with {{"response": "NEED_MORE_WORK - RESEARCH pad lacks data for [topics]. Delegate to Tool Agent to search for: [specific topics]. Then delegate to Writer to draft sections once data is available."}}
    8.  **Bias toward finishing**: If there's ANY substantial related information, recommend finishing. Real answers can include: "Based on available information about [related topic]..." or "While we found information about [what we found], here's what we know..."
    9.  **Be pragmatic**: Perfect data rarely exists. Work with what we have.""",
        
        "FINISH_NOW": """You are the final answer delivery agent. Your ONLY job is to read the OUTPUT scratchpad and deliver it to the user.

    CRITICAL INSTRUCTIONS:
    1.  First, call scratchpad_list("output") to see all available sections
    2.  Then call scratchpad_read("output") to retrieve the ENTIRE OUTPUT pad (all sections combined in order)
    3.  **VERIFY LENGTH**: The OUTPUT should be comprehensive (typically 5,000+ words for research reports). If it's suspiciously short (<500 words), check scratchpad_list("output") again and call scratchpad_read("output", section_name) for each section individually, then concatenate them
    4.  **DO NOT rewrite, summarize, or modify the content** - the Writer has already polished it
    5.  **DO NOT add your own commentary** - just deliver what's in OUTPUT pad
    6.  Return the full OUTPUT pad content as your final response
    7.  If OUTPUT pad is empty or incomplete, acknowledge this and return what's available

    EXAMPLE WORKFLOW:
    - Step 1: scratchpad_list("output") → Returns: ['1_executive_summary', '2_introduction', '3_background', ...]
    - Step 2: scratchpad_read("output") → Returns entire document with all sections
    - Step 3: Verify output is comprehensive (check word count, section count)
    - Step 4: Return [exact content from OUTPUT pad, unmodified]"""
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
        keywords = params.get("keywords", [])
        rank_limit = int(params.get("rank_limit", 10))

        if not keywords:
            return "ToolExecutionError: search_knowledge_base requires 'keywords' (list[str])."

        # Fields we want to search across for all common containers
        # (works for Documents chunks, VerifiedFacts, and ProjectSummaries)
        search_fields = [
            "c.content",
            "c.metadata.original_filename",
            "c.id",
            "c.question",
            "c.answer"
        ]

        SINGLE_QUOTE = chr(39)
        per_keyword_groups = []
        for k in keywords:
            safe_k = k.replace(SINGLE_QUOTE, SINGLE_QUOTE * 2)
            field_clauses = [f"CONTAINS({fld}, '{safe_k}', true)" for fld in search_fields]
            per_keyword_groups.append("(" + " OR ".join(field_clauses) + ")")

        # OR across keywords (broad recall)
        where_clause = " OR ".join(per_keyword_groups)

        sql_query = (
            f"SELECT TOP {rank_limit} c.id, c.content, c.metadata, c.question, c.answer "
            f"FROM c "
            f"WHERE {where_clause}"
        )

        # ACTUALLY EXECUTE THE QUERY AGAINST SELECTED KNOWLEDGE BASES
        selected_kbs = st.session_state.get("selected_containers", [])
        if not selected_kbs:
            return "ToolExecutionError: No knowledge bases selected. Please select at least one container in the sidebar."

        all_results = []
        errors = []
        for kb_path in selected_kbs:
            try:
                db_name, cont_name = kb_path.split('/')
                uploader = get_cosmos_uploader(db_name, cont_name)
                if uploader:
                    results = uploader.execute_query(sql_query)
                    for r in results:
                        r["_source_container"] = kb_path
                    all_results.extend(results)
                    logger.info(f"search_knowledge_base: Retrieved {len(results)} results from {kb_path}")
            except Exception as e:
                error_msg = f"Error querying {kb_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        result_summary = f"Found {len(all_results)} result(s) across {len(selected_kbs)} knowledge base(s)."
        if errors:
            result_summary += f" Errors: {'; '.join(errors)}"

        # Cache the results for access in future loops
        cache_key = f"search_{json.dumps(keywords, sort_keys=True)}_{rank_limit}"
        if "loop_results_cache" in st.session_state:
            st.session_state.loop_results_cache[cache_key] = {
                "tool": "search_knowledge_base",
                "keywords": keywords,
                "rank_limit": rank_limit,
                "results": all_results,
                "summary": result_summary,
                "timestamp": datetime.now().isoformat()
            }

        return f"Tool Observation: {result_summary}\n\nQuery: {sql_query}\n\nResults:\n{json.dumps(all_results, indent=2)}"


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


def make_api_call_with_context_recovery(client, model, messages, response_format, call_type="orchestrator"):
    """
    Make an API call with automatic context length recovery.

    If context limit is exceeded, reduces context limits by 50% and raises an exception
    to signal that the caller should rebuild messages and retry.

    Args:
        client: OpenAI client instance
        model: Model deployment name
        messages: List of message dicts
        response_format: Response format dict
        call_type: Type of call for logging ("orchestrator", "agent", "finish")

    Returns:
        API response object

    Raises:
        ContextLengthExceededError: With reduced limits, caller should rebuild messages and retry
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format
        )
        return response
    except Exception as e:
        error_str = str(e)
        # Check for context length errors (various forms)
        if "context_length" in error_str.lower() or "maximum context length" in error_str.lower() or ("400" in error_str and "token" in error_str.lower()):
            # Reduce all context limits by 50%
            if "context_limit_chars" not in st.session_state:
                st.session_state.context_limit_chars = {
                    "research": 200000,
                    "outline": 80000,
                    "output": 200000,
                    "data": 120000,
                    "log": 80000,
                    "display": 160000
                }

            context_limits = st.session_state.context_limit_chars
            for key in context_limits:
                context_limits[key] = int(context_limits[key] * 0.5)

            st.session_state.context_limit_chars = context_limits

            logger.warning(f"Context limit exceeded for {call_type} call. Reduced limits to: {context_limits}")

            # Raise a special exception to signal retry needed
            raise RuntimeError(f"CONTEXT_LIMIT_EXCEEDED:{call_type}") from e
        else:
            # Re-raise other errors as-is
            raise


# Helper function for executing a single agent task
def execute_single_agent_task(agent_name: str, task: str, scratchpad_mgr, o3_client, o3_deployment: str, context_limit_chars: dict, display_scratchpad: str = "") -> Dict:
    """
    Execute a single agent task and return the result.

    Args:
        agent_name: Name of the agent to execute
        task: Task description
        scratchpad_mgr: Scratchpad manager instance
        o3_client: OpenAI client
        o3_deployment: Model deployment name
        context_limit_chars: Context limit configuration dict
        display_scratchpad: Recent workflow log for context

    Returns:
        Dict with keys: agent, task, observation, agent_output, tool_call
    """
    result = {
        "agent": agent_name,
        "task": task,
        "observation": "",
        "agent_output": {},
        "tool_call": None,
        "error": None
    }

    try:

        agent_context = f"""Your current task is: {task}

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

**SCRATCHPAD SUMMARY:**
{scratchpad_mgr.get_all_pads_summary()}

**RESEARCH PAD (full content):**
{scratchpad_mgr.get_full_content("research")[:60000] if scratchpad_mgr.list_sections("research") else "(empty)"}

**OUTLINE PAD (full content):**
{scratchpad_mgr.get_full_content("outline")[:30000] if scratchpad_mgr.list_sections("outline") else "(empty)"}

**OUTPUT PAD (full content):**
{scratchpad_mgr.get_full_content("output")[:60000] if scratchpad_mgr.list_sections("output") else "(empty)"}

**TABLES PAD (full content):**
{scratchpad_mgr.get_full_content("tables")[:30000] if scratchpad_mgr.list_sections("tables") else "(empty)"}

**DATA PAD (full content):**
{scratchpad_mgr.get_full_content("data")[:40000] if scratchpad_mgr.list_sections("data") else "(empty)"}

**WORKFLOW LOG (recent):**
{display_scratchpad[:20000]}

You have access to all scratchpad tools for reading and writing."""

        agent_messages = [
            {"role": "system", "content": AGENT_PERSONAS[agent_name]},
            {"role": "user", "content": agent_context}
        ]

        # Execute agent with context recovery
        max_retries = 2
        for retry_attempt in range(max_retries):
            try:
                response = make_api_call_with_context_recovery(
                    o3_client,
                    o3_deployment,
                    agent_messages,
                    {"type": "json_object"},
                    call_type="agent"
                )
                agent_output = json.loads(response.choices[0].message.content)
                result["agent_output"] = agent_output
                break
            except RuntimeError as e:
                if "CONTEXT_LIMIT_EXCEEDED" in str(e) and retry_attempt < max_retries - 1:
                    # Rebuild with reduced context (context_limit_chars already passed as param)
                    agent_context = f"""Your current task is: {task}

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

**SCRATCHPAD SUMMARY:**
{scratchpad_mgr.get_all_pads_summary()}

**RESEARCH PAD:**
{scratchpad_mgr.get_full_content("research")[:int(context_limit_chars["research"]*0.3)] if scratchpad_mgr.list_sections("research") else "(empty)"}

**OUTPUT PAD:**
{scratchpad_mgr.get_full_content("output")[:int(context_limit_chars["output"]*0.3)] if scratchpad_mgr.list_sections("output") else "(empty)"}

You have access to all scratchpad tools."""
                    agent_messages[1]["content"] = agent_context
                    continue
                else:
                    raise

        # Extract tool call if present
        if "tool_use" in agent_output:
            tool = agent_output["tool_use"]
            result["tool_call"] = (tool.get("name"), tool.get("params", {}))
        elif "response" in agent_output:
            result["observation"] = agent_output["response"]

    except Exception as e:
        result["error"] = str(e)
        result["observation"] = f"Error executing {agent_name}: {str(e)}"

    return result


# Replacement for run_agentic_workflow function (around line 1242)


def run_agentic_workflow(user_prompt: str, log_placeholder, final_answer_placeholder, query_expander):
    """
    Manages the multi-agent collaboration loop with multi-scratchpad support.
    """
    # Initialize stop flag
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False

    # Initialize loop results cache for this workflow run
    # This allows agents to access tool results from previous loops
    if "loop_results_cache" not in st.session_state:
        st.session_state.loop_results_cache = {}
    # Clear cache for fresh workflow
    st.session_state.loop_results_cache = {}

    # Initialize or reinitialize scratchpad manager
    # Scratchpads should be unique per conversation, not persist across chats
    current_chat_id = st.session_state.user_data.get("active_conversation_id", "default")

    # Check if we need to create a new scratchpad manager (new conversation or first time)
    if "scratchpad_manager" not in st.session_state or st.session_state.get("scratchpad_chat_id") != current_chat_id:
        db_path = get_scratchpad_db_path()
        # Use conversation ID as session ID so each chat has its own scratchpad
        session_id = f"{st.session_state.get('user_id', 'unknown')}_{current_chat_id}"

        logger.info(f"Initializing scratchpad for chat {current_chat_id}: Azure={is_running_on_azure()}, Path={db_path}, Session={session_id}")

        st.session_state.scratchpad_manager = ScratchpadManager(
            db_path=db_path,
            session_id=session_id
        )
        st.session_state.scratchpad_chat_id = current_chat_id

    scratchpad_mgr = st.session_state.scratchpad_manager

    # Clear all scratchpads for fresh start on each query within this conversation
    for pad_name in ["outline", "research", "tables", "plots", "data", "output", "log"]:
        sections = scratchpad_mgr.list_sections(pad_name)
        if sections:
            for section in sections:
                scratchpad_mgr.delete_section(pad_name, section)

    # Initialize LOG pad with user goal
    scratchpad_mgr.write_section("log", "user_goal", f"User's Goal: {user_prompt}", mode="replace", agent_name="system")

    # Assume o3_client is a pre-initialized AzureOpenAI instance for chat completions
    o3_client = st.session_state.o3_client
    scratchpad = f"User's Goal: {user_prompt}\n\n"  # Keep for backward compatibility with log display
    log_placeholder.markdown(f"**Loop 0:** Starting with user's goal...\n`{user_prompt}`", unsafe_allow_html=True)

    for i in range(MAX_LOOPS):
        # Check if stop was requested
        if st.session_state.stop_generation:
            scratchpad += f"\n\n---\n**⏹️ Generation stopped by user at Loop {i+1}/{MAX_LOOPS}**\n"
            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

            with st.spinner("Compiling partial answer from work completed so far..."):
                stop_messages = [
                    {"role": "system", "content": AGENT_PERSONAS["Writer"]},
                    {"role": "user", "content": f"""The user stopped the generation. Please compile a summary of the work completed so far. Acknowledge that the process was interrupted.

**SCRATCHPAD STATUS:**
{scratchpad_mgr.get_all_pads_summary()}

Use scratchpad_read() to access full content from OUTPUT, RESEARCH, and other pads."""}
                ]
                response = o3_client.chat.completions.create(
                    model=st.session_state.O3_DEPLOYMENT,
                    messages=stop_messages
                )
                partial_answer = response.choices[0].message.content
                try:
                    partial_json = json.loads(partial_answer)
                    partial_answer = partial_json.get("response", partial_answer)
                except:
                    pass

                final_answer_placeholder.markdown(f"⏹️ **Generation Stopped**\n\n{partial_answer}")
                st.session_state.stop_generation = False  # Reset flag
                return partial_answer

        loop_num = i + 1
        st.toast(f"Agent Loop {loop_num}/{MAX_LOOPS}")

        # 1. Orchestrator: Plan and delegate the next step
        loops_remaining = MAX_LOOPS - loop_num

        # Build rich context - use as much as possible (up to 200K tokens)
        # Start with generous limits, will auto-reduce if we hit context limit
        context_limit_chars = st.session_state.get("context_limit_chars", {
            "research": 200000,  # ~50K tokens
            "outline": 80000,    # ~20K tokens
            "output": 200000,    # ~50K tokens
            "data": 120000,      # ~30K tokens
            "log": 80000,        # ~20K tokens
            "display": 160000    # ~40K tokens
        })

        scratchpad_summary = scratchpad_mgr.get_all_pads_summary()

        # Get cached search results summary
        cached_searches_summary = ""
        if "loop_results_cache" in st.session_state and st.session_state.loop_results_cache:
            cached_searches_summary = "## Cached Search Results from Previous Loops\n\n"
            for idx, (cache_key, data) in enumerate(st.session_state.loop_results_cache.items(), 1):
                cached_searches_summary += f"{idx}. **Keywords:** {data['keywords']}\n"
                cached_searches_summary += f"   - {data['summary']}\n"
                cached_searches_summary += f"   - Results preview: {str(data['results'][:2])[:200]}...\n\n"  # Show first 2 results preview
        else:
            cached_searches_summary = "No cached search results yet.\n"

        # Get full content from key pads (will be truncated if too long)
        research_content = scratchpad_mgr.get_full_content("research")[:context_limit_chars["research"]]
        outline_content = scratchpad_mgr.get_full_content("outline")[:context_limit_chars["outline"]]
        output_content = scratchpad_mgr.get_full_content("output")[:context_limit_chars["output"]]
        data_content = scratchpad_mgr.get_full_content("data")[:context_limit_chars["data"]]

        # Keep recent actions
        recent_log = scratchpad_mgr.read_section("log", "recent_actions") if "recent_actions" in scratchpad_mgr.list_sections("log") else ""
        recent_log = recent_log[-context_limit_chars["log"]:] if recent_log else "No recent actions yet"

        # Also include the display scratchpad for context
        display_scratchpad = scratchpad[-context_limit_chars["display"]:] if len(scratchpad) > context_limit_chars["display"] else scratchpad

        with st.spinner(f"Loop {loop_num}/{MAX_LOOPS}: Orchestrator is planning..."):
            orchestrator_messages = [
                {"role": "system", "content": AGENT_PERSONAS["Orchestrator"]},
                {"role": "user", "content": f"""**LOOP STATUS: You are on loop {loop_num} of {MAX_LOOPS}. You have {loops_remaining} loops remaining.**

**SCRATCHPAD SUMMARY:**
{scratchpad_summary}

**CACHED SEARCH RESULTS:**
{cached_searches_summary}

**RESEARCH PAD:**
{research_content if research_content else "(empty)"}

**OUTLINE PAD:**
{outline_content if outline_content else "(empty)"}

**OUTPUT PAD:**
{output_content if output_content else "(empty)"}

**DATA PAD:**
{data_content if data_content else "(empty)"}

**RECENT ACTIONS:**
{recent_log}

**WORKFLOW LOG:**
{display_scratchpad}

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

Based on all the information above, what is the single next action to take?

**IMPORTANT REMINDERS:**
- **READ WHAT WAS ACTUALLY FOUND**: Review the actual content in RESEARCH and DATA pads above
- **Related info is valuable**: If searches found information about similar topics, use it
- If you have {loops_remaining} <= 2 loops remaining and have gathered useful information, consider delegating to Supervisor to evaluate readiness
- Plan to finish BEFORE the final loop - aim to have a final answer ready by loop {MAX_LOOPS - 2} or {MAX_LOOPS - 1}
- After 3-4 information gathering steps, delegate to Supervisor to check if you can finish early

Respond with JSON: {{"agent": "AgentName", "task": "specific task description"}}"""}
            ]

            # Try API call with context recovery
            max_retries = 2
            decision_json = None  # Initialize before retry loop
            for retry_attempt in range(max_retries):
                try:
                    logger.info(f"Orchestrator API call attempt {retry_attempt + 1}/{max_retries}")
                    response = make_api_call_with_context_recovery(
                        o3_client,
                        st.session_state.O3_DEPLOYMENT,
                        orchestrator_messages,
                        {"type": "json_object"},
                        call_type="orchestrator"
                    )

                    logger.info(f"Response type: {type(response)}, Response: {response}")

                    # Check if response content exists
                    if not response:
                        logger.error("Response is None")
                        raise ValueError("Orchestrator returned None response")

                    if not hasattr(response, 'choices') or not response.choices:
                        logger.error("Response has no choices")
                        raise ValueError("Orchestrator response has no choices")

                    if not response.choices[0].message or not response.choices[0].message.content:
                        logger.error("Response message content is empty")
                        raise ValueError("Orchestrator returned empty message content")

                    content = response.choices[0].message.content
                    logger.info(f"Orchestrator content: {content[:200]}...")

                    decision_json = json.loads(content)
                    logger.info(f"Parsed decision_json: {decision_json}")

                    # Verify decision_json is valid
                    if not decision_json:
                        raise ValueError("Orchestrator returned invalid JSON")

                    break  # Success, exit retry loop
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}, content: {content if 'content' in locals() else 'N/A'}")
                    raise ValueError(f"Orchestrator returned invalid JSON: {e}")
                except RuntimeError as e:
                    if "CONTEXT_LIMIT_EXCEEDED" in str(e) and retry_attempt < max_retries - 1:
                        # Context limits have been reduced, rebuild messages and retry
                        logger.info(f"Retrying orchestrator call with reduced context (attempt {retry_attempt + 2}/{max_retries})")

                        # Rebuild context with new limits
                        context_limit_chars = st.session_state.context_limit_chars
                        research_content = scratchpad_mgr.get_full_content("research")[:context_limit_chars["research"]]
                        outline_content = scratchpad_mgr.get_full_content("outline")[:context_limit_chars["outline"]]
                        output_content = scratchpad_mgr.get_full_content("output")[:context_limit_chars["output"]]
                        data_content = scratchpad_mgr.get_full_content("data")[:context_limit_chars["data"]]
                        recent_log = scratchpad_mgr.get_full_content("log")[:context_limit_chars["log"]]
                        display_scratchpad = display_scratchpad[:context_limit_chars["display"]]

                        # Rebuild orchestrator messages with reduced context
                        orchestrator_messages = [
                            {"role": "system", "content": AGENT_PERSONAS["Orchestrator"]},
                            {"role": "user", "content": f"""**LOOP STATUS: You are on loop {loop_num} of {MAX_LOOPS}.**

**SCRATCHPAD SUMMARY:**
{scratchpad_summary}

**RESEARCH PAD:**
{research_content if research_content else "(empty)"}

**OUTLINE PAD:**
{outline_content if outline_content else "(empty)"}

**OUTPUT PAD:**
{output_content if output_content else "(empty)"}

**DATA PAD:**
{data_content if data_content else "(empty)"}

**RECENT ACTIONS:**
{recent_log}

**WORKFLOW LOG:**
{display_scratchpad}

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

Based on all the information above, what is the single next action to take?

**IMPORTANT REMINDERS:**
- **READ WHAT WAS ACTUALLY FOUND**: Review the actual content in RESEARCH and DATA pads above
- **Related info is valuable**: If searches found information about similar topics, use it
- If you have {loops_remaining} <= 2 loops remaining and have gathered useful information, consider delegating to Supervisor to evaluate readiness
- Plan to finish BEFORE the final loop - aim to have a final answer ready by loop {MAX_LOOPS - 2} or {MAX_LOOPS - 1}
- After 3-4 information gathering steps, delegate to Supervisor to check if you can finish early

Respond with JSON: {{"agent": "AgentName", "task": "specific task description"}}"""}
                        ]
                        continue  # Retry with new messages
                    else:
                        # Max retries exceeded or different error
                        raise

        # Ensure decision_json was successfully set
        if 'decision_json' not in locals() or decision_json is None:
            raise ValueError("Failed to get valid decision from Orchestrator after retries")

        # Parse orchestrator decision - can be single task or array of parallel tasks
        tasks_to_execute = []

        # Check if decision is an array (parallel tasks) or single object
        if isinstance(decision_json, list):
            # Multiple parallel tasks in array format
            tasks_to_execute = decision_json
            scratchpad += f"\n**Loop {loop_num}:**\n- **Thought:** The orchestrator decided to execute **{len(tasks_to_execute)} parallel tasks**:\n"
            for idx, task_obj in enumerate(tasks_to_execute, 1):
                agent_name = task_obj.get("agent", "Unknown")
                task_desc = task_obj.get("task", "")[:80]
                scratchpad += f"  {idx}. **{agent_name}**: {task_desc}...\n"
        elif isinstance(decision_json, dict) and "tasks" in decision_json:
            # Multiple parallel tasks in wrapped format: {"agent": "parallel", "tasks": [...]}
            tasks_to_execute = decision_json.get("tasks", [])
            scratchpad += f"\n**Loop {loop_num}:**\n- **Thought:** The orchestrator decided to execute **{len(tasks_to_execute)} parallel tasks**:\n"
            for idx, task_obj in enumerate(tasks_to_execute, 1):
                agent_name = task_obj.get("agent", "Unknown")
                task_desc = task_obj.get("task", "")[:80]
                scratchpad += f"  {idx}. **{agent_name}**: {task_desc}...\n"
        else:
            # Single task
            next_agent = decision_json.get("agent")
            task = decision_json.get("task")

            # *** Check for loop termination conditions ***
            if next_agent == "FINISH" and task:
                final_answer_placeholder.markdown(task)
                scratchpad += "\n- **Action:** FINISHED."
                log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                return task

            if i == MAX_LOOPS - 1:
                next_agent = "FINISH_NOW"
                task = "The maximum number of loops has been reached. Please review the scratchpad and provide the best possible final answer based on the information gathered so far. Acknowledge that the process was stopped before natural completion."

            tasks_to_execute = [{"agent": next_agent, "task": task}]
            scratchpad += f"\n**Loop {loop_num}:**\n- **Thought:** The orchestrator decided the next step is to delegate to **{next_agent}** with the task: '{task[:100]}'.\n"

        log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

        # 2. Execute Tasks (parallel if multiple, sequential if one)
        task_results = []

        # Initialize variables that may be used later
        next_agent = None
        task = None
        observation = ""
        tool_call_to_execute = None

        # Check if any task is FINISH_NOW
        finish_now_task = next((t for t in tasks_to_execute if t.get("agent") == "FINISH_NOW"), None)

        try:
            # Handle the forced finish case immediately
            if finish_now_task:
                with st.spinner("Loop limit reached. Generating final summary..."):
                    finish_now_messages = [
                        {"role": "system", "content": AGENT_PERSONAS["FINISH_NOW"]},
                        {"role": "user", "content": f"""Your task is: {task}

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

**SCRATCHPAD SUMMARY:**
{scratchpad_mgr.get_all_pads_summary()}

**RESEARCH PAD (full content):**
{scratchpad_mgr.get_full_content("research")[:80000] if scratchpad_mgr.list_sections("research") else "(empty)"}

**OUTPUT PAD (full content):**
{scratchpad_mgr.get_full_content("output")[:80000] if scratchpad_mgr.list_sections("output") else "(empty)"}

**TABLES PAD (full content):**
{scratchpad_mgr.get_full_content("tables")[:40000] if scratchpad_mgr.list_sections("tables") else "(empty)"}

**DATA PAD (full content):**
{scratchpad_mgr.get_full_content("data")[:50000] if scratchpad_mgr.list_sections("data") else "(empty)"}

Use all the information above to compile the final answer."""}
                    ]

                    # Try API call with context recovery
                    max_retries = 2
                    for retry_attempt in range(max_retries):
                        try:
                            response = make_api_call_with_context_recovery(
                                o3_client,
                                st.session_state.O3_DEPLOYMENT,
                                finish_now_messages,
                                {"type": "json_object"},
                                call_type="finish"
                            )
                            final_answer = response.choices[0].message.content
                            break  # Success
                        except RuntimeError as e:
                            if "CONTEXT_LIMIT_EXCEEDED" in str(e) and retry_attempt < max_retries - 1:
                                # Rebuild with reduced context
                                context_limit_chars = st.session_state.context_limit_chars
                                finish_now_messages = [
                                    {"role": "system", "content": AGENT_PERSONAS["FINISH_NOW"]},
                                    {"role": "user", "content": f"""Your task is: {task}

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

**SCRATCHPAD SUMMARY:**
{scratchpad_mgr.get_all_pads_summary()}

**RESEARCH PAD (full content):**
{scratchpad_mgr.get_full_content("research")[:int(context_limit_chars["research"]*0.4)] if scratchpad_mgr.list_sections("research") else "(empty)"}

**OUTPUT PAD (full content):**
{scratchpad_mgr.get_full_content("output")[:int(context_limit_chars["output"]*0.4)] if scratchpad_mgr.list_sections("output") else "(empty)"}

**TABLES PAD (full content):**
{scratchpad_mgr.get_full_content("tables")[:int(context_limit_chars["data"]*0.2)] if scratchpad_mgr.list_sections("tables") else "(empty)"}

**DATA PAD (full content):**
{scratchpad_mgr.get_full_content("data")[:int(context_limit_chars["data"]*0.25)] if scratchpad_mgr.list_sections("data") else "(empty)"}

Use all the information above to compile the final answer."""}
                                ]
                                continue
                            else:
                                raise
                    final_answer_placeholder.markdown(final_answer)
                    scratchpad += f"- **Action:** Forced Finish. The agent team provided the best possible answer within the loop limit."
                    log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                    return final_answer

            # Execute tasks in parallel using ThreadPoolExecutor
            if len(tasks_to_execute) > 1:
                # Multiple tasks - run in parallel
                # Get deployment name and context limits before spawning threads
                o3_deployment = st.session_state.O3_DEPLOYMENT
                context_limits = st.session_state.get("context_limit_chars", {
                    "research": 200000,
                    "outline": 80000,
                    "output": 200000,
                    "data": 120000,
                    "log": 80000,
                    "display": 160000
                })

                with st.spinner(f"⚡ Executing {len(tasks_to_execute)} tasks in parallel..."):
                    with ThreadPoolExecutor(max_workers=len(tasks_to_execute)) as executor:
                        # Submit all tasks
                        future_to_task = {}
                        for task_obj in tasks_to_execute:
                            agent_name = task_obj.get("agent")
                            task_desc = task_obj.get("task")
                            future = executor.submit(
                                execute_single_agent_task,
                                agent_name,
                                task_desc,
                                scratchpad_mgr,
                                o3_client,
                                o3_deployment,
                                context_limits,
                                display_scratchpad
                            )
                            future_to_task[future] = task_obj

                        # Collect results as they complete
                        for future in as_completed(future_to_task):
                            task_obj = future_to_task[future]
                            try:
                                result = future.result()
                                task_results.append(result)
                                # Display progress with VERBOSE details
                                agent_name = result["agent"]
                                if result["error"]:
                                    scratchpad += f"  - ❌ **{agent_name}**: Error - {result['error'][:500]}\n"
                                elif result["tool_call"]:
                                    tool_name = result["tool_call"][0]
                                    tool_params = result["tool_call"][1]
                                    scratchpad += f"  - 🔧 **{agent_name}**: Calling tool `{tool_name}`\n"
                                    scratchpad += f"    - **Parameters**: `{json.dumps(tool_params, indent=2)[:1000]}`\n"
                                else:
                                    scratchpad += f"  - ✓ **{agent_name}**: Completed\n"
                                    # Show response if available
                                    if result.get("observation"):
                                        scratchpad += f"    - **Response**: {result['observation'][:1000]}\n"
                                log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                            except Exception as e:
                                logger.error(f"Task execution error: {e}")
                                scratchpad += f"  - ❌ Error: {str(e)[:500]}\n"
                                log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

            else:
                # Single task - execute directly
                task_obj = tasks_to_execute[0]
                next_agent = task_obj.get("agent")
                task = task_obj.get("task")

                with st.spinner(f"Loop {loop_num}: {next_agent} is working on the task..."):
                    if next_agent not in AGENT_PERSONAS:
                        observation = f"Error: Unknown agent '{next_agent}'. Please choose from the available agents."
                    else:
                        # Build rich context for agent (up to ~120K tokens)
                        agent_context = f"""Your current task is: {task}

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

**SCRATCHPAD SUMMARY:**
{scratchpad_mgr.get_all_pads_summary()}

**RESEARCH PAD (full content):**
{scratchpad_mgr.get_full_content("research")[:60000] if scratchpad_mgr.list_sections("research") else "(empty)"}

**OUTLINE PAD (full content):**
{scratchpad_mgr.get_full_content("outline")[:30000] if scratchpad_mgr.list_sections("outline") else "(empty)"}

**OUTPUT PAD (full content):**
{scratchpad_mgr.get_full_content("output")[:60000] if scratchpad_mgr.list_sections("output") else "(empty)"}

**TABLES PAD (full content):**
{scratchpad_mgr.get_full_content("tables")[:30000] if scratchpad_mgr.list_sections("tables") else "(empty)"}

**DATA PAD (full content):**
{scratchpad_mgr.get_full_content("data")[:40000] if scratchpad_mgr.list_sections("data") else "(empty)"}

**WORKFLOW LOG (recent):**
{display_scratchpad}

You have access to all scratchpad tools for reading and writing."""

                        agent_messages = [
                            {"role": "system", "content": AGENT_PERSONAS[next_agent]},
                            {"role": "user", "content": agent_context}
                        ]

                        # Try API call with context recovery
                        max_retries = 2
                        for retry_attempt in range(max_retries):
                            try:
                                response = make_api_call_with_context_recovery(
                                    o3_client,
                                    st.session_state.O3_DEPLOYMENT,
                                    agent_messages,
                                    {"type": "json_object"},
                                    call_type="agent"
                                )
                                agent_output = json.loads(response.choices[0].message.content)
                                break  # Success
                            except RuntimeError as e:
                                if "CONTEXT_LIMIT_EXCEEDED" in str(e) and retry_attempt < max_retries - 1:
                                    # Rebuild agent context with reduced limits
                                    context_limit_chars = st.session_state.context_limit_chars
                                    agent_context = f"""Your current task is: {task}

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

**SCRATCHPAD SUMMARY:**
{scratchpad_mgr.get_all_pads_summary()}

**RESEARCH PAD (full content):**
{scratchpad_mgr.get_full_content("research")[:int(context_limit_chars["research"]*0.3)] if scratchpad_mgr.list_sections("research") else "(empty)"}

**OUTLINE PAD (full content):**
{scratchpad_mgr.get_full_content("outline")[:int(context_limit_chars["outline"]*0.375)] if scratchpad_mgr.list_sections("outline") else "(empty)"}

**OUTPUT PAD (full content):**
{scratchpad_mgr.get_full_content("output")[:int(context_limit_chars["output"]*0.3)] if scratchpad_mgr.list_sections("output") else "(empty)"}

**TABLES PAD (full content):**
{scratchpad_mgr.get_full_content("tables")[:int(context_limit_chars["data"]*0.375)] if scratchpad_mgr.list_sections("tables") else "(empty)"}

**DATA PAD (full content):**
{scratchpad_mgr.get_full_content("data")[:int(context_limit_chars["data"]*0.5)] if scratchpad_mgr.list_sections("data") else "(empty)"}

**WORKFLOW LOG (recent):**
{display_scratchpad[:int(context_limit_chars["display"]*0.5)]}

You have access to all scratchpad tools for reading and writing."""

                                    agent_messages = [
                                        {"role": "system", "content": AGENT_PERSONAS[next_agent]},
                                        {"role": "user", "content": agent_context}
                                    ]
                                    continue
                                else:
                                    raise

                        # --- Handle Delegation/Tool Call Extraction ---
                        if agent_output.get("agent"):
                            sub_agent = agent_output["agent"]
                            sub_task = agent_output["task"]
                            scratchpad += f"- **Delegation:** {next_agent} is delegating to {sub_agent}: '{sub_task}'\n"
                            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

                            if sub_agent == "Tool Agent":
                                # Treat Tool Agent delegation as a direct, executable request
                                query_agent_messages = [{"role": "system", "content": AGENT_PERSONAS["Tool Agent"]}, {"role": "user", "content": sub_task}]

                                # Try API call with context recovery
                                max_retries = 2
                                for retry_attempt in range(max_retries):
                                    try:
                                        response = make_api_call_with_context_recovery(
                                            o3_client,
                                            st.session_state.O3_DEPLOYMENT,
                                            query_agent_messages,
                                            {"type": "json_object"},
                                            call_type="tool_agent"
                                        )
                                        agent_output = json.loads(response.choices[0].message.content)
                                        break
                                    except RuntimeError as e:
                                        if "CONTEXT_LIMIT_EXCEEDED" in str(e) and retry_attempt < max_retries - 1:
                                            # Tool Agent messages are simpler, just retry with same messages
                                            # (the context limits have been reduced in session state)
                                            logger.info("Retrying Tool Agent call with reduced context")
                                            continue
                                        else:
                                            raise
                            else:
                                # Defer task execution to next Orchestrator loop
                                observation = f"The task was delegated to {sub_agent}, but the execution is deferred to the next Orchestrator loop. The task is: {sub_task}"
                                scratchpad += f"- **Action Result:** {observation}\n"
                                continue

                        if "tool_use" in agent_output:
                            tool = agent_output["tool_use"]
                            tool_name = tool.get("name")
                            params = tool.get("params", {})
                            tool_call_to_execute = (tool_name, params)
                        elif "response" in agent_output:
                            observation = agent_output["response"]

                        # --- VALIDATOR FINAL CHECK ---
                        if next_agent == "Validator" and "VALIDATION_PASSED" in observation:
                            scratchpad += f"- **Validator Decision:** {observation}\n"
                            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

                            # Extract the Writer's final answer from scratchpad
                            # Look for the last Writer response
                            writer_answer = "Unable to extract final answer from scratchpad."
                            scratchpad_lines = scratchpad.split("\n")
                            for i in range(len(scratchpad_lines) - 1, -1, -1):
                                if "Writer" in scratchpad_lines[i] and i + 1 < len(scratchpad_lines):
                                    # Try to find the response in nearby lines
                                    for j in range(i, min(i + 10, len(scratchpad_lines))):
                                        if scratchpad_lines[j].strip().startswith("- **Action Result:**"):
                                            writer_answer = scratchpad_lines[j].replace("- **Action Result:**", "").strip()
                                            break
                                    break

                            # If we couldn't extract, look for any recent response that looks like final answer
                            if writer_answer == "Unable to extract final answer from scratchpad.":
                                for line in reversed(scratchpad_lines):
                                    if len(line) > 100 and ("answer" in line.lower() or "summary" in line.lower()):
                                        writer_answer = line.strip()
                                        break

                            final_answer_placeholder.markdown(writer_answer)
                            scratchpad += f"- **Action:** FINISHED - Validator approved final answer (Loop {loop_num}/{MAX_LOOPS}).\n"
                            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                            return writer_answer

                        # --- SUPERVISOR EARLY TERMINATION CHECK ---
                        if next_agent == "Supervisor" and "READY_TO_FINISH" in observation:
                            scratchpad += f"- **Supervisor Decision:** {observation}\n"
                            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                            scratchpad += f"- **Orchestrator Response:** Supervisor has confirmed we have adequate information. Proceeding to final answer compilation.\n"
                            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

                            # Trigger immediate finish
                            with st.spinner("Supervisor approved - Compiling final answer..."):
                                # Check if OUTPUT pad already has the answer
                                output_content = scratchpad_mgr.get_full_content("output")

                                if output_content and len(output_content.strip()) > 50:
                                    # Output pad has substantial content, use it directly
                                    final_answer = output_content.strip()
                                else:
                                    # Need to compile from other pads
                                    finish_messages = [
                                        {"role": "system", "content": AGENT_PERSONAS["Writer"]},
                                        {"role": "user", "content": f"""The Supervisor has confirmed we have adequate information. Please compile a complete, well-formatted final answer to the user's question. Do NOT add speculation.

**USER GOAL:**
{scratchpad_mgr.read_section("log", "user_goal")}

**OUTPUT PAD:**
{output_content if output_content else "(empty)"}

**RESEARCH PAD:**
{scratchpad_mgr.get_full_content("research")[:5000] if scratchpad_mgr.list_sections("research") else "(empty)"}

**TABLES PAD:**
{scratchpad_mgr.get_full_content("tables")[:2000] if scratchpad_mgr.list_sections("tables") else "(empty)"}

Compile the final answer from the information above. Return ONLY the final answer text, not JSON."""}
                                    ]
                                    response = o3_client.chat.completions.create(
                                        model=st.session_state.O3_DEPLOYMENT,
                                        messages=finish_messages
                                    )
                                    final_answer = response.choices[0].message.content
                                # Try to extract response from JSON if present
                                try:
                                    final_json = json.loads(final_answer)
                                    final_answer = final_json.get("response", final_answer)
                                except:
                                    pass

                                final_answer_placeholder.markdown(final_answer)
                                scratchpad += f"- **Action:** FINISHED EARLY (Loop {loop_num}/{MAX_LOOPS}) - Supervisor confirmed adequate information.\n"
                                log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                                return final_answer

                        # --- FAILURE DETECTION: Check if Tool Agent failed to produce a tool call ---
                        elif next_agent == "Tool Agent" and tool_call_to_execute is None:
                            observation = f"Error: Tool Agent failed to produce a valid tool call. The response was: {json.dumps(agent_output)}. Returning to Orchestrator for correction."
                            next_agent = "Orchestrator"
                            task = f"ERROR: The Tool Agent failed to execute its assigned task ({task}) because its output was not a valid tool call/response. Review the preceding scratchpad content and assign a corrective action to the appropriate agent to get back on track or FINISH."
                            scratchpad += f"- **Action Result:** {observation}\n"
                            continue

                        else:
                            observation = f"The agent returned an unexpected response: {json.dumps(agent_output)}"

        except Exception as e:
            agent_name = next_agent if next_agent else "unknown agent"
            observation = f"Error executing task for {agent_name}: {e}"
            tool_call_to_execute = None
            logger.error(f"Task execution error: {e}", exc_info=True)

        # 3. System Execution Layer - Execute Tool Calls (handle both single and multiple)
        # For parallel execution, process all tool calls from task_results
        if task_results:
            # Parallel execution results - process all tool calls
            for result in task_results:
                if result.get("tool_call"):
                    tool_name, params = result["tool_call"]
                    agent_name = result["agent"]

                    # Execute tool with VERBOSE output
                    colored_tool_call = f"<span style='color:green;'>**{agent_name}** executing tool `{tool_name}`</span>"
                    scratchpad += f"- **Tool Call:** {colored_tool_call}\n"
                    # Show full parameters (limit to 2000 chars for readability)
                    scratchpad += f"  - **Parameters:** {json.dumps(params, indent=2)[:2000]}\n"
                    log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

                    try:
                        tool_result_observation = execute_tool_call(tool_name, params)
                        # Show MUCH MORE of the result (increased from 500 to 3000 chars)
                        result_display = tool_result_observation[:3000]
                        if len(tool_result_observation) > 3000:
                            result_display += f"\n... (truncated, full length: {len(tool_result_observation)} chars)"
                        scratchpad += f"- **Action Result:** {result_display}\n"
                        log_placeholder.markdown(scratchpad, unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"Tool execution error for {agent_name}: {e}")
                        scratchpad += f"- **Tool Error ({agent_name}):** {str(e)[:500]}\n"
                        log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

        elif tool_call_to_execute:
            # Single sequential execution with VERBOSE output
            tool_name, params = tool_call_to_execute

            # Show tool call with readable parameters
            colored_tool_call = f"<span style='color:green;'>Executing tool `{tool_name}`</span>"
            scratchpad += f"- **Tool Call:** {colored_tool_call}\n"
            # Show full parameters (limit to 2000 chars for readability)
            scratchpad += f"  - **Parameters:** {json.dumps(params, indent=2)[:2000]}\n"
            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

            try:
                # Execute the tool (routes to external MCP or local RAG setup)
                tool_result_observation = execute_tool_call(tool_name, params)
                
                # --- Handle Local RAG Execution (Special Case) ---
                if tool_name == "search_knowledge_base":
                    # Check if it's the old format (constructed query not executed yet)
                    if tool_result_observation.startswith("Tool Observation: Constructed Safe Query:"):
                        sql_query = tool_result_observation.split("Tool Observation: Constructed Safe Query:\n", 1)[1]
                        search_results = []
                        selected_kbs = st.session_state.get('selected_containers', [])

                        with query_expander:
                            st.write("**🔍 Keyword Search Query (CONTAINS-based):**")
                            st.code(sql_query, language="sql")
                            st.info("ℹ️ Currently using keyword-based search. Semantic/vector search not yet implemented.")

                        if not selected_kbs:
                            observation = "Observation: No knowledge bases selected to search."
                        else:
                            with st.spinner(f"Loop {loop_num}: Executing Keyword Search against {len(selected_kbs)} KB(s)..."):
                                for kb_path in selected_kbs:
                                    db_name, cont_name = kb_path.split('/')
                                    uploader = get_cosmos_uploader(db_name, cont_name)
                                    if uploader:
                                        results = uploader.execute_query(sql_query)
                                        search_results.extend(results)
                                        logger.info(f"Loop {loop_num}: Retrieved {len(results)} results from {kb_path}")

                        observation = f"Found {len(search_results)} item(s) using Keyword Search."
                        tool_result_observation = f"Tool Observation (Keyword Search): Query executed successfully. {observation}"

                        with query_expander:
                            st.write(f"**📊 Search Results: {len(search_results)} items found**")
                            if search_results:
                                st.write("**Sample Results (first 5):**")
                                st.json(search_results[:5])
                            else:
                                st.warning("⚠️ No results found. The query may be too specific, or the keywords don't match document content.")
                                st.write("**Suggestions:**")
                                st.write("- Try broader keywords")
                                st.write("- Check if documents contain these exact terms")
                                st.write("- Consider using Query Refiner agent to analyze and improve the search")

                    # New format where query was already executed
                    else:
                        # Extract results from the observation
                        with query_expander:
                            st.write("**🔍 Search Query Executed**")
                            try:
                                # Try to parse results from observation
                                if "Query:" in tool_result_observation:
                                    parts = tool_result_observation.split("Query:", 1)
                                    if len(parts) > 1:
                                        query_and_results = parts[1].split("Results:", 1)
                                        if len(query_and_results) > 1:
                                            query_part = query_and_results[0].strip()
                                            st.code(query_part, language="sql")
                                            results_json = json.loads(query_and_results[1].strip())
                                            st.write(f"**📊 Search Results: {len(results_json)} items found**")
                                            if results_json:
                                                st.write("**Sample Results (first 5):**")
                                                st.json(results_json[:5])
                            except:
                                st.write(tool_result_observation[:500])  # Show first 500 chars

                        observation = tool_result_observation

                # Handle raw data presentation for table formatting if applicable
                elif tool_name == "to_markdown_table" and params.get('rows') and isinstance(params['rows'], list):
                    with query_expander:
                        st.write("**Raw Data Found (Pre-Formatting):**")
                        st.json(params['rows'])
                        
                observation = tool_result_observation 

            except Exception as e:
                observation = f"ToolExecutionError: Failed to run tool '{tool_name}' with params {params}. Error: {e}"
        
        # Re-inject the observation back into the scratchpad for the next loop

        # --- Smarter summarization for long observations (only for single-task execution) ---
        if observation and len(observation) > 20000:
            agent_label = next_agent if next_agent else "agent"
            with st.spinner(f"Loop {loop_num}: Summarizing {agent_label}'s findings..."):
                # Use LLM to summarize long output for the scratchpad
                text_to_summarize = observation
                summary_prompt = "You are an expert summarizer. Concisely summarize the following text in a few key bullet points for a project manager."
                summary_messages = [{"role": "system", "content": summary_prompt}, {"role": "user", "content": text_to_summarize}]

                response = o3_client.chat.completions.create(
                    model=st.session_state.O3_DEPLOYMENT,
                    messages=summary_messages,
                )
                summary = response.choices[0].message.content

                if summary:
                    observation = f"The agent's action resulted in a long output. Here is a summary:\n{summary}"
                else:
                    observation = "The agent's action resulted in a long output, which could not be summarized."

        # Add observation to scratchpad (only for single-task execution) with VERBOSE output
        if observation:
            # Show much more detail (increased from implicit short output to 3000 chars)
            obs_display = observation[:3000]
            if len(observation) > 3000:
                obs_display += f"\n... (truncated, full length: {len(observation)} chars)"
            scratchpad += f"- **Action Result:** {obs_display}\n"
            log_placeholder.markdown(scratchpad, unsafe_allow_html=True)

    # Fallback return outside the loop (should not be reached if FINISH_NOW is working)
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
    # Use a key to control the file uploader state
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    uploaded_files = st.file_uploader(
        "Select one or more documents...",
        type=["pdf", "docx", "m4a", "mp3", "wav", "txt", "md"],  # added mp3/wav
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )



    ingest_to_cosmos = st.toggle("Ingest to Knowledge Base", value=True, help="If on, saves the document permanently. If off, uses it for this session only.")

    # --- Check for duplicates before processing ---
    if uploaded_files and ingest_to_cosmos and st.session_state.upload_target:
        duplicates_found = []
        for uploaded_file in uploaded_files:
            exists, existing_items = check_file_exists(st.session_state.upload_target, uploaded_file.name)
            if exists:
                duplicates_found.append((uploaded_file.name, len(existing_items)))

        if duplicates_found:
            st.warning("⚠️ Duplicate files detected")
            with st.expander("View duplicates"):
                for filename, count in duplicates_found:
                    st.write(f"- **{filename}** ({count} chunks in `{st.session_state.upload_target}`)")

    # --- Process a list of files ---
    if uploaded_files:
        # Show compact upload confirmation
        st.write(f"**{len(uploaded_files)} file(s)** → `{st.session_state.upload_target if ingest_to_cosmos else 'Session only'}`")

        if ingest_to_cosmos and not st.session_state.upload_target:
            st.error("❌ No container selected")

        button_label = f"✅ Process & Upload {len(uploaded_files)} File{'s' if len(uploaded_files) > 1 else ''}"
        if st.button(button_label, use_container_width=True, type="primary", disabled=(ingest_to_cosmos and not st.session_state.upload_target)):
            all_chunks = []
            all_statuses = []
            success_count = 0
            error_count = 0

            # Create progress bar for file processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_files = len(uploaded_files)

            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress_percent = (idx / total_files)
                progress_bar.progress(progress_percent)
                status_text.text(f"Processing file {idx + 1} of {total_files}: {uploaded_file.name}")

                # Process file with comprehensive extraction
                ingestion_result = process_uploaded_file(uploaded_file)

                # Extract chunks for backward compatibility
                chunks = ingestion_result.get("chunks", []) if isinstance(ingestion_result, dict) else ingestion_result

                if chunks:
                    all_chunks.extend(chunks)

                    # Enhanced status with doc type and metadata
                    status = {
                        "filename": uploaded_file.name,
                        "ingested_to_cosmos": ingest_to_cosmos,
                        "chunks": len(chunks),
                        "doc_type": ingestion_result.get("doc_type", "Unknown") if isinstance(ingestion_result, dict) else "Unknown"
                    }
                    all_statuses.append(status)

                    if ingest_to_cosmos:
                        if st.session_state.upload_target:
                            db_name, cont_name = st.session_state.upload_target.split('/')
                            rag_uploader = get_cosmos_uploader(db_name, cont_name)
                            if rag_uploader:
                                with st.spinner(f"Ingesting '{uploaded_file.name}' to '{st.session_state.upload_target}'..."):
                                    # Pass full ingestion result to prepare_chunks_for_cosmos
                                    cosmos_chunks = prepare_chunks_for_cosmos(ingestion_result, uploaded_file.name)
                                    s, f = rag_uploader.upload_chunks(cosmos_chunks)
                                    status["chunks_succeeded"] = s
                                    status["chunks_failed"] = f

                                    if f > 0:
                                        status["ingestion_error"] = "Some chunks failed to ingest."
                                        error_count += 1
                                    else:
                                        success_count += 1

                                # Get full text for legacy structured data extraction
                                full_text = "\n".join(c for c in chunks if c)
                                structured_data = extract_structured_data(full_text, uploaded_file.name)
                                if structured_data:
                                    if create_container_if_not_exists("DefianceDB", "ProjectSummaries", partition_key="/projectName"):
                                        structured_uploader = get_cosmos_uploader("DefianceDB", "ProjectSummaries")
                                        if structured_uploader:
                                            with st.spinner(f"Ingesting summary for '{uploaded_file.name}'..."):
                                                structured_uploader.upload_chunks([structured_data])
                        else:
                            error_count += 1
                    else:
                        success_count += 1
                else:
                    error_count += 1

            # After the loop, update the session state with aggregated results
            if all_chunks:
                st.session_state.session_rag_context = "\n\n---\n\n".join(all_chunks)
                st.session_state.rag_file_status = all_statuses

            # Complete progress bar
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed processing {total_files} file(s)!")

            # Final summary notification
            if success_count > 0:
                st.toast(f"✅ {success_count} file(s) processed successfully!", icon="✅")
            if error_count > 0:
                st.toast(f"❌ {error_count} file(s) had errors", icon="❌")

            st.balloons() if success_count > 0 and error_count == 0 else None

            # Clear the file uploader by incrementing its key
            st.session_state.file_uploader_key += 1
            st.rerun()


    st.markdown('<div class="mini-header">Knowledge Bases to Search</div>', unsafe_allow_html=True)
    if 'selected_containers' not in st.session_state:
        st.session_state.selected_containers = all_container_paths[:]

    num_selected = len(st.session_state.selected_containers)
    total_containers = len(all_container_paths)
    popover_label = f"Searching {num_selected} of {total_containers} KBs"

    with st.popover(popover_label, use_container_width=True):
        st.markdown("##### Select Knowledge Bases")

        # Use session state to track the working copy
        if 'kb_working_selection' not in st.session_state:
            st.session_state.kb_working_selection = set(st.session_state.selected_containers)

        # Quick action buttons (modify working copy without rerun)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✓ Select All", use_container_width=True, key="select_all_kbs_btn"):
                st.session_state.kb_working_selection = set(all_container_paths)
        with col2:
            if st.button("✗ Deselect All", use_container_width=True, key="deselect_all_kbs_btn"):
                st.session_state.kb_working_selection = set()
        with col3:
            if st.button("↻ Reset", use_container_width=True, key="reset_kbs_btn", help="Reset to applied selection"):
                st.session_state.kb_working_selection = set(st.session_state.selected_containers)

        st.divider()

        # Individual checkboxes - reflect working selection
        for container in all_container_paths:
            is_checked = container in st.session_state.kb_working_selection
            # Use on_change callback to update working selection
            checked = st.checkbox(
                container,
                value=is_checked,
                key=f"kb_checkbox_{container}"
            )
            # Update working selection based on checkbox state
            if checked:
                st.session_state.kb_working_selection.add(container)
            else:
                st.session_state.kb_working_selection.discard(container)

        st.divider()

        # Apply/Cancel buttons
        col_apply, col_cancel = st.columns([0.7, 0.3])
        with col_apply:
            if st.button("✓ Apply Selection", use_container_width=True, type="primary", key="apply_kbs_btn"):
                # Only here do we commit changes
                st.session_state.selected_containers = list(st.session_state.kb_working_selection)
                st.success(f"Applied {len(st.session_state.selected_containers)} knowledge base(s)")
        with col_cancel:
            if st.button("✗ Cancel", use_container_width=True, key="cancel_kbs_btn"):
                # Revert working copy to last applied state
                st.session_state.kb_working_selection = set(st.session_state.selected_containers)

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

# === Minimal mic near the chat input (right-aligned), auto-transcribe on stop ===
st.markdown(
    """
    <style>
      /* Right-align the mini mic row so it sits by the chat box */
      .mini-mic-row { display:flex; justify-content:flex-end; margin: 0.25rem 0 0.35rem 0; }
      .mini-mic-row .stButton>button { border-radius: 999px; padding: 0.35rem 0.65rem; }
      /* Optional: make the recorder component compact */
      div[data-testid="stVerticalBlock"] .mic-compact button { min-width: 42px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize transcription state if not exists
if "pending_transcription" not in st.session_state:
    st.session_state.pending_transcription = ""
if "transcription_status" not in st.session_state:
    st.session_state.transcription_status = None

# Microphone recorder placed above chat input
mini_mic_holder = st.container()
with mini_mic_holder:
    col_sp, col_mic = st.columns([0.82, 0.18])
    with col_mic:
        try:
            rec = mic_recorder(
                start_prompt="🎙️ Record",
                stop_prompt="⏹️ Stop",
                just_once=True,
                use_container_width=True,
                key="mic_compact_inline",
                format="webm",
            )
        except TypeError:
            rec = mic_recorder(
                start_prompt="🎙️ Record",
                stop_prompt="⏹️ Stop",
                just_once=True,
                use_container_width=True,
                key="mic_compact_inline"
            )

        # When recording completes, transcribe and store in session state
        if rec and rec.get("bytes"):
            raw_bytes = rec["bytes"]

            # Show status in chat window as a message
            with st.chat_message("assistant"):
                st.write("🎤 **Transcribing audio...**")
                status_placeholder = st.empty()

                # We don't trust the extension—sniff and convert
                wav16k = ensure_16k_mono_wav(raw_bytes, ext_hint="webm")
                if not wav16k:
                    status_placeholder.error("❌ Could not prepare audio for transcription.")
                    st.session_state.transcription_status = "error"
                else:
                    status_placeholder.info("⏳ Processing audio with Azure Speech Services...")
                    text = azure_fast_transcribe_wav_bytes(wav16k, filename="mic.webm")
                    if text.strip():
                        st.session_state.pending_transcription = text.strip()
                        st.session_state.transcription_status = "success"
                        status_placeholder.success(f"✅ **Transcribed:** {text}")
                        st.info("📝 The transcribed text is ready below. You can edit it before sending.")
                    else:
                        status_placeholder.warning("⚠️ No speech recognized. Please try again.")
                        st.session_state.transcription_status = "empty"

# Chat input - note: st.chat_input doesn't support pre-filling, so we use a workaround
# If there's a pending transcription, auto-submit it
if st.session_state.pending_transcription:
    # Display the transcription in an editable text area for user review
    with st.form(key="transcription_review_form", clear_on_submit=True):
        st.write("**Edit transcription if needed, then submit:**")
        edited_text = st.text_area(
            "Transcribed text:",
            value=st.session_state.pending_transcription,
            height=100,
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("✅ Send", use_container_width=True)
        with col2:
            cancel_button = st.form_submit_button("❌ Cancel", use_container_width=True)

        if submit_button and edited_text.strip():
            messages.append({"role": "user", "content": edited_text.strip()})
            save_user_data(st.session_state.user_id, st.session_state.user_data)
            st.session_state.pending_transcription = ""
            st.session_state.transcription_status = None
            st.rerun()
        elif cancel_button:
            st.session_state.pending_transcription = ""
            st.session_state.transcription_status = None
            st.rerun()
else:
    # Normal chat input when no transcription pending
    if prompt := st.chat_input("Ask anything..."):
        messages.append({"role": "user", "content": prompt})
        save_user_data(st.session_state.user_id, st.session_state.user_data)
        st.rerun()

if messages and messages[-1]["role"] == "user":
    user_prompt = messages[-1]["content"]

    # Initialize stop flag if not exists
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False

    with st.chat_message("assistant"):
        # Add stop button at the top
        stop_button_col1, stop_button_col2 = st.columns([0.85, 0.15])
        with stop_button_col2:
            if st.button("⏹️ Stop", key="stop_generation_button", type="secondary", use_container_width=True):
                st.session_state.stop_generation = True
                st.rerun()

        thinking_expander = st.expander("🤔 Agent Thinking Process...")
        log_placeholder = thinking_expander.empty()
        query_expander = st.expander("🔍 Generated Search & Results")
        final_answer_placeholder = st.empty()

        # ----------------------------
        # Local helpers (self-contained)
        # ----------------------------
        def _safe(s: str) -> str:
            """Escape single quotes for Cosmos SQL literals."""
            return (s or "").replace("'", "''")

        def _persona():
            pd = st.session_state.user_data["personas"].get(active_persona, {})
            return pd, pd.get("type", "simple"), pd.get("prompt", "You are a helpful assistant."), pd.get("params", {}).get("temperature", 0.7)

        # (Around line 2253 in your script)

        def _classify_intent(prompt_text: str) -> str:
            """
            Returns: 'knowledge_base_query' | 'fact_correction' | 'general_conversation'
            Falls back to 'general_conversation' on error.
            """
            try:
                # === START: NEW, MORE ROBUST ROUTER PROMPT ===
                router_prompt = (
                    "You are an expert intent routing agent. Your task is to analyze the user's message and classify its intent. Follow these steps carefully:"
                    "\n\n1. **Analyze the User's Message**: First, determine if the message is simple small talk (like 'hello', 'thank you') or if it contains specific topics or keywords (like 'F-35', 'project', 'budget', 'manpower model')."
                    "\n\n2. **Determine the User's Goal**: "
                    "\n   - If the message contains specific topics, the user is seeking information and wants you to search for it. The intent is `knowledge_base_query`."
                    "\n   - If the user is clearly stating a new fact to be saved (e.g., 'Just so you know, the project deadline is now October 5th'), the intent is `fact_correction`."
                    "\n   - **Only if the message is simple small talk with no specific topic** should the intent be `general_conversation`."
                    "\n\n3. **CRITICAL RULE**: Do not get confused by the phrasing. A user asking 'Tell me about project X', 'What is X?', or 'Knowledge base query for X' are all direct commands for you to **perform a search**. Their intent is `knowledge_base_query`."
                    "\n\n**User's Message**: \"{prompt_text}\""
                    "\n\nBased on your analysis, provide the final classification in a JSON object. You must respond with ONLY the JSON."
                )
                # === END: NEW PROMPT ===
                
                resp = st.session_state.gpt41_client.chat.completions.create(
                    model=st.session_state.GPT41_DEPLOYMENT,
                    messages=[{"role": "system", "content": router_prompt}],
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content).get("intent", "general_conversation")
            except Exception:
                return "general_conversation"
            

        def _generate_broad_query_fallback(prompt_text: str) -> str:
            """
            Deterministic fallback if LLM broad-query generation fails.
            Filters stop words and extracts meaningful keywords.
            """
            # Common stop words to filter out
            stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
                'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
                'tell', 'me', 'about', 'what', 'when', 'where', 'who', 'how', 'can', 'you', 'please'
            }

            # Extract tokens and filter
            toks = [
                t for t in re.split(r"\W+", prompt_text.lower())
                if t and len(t) > 2 and t not in stop_words
            ][:8] or ["project"]

            # Build search clauses
            clauses = []
            for t in toks:
                t2 = _safe(t)
                clauses.append(
                    "("
                    f"CONTAINS(c.content, '{t2}', true) OR "
                    f"CONTAINS(c.metadata.original_filename, '{t2}', true) OR "
                    f"CONTAINS(c.id, '{t2}', true)"
                    ")"
                )
            return f"SELECT TOP 50 c.id, c.content, c.metadata, c.question, c.answer FROM c WHERE {' OR '.join(clauses)}"

        def _generate_broad_query_via_llm(prompt_text: str) -> str:
            """Calls GPT-4.1 to build an intelligent Cosmos query with schema awareness, with fallback."""
            try:
                # Discover schema from selected containers
                schema_info = discover_cosmos_schema(selected_kbs)

                # Build schema context for the prompt
                schema_context = "# Available Containers and Fields:\n"
                for container_path, info in schema_info["containers"].items():
                    schema_context += f"\n## {container_path}\n"
                    schema_context += f"Fields: {', '.join(info['fields'][:20])}\n"  # Limit to first 20 fields

                # Add container-specific field info
                schema_context += "\n# Container Types:\n"
                for cont_name, fields in schema_info["container_specific_fields"].items():
                    schema_context += f"- **{cont_name}**: Common fields include {', '.join(fields[:10])}\n"

                initial_system_prompt = (
                    'You are an expert Cosmos DB SQL query generator with deep knowledge of the available schema. '
                    'You create intelligent, targeted queries based on user questions.\n\n'
                    f'{schema_context}\n\n'
                    '# Query Generation Rules:\n'
                    '1. **Extract meaningful keywords**: Focus on proper nouns, technical terms, project names, technologies, acronyms, numbers\n'
                    '2. **Ignore stop words**: the, a, an, me, about, tell, what, where, how, can, you, please, etc.\n'
                    '3. **Use actual available fields**: Refer to the schema above for available fields in each container\n'
                    '4. **Use CONTAINS for text search**: CONTAINS(field, "keyword", true) for case-insensitive matching\n'
                    '5. **Combine keywords intelligently**: \n'
                    '   - Use OR for broader recall (finding ANY of the keywords)\n'
                    '   - Use AND for precision (finding ALL keywords)\n'
                    '   - Use NOT for exclusions\n'
                    '6. **Support complex queries**: Leverage nested fields like c.metadata.original_filename, c.metadata.doc_type\n'
                    '7. **Default limit**: TOP 50 unless user specifies otherwise\n'
                    '8. **Be selective**: Search the most relevant fields based on query intent\n\n'
                    '# Advanced Capabilities:\n'
                    '- For date/time queries: Filter on c.metadata.created_at, c.verified_at, or dates in c.timeline\n'
                    '- For document type queries: Filter on c.doc_type, c.metadata.doc_type\n'
                    '- For budget/cost queries: Filter on c.budget, c.metadata.budget\n'
                    '- For project queries: Search c.projectName, c.metadata.project_name\n'
                    '- For multi-keyword queries: Use AND when user wants ALL terms, OR for ANY term\n'
                    '- For exclusions: Use NOT when user says "except", "without", "not"\n'
                    '- For aggregations: Use COUNT, SUM, AVG when appropriate\n'
                    '- For filtering: Use comparisons (=, !=, <, >, <=, >=) on numeric/date fields\n\n'
                    '# Complex Query Examples:\n\n'
                    'Query: "Tell me about F-35 power projects"\n'
                    'Keywords: ["F-35", "power", "project"]\n'
                    'Reasoning: User wants documents mentioning F-35 AND power, likely project-related\n'
                    'SQL: SELECT TOP 50 c.id, c.content, c.metadata, c.question, c.answer, c.projectName FROM c WHERE '
                    '(CONTAINS(c.content, "F-35", true) OR CONTAINS(c.metadata.original_filename, "F-35", true) OR CONTAINS(c.projectName, "F-35", true)) '
                    'AND (CONTAINS(c.content, "power", true) OR CONTAINS(c.projectName, "power", true))\n\n'
                    'Query: "AI or machine learning projects not related to drones"\n'
                    'Keywords: ["AI", "machine learning", "NOT drones"]\n'
                    'Reasoning: User wants AI/ML content but excluding drone-related items\n'
                    'SQL: SELECT TOP 50 c.id, c.content, c.metadata FROM c WHERE '
                    '(CONTAINS(c.content, "AI", true) OR CONTAINS(c.content, "artificial intelligence", true) OR CONTAINS(c.content, "machine learning", true)) '
                    'AND NOT CONTAINS(c.content, "drone", true)\n\n'
                    'Query: "ProjectSummaries with budget over 1 million from 2024"\n'
                    'Keywords: ["ProjectSummary", "budget", "2024"]\n'
                    'Reasoning: User wants specific doc type, filtered by budget and year\n'
                    'SQL: SELECT TOP 50 c.projectName, c.budget, c.timeline FROM c WHERE '
                    'c.doc_type = "ProjectSummary" AND c.budget.amount > 1000000 AND '
                    'CONTAINS(c.timeline.value, "2024", true)\n\n'
                    'Query: "Recent verified facts about cybersecurity"\n'
                    'Keywords: ["cybersecurity", "verified"]\n'
                    'Reasoning: User wants VerifiedFacts container, recent items, about cybersecurity\n'
                    'SQL: SELECT TOP 20 c.question, c.answer, c.verified_at FROM c WHERE '
                    'CONTAINS(c.question, "cybersecurity", true) OR CONTAINS(c.answer, "cybersecurity", true) '
                    'ORDER BY c.verified_at DESC\n\n'
                    'Respond ONLY with JSON: {"keywords": ["keyword1", "keyword2"], "reasoning": "brief explanation of query strategy", "query_string": "SELECT..."}'
                )
                resp = st.session_state.gpt41_client.chat.completions.create(
                    model=st.session_state.GPT41_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": f"Generate a Cosmos DB query for this user question:\n\n{prompt_text}"},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                result = json.loads(resp.choices[0].message.content)
                q = result.get("query_string", "")
                keywords = result.get("keywords", [])
                reasoning = result.get("reasoning", "")
                logger.info(f"LLM Query Generation - Keywords: {keywords} | Reasoning: {reasoning}")
                return q or _generate_broad_query_fallback(prompt_text)
            except Exception as e:
                logger.error(f"LLM query generation failed: {e}")
                return _generate_broad_query_fallback(prompt_text)

        def _rank_results_by_relevance(results: list, prompt_text: str, top_k: int = 30) -> list:
            """
            Ranks search results by keyword match density.
            Returns top_k most relevant results.
            """
            # Extract meaningful keywords from prompt
            stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
                'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
                'tell', 'me', 'about', 'what', 'when', 'where', 'who', 'how', 'can', 'you', 'please'
            }
            keywords = [
                t.lower() for t in re.split(r"\W+", prompt_text)
                if t and len(t) > 2 and t.lower() not in stop_words
            ]

            if not keywords:
                return results[:top_k]

            # Score each result by keyword matches
            scored_results = []
            for r in results:
                score = 0
                content = str(r.get("content", "")).lower()
                filename = str(r.get("metadata", {}).get("original_filename", "")).lower()
                doc_id = str(r.get("id", "")).lower()
                question = str(r.get("question", "")).lower()
                answer = str(r.get("answer", "")).lower()

                full_text = f"{content} {filename} {doc_id} {question} {answer}"

                for kw in keywords:
                    # Count keyword occurrences (bonus for exact matches)
                    count = full_text.count(kw)
                    score += count * 10
                    # Bonus for filename match
                    if kw in filename:
                        score += 20
                    # Bonus for question match
                    if kw in question:
                        score += 15

                scored_results.append((score, r))

            # Sort by score descending and return top_k
            scored_results.sort(key=lambda x: x[0], reverse=True)
            ranked = [r for score, r in scored_results if score > 0][:top_k]
            logger.info(f"Ranked {len(results)} results down to {len(ranked)} relevant items")
            return ranked

        def _search_selected_kbs(broad_query: str, prompt_text: str) -> tuple[list, list]:
            """
            Executes search across selected containers.
            Returns (verified_facts, document_chunks)
            """
            selected_kbs = st.session_state.get("selected_containers", []) or []
            all_verified_facts, all_document_chunks = [], []

            # Extract meaningful keywords for VerifiedFacts matching
            stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
                'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
                'tell', 'me', 'about', 'what', 'when', 'where', 'who', 'how', 'can', 'you', 'please'
            }
            keywords = [
                t for t in re.split(r"\W+", prompt_text)
                if t and len(t) > 2 and t.lower() not in stop_words
            ][:5]

            if keywords:
                vf_clause = " OR ".join([f"CONTAINS(c.question, '{_safe(k)}', true)" for k in keywords])
            else:
                vf_clause = "CONTAINS(c.question, 'x', true)"

            for kb_path in selected_kbs:
                try:
                    db_name, cont_name = kb_path.split("/")
                except ValueError:
                    continue
                uploader = get_cosmos_uploader(db_name, cont_name)
                if not uploader:
                    continue

                if cont_name == "VerifiedFacts":
                    fact_query = f"SELECT TOP 10 * FROM c WHERE {vf_clause} ORDER BY c.verified_at DESC"
                    res = uploader.execute_query(fact_query)
                    for r in res:
                        r["_source_container"] = kb_path
                    all_verified_facts.extend(res)
                else:
                    res = uploader.execute_query(broad_query)
                    for r in res:
                        r["_source_container"] = kb_path
                    all_document_chunks.extend(res)

            # Rank document chunks by relevance
            if all_document_chunks:
                all_document_chunks = _rank_results_by_relevance(all_document_chunks, prompt_text, top_k=30)

            return all_verified_facts, all_document_chunks

        def _distill(verified_facts: list, chunks: list, prompt_text: str) -> str:
            """LLM distillation - all sources are trusted, with precedence rules."""
            if not verified_facts and not chunks:
                return "No relevant facts were found in the knowledge base."

            distillation_input = {
                "user_confirmed_facts": verified_facts,
                "document_sources": chunks
            }
            distillation_prompt = (
                "You are an expert AI data analyst. Distill the following JSON into key facts relevant "
                "to the user's question.\n\n"
                "# SOURCE TRUST LEVELS:\n"
                "1. **user_confirmed_facts**: Explicitly confirmed by users - HIGHEST priority, use these when available\n"
                "2. **document_sources**: From uploaded documents (PDFs, Word docs, etc.) - TRUSTED sources, treat as authoritative\n\n"
                "# RULES:\n"
                "- ALL sources are trusted and should be used\n"
                "- If user_confirmed_facts conflict with document_sources, prefer user_confirmed_facts (they may be corrections/updates)\n"
                "- Synthesize information from BOTH sources when they complement each other\n"
                "- Cite source types when relevant (e.g., 'According to uploaded documents...' or 'User confirmed that...')\n"
                "- Extract the most relevant information for the user's specific question\n\n"
                f"USER'S QUESTION: \"{prompt_text}\"\n\n"
                f"SEARCH RESULTS:\n{json.dumps(distillation_input, indent=2)}"
            )
            resp = st.session_state.gpt41_client.chat.completions.create(
                model=st.session_state.GPT41_DEPLOYMENT,
                messages=[{"role": "system", "content": distillation_prompt}],
            )
            return (resp.choices[0].message.content or "").strip()

        def _stream_synthesis(system_prompt: str, user_payload: str, placeholder) -> str:
            """Streams O3 synthesis and returns the full concatenated text."""
            stream = st.session_state.o3_client.chat.completions.create(
                model=st.session_state.O3_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                stream=True,
            )
            parts = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    parts.append(token)
                    placeholder.markdown("".join(parts) + " ▌")
            
            full_response = "".join(parts)
            placeholder.markdown(full_response) # Final update without the cursor
            return full_response

        # ----------------------------
        # Mode selection
        # ----------------------------
        persona_details, persona_type, persona_prompt_text, persona_temp = _persona()
        selected_kbs = st.session_state.get("selected_containers", []) or []
        intent = _classify_intent(user_prompt)

        # Debug logging for intent classification and KB selection
        logger.info(f"=== QUERY DEBUG === Intent: {intent} | Persona: {active_persona} ({persona_type}) | Selected KBs: {len(selected_kbs)} | KBs: {selected_kbs}")
        thinking_expander.write(f"Intent: **{intent}**  |  Persona: **{active_persona}** ({persona_type}) | KBs: **{len(selected_kbs)}**")

        # 1) Agentic personas stay agentic
        if persona_type == "agentic":
            logger.info(f"=== AGENTIC WORKFLOW ACTIVATED === User prompt: {user_prompt[:100]}...")
            try:
                final_answer = run_agentic_workflow(
                    user_prompt,
                    log_placeholder,
                    final_answer_placeholder,
                    query_expander
                )
                messages.append({"role": "assistant", "content": final_answer})
                save_user_data(st.session_state.user_id, st.session_state.user_data)
                st.session_state.session_rag_context = ""
                st.session_state.rag_file_status = None
            except Exception as e:
                st.error(f"An error occurred in the agentic workflow: {e}")
            st.stop()

        # 2) Fact correction (all personas)
        if intent == "fact_correction":
            try:
                agent_log = []
                with st.spinner("Interpreting new fact for confirmation..."):
                    fact_extraction_prompt = (
                        "You are an AI assistant. The user provided a new fact. "
                        "Rephrase it into a clear Question and Answer pair. Respond ONLY as "
                        '{"question":"...","answer":"..."} '
                        f'User statement: "{user_prompt}"'
                    )
                    resp = st.session_state.gpt41_client.chat.completions.create(
                        model=st.session_state.GPT41_DEPLOYMENT,
                        messages=[{"role": "system", "content": fact_extraction_prompt}],
                        response_format={"type": "json_object"},
                    )
                    qa_pair = json.loads(resp.choices[0].message.content)
                    question = qa_pair.get("question")
                    answer = qa_pair.get("answer")
                agent_log.append("✅ Fact interpreted for confirmation.")
                log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))

                if question and answer:
                    with final_answer_placeholder.container():
                        st.info("To improve the knowledge base, please review and save this fact:")
                        st.markdown(f"**Question:** `{question}`")
                        st.markdown(f"**Answer:** `{answer}`")
                        if st.button("Confirm and Save Fact", key=f"confirm_save_{active_chat_id}"):
                            save_verified_fact(question, answer)
                            messages.append({"role": "assistant", "content": "Fact saved successfully!"})
                            save_user_data(st.session_state.user_id, st.session_state.user_data)
                            st.rerun()
                st.stop()
            except Exception as e:
                st.error(f"An error occurred in the fact-correction flow: {e}")
                st.stop()

        # 3) KB intent → RAG (for ANY non-agentic persona, e.g., Pirate)
        if intent == "knowledge_base_query" and selected_kbs:
            logger.info(f"=== KB QUERY PATH ACTIVATED === User prompt: {user_prompt[:100]}...")
            try:
                agent_log = []

                # Step 1: Build broad query (LLM + safe fallback)
                with st.spinner("Step 1: Generating comprehensive search query..."):
                    broad_query = _generate_broad_query_via_llm(user_prompt)
                logger.info(f"Generated broad query: {broad_query[:200]}...")
                agent_log.append("✅ Step 1: Comprehensive query generated.")
                log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))
                with query_expander:
                    st.write("**Comprehensive Query for Document Stores:**")
                    st.code(broad_query, language="sql")

                # Step 2: Execute against selected KBs
                with st.spinner("Step 2: Searching selected knowledge bases..."):
                    vf, chunks = _search_selected_kbs(broad_query, user_prompt)
                agent_log.append(
                    f"✅ Step 2: Searched {len(selected_kbs)} KB(s) — {len(vf)} user-confirmed fact(s), {len(chunks)} document chunk(s)."
                )
                log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))

                # Display raw search results
                with query_expander:
                    st.write("**Raw Search Results:**")
                    if vf:
                        st.write(f"_User-Confirmed Facts ({len(vf)}):_")
                        st.json(vf[:5])  # Show first 5
                    if chunks:
                        st.write(f"_Document Chunks from Uploaded Sources ({len(chunks)}):_")
                        st.json(chunks[:10])  # Show first 10
                    if not vf and not chunks:
                        st.write("_No results found._")

                # Step 3: Distill (synthesize all trusted sources)
                with st.spinner("Step 3: Distilling all retrieved data from trusted sources..."):
                    distilled = _distill(vf, chunks, user_prompt)

                with query_expander:
                    st.write("**Distilled Key Facts:**")
                    st.markdown(distilled or "_No relevant facts were distilled._")

                # Merge ephemeral session uploads
                if st.session_state.session_rag_context:
                    distilled = (distilled or "") + "\n\n**From Current Session Upload:**\n" + st.session_state.session_rag_context

                # Step 4: Synthesize in persona voice, grounded ONLY in distilled facts
                synthesis_system_prompt = (
                    f"{persona_prompt_text} First, think step-by-step in a `<think>` block. "
                    "Then, synthesize a clear answer only from the provided key facts. "
                    "**If the provided key facts are insufficient or state 'No relevant facts were found', you MUST reply explicitly that you cannot answer from the knowledge base.**"
                )
                synthesis_user_payload = f"My question was: '{user_prompt}'\n\nHere are the distilled key facts:\n{distilled}"

                with st.spinner("Synthesizing final answer with O3..."):
                    full_response = _stream_synthesis(synthesis_system_prompt, synthesis_user_payload, final_answer_placeholder)

                thinking_content = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
                final_answer = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                final_answer_placeholder.markdown(final_answer)
                if thinking_content:
                    agent_log.append("✅ Final answer synthesized.")
                    log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))
                    thinking_expander.info(thinking_content.group(1).strip())

                messages.append({"role": "assistant", "content": full_response})
                save_user_data(st.session_state.user_id, st.session_state.user_data)
                st.session_state.session_rag_context = ""
                st.session_state.rag_file_status = None
                st.stop()

            except Exception as e:
                st.error(f"An error occurred in the retrieval process: {e}")
                # graceful fallthrough to simple quick path below

        # 4) Simple quick path (fallback/general conversation)
        # BUT: If containers are selected and query looks substantive, try KB search as fallback
        if intent == "general_conversation" and selected_kbs:
            # Heuristic: if prompt has >3 non-stop words, it might be a query misclassified
            substantive_words = [w for w in re.split(r"\W+", user_prompt.lower()) if len(w) > 3]
            if len(substantive_words) >= 2:
                logger.info(f"Intent was 'general_conversation' but {len(selected_kbs)} KB(s) selected and query appears substantive. Attempting KB search as fallback.")
                try:
                    agent_log = []
                    with st.spinner("Searching knowledge bases (fallback)..."):
                        broad_query = _generate_broad_query_via_llm(user_prompt)
                    agent_log.append("✅ Fallback: Query generated.")
                    log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))

                    with st.spinner("Querying selected knowledge bases..."):
                        vf, chunks = _search_selected_kbs(broad_query, user_prompt)
                    agent_log.append(f"✅ Fallback: Retrieved {len(vf)} user-confirmed fact(s), {len(chunks)} document chunk(s).")
                    log_placeholder.markdown("\n".join(f"- {s}" for s in agent_log))

                    # If we found results, proceed with distillation
                    if vf or chunks:
                        with st.spinner("Distilling retrieved data..."):
                            distilled = _distill(vf, chunks, user_prompt)

                        if st.session_state.session_rag_context:
                            distilled = (distilled or "") + "\n\n**From Current Session Upload:**\n" + st.session_state.session_rag_context

                        synthesis_system_prompt = (
                            f"{persona_prompt_text} First, think step-by-step in a `<think>` block. "
                            "Then, synthesize a clear answer only from the provided key facts. "
                            "**If the provided key facts are insufficient, you MUST reply explicitly that you cannot answer from the knowledge base.**"
                        )
                        synthesis_user_payload = f"My question was: '{user_prompt}'\n\nHere are the distilled key facts:\n{distilled}"

                        with st.spinner("Synthesizing final answer..."):
                            full_response = _stream_synthesis(synthesis_system_prompt, synthesis_user_payload, final_answer_placeholder)

                        thinking_content = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
                        final_answer = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                        final_answer_placeholder.markdown(final_answer)
                        if thinking_content:
                            thinking_expander.info(thinking_content.group(1).strip())

                        messages.append({"role": "assistant", "content": full_response})
                        save_user_data(st.session_state.user_id, st.session_state.user_data)
                        st.session_state.session_rag_context = ""
                        st.session_state.rag_file_status = None
                        st.stop()
                except Exception as e:
                    logger.error(f"Fallback KB search failed: {e}")
                    # Continue to simple quick path below

        # 4b) Simple quick path (fallback/general conversation)
        logger.info(f"=== SIMPLE QUICK PATH ACTIVATED === No KB query triggered. Intent: {intent}, KBs: {len(selected_kbs)}")
        try:
            with st.spinner("Synthesizing quick answer..."):
                _, _, system_prompt, temp = _persona()
                user_context = user_prompt
                if st.session_state.session_rag_context:
                    user_context += "\n\nContext from uploaded files:\n" + st.session_state.session_rag_context

                response = st.session_state.gpt41_client.chat.completions.create(
                    model=st.session_state.GPT41_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_context},
                    ],
                    temperature=temp,
                    max_tokens=700,
                )

                final_answer = response.choices[0].message.content.strip()
                final_answer_placeholder.markdown(final_answer)
                messages.append({"role": "assistant", "content": final_answer})
                save_user_data(st.session_state.user_id, st.session_state.user_data)

        except Exception as e:
            st.error(f"Quick mode failed: {e}")

