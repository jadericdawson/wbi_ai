from __future__ import annotations
import json, logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context

from .config import SETTINGS, AzureClients
from .scratchpad.store import ScratchpadStore
from .orchestrator import Orchestrator
from .agents.planner import PlannerAgent
from .agents.actor import ActorAgent
from .agents.critic import CriticAgent
from .agents.aggregator import AggregatorAgent
from .agents.supervisor import SupervisorAgent
from .model_providers.gpt41_provider import GPT41Provider
from .model_providers.deepseek_provider import DeepSeekProvider
from .model_providers.aiproject_provider import AIProjectProvider
from .tools.http_tools import register_http_tools
from .tools.scratchpad_tools import register_scratchpad_tools
from .tools.fs_tools import register_fs_tools
from .tools.csv_tools import register_csv_tools
from .tools.json_tools import register_json_tools
from .tools.md_tools import register_md_tools
from .tools.table_tools import register_table_tools
from .tools.math_tools import register_math_tools
from .tools.schema_tools import register_schema_tools
from .tools.cache_tools import register_cache_tools
from .tools.aiprojects_tools import register_aiprojects_tools
# optional: from .tools.neo4j_tools import register_neo4j_tools

from .agents.router import RouterAgent
from .agents.researcher import ResearcherAgent
from .agents.retriever import RetrieverAgent
from .agents.ranker import RankerAgent
from .agents.normalizer import NormalizerAgent
from .agents.tableizer import TableizerAgent
from .agents.deduplicator import DeduplicatorAgent
from .agents.citationeer import CitationeerAgent
from .agents.pii_scrubber import PIIScrubberAgent
from .agents.security_validator import SecurityValidatorAgent
from .agents.format_director import FormatDirectorAgent

load_dotenv(override=True)
logger = logging.getLogger("actor_critic_mcp")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

mcp = FastMCP("actor-critic-team")

class AppState:
    store: ScratchpadStore
    orchestrator: Orchestrator

def _provider_from_settings(clients: AzureClients):
    mp = SETTINGS.model_provider
    if mp == "gpt41":
        return GPT41Provider(clients.gpt4_client, clients.gpt41_deployment)
    if mp == "deepseek":
        return DeepSeekProvider(clients.deepseek_client, clients.deepseek_deployment)
    if mp == "aiproject":
        return AIProjectProvider(clients.project_client, clients.ai_projects_agent_id)
    raise RuntimeError(f"Unknown MODEL_PROVIDER '{mp}'. Use: gpt41 | deepseek | aiproject")

@mcp.lifespan
async def lifespan(server: FastMCP):
    state = AppState()
    state.store = ScratchpadStore(SETTINGS.scratchpad_db)

    clients = AzureClients()  # EXACT initialization & verification you provided
    provider = _provider_from_settings(clients)

    planner = PlannerAgent("Planner", provider)
    actor = ActorAgent("Actor", provider)
    critic = CriticAgent("Critic", provider)
    aggregator = AggregatorAgent("Aggregator", provider)
    supervisor = SupervisorAgent("Supervisor", provider)
# in lifespan():
    router = RouterAgent("Router", provider)
    researcher = ResearcherAgent("Researcher", provider)
    retriever = RetrieverAgent("Retriever", provider)
    ranker = RankerAgent("Ranker", provider)
    normalizer = NormalizerAgent("Normalizer", provider)
    tableizer = TableizerAgent("Tableizer", provider)
    deduper = DeduplicatorAgent("Deduplicator", provider)
    cite = CitationeerAgent("Citationeer", provider)
    pii = PIIScrubberAgent("PIIScrubber", provider)
    sec = SecurityValidatorAgent("SecurityValidator", provider)
    formatter = FormatDirectorAgent("FormatDirector", provider)

    # Tools
    register_fs_tools(mcp, allow_dirs=["./", "./data", "./uploads"])
    register_csv_tools(mcp)
    register_json_tools(mcp)
    register_md_tools(mcp)
    register_table_tools(mcp)
    register_math_tools(mcp)
    register_schema_tools(mcp)
    register_cache_tools(mcp, state.store)
    register_aiprojects_tools(mcp, state.store, clients.project_client)
    # optional Neo4j:
    # register_neo4j_tools(mcp, uri="bolt://localhost:7687", user="neo4j", password="password", allow=False)

    state.orchestrator = Orchestrator(state.store, planner, actor, critic, aggregator, supervisor)

    register_http_tools(mcp, allowlist=SETTINGS.http_allowlist)
    register_scratchpad_tools(mcp, state.store)

    yield state

# ---------- RESOURCES ----------
@mcp.resource("scratchpad://{session_id}/{section}")
def read_section(session_id: str, section: str, *, context: Context) -> str:
    store: ScratchpadStore = context.server.app_state.store
    if section == "final":
        return store.get_final(session_id) or ""
    notes = store.get_notes(session_id, section=section)
    return "\n\n".join([n["content"] for n in notes]) if notes else ""

# ---------- PROMPTS ----------
@mcp.prompt()
def actor_prompt(section_id: str, objective: str, context_hint: str = "") -> str:
    return f"Section: {section_id}\nObjective: {objective}\nContext:\n{context_hint}\nWrite only this section."

# ---------- TOOLS ----------
@mcp.tool()
async def start_session(task: str, inclusion_policy: str = "all", metadata_json: str = "{}", *, context: Context):
    orch: Orchestrator = context.server.app_state.orchestrator
    sid = await orch.start(task, inclusion_policy, json.loads(metadata_json or "{}"))
    return {"session_id": sid}

@mcp.tool()
def list_sessions(*, context: Context):
    store: ScratchpadStore = context.server.app_state.store
    return store.list_sessions()

@mcp.tool()
async def run_all(session_id: str, max_rounds: int = 2, *, context: Context):
    orch: Orchestrator = context.server.app_state.orchestrator
    return await orch.run_all(session_id, max_rounds=max_rounds)

@mcp.tool()
def get_final(session_id: str, *, context: Context) -> str:
    store: ScratchpadStore = context.server.app_state.store
    return store.get_final(session_id) or ""

@mcp.tool()
def export_markdown(session_id: str, *, context: Context) -> str:
    store: ScratchpadStore = context.server.app_state.store
    return store.export_markdown(session_id)

@mcp.tool()
def add_context_chunk(session_id: str, title: str, content: str, *, context: Context) -> str:
    store: ScratchpadStore = context.server.app_state.store
    store.add_note(session_id, "user", "context", "user", f"### {title}\n\n{content}")
    return "ok"


@mcp.tool()
async def route(task: str, *, context: Context):
    # call RouterAgent, store plan in scratchpad
    store: ScratchpadStore = context.server.app_state.store
    sid = store.create_session(task)
    router_plan = await context.server.app_state.orchestrator.planner.plan(task)  # or router.route(task)
    store.add_note(sid, "router", "plan", "assistant", router_plan, tags=["router"])
    return {"session_id": sid, "router_plan": router_plan}

@mcp.tool()
async def research(session_id: str, topic: str, *, context: Context):
    store: ScratchpadStore = context.server.app_state.store
    researcher = ResearcherAgent("Researcher", _provider_from_settings(AzureClients()))  # quick resolve
    plan = await researcher.plan_research(topic)
    store.add_note(session_id, "researcher", "context", "assistant", plan, tags=["research"])
    return {"plan": plan}

@mcp.tool()
async def tableize_records(json_records: str, *, context: Context) -> str:
    tableizer = TableizerAgent("Tableizer", _provider_from_settings(AzureClients()))
    return await tableizer.tableize(json_records)

@mcp.tool()
async def scrub_pii(text: str, *, context: Context) -> str:
    pii_agent = PIIScrubberAgent("PIIScrubber", _provider_from_settings(AzureClients()))
    return await pii_agent.scrub(text)