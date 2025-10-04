import asyncio, json, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp_team_server.config import AzureClients, SETTINGS
from mcp_team_server.scratchpad.store import ScratchpadStore
from mcp_team_server.agents.planner import PlannerAgent
from mcp_team_server.agents.actor import ActorAgent
from mcp_team_server.agents.critic import CriticAgent
from mcp_team_server.agents.aggregator import AggregatorAgent
from mcp_team_server.model_providers.gpt41_provider import GPT41Provider
from mcp_team_server.model_providers.deepseek_provider import DeepSeekProvider
from mcp_team_server.model_providers.aiproject_provider import AIProjectProvider
from mcp_team_server.orchestrator import Orchestrator

def provider_from_env(clients):
    mp = SETTINGS.model_provider
    if mp == "gpt41":   return GPT41Provider(clients.gpt4_client, SETTINGS.gpt41_deployment)
    if mp == "deepseek":return DeepSeekProvider(clients.deepseek_client, SETTINGS.deepseek_deployment)
    if mp == "aiproject":return AIProjectProvider(clients.project_client, SETTINGS.ai_projects_agent_id)
    raise RuntimeError(f"Unknown MODEL_PROVIDER {mp}")

async def main():
    clients = AzureClients()
    provider = provider_from_env(clients)
    store = ScratchpadStore(SETTINGS.scratchpad_db)
    orch = Orchestrator(store,
                        PlannerAgent("Planner", provider),
                        ActorAgent("Actor", provider),
                        CriticAgent("Critic", provider),
                        AggregatorAgent("Aggregator", provider))
    sid = await orch.start("Three-part output: S1, S2, S3.", metadata={"rerun_demo": True})
    plan = {"sections":[
        {"id":"S1","title":"One","objective":"Write ONE line saying 'alpha'.","acceptance":["single line","contains alpha"]},
        {"id":"S2","title":"Two","objective":"Write ONE line saying 'bravo'.","acceptance":["single line","contains bravo"]},
        {"id":"S3","title":"Three","objective":"Write ONE line saying 'charlie'.","acceptance":["single line","contains charlie"]},
    ]}
    store.add_note(sid, "planner", "plan", "assistant", json.dumps(plan))
    print("Session:", sid)

    # Run only S1 & S3
    r_sel = await orch.run_selected(sid, ["S1","S3"], max_rounds=2)
    print("run_selected:", json.dumps(r_sel, indent=2))

    # Force rerun of S3 with extra reason
    r_re = await orch.rerun_section(sid, "S3", reason="Use uppercase CHARLIE", max_rounds=2)
    print("rerun_section:", json.dumps(r_re, indent=2))

    print("\nFINAL:\n", store.get_final(sid))

if __name__ == "__main__":
    asyncio.run(main())
