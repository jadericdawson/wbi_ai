import asyncio, json, sys
from pathlib import Path

# Ensure package import works when running from repo root
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

def provider_from_env(clients: AzureClients):
    mp = SETTINGS.model_provider
    if mp == "gpt41":
        return GPT41Provider(clients.gpt4_client, SETTINGS.gpt41_deployment)
    if mp == "deepseek":
        return DeepSeekProvider(clients.deepseek_client, SETTINGS.deepseek_deployment)
    if mp == "aiproject":
        return AIProjectProvider(clients.project_client, SETTINGS.ai_projects_agent_id)
    raise RuntimeError(f"Unknown MODEL_PROVIDER '{mp}'")

async def main():
    # 1) Azure clients (your exact setup)
    clients = AzureClients()

    # 2) Provider + agents
    provider = provider_from_env(clients)
    planner = PlannerAgent("Planner", provider)
    actor = ActorAgent("Actor", provider)
    critic = CriticAgent("Critic", provider)
    aggregator = AggregatorAgent("Aggregator", provider)

    # 3) Store + orchestrator
    store = ScratchpadStore(SETTINGS.scratchpad_db)
    orch = Orchestrator(store, planner, actor, critic, aggregator)

    # 4) Start session
    task = "Produce a two-section answer: (S1) SRR entrance criteria checklist; (S2) Explanation of why each matters."
    sid = await orch.start(task, inclusion_policy="all", metadata={"smoke": True})
    print(f"\nSession: {sid}")

    # 5) Seed deterministic plan
    seed_plan = {
        "sections":[
            {
                "id":"S1",
                "title":"SRR Entrance Criteria Checklist",
                "objective":"List the SRR entrance criteria as a bullet checklist only.",
                "acceptance":[
                    "Bulleted list only",
                    "Each bullet is a concise criterion label"
                ]
            },
            {
                "id":"S2",
                "title":"Why Each Entrance Item Matters",
                "objective":"Explain why each item in S1 matters, in 1–2 sentences each.",
                "acceptance":[
                    "One explanation per bullet from S1",
                    "Max 2 sentences each"
                ]
            }
        ]
    }
    store.add_note(sid, "planner", "plan", "assistant", json.dumps(seed_plan), tags=["plan","seeded"])

    # 6) Add context chunk
    example_json = {
        "entrance_minimal": [
            "Requirements baseline available",
            "Stakeholders identified",
            "Risk register drafted",
            "Initial cost/schedule range",
            "Top-level architecture view"
        ]
    }
    store.add_note(sid, "user", "context", "user",
                   f"### Minimal SRR entrance example\n\n```json\n{json.dumps(example_json, indent=2)}\n```",
                   tags=["context"])

    # 7) Run orchestrator
    result = await orch.run_all(sid, max_rounds=2)
    print("\nRun-all report:", json.dumps(result, indent=2))

    # 8) Show final
    final = store.get_final(sid)
    print("\n----- FINAL OUTPUT -----\n")
    print(final or "(no final)")
    print("\n------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
