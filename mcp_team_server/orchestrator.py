from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple

from .scratchpad.store import ScratchpadStore
from .agents.planner import PlannerAgent
from .agents.actor import ActorAgent
from .agents.critic import CriticAgent
from .agents.aggregator import AggregatorAgent
from .agents.supervisor import SupervisorAgent


def _safe_json_loads(text: str) -> Optional[dict]:
    """Lenient JSON parser. Returns dict or None."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # Attempt to find first/last braces if model wrapped with prose
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _extract_plan(plan_notes: List[dict]) -> dict:
    """
    Find the most recent note that contains a valid plan JSON with "sections".
    Each section requires: id, title, objective, acceptance (string or bullet list).
    """
    for n in reversed(plan_notes):
        j = _safe_json_loads(n.get("content", ""))
        if isinstance(j, dict) and "sections" in j and isinstance(j["sections"], list):
            # Normalize section fields
            norm_sections = []
            for s in j["sections"]:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("id") or s.get("section_id") or "").strip()
                title = str(s.get("title") or "").strip()
                objective = str(s.get("objective") or "").strip()
                acceptance = s.get("acceptance")
                if isinstance(acceptance, list):
                    acceptance_text = "\n".join(f"- {str(x)}" for x in acceptance)
                else:
                    acceptance_text = str(acceptance or "").strip()
                if sid and objective:
                    norm_sections.append(
                        {
                            "id": sid,
                            "title": title or sid,
                            "objective": objective,
                            "acceptance": acceptance_text or "Meets the objective exactly; no omissions.",
                        }
                    )
            if norm_sections:
                # keep a deterministic order (S1, S2, ... if present)
                def _sort_key(s: dict) -> Tuple[int, str]:
                    # Extract integer suffix if format like 'S12'
                    sid = s["id"]
                    num = 10**9
                    if len(sid) > 1 and sid[0].lower() == "s":
                        tail = sid[1:]
                        if tail.isdigit():
                            num = int(tail)
                    return (num, sid)
                norm_sections.sort(key=_sort_key)
                return {"sections": norm_sections}
    # Fallback empty if nothing usable found
    return {"sections": []}


def _gather_context(store: ScratchpadStore, sid: str) -> str:
    """
    Assemble prior notes into a context string for the Actor:
    - Any 'context' notes (user/tool-added raw evidence)
    - Prior actor drafts and critic reviews (useful for progressive build)
    """
    chunks: List[str] = []
    for n in store.get_notes(sid):
        sec = n.get("section", "")
        agent = n.get("agent", "")
        content = n.get("content", "")
        if sec in ("context", "actor", "critic"):
            chunks.append(f"[{sec}/{agent}] {content}")
    return "\n\n".join(chunks)


class Orchestrator:
    """
    Drives a multi-agent actor–critic workflow over a planned set of sections.

    Life cycle:
      1) start(task) -> creates session and writes the plan to scratchpad.
      2) run_all(session_id) -> for each planned section:
           Actor drafts -> Critic reviews (accept|revise up to N rounds)
           If accepted -> Aggregator appends verbatim to final; note stored
      3) run_selected(session_id, section_ids)
      4) rerun_section(session_id, section_id, reason)

    The scratchpad captures every step; final output lives in finals table.
    """

    def __init__(
        self,
        store: ScratchpadStore,
        planner: PlannerAgent,
        actor: ActorAgent,
        critic: CriticAgent,
        aggregator: AggregatorAgent,
        supervisor: SupervisorAgent,
    ):
        self.store = store
        self.planner = planner
        self.actor = actor
        self.critic = critic
        self.aggregator = aggregator
        self.supervisor = supervisor

    # ---------- Session ----------
    async def start(self, task: str, inclusion_policy: str = "all", metadata: Optional[dict] = None) -> str:
        sid = self.store.create_session(task, inclusion_policy, metadata or {})
        self.store.add_note(sid, "system", "plan", "system", f"Session created for task: {task}")
        # Plan
        plan_json = await self.planner.plan(task)
        self.store.add_note(sid, "planner", "plan", "assistant", plan_json, tags=["plan"])
        return sid

    # ---------- Core run helpers ----------
    def _load_plan(self, sid: str) -> dict:
        plan_notes = self.store.get_notes(sid, "plan")
        plan = _extract_plan(plan_notes)
        if not plan["sections"]:
            raise RuntimeError("No valid plan found. Ensure PlannerAgent produced a JSON with 'sections'.")
        return plan

    async def _run_section_once(
        self,
        sid: str,
        section: dict,
        extra_objective_suffix: str = "",
        context_hint: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """
        Run a single Actor->Critic round for a section.
        Returns (accepted, draft, critic_json_str)
        """
        sec_id = section["id"]
        objective = section["objective"]
        if extra_objective_suffix:
            objective = f"{objective}\n(Additional instruction: {extra_objective_suffix})"

        context = context_hint if context_hint is not None else _gather_context(self.store, sid)

        # Actor draft
        draft = await self.actor.draft(sec_id, objective, context)
        self.store.add_note(sid, "actor", "actor", "assistant", draft, tags=[sec_id])

        # Critic review
        review = await self.critic.review(sec_id, section["acceptance"], draft)
        self.store.add_note(sid, "critic", "critic", "validator", review, tags=[sec_id])

        # Decide accept / revise
        accepted = False
        instr = "Revise to strictly meet acceptance."
        j = _safe_json_loads(review)
        if isinstance(j, dict):
            dec = (j.get("decision") or "").lower().strip()
            if dec == "accept":
                accepted = True
            else:
                instr = j.get("revision_instructions", instr)

        return accepted, draft, instr

    async def _run_section_until_accept(
        self,
        sid: str,
        section: dict,
        max_rounds: int = 2,
        extra_objective_suffix: str = "",
    ) -> Tuple[bool, str]:
        """
        Iterate Actor↔Critic up to max_rounds. On accept, aggregate and store.
        Returns (accepted, final_draft_text)
        """
        final_text = self.store.get_final(sid) or ""
        sec_id = section["id"]

        accepted, draft, instr = await self._run_section_once(
            sid, section, extra_objective_suffix=extra_objective_suffix
        )
        rounds = 1
        while not accepted and rounds < max_rounds:
            # Ask Actor to revise using critic's instruction
            revised_objective_suffix = f"{extra_objective_suffix} | Revise: {instr}".strip(" |")
            accepted, draft, instr = await self._run_section_once(
                sid,
                section,
                extra_objective_suffix=revised_objective_suffix,
            )
            rounds += 1

        if accepted:
            # Append to final via Aggregator (keeps verbatim)
            assembled = await self.aggregator.assemble(final_text, draft)
            self.store.set_final(sid, assembled)
            self.store.add_note(sid, "aggregator", "final_draft", "assistant", draft, tags=[sec_id, "accepted"])
            return True, draft

        self.store.add_note(
            sid, "orchestrator", "final_draft", "system",
            f"Section {sec_id} not accepted after {max_rounds} round(s).",
            tags=[sec_id, "not_accepted"]
        )
        return False, draft

    # ---------- Public runs ----------
    async def run_all(self, sid: str, max_rounds: int = 2) -> dict[str, Any]:
        """
        Run Actor→Critic→Aggregator over every planned section.
        """
        session = self.store.get_session(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found.")
        task = session["task"]
        scratchpad = self.store.export_markdown(sid)

        supervisor_review = await self.supervisor.review(task, scratchpad)
        j = _safe_json_loads(supervisor_review)
        if isinstance(j, dict) and j.get("status") == "finished":
            return {"status": "finished", "final_available": bool(self.store.get_final(sid)), "report": {"accepted": [], "revised": [], "skipped": []}}


        plan = self._load_plan(sid)
        report: Dict[str, Any] = {"accepted": [], "revised": [], "skipped": []}

        for section in plan["sections"]:
            ok, _ = await self._run_section_until_accept(sid, section, max_rounds=max_rounds)
            if ok:
                report["accepted"].append(section["id"])
            else:
                report["revised"].append(section["id"])

        return {"status": "ok", "final_available": bool(self.store.get_final(sid)), "report": report}

    async def run_selected(self, sid: str, section_ids: List[str], max_rounds: int = 2) -> dict[str, Any]:
        """
        Run only the specified sections by id.
        """
        session = self.store.get_session(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found.")
        task = session["task"]
        scratchpad = self.store.export_markdown(sid)

        supervisor_review = await self.supervisor.review(task, scratchpad)
        j = _safe_json_loads(supervisor_review)
        if isinstance(j, dict) and j.get("status") == "finished":
            return {"status": "finished", "final_available": bool(self.store.get_final(sid)), "report": {"accepted": [], "revised": [], "skipped": section_ids}}


        plan = self._load_plan(sid)
        wanted = set([s.strip() for s in section_ids if s and s.strip()])
        report: Dict[str, Any] = {"accepted": [], "revised": [], "skipped": []}

        for section in plan["sections"]:
            if section["id"] not in wanted:
                report["skipped"].append(section["id"])
                continue
            ok, _ = await self._run_section_until_accept(sid, section, max_rounds=max_rounds)
            if ok:
                report["accepted"].append(section["id"])
            else:
                report["revised"].append(section["id"])

        return {"status": "ok", "final_available": bool(self.store.get_final(sid)), "report": report}

    async def rerun_section(self, sid: str, section_id: str, reason: str = "", max_rounds: int = 2) -> dict[str, Any]:
        """
        Force a single section to re-run with extra instructions (e.g., 'include table XYZ verbatim').
        """
        plan = self._load_plan(sid)
        target = None
        for s in plan["sections"]:
            if s["id"] == section_id:
                target = s
                break
        if not target:
            raise ValueError(f"Section id '{section_id}' not found in plan.")

        ok, draft = await self._run_section_until_accept(
            sid, target, max_rounds=max_rounds, extra_objective_suffix=reason or ""
        )

        return {
            "status": "ok",
            "accepted": ok,
            "section_id": section_id,
            "final_available": bool(self.store.get_final(sid)),
            "last_draft": draft,
        }