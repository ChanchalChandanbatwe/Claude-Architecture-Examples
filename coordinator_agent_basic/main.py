import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import json
from anthropic import AsyncAnthropic, DefaultAioHttpClient

load_dotenv(Path(__file__).parent.parent / ".env")

model = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Use case: job application scanner, built as HUB AND SPOKE.
#
#                       ┌── skill_gap ────┐
#                       ├── resume_tailor ┤
#   candidate ◄──► HUB ─┤                 │   (spokes never talk to each other;
#                       ├── cover_letter ─┤    every message goes through the hub)
#                       └── interview_prep┘
#
# The HUB (Coordinator) owns everything that needs judgment or control:
#   1. DECOMPOSE the request into subtasks
#   2. ASSESS complexity and ROUTE subtasks to spokes
#   3. AGGREGATE spoke outputs into one answer
# The SPOKES only do their one narrow job and return text to the hub.
# ---------------------------------------------------------------------------

JOB_POSTING = """Senior Backend Engineer — Acme Payments
Requirements: 5+ years Python, REST API design, PostgreSQL, AWS (ECS, SQS),
experience leading a small team. Nice to have: Kubernetes, payments domain experience."""

RESUME = """Jane Doe — Backend Engineer, 6 years experience
- Built Python/Flask REST APIs serving 2M requests/day at a logistics startup
- Designed PostgreSQL schemas and tuned slow queries (40% faster)
- Deployed services on AWS EC2 and Lambda
- Mentored 2 junior developers
Skills: Python, Flask, PostgreSQL, Docker, AWS (EC2, Lambda), Git"""

# ---------------------------------------------------------------------------
# SPOKES — each one is a single model call with its own system prompt.
# A spoke knows nothing about other spokes or the hub's plan. It receives a
# task (and optionally context) FROM the hub and returns text TO the hub.
# ---------------------------------------------------------------------------

SPOKE_PROMPTS = {
    "skill_gap": "You are a skill-gap analyst. Compare the résumé to the job posting. "
                 "List matched skills, missing skills, and a match score out of 10. Be concise.",
    "resume_tailor": "You are a résumé coach. Suggest 3 concrete bullet-point rewrites that "
                     "make the résumé fit the job posting. Do not invent experience.",
    "cover_letter": "You are a cover letter writer. Write a short 3-paragraph cover letter "
                    "for this candidate and job. Do not invent experience.",
    "interview_prep": "You are an interview coach. Give the 4 most likely interview questions "
                      "for this role, each with a one-line tip tailored to the candidate.",
}

async def run_spoke(client, name: str, task: str, context: str = "") -> str:
    system = f"{SPOKE_PROMPTS[name]}\n\nJOB POSTING:\n{JOB_POSTING}\n\nRÉSUMÉ:\n{RESUME}"
    if context:
        system += f"\n\nEARLIER FINDINGS (provided by the coordinator):\n{context}"
    response = await client.messages.create(
        model=model,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": task}],
    )
    return "".join(b.text for b in response.content if b.type == "text")

# ---------------------------------------------------------------------------
# HUB — the coordinator. The only component that sees the whole picture.
# ---------------------------------------------------------------------------

PLAN_TOOL = {
    "name": "create_plan",
    "description": "Break a job-application request into subtasks and assign each to a spoke.",
    "input_schema": {
        "type": "object",
        "properties": {
            "complexity": {
                "type": "string",
                "enum": ["simple", "complex"],
                "description": "simple = ONE spoke can handle it. complex = needs several spokes.",
            },
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "spoke": {"type": "string", "enum": list(SPOKE_PROMPTS)},
                        "task": {"type": "string", "description": "A self-contained instruction for that spoke"},
                    },
                    "required": ["spoke", "task"],
                },
            },
        },
        "required": ["complexity", "subtasks"],
    },
}

DECOMPOSE_PROMPT = """You are the coordinator of a job application team. You never do the work yourself.
Spokes available:
- skill_gap: compares résumé to posting, finds matched/missing skills
- resume_tailor: rewrites résumé bullets for the posting
- cover_letter: writes a cover letter
- interview_prep: predicts interview questions and gives tips

Decompose the request into the smallest set of subtasks needed.
A narrow request needs exactly one subtask. A broad request ("fully prepare me")
needs one subtask per relevant spoke."""

AGGREGATE_PROMPT = """You are the coordinator of a job application team. Combine the spoke
outputs into one application package with clear section headings.
Start with a 2-line overall verdict on the candidate's fit. Use only what the
spokes produced; do not add new claims about the candidate."""

class Coordinator:
    def __init__(self, client):
        self.client = client

    # --- Hub responsibility 1: DECOMPOSE + ASSESS COMPLEXITY ---------------
    async def decompose(self, request: str) -> dict:
        response = await self.client.messages.create(
            model=model,
            max_tokens=1024,
            system=DECOMPOSE_PROMPT,
            tools=[PLAN_TOOL],
            tool_choice={"type": "tool", "name": "create_plan"},
            messages=[{"role": "user", "content": request}],
        )
        return next(b.input for b in response.content if b.type == "tool_use")

    # --- Hub responsibility 2: ROUTE ---------------------------------------
    # Decided in code, from complexity and dependencies:
    #   simple  -> call the single spoke
    #   complex -> skill_gap first (others depend on it), then the rest in parallel.
    #              The hub relays skill_gap's findings; spokes never talk directly.
    async def route(self, plan: dict) -> list[dict]:
        subtasks = plan["subtasks"]

        async def call_spoke(subtask, context=""):
            print(f"  → Hub sends to [{subtask['spoke']}]: {subtask['task']}")
            answer = await run_spoke(self.client, subtask["spoke"], subtask["task"], context)
            print(f"  ← [{subtask['spoke']}] replied to hub")
            return {**subtask, "answer": answer}

        if plan["complexity"] == "simple":
            return [await call_spoke(subtasks[0])]

        results = []
        gap_tasks = [s for s in subtasks if s["spoke"] == "skill_gap"]
        other_tasks = [s for s in subtasks if s["spoke"] != "skill_gap"]

        context = ""
        if gap_tasks:
            gap_result = await call_spoke(gap_tasks[0])
            results.append(gap_result)
            context = gap_result["answer"]

        results += await asyncio.gather(*(call_spoke(s, context) for s in other_tasks))
        return results

    # --- Hub responsibility 3: AGGREGATE -----------------------------------
    async def aggregate(self, request: str, results: list[dict]) -> str:
        findings = "\n\n".join(f"### {r['spoke']}\n{r['answer']}" for r in results)
        response = await self.client.messages.create(
            model=model,
            max_tokens=2048,
            system=AGGREGATE_PROMPT,
            messages=[{"role": "user", "content": f"Request: {request}\n\nSpoke outputs:\n{findings}"}],
        )
        return next(b.text for b in response.content if b.type == "text")

    # --- Entry point: the only way in or out of the system ------------------
    async def handle(self, request: str) -> str:
        plan = await self.decompose(request)
        print(f"[1] Decomposed: complexity={plan['complexity']}, "
              f"spokes={json.dumps([s['spoke'] for s in plan['subtasks']])}")

        print("[2] Routing:")
        results = await self.route(plan)

        print("[3] Aggregating...")
        return await self.aggregate(request, results)

# ---------------------------------------------------------------------------
# Two requests with different complexity, to show the routing difference.
# ---------------------------------------------------------------------------

REQUESTS = [
    "Which skills am I missing for this job?",
    "I'm applying to this job tomorrow. Fully prepare me: fit check, résumé fixes, cover letter, and interview prep.",
]

async def main() -> None:
    async with AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        http_client=DefaultAioHttpClient(),
    ) as client:
        hub = Coordinator(client)
        for request in REQUESTS:
            print("=" * 70)
            print(f"Candidate: {request}\n")
            print(f"\nCoordinator:\n{await hub.handle(request)}\n")

asyncio.run(main())
