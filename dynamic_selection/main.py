import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from anthropic import AsyncAnthropic, DefaultAioHttpClient

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Use case: Job Application Screener — Fixed Pipeline vs. Dynamic Selection
#
# The narrow coordinator always runs the same three checks, no matter who
# applied. A clear mismatch gets the same scrutiny as a borderline candidate,
# which wastes calls (and money).
#
# The dynamic coordinator first analyzes the application, picks the pipeline
# that fits, explains its selection, and only then delegates:
#   - Clear knockout            → one knockout check, PASS
#   - Clear strong match        → skills + experience depth, short pipeline
#   - Borderline / unusual      → full pipeline (red flags, trade-offs, growth,
#                                 working style, compensation, team fit)
#
# We run both on three different applications and compare how many screening
# calls each one spends.
# ---------------------------------------------------------------------------

model = "claude-haiku-4-5-20251001"

SCREENING_AGENT_PROMPT = """You are a specialist hiring analyst. You will be given a specific
screening question about a candidate along with the job posting and resume.

Answer the question in 2-3 focused sentences. Be concrete and specific to this candidate."""

tools = [
    {
        "name": "screening_agent",
        "description": (
            "Runs a specialist screening agent on one specific angle of candidate fit. "
            "Call once per screening dimension with a focused question. "
            "Each call receives the full job posting and resume."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific screening question to evaluate",
                }
            },
            "required": ["question"],
        },
    }
]


async def call_screening_agent(client, question: str, job_posting: str, resume: str) -> str:
    r = await client.messages.create(
        model=model,
        max_tokens=200,
        system=SCREENING_AGENT_PROMPT,
        messages=[{"role": "user", "content": (
            f"QUESTION: {question}\n\n"
            f"JOB POSTING:\n{job_posting}\n\n"
            f"RESUME:\n{resume}"
        )}],
    )
    return r.content[0].text


# ---------------------------------------------------------------------------
# NARROW coordinator — fixed pipeline, runs every check on every candidate.
# ---------------------------------------------------------------------------

NARROW_COORDINATOR = """You are a job application screening coordinator.

Screen this candidate by running the following checks using the screening_agent tool:
1. Keyword match — do required skills appear in the resume?
2. Experience depth — does the candidate's background match the seniority level?
3. Red flags — are there disqualifying signals (gaps, job hopping, seniority mismatch)?

After all checks, synthesize a final HIRE / MAYBE / PASS recommendation."""

# ---------------------------------------------------------------------------
# DYNAMIC coordinator — analyzes the application, then picks the pipeline.
# ---------------------------------------------------------------------------

DYNAMIC_COORDINATOR = """You are a job application screening coordinator.

Before delegating, analyze the application and classify it:

- Clear knockout (missing a hard requirement, or far outside the seniority band)
  → ONE knockout check on the disqualifying gap, then PASS. No further screening.
- Clear strong match (all hard requirements visibly met, seniority fits, no odd signals)
  → skills match + experience depth only. Skip soft-skill and culture checks.
- Borderline or unusual (gaps in history, job hopping, partially met requirements,
  career changes, seniority at the edge of the band)
  → run the deeper pipeline, choosing only the angles that apply:
    red flags AND compensating strengths, growth trajectory,
    nice-to-have trade-offs, working style / team fit.
- Compensation, motivation, and timeline checks → only if the resume or posting
  gives something concrete to evaluate (e.g. a salary band, a relocation, a notice period).

Never call the screening_agent for an angle unless it adds value to THIS application.
Always explain your selection in a short paragraph (classification + chosen angles)
BEFORE making your first tool call.

After the chosen angles are covered, synthesize a final HIRE / MAYBE / PASS recommendation."""


async def run_coordinator(
    client, coordinator_prompt: str, label: str,
    job_posting: str, resume: str
) -> list[str]:
    print(f"\n  --- {label} ---")

    messages = [{"role": "user", "content": (
        f"Please screen this application.\n\n"
        f"JOB POSTING:\n{job_posting}\n\n"
        f"RESUME:\n{resume}"
    )}]

    delegated: list[str] = []
    step = 0

    while step < 30:
        step += 1
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=coordinator_prompt,
            tools=tools,
            messages=messages,
        )

        # Surface the coordinator's reasoning (selection explanation) as it happens
        for block in response.content:
            if block.type == "text" and block.text.strip():
                heading = "Final recommendation" if response.stop_reason == "end_turn" else "Selection reasoning"
                print(f"\n  {heading}:\n    " + block.text.strip().replace("\n", "\n    "))

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "screening_agent":
                    question = block.input["question"]
                    delegated.append(question)
                    print(f"\n    [{len(delegated):2d}] {question}")

                    result = await call_screening_agent(client, question, job_posting, resume)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages += [
                {"role": "assistant", "content": response.content},
                {"role": "user",      "content": tool_results},
            ]
    else:
        print("[WARN] Reached step limit.")

    return delegated


# ---------------------------------------------------------------------------
# Test applications — each calls for a different pipeline.
# ---------------------------------------------------------------------------

JOB_POSTING = """\
Title: Senior Python Backend Engineer
Required skills: Python (5+ years), FastAPI, PostgreSQL, Docker, REST API design
Nice to have: Kubernetes, Redis, system design at scale
Seniority: Senior (5-8 years total experience)"""

CASES = [
    (
        "Clear knockout — junior frontend developer, no Python",
        """\
Sam Rivera — 1 year experience
2024–present: Junior Frontend Developer, agency — React, CSS, JavaScript
Skills: React, HTML, CSS, JavaScript
Education: Coding bootcamp graduate""",
    ),
    (
        "Clear strong match — all requirements met, steady career",
        """\
Priya Nair — 7 years experience
2021–present: Senior Backend Engineer, logistics SaaS — Python, FastAPI, PostgreSQL, Docker, Kubernetes
2018–2021:    Backend Engineer, e-commerce — Python, FastAPI, PostgreSQL, Redis
2017–2018:    Junior Developer, startup — Python, Flask
Skills: Python, FastAPI, PostgreSQL, Docker, Kubernetes, Redis, REST APIs, system design
Education: B.Sc. Computer Science""",
    ),
    (
        "Borderline — job hopping, a gap, and a career change",
        """\
Alex Chen — 7 years experience
2024–present: Backend Engineer, FinTech startup — Python, FastAPI, PostgreSQL, Docker
2023–2024:    Career break (caring for family)
2022–2023:    Software Engineer, SaaS company — Python, Django, MySQL (8 months)
2021–2022:    Data Analyst, retail — SQL, Python (pandas)
2017–2021:    Junior Developer, agency — PHP, JavaScript
Skills: Python, FastAPI, PostgreSQL, Docker, REST APIs, some Redis
Education: B.Sc. Computer Science""",
    ),
]


async def main() -> None:
    async with AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        http_client=DefaultAioHttpClient(),
    ) as client:

        print("Same screening_agent spoke, same three applications.")
        print("Narrow coordinator: fixed 3-check pipeline every time.")
        print("Dynamic coordinator: picks the pipeline that fits each application.")

        results: list[tuple[str, int, int]] = []

        for title, resume in CASES:
            print(f"\n{'='*70}")
            print(f"CASE: {title}")
            print(f"{'='*70}")

            narrow_qs = await run_coordinator(
                client, NARROW_COORDINATOR, "NARROW (fixed pipeline)", JOB_POSTING, resume
            )
            dynamic_qs = await run_coordinator(
                client, DYNAMIC_COORDINATOR, "DYNAMIC (selects per application)", JOB_POSTING, resume
            )
            results.append((title, len(narrow_qs), len(dynamic_qs)))

        print(f"\n{'='*70}")
        print("SCREENING CALLS PER CASE")
        print(f"{'='*70}")
        print(f"\n  {'Case':<58} {'Narrow':>7} {'Dynamic':>8}")
        print(f"  {'-'*58} {'-'*7} {'-'*8}")
        for title, n, d in results:
            print(f"  {title[:58]:<58} {n:>7} {d:>8}")
        print(f"  {'-'*58} {'-'*7} {'-'*8}")
        print(f"  {'Total':<58} {sum(r[1] for r in results):>7} {sum(r[2] for r in results):>8}")

        print()
        print("  The fix is a coordinator prompt that:")
        print("  1. Classifies the application BEFORE delegating")
        print("  2. Maps each class to a pipeline (knockout / strong match / borderline)")
        print("  3. Forbids calling a spoke that adds no value to this application")
        print("  4. Explains its selection before the first tool call")


asyncio.run(main())
