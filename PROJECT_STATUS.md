# PROJECT_STATUS.md — Project Dashboard (update at every session close)
# Records state; never governs. Rules live in PROJECT_RULES.md.

Project: Agentic HR Policy Assistant
Current phase: S1 — Foundation
Overall completion: ░░░░░░░░░░ 0%
Last session: YYYY-MM-DD

## Timeline
| Day | Date       | Focus                        | Status   |
|-----|------------|------------------------------|----------|
| 1   | YYYY-MM-DD | S1 repo + env                | planned  |
| 2   | YYYY-MM-DD | S2 corpus + S3 mock data     | planned  |
| 3   | YYYY-MM-DD | S4 RAG pipeline              | planned  |
| 4   | YYYY-MM-DD | S5 MCP + S6 agent            | planned  |
| 5   | YYYY-MM-DD | S7 web app                   | planned  |
| 6   | YYYY-MM-DD | S8 deploy + CI               | planned  |
| 7   | YYYY-MM-DD | S9 eval + S10 video/submit   | planned  |

## Phase Progress
- S1 Foundation   ⬜ not started
- S2 Corpus       ⬜ not started
- S3 Mock data    ⬜ not started
- S4 RAG          ⬜ not started
- S5 MCP          ⬜ not started
- S6 Agent        ⬜ not started
- S7 Web          ⬜ not started
- S8 Deploy+CI    ⬜ not started
- S9 Evaluation   ⬜ not started
- S10 Video+submit ⬜ not started

## Current Objective
Create a reproducible development environment.

## G-items Served (this phase)
G7 — CI/testing foundations
G9 — documentation/reproducibility

## Completed
- [x] Governance files committed (rules v3.2, spec v3.3,
      status v1.3, checklist v1.0)

## Evidence
Commit: <hash>
Environment: Python 3.11.x, venv
Tests: pytest — 1 passed
Files verified:
- [x] PROJECT_RULES.md
- [x] IMPLEMENTATION_SPEC.md
- [x] PROJECT_STATUS.md
- [x] SUBMISSION_CHECKLIST.md

## Risks
| Risk                          | Probability | Mitigation                  |
|-------------------------------|-------------|-----------------------------|
| MCP SDK lacks annotations     | medium      | S5 checkpoint + evidence    |
| Render cold-start slowness    | medium      | build-step ingestion (§6)   |
| LLM API failure/rate limit    | medium      | OpenRouter fallback (§1)    |
| Eval harness eats day 6       | medium      | required/optional split §11 |

## Blockers
None

## Next Action
Create venv, install dependencies, generate pinned
requirements.txt, get Groq API key into local .env only.

## Last Updated
YYYY-MM-DD
