# Agentic HR Policy Assistant

A deployed LLM-based HR assistant demonstrating retrieval-augmented generation
(RAG), agentic tool use, MCP integration, structured mock HR data, grounded
citations, confirmation-gated actions, evaluation, CI, and hosted deployment.

The project models a hypothetical Australian technology company,
**Promote Health Analytics Pty Ltd**.

## 1. Project Objective

The system helps employees answer HR policy questions and complete realistic HR
operations while remaining grounded in company policy and structured employee
data.

The final system demonstrates:

- policy RAG over an internal synthetic HR corpus;
- an LLM-based agent that plans and selects tools;
- eight MCP tools exposed through the official MCP SDK;
- structured mock employee, PTO, benefits, and ticket data;
- deterministic grounding and workflow guards where correctness is critical;
- confirmation-gated HR ACTION tools;
- structured citations and an observable operational trace;
- automated evaluation of answer quality and agent behavior;
- GitHub Actions CI;
- deployment to Google Cloud Run;
- browser-based demonstration workflows.

## 2. Architecture

```text
Browser UI
    |
    v
FastAPI (/chat, /health, static UI)
    |
    v
Agent Orchestrator
    |-- LLM planning and tool loop
    |-- grounding/workflow guards
    |-- trace and confirmation state
    |
    +--> LLM Provider
    |
    v
MCP Client (stdio)
    |
    v
FastMCP Server
    |
    +--> Policy/RAG tools --> Chroma --> BAAI/bge-small-en-v1.5
    |
    +--> Structured HR tools --> mock JSON data
```

The application is deployed as a single service. MCP communication remains
**stdio-only** inside the service; HTTP MCP was intentionally not introduced.

## 3. Core Technology

- Python 3.11
- FastAPI / Uvicorn
- official MCP SDK with FastMCP
- Chroma vector store
- `BAAI/bge-small-en-v1.5` sentence-transformer embeddings
- OpenAI-compatible LLM client
- pytest
- GitHub Actions
- Google Cloud Run
- Docker only as the Cloud Run packaging boundary

The architecture deliberately avoids unnecessary orchestration frameworks so
the RAG, MCP, agent, grounding, and confirmation behavior remain inspectable.

## 4. RAG Pipeline

The policy corpus is parsed into heading-aware chunks and embedded using
`BAAI/bge-small-en-v1.5`.

The deployed policy index contains:

- corpus version: `1.2`
- policy chunks: `400`
- embedding dimension: `384`
- vector store: Chroma
- retrieval default: `k=5`

The production container uses a build-produced policy index. It does not
rebuild or re-embed the policy corpus during runtime startup.

Policy answers expose structured citation data including document ID, title,
section, and supporting snippet.

Grounding guards prevent unsupported exact-section lookups from reaching MCP.

## 5. MCP Tools

The production MCP server exposes eight tools.

| Tool | Type | Purpose |
|---|---|---|
| `search_policy_documents` | READ | Semantic search over the policy corpus |
| `get_policy_section` | READ | Retrieve an exact grounded policy section |
| `lookup_employee_profile` | READ | Retrieve mock employee profile data |
| `lookup_benefits_status` | READ | Retrieve mock employee benefits state |
| `check_pto_balance` | CALCULATION | Retrieve PTO balance state |
| `check_policy_compliance` | CALCULATION | Apply deterministic policy-compliance logic |
| `create_mock_hr_ticket` | ACTION | Create a mock HR ticket |
| `draft_hr_email` | ACTION | Create a mock HR email draft |

ACTION tools do not implement user confirmation themselves. Confirmation is
enforced by the agent/API orchestration layer.

## 6. Agentic Workflows

### WF1 — International Remote Work

Demo request:

```text
I'm employee E003. Can I work remotely from overseas for six weeks?
```

The production workflow:

1. retrieves employee `E003`;
2. retrieves relevant remote-work policy evidence;
3. checks international remote-work compliance;
4. verifies required grounded policy evidence;
5. returns a deterministic cited outcome.

The deployed answer is grounded in `HR-POL-004 §4.4` and `HR-POL-004 §5.3`.
A six-week arrangement exceeds the ordinary 30-calendar-day pathway and
requires formal exception review.

### WF2 — PTO Request

Demo request:

```text
I'm employee E001. Can I take 3 days of PTO next week?
```

The production workflow:

1. retrieves employee `E001`;
2. checks PTO balance;
3. retrieves applicable PTO policy;
4. determines whether the request can proceed to manager review;
5. prepares a `draft_hr_email` ACTION proposal;
6. requires explicit confirmation;
7. executes only the exact action bound to that confirmation.

The demonstrated employee has `8.0` available PTO days for a `3` day request.

After confirmation, the exact pending action executes, the pending
confirmation is cleared, HR-POL-002 grounding is retained, and the generated
draft remains explicitly marked `MOCK — not sent`.

## 7. Evaluation

Evaluation uses the frozen 24-item set in `evaluation/eval_set.jsonl`.

It measures expected behavior, retrieval recall, groundedness, citation
accuracy, tool selection, workflow completion, action safety, runtime failure,
and end-to-end latency.

Published evidence includes:

- `evaluation/results/canonical-k5.json`
- `evaluation/results/postfix-subset-k5.json`
- `evaluation/results/postfix-subset-k5-final.json`
- `evaluation/results/ablation-k3.json`
- `evaluation/results/ablation-control-k5.json`
- `evaluation/results/ablation-k8.json`

### Retrieval-depth ablation

| Metric | k=3 | k=5 | k=8 |
|---|---:|---:|---:|
| Expected-behavior matches | 21/24 | 20/24 | 23/24 |
| Recall@k | 0.8889 | 0.9167 | 0.9167 |
| Groundedness | 0.6304 | 0.6250 | 0.5833 |
| Citation accuracy | 0.8402 | 0.8361 | 0.8433 |
| Tool-selection accuracy | 0.8750 | 0.7917 | 0.8750 |
| Workflow completion | 0.8750 | 0.8333 | 0.9583 |
| Action-safety pass rate | 1.0000 | 1.0000 | 1.0000 |
| Runtime failures | 1 | 0 | 0 |
| Mean item latency | 33.98 s | 23.16 s | 24.04 s |

The V1 system retains `k=5`: it matched k=8 on retrieval recall, had higher
groundedness, lower latency, and zero runtime failures. The stronger single-run
behavioral result at k=8 was not sufficient evidence to change the validated
default.

Detailed methodology, residuals, and limitations are recorded in
`design-and-evaluation.md`.

## 8. Final Verification

- agent regression: `131 passed`
- complete repository regression: `1435 passed`
- dependency consistency: PASS
- `git diff --check`: PASS
- GitHub Actions CI: PASS
- final CI run: `33494287703`
- candidate WF1: PASS
- candidate WF2 confirmation workflow: PASS
- production WF1: PASS
- production WF2 pre-confirmation: PASS
- production WF2 confirmation: PASS

## 9. Deployment

Production URL:

https://agentic-hr-policy-assistant-ykwvm3nhfq-ts.a.run.app

Health endpoint:

https://agentic-hr-policy-assistant-ykwvm3nhfq-ts.a.run.app/health

Final validated release:

- source commit: `aef0ddab770e17a7750971f365dd54f204517930`
- Cloud Run revision: `agentic-hr-policy-assistant-00020-kuv`
- immutable image digest: `sha256:3802b5da57611fb007e9f9061c12a3adb2a7b8463925c51141cd107b42185763`
- region: `australia-southeast1`
- production traffic: `100%`
- policy index: ready
- indexed chunks: `400`
- corpus version: `1.2`

Production health reports `status=ok`, `mcp=connected`, `index=ready`,
`index_chunks=400`, `corpus_version=1.2`, and `llm=ok`.

See `deployed.md` for deployment evidence and operational limitations.

## 10. Local Setup

Create and activate the Python 3.11 environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Configure environment variables from `.env.example`. Do not commit `.env`.

Build and publish the policy index:

```bash
python -m rag.index build
python -m rag.index publish
```

Run the application:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run the full regression:

```bash
python -m pytest -q
python -m pip check
git diff --check
```

## 11. Key Repository Areas

- `agent/` — LLM client, prompts, orchestrator, trace
- `app/` — FastAPI application and browser UI
- `mcp/` — FastMCP server and eight HR tools
- `rag/` — parsing, chunking, embedding, indexing, retrieval
- `corpus/` — synthetic HR policy corpus
- `mock_data/` — structured synthetic HR records
- `evaluation/` — gold set, evaluator, and published results
- `tests/` — regression, integration, and architecture tests

## 12. Known Limitations

- synthetic HR and policy data only;
- in-memory conversation/session state;
- mock ACTION tools do not update a real HR system;
- email drafts are not actually sent;
- no production authentication or authorization layer;
- single-service architecture;
- remote-provider latency can be significant;
- model-selected tool sequences can vary;
- evaluation includes same-model judging and single-run ablation comparisons;
- documented evaluation residuals were retained rather than overfit.

## 13. Project Outcome

The final hosted system combines grounded RAG, MCP-based tool use, agentic
workflow execution, human confirmation for ACTION tools, observable traces,
evaluation evidence, CI, and reproducible deployment in one inspectable
project.
