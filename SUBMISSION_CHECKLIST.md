# Submission Checklist

Project: Agentic HR Policy Assistant
Status: final documentation and submission closure
Last updated: 2026-09-02

## 1. Core System Requirements

- [x] Working LLM-based agentic HR assistant implemented.
- [x] Retrieval-augmented generation over internal synthetic HR policy documents.
- [x] Structured mock HR data used for employee, PTO, benefits, and ticket workflows.
- [x] Agent can plan, select tools, and call MCP tools.
- [x] Official MCP SDK used with FastMCP.
- [x] MCP transport remains stdio-only.
- [x] Eight MCP tools implemented and discoverable.
- [x] Grounded policy citations returned to users.
- [x] Observable operational trace exposed through the API/UI.
- [x] Confirmation gate implemented for ACTION tools.

## 2. MCP Tool Surface

- [x] `search_policy_documents`
- [x] `get_policy_section`
- [x] `lookup_employee_profile`
- [x] `lookup_benefits_status`
- [x] `check_pto_balance`
- [x] `check_policy_compliance`
- [x] `create_mock_hr_ticket`
- [x] `draft_hr_email`

## 3. Demonstration Workflows

### WF1 — International Remote Work

- [x] E003 employee profile lookup.
- [x] Remote-work policy retrieval.
- [x] `check_policy_compliance` executed.
- [x] Required policy/compliance evidence enforced before completion.
- [x] Final grounding includes `HR-POL-004 §4.4` and `HR-POL-004 §5.3`.
- [x] Six-week request correctly exceeds the ordinary 30-calendar-day pathway.
- [x] Production WF1 acceptance passed.

### WF2 — PTO Request

- [x] E001 employee profile lookup.
- [x] PTO balance lookup shows 8.0 available days for a three-day request.
- [x] HR-POL-002 policy evidence retrieved.
- [x] Weak first retrieval can recover through one bounded deterministic retry.
- [x] `draft_hr_email` remains pending before explicit confirmation.
- [x] Matching confirmation executes only the bound pending action.
- [x] Pending confirmation clears after execution.
- [x] HR-POL-002 grounding retained after confirmation.
- [x] Result remains `MOCK — not sent`.
- [x] Production WF2 acceptance passed.

## 4. RAG and Evaluation

- [x] Policy corpus version `1.2`.
- [x] Heading-aware chunking implemented.
- [x] BAAI/bge-small-en-v1.5 embeddings used.
- [x] 384-dimensional normalized embeddings.
- [x] Chroma vector store used.
- [x] Published index contains 400 policy chunks.
- [x] Default retrieval depth remains `k=5`.
- [x] Frozen 24-item evaluation set present.
- [x] Canonical k=5 evaluation artifact present.
- [x] Controlled k=3/k=5/k=8 retrieval-depth ablation present.
- [x] Answer quality, retrieval recall, groundedness, citation accuracy, tool selection, workflow completion, action safety, failures, and latency measured.
- [x] Evaluation limitations documented.
- [x] Residual failures documented rather than hidden or overfit.

## 5. Verification and CI

- [x] Agent regression: 131 passed.
- [x] Full repository regression: 1435 passed.
- [x] `python -m pip check`: PASS.
- [x] `git diff --check`: PASS.
- [x] GitHub Actions CI run `33494287703`: success.
- [x] CI installs pinned dependencies.
- [x] CI checks dependency consistency.
- [x] CI validates FastAPI import.
- [x] CI builds and publishes the policy index.
- [x] CI runs the complete pytest suite.

## 6. Deployment

- [x] Hosted deployment available on Google Cloud Run.
- [x] Deployment remains compatible with the approved single-service architecture.
- [x] Docker used only as the Cloud Run packaging boundary.
- [x] Final source commit: `aef0ddab770e17a7750971f365dd54f204517930`.
- [x] Final Cloud Run revision: `agentic-hr-policy-assistant-00020-kuv`.
- [x] Final immutable image digest: `sha256:3802b5da57611fb007e9f9061c12a3adb2a7b8463925c51141cd107b42185763`.
- [x] Final validated revision receives 100% production traffic.
- [x] Production health reports MCP connected, index ready, 400 chunks, corpus 1.2, and LLM ok.
- [x] Candidate validated at zero production traffic before promotion.
- [x] Production WF1 smoke/acceptance passed after promotion.
- [x] Production WF2 confirmation workflow passed after promotion.

## 7. Required Submission Documentation

- [x] `README.md` — project overview, architecture, setup, demo, evaluation, deployment.
- [x] `design-and-evaluation.md` — architecture decisions, evaluation, remediation, limitations, S10 qualification.
- [x] `deployed.md` — hosted deployment and runtime evidence.
- [x] `ai-tooling.md` — AI-assisted engineering process and impact.
- [x] `PROJECT_STATUS.md` — current status and project history.
- [x] `PROJECT_RULES.md` — engineering governance.
- [x] `IMPLEMENTATION_SPEC.md` — frozen implementation specification.
- [x] `evaluation/eval_set.jsonl` — frozen gold set.
- [x] `evaluation/results/` — published evaluation outputs.
- [x] `.github/workflows/ci.yml` — reproducible CI definition.

## 8. Known Limitations Declared

- [x] Synthetic HR data and synthetic policy corpus only.
- [x] In-memory conversation/session state.
- [x] Mock HR ACTION tools only.
- [x] Email drafts are not sent.
- [x] No production authentication or authorization layer.
- [x] Single-service demonstration architecture.
- [x] Remote-provider latency can be significant.
- [x] Model-selected tool trajectories can vary.
- [x] Same-model judging and single-run ablation limitations documented.

## 9. Final Submission State

- [x] S1 Foundation complete.
- [x] S2 Policy Corpus complete.
- [x] S3 Mock Data complete.
- [x] S4 RAG complete.
- [x] S5 MCP complete.
- [x] S6 Agent complete.
- [x] S7 Web/API complete.
- [x] S8 Deployment and CI complete.
- [x] S9 Evaluation complete.
- [x] S10 production stabilization and runtime acceptance complete.
- [x] Final documentation-only commit published and synchronized.
- [x] Final post-publication submission audit completed.

The implementation is frozen. Remaining work is documentation publication,
final repository synchronization, and submission handoff only.
