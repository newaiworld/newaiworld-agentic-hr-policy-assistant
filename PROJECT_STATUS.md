# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S10 — Demo and submission closure
Current checkpoint: S10 production runtime acceptance complete
Previous checkpoint: S9 evaluation and retrieval ablation complete
Next checkpoint: final documentation audit, publication, and submission handoff
Last updated: 2026-09-02

## Phase Progress

- S1 Foundation — complete
- S2 Policy Corpus — complete
- S3 Mock Data — complete
- S4 RAG — complete
  - Repository/engineering readiness — complete
  - Manifest/source resolution — complete
  - Markdown/PDF parsing — complete
  - Deterministic normalization — complete
  - Exact token counting — complete
  - Heading-aware chunking — complete
  - Deterministic chunk IDs — complete
  - Canonical `corpus/processed/chunks.json` — complete
  - CP7 Embeddings — complete
  - CP8 Chroma — complete
  - Retrieval and citation-ready results — complete
  - Exact policy-section lookup — complete
  - WF1/WF2 real-corpus retrieval validation — complete
- S5 MCP — complete
  - Official MCP SDK dependency gate — complete
  - FastMCP stdio server foundation — complete
  - Policy retrieval adapter/bootstrap foundation — complete
  - R6E-C4 `search_policy_documents` composition — complete
  - R6E-C5 FastMCP READ registration — complete and published
  - R6E-C6 live MCP invocation — complete and published
  - R6E-D `get_policy_section` READ capability — complete and published

  - R6E-E `lookup_employee_profile` READ capability — complete and published
  - R6E-F0 S5 reviewer/compliance remediation — complete and published
  - R6E-F1 `lookup_benefits_status` READ capability — complete and published
  - R6E-F2 `check_pto_balance` CALCULATION capability — complete and published
  - R6E-F3 `check_policy_compliance` CALCULATION capability — complete and published
  - R6E-F4 `create_mock_hr_ticket` ACTION capability — complete and published at `cf3e3f8`
  - R6E-F5 `draft_hr_email` ACTION capability — complete and published at `3b04e21`
- S6 Agent — complete
- S7 Web — complete and published
  - FastAPI application lifecycle and shared S6 resources — complete
  - `POST /chat` API boundary and conversation IDs — complete
  - in-memory session and pending-confirmation state — complete
  - confirmation replay through existing S6 contracts — complete
  - static browser UI — complete
  - citation and operational-trace presentation — complete
  - WF1/WF2 browser controls — complete
  - observational `/health` endpoint — complete
  - S7 implementation published at `8c7d4f2`
- S8 Deployment and CI — complete
- S9 Evaluation — complete
- S10 Demo and submission — in progress; production runtime acceptance complete

## Current Objective

S1 through S9 are complete. S10 production stabilization and runtime acceptance are also complete.

The final validated source baseline is
`aef0ddab770e17a7750971f365dd54f204517930`.

Google Cloud Run production is serving revision
`agentic-hr-policy-assistant-00020-kuv` from immutable image digest
`sha256:3802b5da57611fb007e9f9061c12a3adb2a7b8463925c51141cd107b42185763`
with 100% production traffic.

Production health reports `status=ok`, `mcp=connected`, `index=ready`,
`index_chunks=400`, `corpus_version=1.2`, and `llm=ok`.

Final production WF1 acceptance passed with grounded HR-POL-004 §4.4 and
§5.3 evidence plus the required compliance check. Final production WF2
acceptance passed through employee lookup, PTO balance lookup, policy RAG,
confirmation-gated `draft_hr_email`, exact pending-action execution,
citation retention, pending-confirmation clearing, and the `MOCK — not sent`
safeguard.

The current objective is documentation and submission closure only. No further
application, RAG, MCP, corpus, mock-data, dependency, or deployment changes are
planned unless the final audit exposes a genuine submission blocker.

Historical S7 completion evidence follows.

Final S7 completion evidence:

- FastAPI owns the web/API boundary while S6 retains agent orchestration;
- `GET /` serves the single static browser application;
- `POST /chat` delegates normal turns to the existing S6 `run_turn()` contract;
- browser conversations use generated `conversation_id` values and an
  in-memory session store;
- pending ACTION confirmation state is stored server-side by conversation;
- confirmation requests carry only `conversation_id`, `confirmed=true`, and
  the matching `confirmation_id`;
- confirmed execution delegates to the existing S6
  `confirm_pending_action()` contract;
- the browser does not supply MCP tool names or business arguments for
  confirmed actions;
- citations and observable operational trace items are serialized through
  the API and rendered in the browser UI;
- WF1 and WF2 browser controls are present;
- `GET /health` observes MCP connectivity, corpus version, policy-index
  availability, and indexed chunk count without executing agent or HR
  business workflows;
- the web layer contains no direct mock-data access and no direct MCP
  business-tool imports;
- unsafe HTML insertion APIs are absent from the static UI;
- S7 application tests: 12 passed;
- S6 agent regression: 44 passed;
- complete repository regression: 1178 passed;
- dependency health, compile checks, architecture guards, environment-
  independence checks, and diff hygiene: pass;
- frozen S6 `agent/llm.py`, `agent/orchestrator.py`, and `requirements.txt`
  hashes remained unchanged throughout S7;
- published S7 implementation:
  `8c7d4f2` — `feat(app): add S7 web and API boundary`;
- `HEAD`, `main`, `origin/main`, and `origin/HEAD` are synchronized at the
  published S7 implementation baseline.

S7 therefore satisfies the local Web/API integration scope without
duplicating MCP business logic or S6 confirmation semantics.

Deployment-specific build behavior, hosted-service validation, CI/deployment
evidence, cold-start measurement, and deployment documentation remain S8
responsibilities.

## Historical Published R6E-F5 Baseline

R6E-F5 `draft_hr_email(to_role, subject, context)` is complete and
published at commit `3b04e21`.

Published R6E-F5 implementation:

- commit:
  `3b04e21` — `feat(mcp): add mock HR email draft action tool`;

- push to `origin/main`:
  successful;

- synchronized refs:
  - `HEAD`: `3b04e21fd014a8d4a6c473211db97382fe85e238`;
  - `main`: `3b04e21fd014a8d4a6c473211db97382fe85e238`;
  - `origin/main`: `3b04e21fd014a8d4a6c473211db97382fe85e238`;
  - `origin/HEAD`: `3b04e21fd014a8d4a6c473211db97382fe85e238`;

- working tree after implementation push:
  clean.

Published R6E-F5 technical evidence:

- capability:
  `draft_hr_email(to_role, subject, context)`;

- semantic class:
  ACTION;

- MCP side-effect classification:
  `readOnlyHint=False`;

- production MCP tools:
  8;

- current completed MCP tools:
  8;

- final required MCP tools:
  8;

- READ/CALCULATION tools:
  6;

- ACTION tools:
  2;

- permanent R6E-F5 test functions:
  7;

- net-new collected MCP test items:
  15;

- real stdio F5 tests:
  2 passed;

- MCP collection:
  162;

- complete MCP regression:
  162 passed;

- complete repository regression:
  1122 passed;

- dependency health:
  pass;

- compile checks:
  pass;

- architecture boundary guards:
  pass;

- `git diff --check`:
  pass.

Published ACTION behavior:

- public response:
  `{draft_text, note}`;

- fixed mock marker:
  `note: "MOCK — not sent"`;

- draft behavior:
  deterministic formatting of exact caller-supplied `to_role`, `subject`,
  and `context`;

- LLM/API calls:
  none;

- policy retrieval / RAG / Chroma access:
  none;

- persistence:
  none;

- recipient resolution:
  none;

- confirmation/session state:
  deliberately outside the MCP primitive under AD-06 and AD-10.

Real stdio verification covers:

- successful ACTION discovery and invocation;
- `readOnlyHint=False`;
- exact three-field schema;
- protocol-visible validation failure;
- no traceback exposure;
- same-session recovery through a successful READ.

R6E-F5 is historical published evidence.

All eight frozen S5 MCP capabilities are now published.


## Historical Published R6E-F4 Baseline

R6E-F4 `create_mock_hr_ticket(employee_id, category, summary)` is complete
and published at commit `cf3e3f8`.

Published R6E-F4 implementation:

- commit:
  `cf3e3f8` — `feat(mcp): add mock HR ticket action tool`;

- push to `origin/main`:
  successful;

- synchronized refs:
  - `HEAD`: `cf3e3f8dc32ececd33409240ec30b6c21d571e7a`;
  - `main`: `cf3e3f8dc32ececd33409240ec30b6c21d571e7a`;
  - `origin/main`: `cf3e3f8dc32ececd33409240ec30b6c21d571e7a`;
  - `origin/HEAD`: `cf3e3f8dc32ececd33409240ec30b6c21d571e7a`;

- working tree after implementation push:
  clean.

Published R6E-F4 technical evidence:

- capability:
  `create_mock_hr_ticket(employee_id, category, summary)`;

- semantic class:
  ACTION;

- MCP side-effect classification:
  `readOnlyHint=False`;

- production MCP tools:
  7;

- current completed MCP tools:
  7;

- final required MCP tools:
  8;

- net-new R6E-F4 tests:
  33;

- ACTION-focused tests:
  9 passed;

- real stdio ACTION tests:
  2 passed;

- MCP collection:
  147;

- complete MCP regression:
  147 passed;

- complete repository regression:
  1107 passed;

- dependency health:
  pass;

- compile checks:
  pass;

- production ticket fixture integrity:
  pass;

- residual repository temporary-file check:
  pass;

- `git diff --check`:
  pass.

Published ACTION behavior:

- public response:
  `{ticket_id, status}`;

- public success marker:
  `status: "MOCK"`;

- persisted lifecycle state:
  `status: "open"`;

- timestamps:
  offset-aware UTC ISO-8601;

- ticket allocation:
  sequential from authoritative fixture state;

- persistence:
  validated complete-state publication through sibling temporary file and
  `os.replace()`;

- confirmation:
  deliberately outside the MCP primitive under AD-06 and AD-10.

The repository production ticket fixture remained pristine throughout
verification:

- ticket count: 4;
- next ticket number: 1005;
- ticket IDs:
  `TKT-1001` through `TKT-1004`.

R6E-F4 is historical published evidence.

The remaining frozen S5 MCP capability is:

`draft_hr_email`.


## Historical Published R6E-F3 Baseline

R6E-F3 `check_policy_compliance(topic, employee_id)` is complete and
published at commit `5987cdc`.

Published R6E-F3 implementation:

- commit:
  `5987cdc` — `feat(mcp): add policy compliance calculation tool`;

- push to `origin/main`:
  successful;

- remote advanced from:
  `8bd2962` to `5987cdc`;

- synchronized refs:
  - `HEAD`: `5987cdc`;
  - `main`: `5987cdc`;
  - `origin/main`: `5987cdc`;
  - `origin/HEAD`: `5987cdc`;

- working tree after push:
  clean.

Published R6E-F3 technical evidence:

- capability:
  `check_policy_compliance(topic, employee_id)`;

- semantic class:
  CALCULATION;

- MCP side-effect classification:
  `readOnlyHint=True`;

- production MCP tools:
  6;

- current completed MCP tools:
  6;

- final required MCP tools:
  8;

- net-new R6E-F3 tests:
  19;

- MCP collection:
  114;

- complete MCP regression:
  114 passed;

- repository collection:
  1074;

- complete repository regression:
  1074 passed;

- dependency health:
  pass;

- compile checks:
  pass;

- architecture review:
  pass;

- architecture decision:
  `AD-F3-001`;

- runtime policy boundary:
  no runtime RAG, Chroma, embeddings, or policy retrieval;

- `git diff --check`:
  pass.

The published `check_policy_compliance(topic, employee_id)` capability
preserves the frozen V1 policy-compliance boundary:

- supported topic:
  `remote_work_international`;

- employee identity is validated against authoritative mock data;

- governing policy facts were verified through the retrieval layer during
  engineering;

- production runtime executes deterministic frozen compliance logic;

- public response:
  `{compliant, reasons, policy_refs}`;

- MCP classification:
  `readOnlyHint=True`.

R6E-F3 is historical published evidence.

The subsequent R6E-F4 ACTION capability is tracked separately as the current
verified-local, publication-pending checkpoint.


## Historical Published R6E-F2 Baseline

R6E-F2 `check_pto_balance(employee_id)` is complete and published
at commit `60ec09b`.

Published R6E-F2 implementation:

- commit:
  `60ec09b` — `feat(mcp): add pto balance calculation tool`;

- push to `origin/main`:
  successful;

- remote advanced from:
  `101152e` to `60ec09b`;

- synchronized refs:
  - `HEAD`: `60ec09b`;
  - `main`: `60ec09b`;
  - `origin/main`: `60ec09b`;
  - `origin/HEAD`: `60ec09b`;

- working tree after push:
  clean.

Published R6E-F2 technical evidence:

- capability:
  `check_pto_balance(employee_id)`;

- semantic class:
  CALCULATION;

- MCP side-effect classification:
  `readOnlyHint=True`;

- production MCP tools:
  5;

- current completed MCP tools:
  5;

- final required MCP tools:
  8;

- net-new R6E-F2 tests:
  20;

- permanent-test ledger:
  - behavior/public contract: 10;
  - loader failures: 3;
  - fixture/policy consistency: 3;
  - discovery/registration: 2;
  - real stdio: 2;

- MCP collection:
  95;

- complete MCP regression:
  95 passed;

- repository collection:
  1055;

- full repository regression:
  1055 passed;

- dependency health:
  pass;

- compile checks:
  pass;

- architecture review:
  pass;

- `git diff --check`:
  pass.

The published `check_pto_balance(employee_id)` capability preserves:

- exact public response:
  `{available_days, accrual_rate, next_accrual_date}`;

- framework-agnostic structured-data access;

- authoritative stored `mock_data/pto.json` state;

- no employee-data runtime join;

- no policy/RAG runtime retrieval;

- no runtime entitlement calculation;

- no runtime FTE multiplication;

- no runtime date derivation;

- no synthetic contractor PTO record;

- `readOnlyHint=True`;

- real stdio invocation and same-session recovery verification.

Historical R6E-F1 publication evidence remains preserved:

- implementation commit:
  `755768f`;

- published MCP regression:
  75 passed;

- published full repository regression:
  1035 passed.


## S9 Evaluation Evidence — 2026-08-30

Status: evaluation execution, bounded remediation, and retrieval ablation complete; documentation closure in progress.

Published evidence:

- frozen 24-item gold set: `evaluation/eval_set.jsonl`;
- canonical k=5 baseline: `evaluation/results/canonical-k5.json`;
- bounded post-remediation subsets:
  - `evaluation/results/postfix-subset-k5.json`;
  - `evaluation/results/postfix-subset-k5-final.json`;
- controlled retrieval-depth evidence:
  - `evaluation/results/ablation-k3.json`;
  - `evaluation/results/ablation-control-k5.json`;
  - `evaluation/results/ablation-k8.json`.

Final controlled retrieval-depth results:

| Metric | k=3 | k=5 | k=8 |
|---|---:|---:|---:|
| Expected-behavior matches | 21/24 | 20/24 | 23/24 |
| Recall@k | 0.8889 | 0.9167 | 0.9167 |
| Groundedness | 0.6304 | 0.6250 | 0.5833 |
| Citation accuracy | 0.8402 | 0.8361 | 0.8433 |
| Tool selection | 0.8750 | 0.7917 | 0.8750 |
| Workflow completion | 0.8750 | 0.8333 | 0.9583 |
| Action safety | 1.0000 | 1.0000 | 1.0000 |
| Runtime failures | 1 | 0 | 0 |
| Mean item latency | 33.98 s | 23.16 s | 24.04 s |

Decision:

- retain frozen V1 retrieval default `k=5`;
- k=8 produced the strongest single-run expected-behavior/workflow result;
- k=5 matched k=8 on retrieval recall, had higher groundedness, lower mean/median/p95 latency, and zero runtime failures;
- no production-default change is justified from one run per setting;
- action safety remained 100% for all three retrieval depths.

Bounded post-remediation validation retained two explicit residual limitations:

- `MD05`: semantically safe escalation response remained deterministically classified as `answer` in the final seven-item subset;
- `OOS02`: repeated nominal document novelty allowed the bounded loop to terminate at `max_iterations`.

These residuals are documented rather than subjected to further item-specific tuning.

Evaluation limitations include same-model generation/judging and single-run remote-provider inference.

S9 technical evaluation work is therefore complete. Remaining S9 work is documentation consistency, publication, and handoff.

## Current Risks

| Risk | Probability | Mitigation |
|---|---|---|
| Stale or partially published Chroma index | medium | inspect persistence semantics before publication; validate temporary index before publishing |
| Index metadata mismatch | medium | compare corpus version and embedding/chunk configuration before reuse |
| Retrieval misses workflow-critical rules | medium | known-question and WF1/WF2 retrieval validation |
| Similarity threshold poorly calibrated | medium | inspect score distribution during retrieval evaluation |
| Model unavailable on fresh deployment | low-medium | deployment build pre-downloads frozen embedding model |

## Blockers

None.

## Next Action

Begin S8 Deployment and CI with inspection and contract reconciliation only.

Inspect the frozen deployment and CI requirements before changing production
code or configuration, including:

1. the existing GitHub Actions workflow and required test gates;
2. the Render/free-tier deployment contract;
3. the build command for dependency installation, embedding-model
   pre-download, and policy-index construction;
4. startup behavior against the build-produced Chroma index;
5. `/health` behavior during deployment readiness and cold start;
6. `/chat` graceful behavior while the index is unavailable or warming;
7. required environment variables and `.env.example` parity;
8. deployment persistence assumptions and ephemeral runtime writes;
9. cold-start timing evidence required for `deployed.md`;
10. preservation of stdio-only MCP and the existing S6/S7 boundaries.

Continue the established discipline:

`inspect → freeze contract → implement one capability → focused test →
real integration validation → full regression → architecture review →
governance review → commit → push`.

Do not add deployment complexity beyond the frozen S8 requirements.

## Last Updated

2026-08-24
## S8 Deployment and CI — Local Progress (2026-08-24)

S8 is in progress.

Verified completed local checkpoints:

- B2 deployment contract and policy-index lifecycle:
  - canonical `python -m rag.index {build,publish}` CLI added;
  - isolated hidden-build and publication behavior verified;
  - published index contains 400 policy chunks;
  - semantic retrieval against the isolated published index passed.
- B3 CI strengthening:
  - existing GitHub Actions CI retained;
  - explicit `from app.main import app` import gate added;
  - dependency consistency and full regression remain required.
- B4 deployment-index rehearsal:
  - isolated build, metadata validation, safe publication, currentness check,
    collection count, and real retrieval all passed;
  - repository-local `chroma_db` remained unchanged.
- B5 production-runtime rehearsal:
  - FastAPI startup passed;
  - MCP stdio discovery passed;
  - `/health` returned `status=ok`, `mcp=connected`, `index=ready`,
    `index_chunks=400`, `corpus_version=1.2`, and `llm=ok`.
- B5 provider compatibility:
  - the originally documented Groq model
    `llama-3.3-70b-versatile` was unavailable to the configured account;
  - Groq `openai/gpt-oss-120b` passed basic completion and function-tool
    compatibility but canonical multi-step execution repeatedly hit provider
    free-tier limits;
  - the frozen OpenRouter fallback was exercised through the existing
    OpenAI-compatible LLM boundary;
  - `openrouter/free` passed basic completion and function-tool compatibility
    and completed the canonical E003 six-week international-remote-work case.
- Prompt v1.2:
  - exact policy-section tools must use exact identifiers returned by evidence;
  - invented or paraphrased section names are prohibited;
  - live exact-section calls used valid identifiers.
- B5.7 structured citation propagation:
  - exact-section MCP responses remain frozen as `{title, section, text}`;
  - the orchestrator composes citation `doc_id` from the successful tool
    invocation arguments and the remaining citation fields from returned
    evidence;
  - full returned section `text` remains the V1 citation `snippet`;
  - no MCP response-contract, RAG, or retrieval schema change was required.

B5.7 verification evidence:

- focused citation verification: 6 passed;
- complete agent regression: 49 passed;
- complete repository regression: 1186 passed;
- dependency health: pass;
- diff hygiene: pass;
- strict live WF1 structured-citation gate: pass;
- live WF2 citation smoke: pass;
- post-request service health: pass.

Open items:

- B5.8 live workflow/tool-selection measurement is complete.
  Model-selected sequences remain variable, but the critical exact-section
  groundedness invariant is now enforced deterministically at the
  orchestration boundary rather than depending on prompt compliance alone.

- WF2 live retrieval produced broad citation accumulation; retrieval/citation
  precision remains deferred to formal S9 evaluation rather than altered
  inside the S8 groundedness repair.

- Render deployment, hosted health/WF1/WF2 validation, cold-start timing,
  and `deployed.md` remain pending.

Next checkpoint:

S8-B6 — Render deployment and hosted validation.

## S8-B5.8A R9 — MCP Environment Propagation Repair

Status: complete and remotely verified on 2026-08-25.

R9 resolved a cross-process runtime-configuration defect discovered while
making the real-MCP WF1/WF2 tests hermetic.

Root cause:

- the official MCP stdio client launches the child process with a restricted
  safe environment rather than the complete parent-process environment;
- `CHROMA_DIR` is a sanctioned V1 runtime setting but is not included in that
  default safe environment;
- therefore the FastAPI / agent parent process and MCP subprocess could
  resolve different policy indexes when `CHROMA_DIR` was explicitly set.

Repair:

- `AgentMCPClient` now forwards only the configured `CHROMA_DIR` value through
  `StdioServerParameters.env`;
- the MCP SDK continues to merge its own safe default environment;
- no arbitrary parent environment, LLM credentials, API keys, tool contracts,
  MCP transport settings, RAG behavior, prompt behavior, dependencies, or CI
  workflow behavior were changed.

Verification evidence:

- focused MCP environment contract tests: 4 passed;
- real isolated-index MCP subprocess proof: PASS;
- WF1/WF2 with repository `chroma_db` physically absent: 2 passed;
- full suite with repository `chroma_db` physically absent: 1190 passed;
- normal full regression after index restoration: 1190 passed;
- `python -m pip check`: PASS;
- local policy-index metadata SHA-256 remained
  `d2157dd266bc57121d8ccea029f7d433ee6b320e56720625bd28de41a8ad1e27`;
- technical commit:
  `df8ebdb` — `fix(agent): propagate Chroma config to MCP subprocess`;
- GitHub Actions CI run `32798529391`:
  `status=completed`, `conclusion=success`,
  `headSha=df8ebdbe035519d9c82449aa94df8adcd230706a`.

The minimal three-chunk workflow index is an integration fixture used to prove
real MCP / real retrieval plumbing and configuration isolation. It is not used
as evidence of retrieval ranking quality; retrieval quality remains owned by
the S4 retrieval tests and the formal S9 evaluation.

## S8-B5.8B-D — Live Workflow Measurement and Grounded Exact-Section Enforcement

Status: complete and remotely verified on 2026-08-25.

B5.8 measured live OpenRouter-selected WF1/WF2 behavior rather than assuming
that scripted workflow tests predicted model-selected tool sequences.

Observed live behavior:

- WF1 remained functionally correct while model-selected tool ordering varied;
- `search_policy_documents` was not consistently selected during WF1;
- WF2 initially used a weak semantic-search query and then proposed a
  nonexistent exact section;
- prompt v1.3 improved concrete semantic retrieval and required search recovery
  rather than unsupported section-name inference;
- repeated WF2 measurement improved materially under prompt v1.3 but still
  produced one unsupported exact-section proposal;
- this proved that prompt-only enforcement was insufficient for a
  correctness-critical grounding invariant.

Final orchestration decision:

- `run_turn()` maintains turn-local grounded policy selectors;
- successful `search_policy_documents` results establish exact
  `(doc_id, section)` selectors plus canonical numeric identifiers;
- `check_policy_compliance.policy_refs` establish exact numeric selectors;
- successful `get_policy_section` results may extend established selectors;
- unrelated tools establish no policy-section provenance;
- selector comparison is exact: no fuzzy matching, case folding, semantic
  equivalence, paraphrase matching, or inferred neighbouring sections;
- unsupported `get_policy_section` proposals are rejected before MCP
  execution;
- the trace records `section_guard_rejected`;
- the model receives a bounded corrective tool result and may search again
  or answer from existing grounded evidence.

Verification evidence:

- pure selector tests: 10 passed;
- guard/recovery integration tests: 3 passed;
- compatibility verification including real-MCP WF1/WF2: 6 passed;
- complete agent regression: 67 passed;
- complete repository regression: 1204 passed;
- dependency consistency: PASS;
- diff hygiene: PASS;
- guarded live WF2 reached `draft_hr_email` confirmation with zero tool errors;
- explicit confirmation executed exactly one mock `draft_hr_email` ACTION;
- guarded live WF1 used compliance-derived selectors for
  `HR-POL-004 §4.4`, `HR-POL-004 §8`, and `HR-POL-005 §4.5`;
- live WF1 completed with zero tool errors and zero incorrect guard rejections;
- post-workflow `/health` remained healthy;
- technical commit:
  `54a1a9b` — `fix(agent): enforce grounded policy section lookups`;
- GitHub Actions run `32802575960` completed successfully for
  `54a1a9b6ac65845f5fc09866c70225e7c281b16c`.

Evaluation scope:

B5.8 establishes deterministic exact-section groundedness and bounded recovery.
It does not claim optimal retrieval ranking, citation precision, or
deterministic LLM tool ordering. Those remain S9 evaluation dimensions.

Next:

S8-B6 — Render deployment and hosted validation.

## C8 Late-Stage Release Qualification — 2026-08-28

Status: in final S8 release qualification.

Qualified release evidence:

- production repair source published at `d79e5aa`;
- GitHub Actions CI run `33134626171`: success;
- MCP runtime contract:
  - startup/discovery outer bound: 30 seconds;
  - session read/runtime tool call: 60 seconds;
  - LLM call timeout: 30 seconds;
- `check_policy_compliance` retains the frozen V1 topic
  `remote_work_international`;
- its LLM-facing MCP schema now exposes that single allowed value as a
  machine-readable JSON Schema `const`;
- complete MCP regression: 163 passed;
- complete agent regression: 85 passed;
- complete repository regression: 1244 passed;
- exact frozen WF2 live acceptance: pass;
- invalid PTO use of `check_policy_compliance`: eliminated;
- `draft_hr_email` confirmation gate: pass;
- explicit confirmation: pass;
- confirmed mock ACTION execution count: exactly one.

The remaining S8 work is deployment closure, not feature development:
one final linux/amd64 release image, registry publication, Cloud Run deployment,
hosted acceptance, cold-start evidence, and `deployed.md`.

Retrieval ranking, citation precision, and model-selected sequence metrics remain
owned by S9 and are not tuned during S8.

## S8 Deployment and CI — Hosted Release Acceptance (2026-08-28)

Status: deployment acceptance complete.

Qualified release identity:

- published source: `2e315ce104824fb6759596b5333f47b68ed9231f`
- Cloud Run revision: `agentic-hr-policy-assistant-00002-9tw`
- immutable image digest: `sha256:c4c22fda7b5906f9832a7fe538edb9834585754306152db019e100a36e517afc`
- hosted URL: https://agentic-hr-policy-assistant-ykwvm3nhfq-ts.a.run.app
- region: `australia-southeast1`
- release platform: `linux/amd64`

Hosted acceptance evidence:

- Cloud Run Ready: PASS
- 100% traffic routed to the qualified ready revision
- root endpoint HTTP 200
- health endpoint HTTP 200
- MCP runtime connected
- Chroma policy index ready
- published policy chunks: 400
- corpus version: 1.2
- LLM runtime healthy
- hosted WF1 acceptance: PASS
- hosted WF2 acceptance: PASS
- confirmation gating: PASS
- confirmation-bound action execution: PASS
- exactly-once action execution: PASS
- confirmation replay protection: PASS

True request-driven scale-from-zero evidence was also captured. The
single cold-start health request completed successfully in 10.032
seconds client-observed time. Cloud Run logged an AUTOSCALING instance
start, with application readiness reached 8.927 seconds after instance
start and the startup probe succeeding after 8.935 seconds.

The earlier failed Cloud Run image was diagnosed as an image/runtime
architecture problem: the container terminated before application
startup with `failed to load /usr/bin/sh: exec format error`. No
application-source repair was required. A qualified `linux/amd64`
image was subsequently built, published, deployed by immutable digest,
and accepted successfully.

The deployed production runtime does not rebuild the RAG index during
startup. The 400-chunk Chroma index and required model artifacts are
prepared during the deployment image build lifecycle and used offline
at runtime.

S8 feature and deployment implementation is therefore complete at the
hosted release acceptance boundary. Remaining work proceeds to the next
governed project phase for evaluation, evidence consolidation, final
documentation, and submission readiness.

## S10 Final Production Acceptance — 2026-09-01

Status: production stabilization and runtime acceptance complete; documentation closure in progress.

Final release identity:

- source commit: `aef0ddab770e17a7750971f365dd54f204517930`;
- GitHub Actions CI run `33494287703`: success;
- Cloud Run revision: `agentic-hr-policy-assistant-00020-kuv`;
- immutable image digest: `sha256:3802b5da57611fb007e9f9061c12a3adb2a7b8463925c51141cd107b42185763`;
- production traffic: 100% on the validated revision;
- region: `australia-southeast1`.

Final local qualification before publication:

- agent regression: 131 passed;
- complete repository regression: 1435 passed;
- dependency consistency: PASS;
- `git diff --check`: PASS.

Production runtime health:

- `status=ok`;
- `mcp=connected`;
- `index=ready`;
- `index_chunks=400`;
- `corpus_version=1.2`;
- `llm=ok`.

WF1 production acceptance:

- E003 six-week international remote-work workflow: PASS;
- employee lookup: PASS;
- policy RAG: PASS;
- `check_policy_compliance`: PASS;
- grounded citations: `HR-POL-004 §4.4` and `HR-POL-004 §5.3`;
- premature completion defect: resolved;
- final pending confirmation: none.

WF2 production acceptance:

- E001 three-day PTO request with 8.0 available days: PASS;
- employee lookup and PTO balance lookup: PASS;
- HR-POL-002 policy grounding: PASS;
- `draft_hr_email` remained pending before confirmation: PASS;
- exact confirmation-bound action execution: PASS;
- pending confirmation cleared after execution: PASS;
- HR-POL-002 citation retained after confirmation: PASS;
- `MOCK — not sent` safeguard retained: PASS.

S10 runtime engineering is therefore frozen. Remaining work is documentation,
submission verification, documentation-only publication, and final handoff.
