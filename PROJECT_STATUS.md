# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S5 — MCP Integration
Current checkpoint: R6E-F5 — draft_hr_email ACTION capability complete and published
Previous checkpoint: R6E-F4 — create_mock_hr_ticket ACTION capability complete and published
Next checkpoint: S5 MCP completion reconciliation / transition to S6
Last updated: 2026-08-21

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
- S5 MCP — in progress
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
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

Reconcile S5 MCP completion and prepare the governed transition to S6 Agent.

R6E-F5 `draft_hr_email(to_role, subject, context)` is complete and published.

Published implementation commit:

- `3b04e21` — `feat(mcp): add mock HR email draft action tool`.

Post-push synchronization verified:

- `HEAD`: `3b04e21fd014a8d4a6c473211db97382fe85e238`;
- `main`: `3b04e21fd014a8d4a6c473211db97382fe85e238`;
- `origin/main`: `3b04e21fd014a8d4a6c473211db97382fe85e238`;
- `origin/HEAD`: `3b04e21fd014a8d4a6c473211db97382fe85e238`;
- working tree after implementation publication: clean.

Published S5 MCP state:

- production MCP tools: 8;
- current completed MCP tools: 8;
- final required MCP tools: 8;
- READ/CALCULATION tools: 6;
- ACTION tools: 2;
- both ACTION tools expose `readOnlyHint=False`;
- `draft_hr_email` accepts only `to_role`, `subject`, and `context`;
- permanent R6E-F5 test functions: 7;
- net-new collected MCP test items: 15;
- real stdio F5 tests: 2 passed;
- MCP collection: 162;
- complete MCP regression: 162 passed;
- complete repository regression: 1122 passed;
- dependency health: pass;
- compile checks: pass;
- diff hygiene: pass.

All eight frozen S5 MCP capabilities are now implemented, registered,
discoverable, tested through their required boundaries, and published.

The next objective is governance reconciliation of S5 completion before
production work begins on S6 Agent orchestration.


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

Reconcile and close S5 MCP Integration, then prepare the governed transition
to S6 Agent.

Published S5 state:

- all 8 frozen MCP tools are implemented;
- all 8 frozen MCP tools are registered and discoverable;
- all 8 frozen MCP tools are published;
- six READ/CALCULATION tools expose `readOnlyHint=True`;
- two ACTION tools expose `readOnlyHint=False`;
- F5 implementation publication is synchronized across all refs;
- MCP collection: 162;
- complete MCP regression: 162 passed;
- complete repository regression: 1122 passed;
- dependency health: pass;
- compile checks: pass;
- diff hygiene: pass.

Before beginning S6 production implementation:

1. review this governance-only R6E-F5 publication closure;
2. commit the closure documentation;
3. push the closure commit to `origin/main`;
4. verify `HEAD`, `main`, `origin/main`, and `origin/HEAD` synchronization;
5. perform explicit S5 completion reconciliation against the frozen project
   rules, implementation specification, and project status;
6. only then advance the active phase/checkpoint to S6 Agent.

Do not begin S6 production implementation before the governance closure and
S5 reconciliation are complete.


## Last Updated

2026-08-21
