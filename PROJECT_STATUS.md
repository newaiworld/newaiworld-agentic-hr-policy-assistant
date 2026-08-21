# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S5 — MCP Integration
Current checkpoint: R6E-F4 — create_mock_hr_ticket ACTION capability implemented and fully verified locally
Previous checkpoint: R6E-F3 — check_policy_compliance CALCULATION capability complete and published
Next checkpoint: R6E-F4 governance and publication closure
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
  - R6E-F4 `create_mock_hr_ticket` ACTION capability — implemented and fully verified locally; publication pending
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

Close and publish the fully verified R6E-F4
`create_mock_hr_ticket(employee_id, category, summary)` ACTION capability.

Current verified-local S5 MCP state:

- production MCP tools: 7;
- current completed MCP tools: 7;
- final required MCP tools: 8;
- completed ACTION tools: 1;
- `create_mock_hr_ticket` MCP classification: `readOnlyHint=False`;
- remaining frozen MCP capability: `draft_hr_email`;
- MCP collection: 147;
- complete MCP regression: 147 passed;
- complete repository regression: 1107 passed.

R6E-F4 confirmation semantics remain deliberately separated from the MCP
primitive.

Under AD-06 and AD-10, the later agent/web confirmation layer owns action
preview generation, `pending_confirmation`, server-generated
`confirmation_id` binding, explicit confirmation, and gated execution.

The immediate objective is governance preservation, publication, and
local/remote synchronization. No further F4 production implementation is
required unless governance review exposes a genuine defect.


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

Close and publish R6E-F4 `create_mock_hr_ticket`.

R6E-F4 is implemented and fully verified locally.

Verified local baseline:

- production MCP tools: 7;
- current completed MCP tools: 7;
- final required MCP tools: 8;
- ACTION classification: `readOnlyHint=False`;
- ACTION-focused tests: 9 passed;
- real stdio ACTION tests: 2 passed;
- MCP collection: 147;
- complete MCP regression: 147 passed;
- complete repository regression: 1107 passed;
- dependency health: pass;
- compile checks: pass;
- production ticket fixture integrity: pass;
- residual temporary-file check: pass;
- `git diff --check`: pass.

The production ACTION primitive does not implement confirmation internally.

Under AD-06 and AD-10, preview generation, pending-confirmation state,
`confirmation_id` binding, explicit user confirmation, and gated execution
belong to the later agent/web confirmation layer.

Publication sequence:

1. complete F4 governance documentation;
2. verify preservation of historical F1/F2/F3 publication evidence;
3. verify production/test artifact hashes remain unchanged;
4. run final governance and diff review;
5. commit the complete R6E-F4 change set;
6. push to `origin/main`;
7. verify synchronized local and remote refs;
8. verify a clean post-push working tree.

Only after successful R6E-F4 publication begin the final frozen S5 MCP
capability:

`draft_hr_email`.

Do not begin agent orchestration or broader confirmation-workflow
implementation during this publication checkpoint.

## Last Updated

2026-08-21
