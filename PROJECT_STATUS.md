# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S7 — Web
Current checkpoint: S7 — pre-implementation inspection pending
Previous checkpoint: S6 — Agent complete
Next checkpoint: S7 web/API architecture and contract inspection
Last updated: 2026-08-24

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
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

S6 Agent is complete.

Final S6 completion evidence:

- MCP tools are discovered dynamically through the real stdio MCP boundary;
- production agent code contains no hardcoded MCP tool registry;
- `agent/llm.py` remains the sole LLM/API owner;
- LLM timeout handling is bounded and controlled;
- MCP tool calls are bounded to the frozen timeout and runtime timeout /
  transport failure moves the MCP client to degraded state;
- the orchestration loop is bounded to six iterations with explicit
  `max_iterations` termination;
- operational traces record observable tool execution facts without hidden
  chain-of-thought;
- policy citations are accumulated through the real FastMCP result shape;
- ACTION classification derives from discovered `readOnlyHint=False`;
- unconfirmed ACTIONs do not execute;
- confirmation binds to the exact previewed tool and argument snapshot;
- wrong or detached confirmation IDs are rejected;
- WF1 Remote Work completes through real MCP, real retrieval, and compliance
  calculation;
- WF2 PTO completes through real MCP, policy retrieval, confirmation gating,
  and real mock ACTION execution;
- unknown-employee, MCP timeout/down, no-evidence, ambiguous-request, and
  sensitive-topic behaviors are covered by deterministic tests;
- sensitive-topic handling uses `PROMPT_VERSION = "1.1"` under `AD-S6-001`;
- agent regression: 44 passed;
- MCP regression: 162 passed;
- complete repository regression: 1166 passed;
- dependency health, compile checks, architecture guards, and diff hygiene:
  pass;
- published S6 implementation baseline:
  `2cc770b` — `feat(agent): add failure recovery and safety handling`;
- `HEAD`, `main`, `origin/main`, and `origin/HEAD` were synchronized at the
  published S6 implementation baseline.

S6 Agent therefore satisfies the frozen Agent Contract.

The remaining `POST /chat`, `conversation_id` session store,
`pending_confirmation` persistence by conversation, `/health` presentation,
and browser UI responsibilities belong to S7 Web/API integration. S7 must
consume the S6 agent contracts rather than duplicate MCP business logic or
confirmation semantics.

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

Begin S7 Web/API with inspection and contract reconciliation only.

Inspect and freeze:

1. the `POST /chat` request/response contract;
2. `conversation_id` generation and in-memory session ownership;
3. conversation history and `pending_confirmation` storage;
4. confirmation replay/binding at the HTTP boundary using the existing S6
   `PendingConfirmation` and `confirm_pending_action()` contracts;
5. `/health` reporting for MCP, index, corpus, and LLM state;
6. serialization of S6 citations and operational trace items;
7. browser UI requirements for chat, citation chips, trace display,
   WF1/WF2 buttons, and confirmation dialog;
8. preservation of stdio-only MCP transport;
9. the prohibition on duplicating MCP business logic in the web layer;
10. failure behavior across API/session boundaries.

Continue the established discipline:

`inspect → freeze contract → implement one capability → focused test →
real integration validation → full regression → architecture review →
governance review → commit → push`.

Do not implement S7 production code until its architecture and ownership
boundaries are inspected against the frozen specification.


## Last Updated

2026-08-24
