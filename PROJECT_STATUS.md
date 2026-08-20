# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S5 — MCP Integration
Current checkpoint: R6E-F3 — check_policy_compliance CALCULATION capability implemented and fully verified locally; publication pending
Previous checkpoint: R6E-F2 — check_pto_balance CALCULATION capability complete and published
Next checkpoint: R6E-F3 publication closure — governance review, commit, push, and synchronization verification
Last updated: 2026-08-20

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
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

Advance S5 MCP after publication of the verified R6E-F0
reviewer/compliance remediation.

R6E-F0 is complete and published at commit `c4783d3`:

`test(mcp): freeze final eight-tool contract`.

Publication verification:

- push to `origin/main`: successful;

- remote advanced from `1d369e2` to `c4783d3`;

- `HEAD`: `c4783d3`;

- `main`: `c4783d3`;

- `origin/main`: `c4783d3`;

- `origin/HEAD`: `c4783d3`;

- local `main` tracks `origin/main` without ahead/behind divergence;

- working tree after push: clean.

Published R6E-F0 technical evidence:

- exact `mcp==1.29.0` dependency evidence remains committed;

- FastMCP annotation support remains verified;

- live `list_tools()` discovery preserves `readOnlyHint=True`;

- committed CI discovery/annotation assertions remain valid;

- current production MCP tool count remains 3;

- final required S5 MCP tool count is frozen at 8;

- the final eight-tool source-level CI contract is now committed;

- benefits-policy consistency was verified across all 12 records;

- production runtime behavior is unchanged;

- MCP collection: 59 tests;

- MCP regression: 59 passed;

- repository collection: 1019 tests;

- full repository regression: 1019 passed;

- dependency health: pass;

- compile checks: pass;

- `git diff --check`: pass.

The reviewer/compliance checkpoint is therefore closed.

The next frozen MCP capability is:

`lookup_benefits_status(employee_id)`.


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

Close and publish the verified R6E-F3
`check_policy_compliance(topic, employee_id)` CALCULATION capability.

R6E-F3 is implemented and fully verified locally. Publication remains
pending.

### Verified R6E-F3 evidence

Capability:

`check_policy_compliance(topic, employee_id)`.

Supported V1 topic:

`remote_work_international`.

Frozen public response:

- `compliant`;
- `reasons`;
- `policy_refs`.

Semantic classification:

- project semantic class:
  CALCULATION;
- MCP side-effect classification:
  `readOnlyHint=True`.

Verified production architecture:

- framework-agnostic implementation in `mcp/tools_data.py`;
- existing `_load_data_tool()` registration path in `mcp/server.py`;
- employee identity validated through authoritative mock employee data;
- compliance decision does not branch on employee profile attributes;
- no runtime RAG or policy retrieval;
- no Chroma or embedding dependency;
- no environment reads;
- fresh mutation-isolated result projection;
- frozen stdio transport preserved.

Verified policy grounding used during engineering:

- reason 1:
  - `HR-POL-004 §4.4`;
  - `HR-POL-004 §8`;
- reason 2:
  - `HR-POL-004 §8`;
  - `HR-POL-005 §4.5`.

Verified MCP state:

- production MCP tools:
  6;
- current completed MCP contract:
  6;
- final required MCP contract:
  8.

Verified R6E-F3 test ledger:

- behavior/public-contract tests:
  13;
- architecture tests:
  2;
- discovery/registration tests:
  2;
- real stdio tests:
  2;
- net-new R6E-F3 tests:
  19.

Verified regression baseline:

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
- `git diff --check`:
  pass.

R6E-F3 publication closure still requires:

1. record the consequential F3 architecture decision in
   `design-and-evaluation.md`;
2. record the explicit reason-to-policy-reference grounding map and
   corpus-change revalidation trigger;
3. append the R6E-F3 development record and engineering lessons to
   `ai-tooling.md`;
4. perform cross-document governance consistency review;
5. stage only the coherent R6E-F3 implementation, tests, and governance
   evidence;
6. commit the implementation/governance state;
7. push to `origin/main`;
8. verify `HEAD == main == origin/main == origin/HEAD`;
9. reconcile governance to the final published state in a separate
   governance-closure commit.

After R6E-F3 publication closure, the next frozen MCP capability is:

`create_mock_hr_ticket`.

That capability is an ACTION tool.

Do not begin `create_mock_hr_ticket`, `draft_hr_email`, agent orchestration,
or confirmation-workflow implementation until R6E-F3 is fully governed,
committed, pushed, synchronized, and publication-closed.

## Last Updated

2026-08-20
