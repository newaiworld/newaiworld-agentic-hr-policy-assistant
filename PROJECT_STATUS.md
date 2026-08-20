# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S5 — MCP Integration
Current checkpoint: R6E-F1 — lookup_benefits_status READ capability implemented and verified locally; publication pending
Previous checkpoint: R6E-F0 — S5 reviewer/compliance remediation complete and published
Next checkpoint: R6E-F1 publication closure — governance review, commit, push, and synchronization verification
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

Close and publish the verified R6E-F1
`lookup_benefits_status(employee_id)` READ capability.

R6E-F1 is implemented and verified locally. Publication remains pending
until the complete implementation, tests, and governance evidence are
committed, pushed, and synchronized.

Verified R6E-F1 evidence:

- public response contract:
  `{elections, eligibility, coverage_start}`;

- structured-data source:
  `mock_data/benefits.json`;

- framework-agnostic implementation in `mcp/tools_data.py`;

- no employee-data runtime join;

- no policy or RAG recomputation;

- production registration through the existing `_load_data_tool()`
  mechanism;

- `lookup_benefits_status` exposes `readOnlyHint=True`;

- production MCP discovery now exposes exactly four completed READ tools:
  - `search_policy_documents`;
  - `get_policy_section`;
  - `lookup_employee_profile`;
  - `lookup_benefits_status`;

- frozen current/final MCP contract:
  - current completed tools: 4;
  - final required tools: 8;

- F1 permanent test ledger:
  - behavior: 9;
  - loader failures: 3;
  - discovery/registration: 2;
  - real stdio: 2;
  - total net new: 16;

- MCP collection: 75 tests;

- complete MCP regression: 75 passed;

- repository collection: 1035 tests;

- full repository regression: 1035 passed;

- dependency health: pass;

- compile checks: pass;

- architecture review: pass;

- `git diff --check`: pass.

R6E-F1 technical scope remains exactly:

- `mcp/tools_data.py`;
- `mcp/server.py`;
- `tests/test_mcp.py`.

Publication closure must now:

1. update `design-and-evaluation.md` with the verified R6E-F1
   design and evaluation evidence;

2. append the R6E-F1 development record to `ai-tooling.md`;

3. review all three governance files for consistent
   verified-local / publication-pending wording;

4. rerun the required verification gates if any executable code changes;

5. stage only the coherent R6E-F1 change set;

6. verify the staged file set and staged diff hygiene;

7. commit the R6E-F1 implementation and governance evidence;

8. push to `origin/main`;

9. verify `HEAD`, `main`, `origin/main`, and `origin/HEAD`
   are synchronized;

10. reconcile governance to complete-and-published only after
    publication is verified.

After R6E-F1 publication closure, the next frozen MCP capability is:

`check_pto_balance(employee_id)`.

Do not begin `check_pto_balance`, `check_policy_compliance`, agent
integration, or ACTION tools while R6E-F1 publication remains open.


## Last Updated

2026-08-20
