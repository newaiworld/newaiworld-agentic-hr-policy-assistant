# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S5 — MCP Integration
Current checkpoint: R6E-F0 — S5 reviewer/compliance remediation implemented and verified locally; publication pending
Previous checkpoint: R6E-E — lookup_employee_profile READ capability complete and published
Next checkpoint: R6E-F0 closure — governance review, commit, push, and synchronization verification
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
  - R6E-F0 S5 reviewer/compliance remediation — implemented and verified locally; publication pending
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

Close and publish the verified R6E-F0 S5 reviewer/compliance
remediation before beginning the next MCP tool capability.

R6E-F0 is implemented and verified locally. Publication remains pending
until the test-only change and governance evidence are reviewed,
committed, pushed, and synchronized.

Verified R6E-F0 findings:

- exact dependency checkpoint evidence for `mcp==1.29.0` was already
  committed and remains valid;

- production FastMCP registrations preserve
  `ToolAnnotations(readOnlyHint=True)`;

- live `list_tools()` discovery exposes `readOnlyHint=True` for all
  three currently published READ tools;

- committed MCP discovery tests already verify annotation propagation
  and the exact current production surface;

- READ, CALCULATION, and ACTION classifications remain frozen in
  `IMPLEMENTATION_SPEC.md`;

- confirmation behavior remains designed to consume discovered
  `readOnlyHint` metadata rather than a hardcoded action-tool registry;

- all 12 benefits records were cross-checked against employee data and
  HR-POL-007 benefit eligibility/coverage rules;

- all 12 benefit records satisfy the employment-type and 30-day
  coverage-commencement rules;

- benefit election states are internally consistent for eligible,
  pending, and ineligible records;

- the only verified reviewer/compliance gap was the absence of a
  source-level final eight-tool CI contract;

- that gap is now closed in `tests/test_mcp.py` through separate
  current-completed and final-required MCP tool-name contracts;

- production MCP behavior is unchanged;

- current production tool count remains 3;

- final required S5 MCP tool count is frozen at 8;

- MCP collection: 59 tests;

- MCP regression: 59 passed;

- repository collection: 1019 tests;

- full repository regression: 1019 passed;

- dependency health: pass;

- compile checks: pass;

- `git diff --check`: pass.

R6E-F0 technical scope is deliberately limited to:

`tests/test_mcp.py`.

No production source file, dependency, frozen specification, or runtime
tool registration was changed.

After R6E-F0 publication closure, the next frozen MCP capability is:

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

Close and publish R6E-F0.

Required closure sequence:

1. update `PROJECT_STATUS.md`, `design-and-evaluation.md`, and
   `ai-tooling.md` with the verified reviewer/compliance evidence;

2. preserve the distinction between:
   - the current completed three-tool production surface; and
   - the frozen final eight-tool S5 contract;

3. verify that no production runtime behavior changed;

4. reconfirm the 59 MCP / 1019 repository test baseline;

5. review the complete intended F0 file set;

6. stage only the verified F0 compliance/test and governance files;

7. commit the coherent R6E-F0 closure;

8. push to `origin/main`;

9. verify `HEAD`, `main`, `origin/main`, and `origin/HEAD`
   synchronization;

10. only after R6E-F0 is published, begin
    `lookup_benefits_status(employee_id)` with inspection and exact
    contract freeze.

Do not begin benefits implementation, calculation tools, agent
integration, or ACTION tools while R6E-F0 publication remains open.


## Last Updated

2026-08-20
