# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S5 — MCP Integration
Current checkpoint: R6E-E — lookup_employee_profile READ capability complete and published
Previous checkpoint: R6E-D — get_policy_section READ capability complete and published
Next checkpoint: S5 MCP — lookup_benefits_status READ tool implementation
Last updated: 2026-08-19

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
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

Advance S5 MCP after publication of the verified
`lookup_employee_profile(employee_id)` READ capability.

R6E-E is complete and published at commit `4b5e561`:

`feat(mcp): add employee profile read tool`.

Publication verification:

- push to `origin/main`: successful;

- remote advanced from `1806db0` to `4b5e561`;

- `HEAD`: `4b5e561`;

- `main`: `4b5e561`;

- `origin/main`: `4b5e561`;

- `origin/HEAD`: `4b5e561`;

- local `main` tracks `origin/main` without ahead/behind divergence;

- working tree after push: clean.

Published R6E-E technical evidence:

- `lookup_employee_profile(employee_id: str)` is production-registered;

- exact six-field response preserved:

  - `name`;

  - `role`;

  - `employment_type`;

  - `location`;

  - `manager_id`;

  - `start_date`;

- employee IDs remain exact and case-sensitive;

- nullable `manager_id` preserved;

- `start_date` preserved verbatim;

- framework-agnostic structured-data layer preserved;

- production MCP tool count: 3;

- production READ tools:

  - `search_policy_documents`;

  - `get_policy_section`;

  - `lookup_employee_profile`;

- all three expose `readOnlyHint=True`;

- real stdio MCP success invocation: verified;

- unknown-employee MCP error translation: verified;

- error `structuredContent=None`;

- no traceback leakage;

- same-session recovery after handled employee error: verified;

- E3 behavior tests: 9;

- E4 loader-failure tests: 3;

- E5 discovery/registration tests: 2;

- E6 live stdio tests: 2;

- total net-new E-series tests: 16;

- MCP collection: 58 tests;

- MCP regression: 58 passed;

- repository collection: 1018 tests;

- full repository regression: 1018 passed;

- dependency health: pass;

- compile checks: pass;

- diff hygiene: pass.

G3 is materially advanced but not complete. Three production READ tools
are now implemented, registered, discovered, and exercised through the
real MCP protocol boundary.

The remaining mock-data READ tools, calculation tools,
confirmation-gated ACTION tools, and later agent-through-MCP execution
remain pending.

The next frozen MCP capability is:

`lookup_benefits_status(employee_id)`.


## CP7 Embedding Completion

### Model contract

- Embedding model: `BAAI/bge-small-en-v1.5`
- Embedding dimension: 384
- Model context length: 512 tokens
- Default embedding batch size: 32
- Local inference via `sentence-transformers`
- Document embeddings are normalized
- Query embeddings use the BGE retrieval instruction
- Model loading is lazy and cached per Python process

### Canonical corpus validation

- Canonical chunks: 400
- Unique chunk IDs: 400
- Duplicate chunk IDs: 0
- Invalid chunk texts: 0
- Invalid token counts: 0
- Maximum real-corpus chunk length: 141 tokens
- Full embedding matrix: `(400, 384)`
- Embedding dtype: `float32`
- Non-finite embedding values: 0
- All corpus embeddings normalized: pass
- Canonical chunk-to-row association: pass
- First real chunk repeatability: pass
- Batch-size stability (16 vs 32): pass

### Numerical-stability evidence

Verified on:

- device: Apple MPS
- PyTorch: 2.13.0
- sentence-transformers: 5.6.1
- NumPy: 2.4.6

Acceptance tolerance:

- `rtol=1e-5`
- `atol=1e-5`

Results:

- repeated document embeddings: pass
- repeated query embeddings: pass
- ordering stability: pass
- batch-size invariance: pass
- maximum observed full-corpus batch difference: 0.0
- maximum observed first-chunk repeat difference:
  `1.1920928955078125e-07`

Cross-device or cross-platform byte-identical floating-point
results are not claimed.

### Performance evidence

Observed full-corpus embedding runs:

- 400 chunks in 7.256 seconds
- 400 chunks in 6.376 seconds
- measured throughput: approximately 62.73 chunks/second
- embedding matrix size: 614,400 bytes (~0.586 MiB)

These are observed local measurements, not performance
guarantees.

### Verification

- Focused embedding tests: 67 passed
- Full repository regression: 268 passed
- `git diff --check`: pass
- Working tree after CP7 validation: clean

### CP7 commits

- `e9ee1c2` — `feat(rag): add embedding model lifecycle`
- `508f9bc` — `feat(rag): add document embedding`
- `a872eac` — `feat(rag): add query embedding`

E4 numerical stability and E5 full-corpus validation required no
production-code changes and therefore no artificial commits were
created.

## S5 MCP Progress — 2026-08-18

### Verified milestones

- `93d226d` — `deps: add official MCP SDK`
  - pinned `mcp==1.29.0` and required transitive dependencies;
  - `pip check`: pass;
  - FastMCP, stdio, ToolAnnotations and `readOnlyHint` API verified.
- `3e1177a` — `feat(mcp): add stdio server foundation`
  - one FastMCP server;
  - explicit V1 `stdio` transport;
  - local `mcp/` remains non-package to avoid SDK shadowing.
- `a4ab00b` — `feat(mcp): add policy retrieval adapter foundation`
  - repository-root runtime bootstrap verified;
  - pure `RetrievalResult` to MCP response projection;
  - exact five-field schema: `doc_id`, `title`, `section`,
    `snippet`, `score`.
- `c0e3759` — `feat(mcp): add policy search composition`
  - `search_policy_documents(query, k=5)` implemented;
  - delegates validation/retrieval to `rag.retrieve`;
  - preserves retrieval ordering and similarity score;
  - lower-layer errors propagate unchanged.

### Verification evidence

- Active Chroma collection: `policy_chunks`, 400 records.
- Focused MCP suite: 21 passed.
- Full repository regression: 936 passed.
- `git diff --check`: pass before C4 commit.
- WF1 real-corpus search:
  - `HR-POL-004` Remote and Flexible Work Policy in top 5;
  - `HR-POL-005` Information Security and Acceptable Use Policy
    in top 5.
- WF2 real-corpus search:
  - `HR-POL-002` Paid Time Off Policy at ranks 1, 2, and 4.
- R6E-C5 local verification:
  - production `search_policy_documents` registration: pass;
  - `ToolAnnotations(readOnlyHint=True)`: pass;
  - production `list_tools()` discovery: exactly one tool;
  - discovered `query`: required string;
  - discovered `k`: optional integer with default 5;
  - focused registration/discovery tests: 4 passed;
  - complete MCP suite: 24 passed;
  - full repository regression: 984 passed;
  - `pip check`: pass;
  - `git diff --check`: pass.
- R6E-C6 local verification:
  - real stdio MCP subprocess initialization: pass;
  - `ClientSession.call_tool()` against the production server: pass;
  - successful production result: `isError=False`;
  - production structured result envelope:
    `structuredContent["result"]`;
  - frozen five-field policy evidence preserved:
    `doc_id`, `title`, `section`, `snippet`, `score`;
  - invalid `k=0`: MCP `CallToolResult(isError=True)`;
  - validation message preserved without traceback leakage;
  - same MCP session remained usable after a tool error;
  - CI-safe fixture-backed stdio invocation tests: 3 passed;
  - complete MCP suite: 27 passed;
  - full repository collection: 987 tests;
  - full repository regression: 987 passed;
  - no production code changed;
  - `pip check`: pass;
  - `git diff --check`: pass.
- R6E-D `get_policy_section` local verification:
  - existing S4 exact-section retrieval reused rather than reimplemented;
  - frozen MCP response `{title, section, text}` preserved;
  - complete section text preserved without truncation;
  - exact lookup remains catalogue-backed rather than Chroma-backed;
  - production tool count: 2;
  - production tools:
    - `search_policy_documents`;
    - `get_policy_section`;
  - both tools expose `readOnlyHint=True`;
  - `get_policy_section` discovery schema:
    - required `doc_id`: string;
    - required `section`: string;
  - real production stdio MCP invocation: pass;
  - clean missing-section MCP error translation: pass;
  - no traceback leakage;
  - same-session recovery after tool error: pass;
  - WF1 real-corpus exact-section validation: pass;
  - WF2 real-corpus exact-section validation: pass;
  - complete MCP collection: 42 tests;
  - complete MCP regression: 42 passed;
  - full repository collection: 1002 tests;
  - full repository regression: 1002 passed;
  - `python -m pip check`: pass;
  - `git diff --check`: pass;
  - publication:
    - commit: `281a5db` — `feat(mcp): add exact policy section read tool`;
    - push to `origin/main`: successful;
    - `HEAD`, `main`, `origin/main`, and `origin/HEAD` synchronized at
      `281a5db`;
    - working tree after push: clean.

- R6E-E `lookup_employee_profile` local verification:

  - framework-agnostic structured-data implementation added in
    `mcp/tools_data.py`;

  - repository-relative employee fixture loading preserved;

  - no environment-variable configuration added;

  - V1 no-cache decision preserved;

  - employee IDs remain exact and case-sensitive;

  - frozen response fields:

    - `name`;

    - `role`;

    - `employment_type`;

    - `location`;

    - `manager_id`;

    - `start_date`;

  - `manager_id` preserves `str | None`;

  - `start_date` is copied verbatim from the source fixture;

  - unknown employee error:
    `Employee not found: 'E999'.`;

  - missing-file loader failure: verified;

  - malformed-JSON loader failure: verified;

  - duplicate-employee-ID failure: verified;

  - production tool count: 3;

  - production READ tools:

    - `search_policy_documents`;

    - `get_policy_section`;

    - `lookup_employee_profile`;

  - all three expose `readOnlyHint=True`;

  - `lookup_employee_profile` discovery schema:

    - required `employee_id`: string;

  - real production stdio MCP invocation: pass;

  - clean unknown-employee MCP error translation: pass;

  - error `structuredContent=None`;

  - no traceback leakage;

  - same-session recovery after tool error: pass;

  - E3 behavior tests: 9;

  - E4 loader-failure tests: 3;

  - E5 discovery/registration tests: 2;

  - E6 live stdio tests: 2;

  - total net-new E-series tests: 16;

  - complete MCP collection: 58 tests;

  - complete MCP regression: 58 passed;

  - full repository collection: 1018 tests;

  - full repository regression: 1018 passed;

  - `python -m pip check`: pass;

  - compile checks: pass;

  - `git diff --check`: pass;

  - publication:

    - commit: `4b5e561` — `feat(mcp): add employee profile read tool`;

    - push to `origin/main`: successful;

    - remote advanced from `1806db0` to `4b5e561`;

    - `HEAD`, `main`, `origin/main`, and `origin/HEAD` synchronized at
      `4b5e561`;

    - working tree after push: clean.

- G3 status: advanced, not complete.
  - Python MCP-facing composition, registration, discovery, and live
    MCP invocation are verified.
  - The remaining MCP tools and agent-through-MCP execution remain
    pending.

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

Begin the next frozen S5 READ-tool capability:

`lookup_benefits_status(employee_id)`.

Continue the established engineering sequence:

1. inspect the frozen schema and mock benefits data;

2. inspect existing structured-data and MCP registration patterns;

3. define the smallest framework-agnostic composition contract;

4. implement one capability only;

5. add focused synthetic tests for genuine coverage gaps;

6. validate representative real mock-data cases;

7. register the completed capability with the correct READ annotation;

8. verify real MCP discovery and invocation;

9. run complete MCP and repository regression;

10. review, document, commit, push, and synchronize before advancing.

Do not begin calculation tools, agent integration, or ACTION tools while
the remaining READ MCP surface is incomplete.


## Last Updated

2026-08-19
