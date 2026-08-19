# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S5 — MCP Integration
Current checkpoint: R6E-D9 — get_policy_section READ tool implemented and verified locally; publication pending
Previous checkpoint: R6E-C6 — live MCP invocation complete and published
Next checkpoint: R6E-D10 — governance closure, commit, push, and synchronization verification
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
  - R6E-D `get_policy_section` READ capability — implemented and verified locally; publication pending
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

Close and publish the verified `get_policy_section` READ-tool capability
before advancing to the next frozen MCP tool.

The new exact-section capability is implemented through the existing
framework-agnostic MCP boundary:

`ClientSession` → stdio subprocess → FastMCP →
`get_policy_section` → `rag.retrieve.get_policy_section`.

The implementation preserves the frozen response contract:

- `title`;
- `section`;
- complete `text`.

Exact-section lookup remains catalogue-backed. No Chroma query,
embedding step, section matching, normalization logic, or truncation
was duplicated in the MCP layer.

Production MCP discovery now exposes exactly two READ tools:

- `search_policy_documents`;
- `get_policy_section`.

Both advertise `readOnlyHint=True`.

Verified `get_policy_section` evidence:

- plain-Python composition: pass;
- WF1 exact lookup:
  - `HR-POL-004`;
  - `5.3 International approval`;
- WF2 exact lookup:
  - `HR-POL-002`;
  - `9.1 Three-day request with sufficient balance`;
- production FastMCP registration/discovery: pass;
- real stdio MCP success invocation: pass;
- structured result `{title, section, text}`: pass;
- complete exact-section text preserved: pass;
- missing-section `RetrievalError` translated to a clean MCP
  `CallToolResult(isError=True)`;
- error `structuredContent=None`;
- no traceback leakage;
- same initialized MCP session remained usable after the error.

Current verified baseline:

- MCP collection: 42 tests;
- complete MCP regression: 42 passed;
- repository collection: 1002 tests;
- full repository regression: 1002 passed;
- dependency health: pass;
- `git diff --check`: pass.

The current D2–D8 implementation is verified locally but is not yet
claimed as published. The technical change set remains limited to:

- `mcp/server.py`;
- `mcp/tools_policy.py`;
- `tests/test_mcp.py`.

G3 is materially advanced but not complete. Two production READ tools
are now composed, registered, discovered, and exercised through MCP.
The remaining READ/CALCULATION/ACTION tools and later agent-through-MCP
execution remain pending.

After R6E-D publication closure, the next frozen MCP capability is:

`lookup_employee_profile(employee_id)`.

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
  - implementation publication: pending.

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

Close and publish the verified R6E-D `get_policy_section` capability:

1. update `design-and-evaluation.md` with the D2–D8 architecture,
   discovery, live-MCP, error, and regression evidence;
2. append the verified R6E-D development session to `ai-tooling.md`;
3. review the complete implementation, tests, and governance diff;
4. run consolidated MCP and repository verification;
5. stage only the intended implementation, test, and governance files;
6. commit the coherent R6E-D change set;
7. push to `origin/main`;
8. verify `HEAD`, `main`, `origin/main`, and `origin/HEAD` are
   synchronized;
9. only after publication closure, begin the next frozen READ capability:
   `lookup_employee_profile(employee_id)`.

Do not begin agent integration, calculation tools, or ACTION tools while
the remaining READ MCP surface is incomplete.

## Last Updated

2026-08-19
