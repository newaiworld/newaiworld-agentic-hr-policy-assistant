# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S5 — MCP Integration
Current checkpoint: R6E-C4 — policy search composition complete
Previous checkpoint: R6E-C2/C3 — policy adapter/bootstrap foundation complete
Next checkpoint: R6E-C5 — FastMCP READ registration
Last updated: 2026-08-18

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
  - R6E-C5 FastMCP READ registration — next
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

Advance S5 MCP from the verified plain-Python policy-search
composition to FastMCP READ-tool registration.

R6E-C4 is implemented and verified. The next checkpoint is
R6E-C5: register `search_policy_documents` on the existing
FastMCP stdio server with `readOnlyHint=true`, without changing
retrieval semantics or the frozen tool schema.

G3 is advanced but not complete. MCP registration, discovery,
live MCP invocation, and later agent-through-MCP execution remain
to be verified.

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
- G3 status: advanced, not complete.
  - Python MCP-facing composition is verified.
  - FastMCP registration/discovery, live MCP invocation, and
    agent-through-MCP execution remain pending.

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

Begin R6E-C5 — FastMCP READ registration:

1. inspect the existing FastMCP server and pinned registration API;
2. keep `mcp/tools_policy.py` framework-agnostic;
3. register only `search_policy_documents` on the existing server;
4. set `ToolAnnotations(readOnlyHint=True)`;
5. verify discovery name, input schema, default `k=5`, and annotation;
6. run focused MCP and full repository regression tests;
7. review, commit, and push before advancing.

Do not begin live MCP `call_tool()` or agent integration until
R6E-C5 registration and discovery are verified.

## Last Updated

2026-08-18
