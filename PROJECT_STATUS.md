# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant
Company: Promote Health Analytics Pty Ltd
Current phase: S4 — RAG Pipeline
Current checkpoint: CP7 Embeddings — complete
Next checkpoint: CP8 Chroma
Last updated: 2026-08-11

## Phase Progress

- S1 Foundation — complete
- S2 Policy Corpus — complete
- S3 Mock Data — complete
- S4 RAG — in progress
  - Repository/engineering readiness — complete
  - Manifest/source resolution — complete
  - Markdown/PDF parsing — complete
  - Deterministic normalization — complete
  - Exact token counting — complete
  - Heading-aware chunking — complete
  - Deterministic chunk IDs — complete
  - Canonical `corpus/processed/chunks.json` — complete
  - CP7 Embeddings — complete
  - CP8 Chroma — next
  - CP9 Retrieval and citations — pending
  - CP10 Retrieval validation — pending
- S5 MCP — not started
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## Current Objective

Close CP7 Embeddings and begin CP8 Chroma vector-index
construction and validation.

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

Begin S4 CP8 — Chroma:

1. inspect the pinned Chroma API and persistence behavior;
2. freeze collection/index metadata;
3. build a disposable index from canonical chunks and validated
   embeddings;
4. verify record count, IDs, metadata, and semantic query behavior;
5. publish the validated index safely.

Do not begin retrieval until CP8 is fully verified.

## Last Updated

2026-08-11
