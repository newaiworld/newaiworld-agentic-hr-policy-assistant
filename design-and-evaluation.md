# Design and Evaluation

## Project Status

S1–S4 are complete and verified. The project is now in S5 — MCP Integration, with the FastMCP stdio server foundation, policy retrieval adapter, and `search_policy_documents(query, k=5)` composition completed. The next checkpoint is R6E-C5 — FastMCP READ registration. S6–S10 remain pending and are not yet claimed as implemented. Architecture and evaluation evidence will continue to be added only as each capability is implemented and verified.

## Architecture Decision Log

| ID | Context | Decision | Consequence |
|---|---|---|---|
| AD-01 | The project must remain free-tier compatible and simple to deploy. | Use stdio-only MCP transport in v1. | The MCP server runs as a subprocess within the single deployed service. |
| AD-02 | The earlier v2.1 design allowed partial index rebuilds, which could leave parser, chunking, embedding, or metadata state out of sync after a corpus/configuration change. | Amend the design from partial rebuild to version/configuration-triggered full re-ingest, while committing deterministic `corpus/processed/chunks.json`. | Full rebuilds trade a small local rebuild cost for a simpler invariant: source corpus, canonical chunks, embeddings, metadata, and generated Chroma state always move together; the committed chunk artifact remains reviewable and byte-testable. |
| AD-03 | The v1 deployment is a single-process free-tier service and does not require durable multi-user state. | Use an in-memory session store keyed by `conversation_id`. | Simple and zero-cost for v1; pending state is lost on restart and this remains a documented limitation. **Frozen design; implementation belongs to the later agent/web phases.** |
| AD-04 | AI-assisted implementation must remain explainable, evidence-based, and resistant to accidental duplicate architecture. | Use inspect-before-change, one verified capability at a time, with tests and evidence before advancing. | Changes remain auditable and regressions are found early; this working discipline is enforced by `PROJECT_RULES.md`. |
| AD-05 | Free-tier cold starts must not repeatedly parse and embed the policy corpus. | Perform ingestion during the deploy build step; retain startup version checking and rebuild only as a local-development/fallback path, with `/health` exposing index-building state. | Production cold starts load a prepared index while development can recover automatically. **Deploy/startup integration remains pending later phases.** |
| AD-06 | Tool safety classification must be discoverable through MCP rather than duplicated in a custom registry. | Use MCP tool annotations, especially `readOnlyHint`, to distinguish read/calculation tools from action tools. | Confirmation middleware can derive safety behavior from discovered MCP metadata instead of hardcoded tool names. **Registration/discovery completion belongs to S5.** |
| AD-07 | S5 depends on SDK support for MCP annotations and must not silently emulate missing MCP capabilities. | Verify the exact pinned `mcp` SDK supports `ToolAnnotations` / `readOnlyHint` before building registered tools. | Unsupported SDK behavior requires a deliberate spec amendment; verified support permits S5 tool registration to proceed. |
| AD-08 | `chunks.json` must be deterministic enough for byte-for-byte CI verification. | Serialize canonically as UTF-8/LF JSON with sorted keys, normalized whitespace, no timestamps or score floats, and one trailing newline. | Repeated ingestion of the same corpus/configuration produces a stable committed artifact suitable for CI determinism tests. |
| AD-09 | The embedding model must cover the full hard-max policy chunk without silent truncation. | Use `BAAI/bge-small-en-v1.5` with 512-token context and size chunks to approximately 350 tokens with a 450-token hard maximum. | Full policy-section tails remain available to embeddings while retaining a small local model suitable for free-tier deployment. |
| AD-10 | User confirmation must authorize the exact action preview rather than a detached later action. | Bind confirmation to a server-generated `confirmation_id` associated with the pending preview. | A confirmation can execute only the matching pending action; restart loss remains acceptable for mock v1 actions. **Frozen design; implementation belongs to the agent/web confirmation phase.** |


## Verified Engineering Evidence

### S4 — RAG Pipeline

S4 is implemented and verified against the frozen RAG contract.

- Corpus ingestion covers Markdown and PDF sources.
- Canonical `corpus/processed/chunks.json` is committed and contains
  400 citation-bearing chunks.
- Chunking is heading-aware, deterministic, and enforces the
  450-token hard maximum.
- Embeddings use `BAAI/bge-small-en-v1.5`, producing normalized
  384-dimensional vectors.
- Chroma persistence, freshness metadata, and safe index publication
  are implemented.
- Semantic retrieval returns citation-ready document ID, title,
  section, snippet, and similarity data.
- Exact policy-section lookup remains a separate retrieval API.
- `python -m rag.retrieve` is operational:
  `--help` returns 0; invalid CLI input returns 1 with clean stderr
  and no traceback.
- Real WF1 retrieval returned both the remote-work and
  information-security policies in the top five.
- Real WF2 retrieval returned the Paid Time Off Policy in the top five.
- Final retrieval regression: 437 tests passed.
- Final repository regression after the S4 CLI closure:
  981 tests passed.

### S5 — MCP Dependency Readiness and Current Boundary

The SDK-readiness portion of the S5 dependency checkpoint is verified.
The full §5 checkpoint evidence is intentionally still open until
R6E-C5 registers the first production READ tool and produces the
required annotation/discovery evidence.

- Frozen dependency: `mcp==1.29.0`.
- Current environment and a separately created Python 3.11 clean
  virtual environment both resolve the exact pinned requirements with
  `pip check` reporting no broken requirements.
- The pinned SDK exposes FastMCP and the annotation mechanism required
  for `ToolAnnotations` / `readOnlyHint`.
- The MCP server foundation runs explicitly over stdio.
- The policy adapter and
  `search_policy_documents(query: str, k: int = 5)` composition are
  implemented and tested.
- Remaining §5 evidence to be generated in R6E-C5:
  an annotated production tool declaration, `list_tools()` output
  showing `readOnlyHint=true`, and the corresponding
  `tests/test_mcp.py` discovery assertion.
- Therefore S5 is in progress; the full MCP dependency/evidence gate
  is not yet claimed complete.
