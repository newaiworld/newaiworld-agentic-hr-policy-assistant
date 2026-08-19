# Design and Evaluation

## Project Status

S1–S4 are complete and verified. The project is now in S5 — MCP Integration. R6E-C5 FastMCP READ registration and R6E-C6 live invocation of `search_policy_documents(query, k=5)` are complete and published. The R6E-D `get_policy_section(doc_id, section)` READ capability is implemented and verified locally through composition, FastMCP registration/discovery, real stdio invocation, clean MCP error translation, and same-session recovery. The current D implementation is not yet claimed as published until its code, tests, and governance evidence are committed and pushed. Production discovery now exposes two verified READ tools: `search_policy_documents` and `get_policy_section`. The remaining MCP tools and agent-through-MCP execution remain pending. S6–S10 remain pending and are not yet claimed as implemented.

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

### S5 — MCP Dependency Readiness and R6E-C5 Evidence

The SDK-readiness portion of the S5 dependency checkpoint and the
R6E-C5 production READ registration are implemented, verified, and
published. The C5 implementation, tests, and governance evidence were
committed and pushed at `a6a6a8c`.

- Frozen dependency: `mcp==1.29.0`.
- Current environment and a separately created Python 3.11 clean
  virtual environment both resolve the exact pinned requirements with
  `pip check` reporting no broken requirements.
- The pinned SDK exposes FastMCP and the annotation mechanism required
  for `ToolAnnotations` / `readOnlyHint`.
- The MCP server continues to run explicitly over the frozen stdio
  transport.
- `mcp/tools_policy.py` remains framework-agnostic and retains
  `search_policy_documents(query: str, k: int = 5)` as the existing
  policy-search composition.
- The production FastMCP server registers that existing function rather
  than reimplementing retrieval behavior.
- The production declaration uses
  `ToolAnnotations(readOnlyHint=True)`.
- Production `list_tools()` discovery returned exactly one registered
  tool: `search_policy_documents`.
- Discovery exposed `readOnlyHint=true`.
- The generated MCP input schema preserved:
  - required `query` with type `string`;
  - optional `k` with type `integer`;
  - literal default `k=5`.
- The obsolete foundation assertion that no tools were registered was
  deliberately replaced with the production registration contract.
- Focused registration/discovery tests: 4 passed.
- Complete `tests/test_mcp.py` regression: 24 passed.
- Full repository regression after R6E-C5 implementation: 984 passed.
- `python -m pip check`: pass.
- `git diff --check`: pass.
- R6E-C5 publication:
  - commit: `a6a6a8c` — `feat(mcp): register policy search read tool`;
  - push to `origin/main`: successful;
  - `HEAD`, `main`, `origin/main`, and `origin/HEAD` synchronized at
    `a6a6a8c`;
  - working tree after push: clean.
- The local `mcp/` directory remains without `__init__.py`, and the
  official SDK continues to resolve from `site-packages/mcp`.
- G3 is advanced, not complete. Production registration and discovery
  are verified. R6E-C6 separately verifies live MCP invocation; the
  remaining MCP tools and later agent-through-MCP execution remain
  pending.

### S5 — R6E-C6 Live MCP Invocation Evidence

R6E-C6 verifies that the registered policy-search capability can be
invoked through the actual MCP protocol boundary rather than through a
direct Python function call. The C6 implementation, tests, and governance
evidence are complete and published at commit `0d87ac9`.

#### Protocol path verified

The production-path probe exercised:

`ClientSession` → `stdio_client` → separate Python subprocess →
`mcp/server.py` → FastMCP → `search_policy_documents` →
`rag.retrieve` → embeddings → Chroma.

Evidence:

- `ClientSession.initialize()`: pass.
- `ClientSession.list_tools()`: pass.
- `ClientSession.call_tool("search_policy_documents", ...)`: pass.
- The server processed a real MCP `CallToolRequest`.
- Successful production call returned
  `CallToolResult(isError=False)`.
- Successful structured result envelope:
  `structuredContent["result"]`.
- Every policy-evidence record preserved the frozen five-field MCP
  schema:
  - `doc_id`;
  - `title`;
  - `section`;
  - `snippet`;
  - `score`.
- Real production retrieval returned policy evidence including
  `HR-POL-004` Remote and Flexible Work Policy and
  `HR-POL-005` Information Security and Acceptable Use Policy.

#### MCP error semantics

A production MCP call with invalid `k=0` established the protocol-level
failure contract:

- the client received `CallToolResult`, not a client-side exception;
- `isError=True`;
- `structuredContent=None`;
- error content was returned as `TextContent`;
- the lower-layer validation message `k must be positive.` was
  preserved inside FastMCP's clean tool-error message;
- no traceback text was exposed to the client;
- the same initialized MCP session remained usable afterward.

#### Automated CI strategy

The production Chroma index is gitignored and the current CI workflow
does not build the production index. Therefore R6E-C6 deliberately
separates protocol regression from real-corpus validation:

- automated tests use temporary fixture-backed FastMCP servers created
  under pytest `tmp_path`;
- those tests still use the official `stdio_client`,
  `StdioServerParameters`, `ClientSession`, and `call_tool()` APIs;
- fixture servers run as separate Python subprocesses;
- subprocess identity is verified by asserting the server PID differs
  from the pytest/client PID;
- the automated tests do not depend on `chroma_db/`, model downloads,
  or external network availability;
- a separate local production probe validates the actual
  `mcp/server.py` → RAG → Chroma execution path.

This split keeps CI hermetic while still preserving direct evidence that
the production policy-search tool works through MCP.

#### Timeout and subprocess lifecycle

Each automated live MCP test uses:

- an outer `anyio.fail_after(20)` deadline;
- a `ClientSession` read timeout of 10 seconds;
- the SDK `stdio_client` async context manager for subprocess ownership
  and cleanup.

No `pytest-asyncio` dependency was added.

#### Automated test evidence

Three live-stdio tests were added to `tests/test_mcp.py`:

- `test_stdio_client_calls_policy_search_through_mcp`;
- `test_stdio_client_receives_clean_mcp_error_result`;
- `test_stdio_client_session_recovers_after_tool_error`.

Verification:

- pre-C6 MCP suite: 24 tests;
- C6 live invocation tests added: 3;
- current MCP collection: 27 tests;
- complete MCP regression: 27 passed;
- pre-C6 repository collection: 984 tests;
- current repository collection: 987 tests;
- full repository regression: 987 passed;
- `python -m pip check`: pass;
- `git diff --check`: pass;
- no production source file changed during R6E-C6.
- R6E-C6 publication:
  - commit: `0d87ac9` —
    `test(mcp): verify live stdio tool invocation`;
  - push to `origin/main`: successful;
  - `HEAD`, `main`, `origin/main`, and `origin/HEAD` synchronized at
    `0d87ac9`;
  - local `main` tracks `origin/main` with no ahead/behind divergence;
  - working tree after push: clean.

#### Current grading boundary

G3 is materially advanced because real MCP tool invocation is now
verified across a subprocess/stdin-stdout protocol boundary rather than
through a hard-coded direct function call.

G3 is not yet complete. The remaining MCP tools still need to be
implemented and exposed, and later the agent must discover tool schemas
and execute selected tools through the MCP client rather than through
direct imports.

### S5 — R6E-D `get_policy_section` READ Capability Evidence

R6E-D adds the second frozen RAG-backed READ tool:

`get_policy_section(doc_id: str, section: str) -> {title, section, text}`.

The capability is implemented and verified locally. Publication remains
pending until the current implementation, tests, and governance changes
are committed and pushed.

#### Architecture

The MCP layer reuses the existing S4 exact-section retrieval capability
rather than implementing another lookup path.

Verified composition:

`rag.retrieve.get_policy_section`
→ `PolicySection`
→ `mcp.tools_policy.get_policy_section`
→ `{title, section, text}`.

Responsibilities remain separated:

- `rag.retrieve` owns:
  - argument validation;
  - corpus resolution;
  - catalogue construction and caching;
  - exact document lookup;
  - full-heading matching;
  - numeric-section matching;
  - ambiguity handling;
  - normalized full-section text;
- `mcp/tools_policy.py` owns:
  - projection of `PolicySection` to the frozen MCP response;
  - framework-agnostic composition only;
- `mcp/server.py` owns:
  - project-tool loading;
  - FastMCP registration;
  - MCP annotations;
  - the frozen stdio transport.

Exact-section lookup is catalogue-backed and does not require the
Chroma vector index or query embeddings. No section-matching,
normalization, retrieval, or Chroma logic is duplicated in the MCP
layer.

#### Frozen response contract

The public tool returns exactly:

- `title: str`;
- `section: str`;
- `text: str`.

The complete normalized section text is preserved. The MCP adapter does
not convert it to a snippet, truncate it, or apply a separate token cap.

#### Real-corpus evidence

WF1 exact-section validation:

- document: `HR-POL-004`;
- section: `5.3 International approval`;
- title: `Remote and Flexible Work Policy`;
- exact-section text returned successfully.

WF2 exact-section validation:

- document: `HR-POL-002`;
- section: `9.1 Three-day request with sufficient balance`;
- title: `Paid Time Off Policy`;
- exact-section text returned successfully.

The D1 contract probe also verified:

- numeric section lookup;
- case-insensitive complete-heading matching;
- unnumbered root-heading lookup;
- exact/case-sensitive `doc_id` behavior;
- clean missing-document and missing-section `RetrievalError` behavior.

#### FastMCP registration and discovery

The production FastMCP server now registers exactly two completed policy
READ tools:

- `search_policy_documents`;
- `get_policy_section`.

Both expose:

`ToolAnnotations(readOnlyHint=True)`.

Production `list_tools()` discovery for `get_policy_section` preserves:

- required `doc_id` with type `string`;
- required `section` with type `string`.

The generated output schema reflects the Python `dict[str, str]`
return type.

The server reuses the existing framework-agnostic implementations from
`mcp/tools_policy.py`; retrieval/business logic is not reimplemented in
`mcp/server.py`.

The project-local `mcp/` directory remains a non-package, and the
official MCP SDK continues to resolve from `site-packages/mcp`.

#### Live MCP protocol evidence

A real production stdio probe executed:

`ClientSession`
→ stdio subprocess
→ `mcp/server.py`
→ FastMCP
→ `get_policy_section`
→ exact S4 section retrieval.

Successful invocation returned:

- MCP result type: `CallToolResult`;
- `isError=False`;
- direct `structuredContent` with exactly:
  - `title`;
  - `section`;
  - `text`;
- complete non-empty section text.

For `HR-POL-004` section `5.3 International approval`, the returned
normalized text length was 308 characters.

#### MCP error and recovery behavior

A real production call for missing section `99.99` returned:

- `CallToolResult`;
- `isError=True`;
- `structuredContent=None`;
- error content as `TextContent`;
- lower-layer policy-section-not-found message preserved;
- no traceback exposed to the MCP client.

The same initialized MCP session then successfully executed a valid
`get_policy_section` request. A handled tool error therefore does not
poison the stdio MCP session.

#### Automated subprocess evidence

Two D6 tests exercise the exact-section tool through the real MCP client
boundary:

- `test_stdio_client_calls_get_policy_section_through_mcp`;
- `test_stdio_client_get_policy_section_recovers_after_error`.

The tests use:

- `StdioServerParameters`;
- `stdio_client`;
- `ClientSession`;
- `ClientSession.call_tool`;
- a 20-second outer timeout;
- a 10-second session read timeout;
- temporary fixture-backed FastMCP subprocesses.

Fixture-server PIDs are asserted to differ from the pytest/client PID,
providing evidence that the tests cross an actual subprocess/stdin-stdout
boundary rather than directly calling project Python functions.

#### Test progression and regression evidence

Published pre-D baseline:

- MCP collection: 27 tests;
- repository collection: 987 tests.

Verified progression:

- D2 converter tests: `27 -> 31`;
- D3 composition tests: `31 -> 37`;
- D5 registration/discovery net tests: `37 -> 40`;
- D6 live-MCP tests: `40 -> 42`.

Current verification:

- MCP collection: 42 tests;
- complete MCP regression: 42 passed;
- repository collection: 1002 tests;
- full repository regression: 1002 passed;
- `python -m pip check`: pass;
- `git diff --check`: pass.

The current technical change set remains limited to:

- `mcp/server.py`;
- `mcp/tools_policy.py`;
- `tests/test_mcp.py`.

#### Current grading boundary

G3 is materially advanced but not complete.

Two RAG-backed production READ tools are now implemented, registered,
discoverable through MCP metadata, and exercised across the real stdio
protocol boundary:

- `search_policy_documents`;
- `get_policy_section`.

The remaining mock-data READ tools, calculation tools, confirmation-gated
ACTION tools, and later agent-through-MCP execution are still pending.

After publication of R6E-D, the next frozen MCP capability is:

`lookup_employee_profile(employee_id)`.
