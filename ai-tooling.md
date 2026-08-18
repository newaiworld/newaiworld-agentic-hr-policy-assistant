# AI Tooling

## Session Log

### 2026-08-04 — S1 Foundation

- Tool: ChatGPT
- Use: Repository setup guidance, macOS permission troubleshooting, Git configuration, and reproducible Python environment planning.
- Verified by: Terminal outputs, Git status, pinned requirements, and pytest.
- Limitations: Commands were reviewed and executed manually by the project author.

## 2026-08-05 — S3 Synthetic Mock Data

### AI tools used

- ChatGPT was used to design the S3 data model and controlled vocabulary.
- ChatGPT was used to define the four frozen dataset schemas:
  - `mock_data/employees.json`
  - `mock_data/pto.json`
  - `mock_data/benefits.json`
  - `mock_data/tickets.json`
- ChatGPT was used to generate the initial synthetic employee, PTO, benefits, and ticket records.
- ChatGPT was used to produce validation commands for JSON syntax, referential integrity, manager hierarchy, policy consistency, benefits dates, part-time PTO accrual, ticket sequencing, and synthetic-data safety.

### Human review and decisions

- Confirmed the S3 scope remained limited to the four files required by `IMPLEMENTATION_SPEC.md`.
- Reviewed and accepted the controlled vocabulary for employment type, employment status, location, benefits eligibility, benefits election status, ticket status, and ticket category.
- Fixed E005's start date at `2026-07-15` so the pending benefits state is consistent with the 30-day commencement rule.
- Fixed part-time FTE values:
  - E002 at `0.6`
  - E008 at `0.4`
- Confirmed the corresponding monthly PTO accrual rates:
  - E002 at `1.0` day per month
  - E008 at `0.6667` days per month
- Confirmed E006 is represented as a known contractor who is PTO-ineligible rather than an unknown employee.
- Confirmed E001 supports the PTO workflow with 8.0 available days and no existing PTO ticket.
- Confirmed E003 supports the international remote-work workflow as an active full-time employee with a valid manager and domestic location.

### Validation performed

- All four JSON files passed `python -m json.tool`.
- Cross-file referential-integrity validation passed.
- Manager-reference validation passed.
- Manager-cycle validation passed.
- PTO policy-consistency validation passed.
- Benefits eligibility and commencement-date validation passed.
- Part-time accrual validation passed.
- Ticket sequencing and mock-action preconditions passed.
- Synthetic email-domain validation passed.
- No phone-number-like values were found.
- No legacy company names were found.
- No final policy decisions were stored directly in the mock data.

### Impact of AI assistance

AI assistance reduced the time required to design the schemas, generate consistent synthetic records, and produce repeatable validation checks. Human review was used to verify policy alignment, approve controlled vocabulary, resolve edge-case dates and FTE values, and confirm that the final datasets support the frozen workflows without introducing specification drift.

## 2026-08-11 — S4 CP7 Embeddings

### AI tools used

- ChatGPT was used as an AI engineering assistant for the S4
  embedding phase.
- AI assistance was used to:
  - inspect the existing RAG architecture before implementation;
  - decompose CP7 into model lifecycle, document embedding,
    query embedding, numerical validation, and full-corpus
    validation checkpoints;
  - generate focused Python implementation and pytest cases;
  - review terminal, pytest, Git, and real-model validation
    outputs before each checkpoint advanced;
  - identify and correct source-formatting issues before commits;
  - design real-corpus numerical and ordering validation scripts.

### Human review and decisions

- All commands and code changes were reviewed and executed
  manually by the project author.
- The project author retained the frozen
  `BAAI/bge-small-en-v1.5` embedding model and did not introduce
  additional frameworks or embedding services.
- The project author approved the separation between document
  and query embedding:
  - documents are embedded without a query prefix;
  - queries use the BGE retrieval instruction.
- The project author retained fail-fast behavior rather than
  skipping malformed chunks or invalid vectors.
- No intermediate persistent embedding artifact was introduced;
  `corpus/processed/chunks.json` remains the canonical source
  from which the generated Chroma index will be rebuilt.

### Validation performed

- Embedding model lifecycle:
  - lazy model loading: pass;
  - cached per-process reuse: pass;
  - embedding dimension: 384;
  - maximum model sequence length: 512.
- Document embedding:
  - focused tests: pass;
  - finite 384-dimensional vectors: pass;
  - normalization: pass;
  - ordered row mapping: pass.
- Query embedding:
  - BGE retrieval instruction: verified;
  - query vector shape `(384,)`: pass;
  - finite/normalized query vectors: pass.
- Numerical validation:
  - repeated document embeddings: pass;
  - repeated query embeddings: pass;
  - tested batch-size invariance: pass;
  - ordering stability: pass;
  - tolerance: `rtol=1e-5`, `atol=1e-5`.
- Full canonical corpus:
  - chunks embedded: 400/400;
  - resulting matrix: `(400, 384)`;
  - non-finite values: 0;
  - all vectors normalized: pass;
  - first real chunk repeatability: pass;
  - full-corpus batch-size stability: pass.
- Final focused embedding suite: 67 passed.
- Final repository regression: 268 passed.

### Impact of AI assistance

AI assistance reduced implementation and debugging time by
providing checkpoint-specific code, focused synthetic tests, and
repeatable validation commands. The inspect-before-change
workflow was retained throughout: implementation recommendations
were not accepted until the current repository state was
verified, and each capability was tested independently before
commit.

Human verification remained the acceptance gate. AI-generated
code and commands were checked through compilation, focused
pytest runs, real-model inference, full regression, Git diff
inspection, and remote synchronization before the project
advanced.

### Limitations

- AI-generated implementation suggestions required manual review
  against `PROJECT_RULES.md`, `IMPLEMENTATION_SPEC.md`, and the
  S4 engineering blueprint.
- Real embedding repeatability was validated on the local Apple
  MPS environment. Cross-device and cross-platform
  floating-point byte identity was not assumed.
- Hugging Face model availability remains an external dependency
  for an uncached development environment; deployment will
  pre-download the frozen model during the build phase.


## 2026-08-18 — S5 MCP Integration through R6E-C4

### AI tools used

- ChatGPT was used as an AI engineering assistant for the S5 MCP phase.
- AI assistance was used to:
  - inspect the frozen MCP architecture and project governance before implementation;
  - verify the official `mcp==1.29.0` SDK dependency and FastMCP APIs;
  - inspect `ToolAnnotations`, `readOnlyHint`, stdio transport, tool discovery,
    and runtime import behavior;
  - design and review the FastMCP stdio server foundation;
  - design the repository-root bootstrap while preserving ownership of the
    installed `mcp` SDK namespace;
  - design the pure `RetrievalResult` to MCP response adapter;
  - design and test `search_policy_documents(query, k=5)`;
  - generate focused pytest cases for schema projection, ordering, default
    values, error propagation, runtime bootstrap, and SDK namespace safety;
  - diagnose and recover from an accidental shell/heredoc insertion into
    `tests/test_mcp.py`;
  - guide real-corpus validation for the frozen WF1 and WF2 policy-search
    scenarios.

### Human review and decisions

- All commands and code changes were reviewed and executed manually by the
  project author.
- The project author retained the frozen stdio-only MCP architecture.
- The local `mcp/` directory was deliberately kept without `__init__.py` so it
  does not shadow the installed MCP SDK package.
- The project author retained `mcp/tools_policy.py` as a framework-agnostic
  adapter/composition layer and deferred FastMCP registration to R6E-C5.
- The public MCP search contract was kept at literal `k=5`, independent of the
  lower-level retrieval default.
- Retrieval validation, embedding, Chroma access, ranking, and retrieval error
  semantics remain owned by `rag.retrieve`; these were not duplicated in the
  MCP adapter.
- G3 was recorded as advanced rather than complete because FastMCP
  registration, discovery, live MCP invocation, and later agent-through-MCP
  execution remain pending.

### Validation performed

- Official MCP SDK:
  - `mcp==1.29.0` installed and pinned;
  - `pip check`: pass;
  - FastMCP import: pass;
  - stdio transport API: verified;
  - `ToolAnnotations(readOnlyHint=True)`: verified.
- FastMCP server foundation:
  - explicit `transport="stdio"`: pass;
  - project server import: pass;
  - installed MCP SDK namespace preserved: pass.
- MCP policy adapter:
  - exact five-field response schema:
    `doc_id`, `title`, `section`, `snippet`, `score`;
  - retrieval order preserved;
  - score copied from retrieval similarity;
  - empty result handling: pass;
  - invalid container/member validation: pass.
- `search_policy_documents` composition:
  - signature `(query: str, k: int = 5)`: verified;
  - retrieval composition: pass;
  - default `k=5`: pass;
  - delegated `TypeError`, `ValueError`, and `RetrievalError`
    propagation: pass.
- Real-corpus validation:
  - active Chroma collection: `policy_chunks`;
  - indexed records: 400;
  - WF1 top-5 evidence included:
    - `HR-POL-004` Remote and Flexible Work Policy;
    - `HR-POL-005` Information Security and Acceptable Use Policy;
  - WF2 top-5 evidence included `HR-POL-002` Paid Time Off Policy at
    ranks 1, 2, and 4.
- Final focused MCP suite for R6E-C4: 21 passed.
- Final repository regression for R6E-C4: 936 passed.
- `git diff --check`: pass before commit.
- R6E-C4 commit pushed:
  - `c0e3759` — `feat(mcp): add policy search composition`.

### Impact of AI assistance

AI assistance reduced implementation and debugging time by decomposing the MCP
work into small, verifiable checkpoints and by generating focused inspection,
test, and validation commands. It also helped identify namespace-shadowing and
runtime-import risks before FastMCP tool registration.

Human verification remained the acceptance gate. AI-generated recommendations
were accepted only after compilation, focused pytest execution, full regression,
real-corpus validation, Git diff review, and remote synchronization.

### Limitations

- FastMCP registration of `search_policy_documents` is not yet implemented.
- MCP discovery of the production policy tool is not yet verified.
- Live MCP `call_tool()` execution is not yet verified.
- Agent-through-MCP execution is not yet implemented, so G3 is not complete.
- Hugging Face emitted an unauthenticated-request warning during real-model
  loading, but model loading and retrieval completed successfully; no new
  authentication dependency was introduced.


## 2026-08-18 — S5 MCP Integration R6E-C5 READ Registration

### AI tools used

- ChatGPT was used as an AI engineering assistant for R6E-C5
  FastMCP READ-tool registration.
- AI assistance was used to:
  - review the current project governance and verified S4/S5
    checkpoint state before implementation;
  - identify and verify the previously repaired S4 retrieval CLI
    before allowing S5 work to continue;
  - inspect the installed `mcp==1.29.0` FastMCP registration API
    rather than assuming SDK behavior;
  - probe `FastMCP.tool()` and
    `ToolAnnotations(readOnlyHint=True)` against the existing
    `search_policy_documents(query: str, k: int = 5)` function;
  - inspect the generated MCP discovery schema before production
    registration;
  - design the minimal production registration while preserving
    ownership of the installed `mcp` SDK namespace;
  - identify the deliberate contract change required for the
    earlier `test_server_foundation_has_no_registered_tools`
    foundation test;
  - design focused discovery tests for tool registration,
    annotations, input schema, and implementation reuse;
  - generate production `list_tools()` evidence;
  - review the complete production/test diff for scope and
    architecture drift;
  - guide focused MCP regression and full repository regression
    checks.

### Human review and decisions

- All commands and code changes were reviewed and executed manually
  by the project author.
- The project author retained the frozen stdio-only MCP transport.
- The project author retained one existing FastMCP server rather
  than adding another MCP server or transport.
- `mcp/tools_policy.py` remains framework-agnostic. Retrieval,
  embedding, Chroma access, ranking, result validation, and response
  projection were not reimplemented in `mcp/server.py`.
- The local `mcp/` directory remains without `__init__.py` so the
  installed official MCP SDK continues to own the Python `mcp`
  namespace.
- The existing `search_policy_documents` function was registered
  directly rather than duplicated.
- The production policy-search tool was classified as READ using
  `ToolAnnotations(readOnlyHint=True)`.
- The old foundation test asserting that the server exposed no tools
  was deliberately replaced because R6E-C5 intentionally changes
  that contract.
- No live MCP `call_tool()` execution or agent integration was added
  in this checkpoint; those remain separate later capabilities.
- G3 remains advanced rather than complete.

### Validation performed

- Pre-implementation baseline:
  - repository baseline `71a65fe`: verified;
  - local `main`, `origin/main`, and `origin/HEAD`: synchronized;
  - S4 retrieval CLI repair commit `add9f58`: verified;
  - retrieval CLI entry point and CLI tests: verified;
  - `mcp==1.29.0`: installed and pinned;
  - `pip check`: pass;
  - pre-change MCP suite: 21 passed.
- FastMCP registration-contract probe:
  - existing synchronous `search_policy_documents` accepted by
    `FastMCP.tool()`;
  - registration returned the same function object;
  - discovered tool count: 1;
  - discovered name: `search_policy_documents`;
  - `readOnlyHint=True`;
  - generated input schema preserved required string `query`;
  - generated input schema preserved optional integer `k`;
  - generated default `k=5`.
- Production registration:
  - exactly one production tool registered;
  - explicit stdio server transport preserved;
  - official `mcp` SDK continued to resolve from `site-packages`;
  - local `mcp/` remained a non-package;
  - no retrieval, embedding, Chroma, similarity, or distance logic
    was duplicated into `mcp/server.py`.
- Focused registration/discovery tests:
  - exact registered tool set: pass;
  - READ annotation discovery: pass;
  - discovered input schema/default contract: pass;
  - existing policy-search implementation reuse: pass;
  - focused registration tests: 4 passed.
- Complete MCP regression:
  - `tests/test_mcp.py`: 24 passed.
- Production discovery evidence:
  - tool count: 1;
  - name: `search_policy_documents`;
  - `query`: required string;
  - `k`: optional integer with default 5;
  - `readOnlyHint=true`;
  - acceptance: pass.
- Full repository regression:
  - 984 passed.
- Repository hygiene:
  - `python -m pip check`: pass;
  - `git diff --check`: pass;
  - implementation review confirmed only intended C5 source/test
    files changed before governance closure.

### Impact of AI assistance

AI assistance reduced implementation risk by separating SDK
inspection, contract probing, production registration, automated
discovery tests, runtime discovery evidence, and regression into
independent verification gates. This prevented the implementation
from relying on assumed FastMCP behavior and exposed the expected
foundation-test collision before the production change was made.

The inspect-before-change discipline also prevented unnecessary
changes to the already verified RAG and policy-composition layers.
Human review remained the acceptance gate for every generated command,
code change, test result, and architectural claim.

### Limitations and remaining work

- R6E-C5 verifies FastMCP registration and discovery of the first
  production READ tool only.
- Live MCP `call_tool()` execution has not yet been verified.
- The remaining seven frozen MCP tools have not yet been registered.
- Agent startup discovery and conversion of MCP schemas into the LLM
  tool-calling format have not yet been implemented.
- Agent-through-MCP execution has not yet been verified.
- ACTION-tool confirmation behavior remains a later checkpoint.
- G3 is therefore advanced but not complete.


### R6E-C5 publication closure

The verified R6E-C5 implementation, tests, and governance evidence were
published after the local verification gates completed.

- Commit:
  - `a6a6a8c` — `feat(mcp): register policy search read tool`.
- Push:
  - `git push origin main`: successful;
  - remote advanced from `71a65fe` to `a6a6a8c`.
- Post-push synchronization:
  - `HEAD`: `a6a6a8c`;
  - `main`: `a6a6a8c`;
  - `origin/main`: `a6a6a8c`;
  - `origin/HEAD`: `a6a6a8c`.
- Branch state:
  - local `main` tracks `origin/main` with no ahead/behind divergence.
- Working tree after push:
  - clean.
- R6E-C5 status:
  - FastMCP READ registration: complete and published;
  - production `list_tools()` discovery: verified;
  - `ToolAnnotations(readOnlyHint=True)`: verified;
  - focused MCP suite: 24 passed;
  - full repository regression: 984 passed.
- G3 remains advanced rather than complete because live MCP
  `call_tool()` execution, the remaining MCP tools, and
  agent-through-MCP execution remain pending.


## 2026-08-18 — S5 MCP Integration R6E-C6 Live MCP Invocation

### AI tools used

- ChatGPT was used as an AI engineering assistant for R6E-C6
  live MCP invocation.
- AI assistance was used to:
  - review the R6E-C6 plan against project governance and reviewer
    feedback before implementation;
  - inspect the exact pinned `mcp==1.29.0` client-side API rather
    than assuming MCP client behavior;
  - inspect `ClientSession`, `StdioServerParameters`,
    `stdio_client`, `CallToolResult`, and structured-content
    semantics;
  - identify the CI risk created by the gitignored production
    Chroma index;
  - separate automated protocol testing from real-production
    RAG/Chroma validation;
  - design timeout and subprocess-cleanup boundaries before adding
    live tests;
  - probe the actual MCP error translation before freezing the
    automated error contract;
  - probe a successful production `call_tool()` before freezing
    the structured-result contract;
  - design focused subprocess-level tests that exercise MCP through
    stdio instead of direct Python function calls;
  - pin MCP and repository test counts before regression;
  - review the final C6 diff for direct-call shortcuts, production
    scope drift, timeout handling, and subprocess-boundary evidence.

### Human review and decisions

- All generated commands, test changes, probes, and evidence were
  manually reviewed and executed by the project author.
- The frozen stdio-only MCP architecture was retained.
- No new MCP transport, server, production tool, or agent behavior
  was introduced during C6.
- No production Python source file required modification.
- The project author selected a two-part C6 verification strategy:
  - automated CI-safe tests use temporary fixture-backed FastMCP
    servers under pytest `tmp_path`;
  - a separate local production probe validates the actual
    `mcp/server.py` → `search_policy_documents` → RAG → Chroma
    execution path.
- This strategy avoids making CI depend on the gitignored
  `chroma_db/`, model downloads, or external network access while
  still retaining real-production MCP evidence.
- No `pytest-asyncio` dependency was added. Existing synchronous
  pytest functions continue to use an explicit async runner.
- Every live automated MCP test uses:
  - an outer 20-second deadline;
  - a 10-second `ClientSession` read timeout;
  - the SDK `stdio_client` context manager for subprocess
    ownership and cleanup.
- New C6 tests never directly call the project
  `search_policy_documents` function or `retrieve_policy`.
  Tool execution occurs through `ClientSession.call_tool()`.
- G3 remains advanced rather than complete because the remaining
  MCP tools and later agent-through-MCP execution are still
  pending.

### Validation performed

- Pre-C6 published baseline:
  - `HEAD`: `9507172`;
  - `main`: `9507172`;
  - `origin/main`: `9507172`;
  - `origin/HEAD`: `9507172`;
  - working tree: clean;
  - MCP suite: 24 passed;
  - repository collection: 984 tests;
  - repository regression: 984 passed.
- C5 closure was reverified before C6:
  - committed `list_tools()` evidence present;
  - retired
    `test_server_foundation_has_no_registered_tools`
    test count: 0;
  - `mcp==1.29.0`: installed and pinned.
- Client/API inspection:
  - `ClientSession.initialize()`: verified;
  - `ClientSession.list_tools()`: verified;
  - `ClientSession.call_tool()`: verified;
  - `StdioServerParameters`: verified;
  - `stdio_client`: verified;
  - `CallToolResult.content`: verified;
  - `CallToolResult.structuredContent`: verified;
  - `CallToolResult.isError`: verified.
- Real stdio lifecycle probe:
  - actual production `mcp/server.py` launched as a subprocess;
  - session initialization: pass;
  - real MCP `ListToolsRequest`: pass;
  - exactly one tool discovered;
  - tool: `search_policy_documents`;
  - `readOnlyHint=True`;
  - frozen input schema preserved.
- Production error-surface probe:
  - real MCP `CallToolRequest`: pass;
  - invalid `k=0` returned `CallToolResult`;
  - `isError=True`;
  - `structuredContent=None`;
  - error payload returned as `TextContent`;
  - `k must be positive.` preserved;
  - no traceback exposed;
  - same MCP session remained usable afterward.
- Production success probe:
  - real MCP `CallToolRequest`: pass;
  - actual production `mcp/server.py`: used;
  - actual embedding/retrieval/Chroma path: used;
  - result `isError=False`;
  - structured envelope:
    `structuredContent["result"]`;
  - three policy results returned for the test query;
  - frozen result keys preserved:
    `doc_id`, `title`, `section`, `snippet`, `score`;
  - returned evidence included `HR-POL-004` and
    `HR-POL-005`.
- Automated C6 tests added:
  - `test_stdio_client_calls_policy_search_through_mcp`;
  - `test_stdio_client_receives_clean_mcp_error_result`;
  - `test_stdio_client_session_recovers_after_tool_error`.
- Subprocess-boundary proof:
  - fixture servers are launched through `stdio_client`;
  - returned fixture PID differs from pytest/client PID;
  - no direct project-tool invocation occurs in the C6 live-test
    region.
- C6 focused live tests:
  - 3 passed.
- Complete MCP suite:
  - 27 tests collected;
  - 27 passed.
- Full repository:
  - 987 tests collected;
  - 987 passed.
- Dependency health:
  - `python -m pip check`: pass.
- Repository hygiene:
  - `git diff --check`: pass.
- Current C6 implementation scope:
  - `tests/test_mcp.py` only before governance updates;
  - no production source files changed.

### Impact of AI assistance

AI assistance was most useful in identifying protocol and environment
assumptions before they became committed test behavior. In particular,
the review process exposed the production-index dependency, async and
subprocess timeout requirements, MCP-level error semantics, and the
need to distinguish an actual stdio invocation from an in-process
Python call.

The staged inspection → probe → contract freeze → one-test-at-a-time
implementation sequence reduced debugging ambiguity. Transport
lifecycle, error translation, successful production invocation, and
automated CI behavior were verified independently before the complete
MCP and repository regressions were run.

Human verification remained the acceptance gate for every AI-generated
instruction and engineering claim.

### Limitations and remaining work

- R6E-C6 verifies live MCP invocation for the first production READ
  tool only.
- The remaining frozen MCP tools have not yet all been implemented,
  registered, discovered, or invoked.
- Agent startup discovery and MCP-schema conversion into the LLM
  tool-calling format remain pending.
- The agent has not yet selected and invoked tools through MCP.
- ACTION-tool confirmation behavior remains a later checkpoint.
- The production success probe depended on the developer's existing
  local Chroma index and embedding environment; this is intentionally
  separate from the hermetic automated CI tests.
- Hugging Face network requests were observed during the production
  embedding-model load. The automated C6 subprocess tests do not
  depend on external network availability.
- G3 is therefore materially advanced but not complete.


### R6E-C6 publication closure

The verified R6E-C6 live MCP invocation tests and governance evidence
were published after all local verification gates completed.

- Commit:
  - `0d87ac9` — `test(mcp): verify live stdio tool invocation`.
- Push:
  - `git push origin main`: successful;
  - remote advanced from `9507172` to `0d87ac9`.
- Post-push synchronization:
  - `HEAD`: `0d87ac9`;
  - `main`: `0d87ac9`;
  - `origin/main`: `0d87ac9`;
  - `origin/HEAD`: `0d87ac9`.
- Branch state:
  - local `main` tracks `origin/main` with no ahead/behind divergence.
- Working tree after push:
  - clean.
- Published R6E-C6 verification:
  - real MCP stdio invocation: verified;
  - production `ClientSession.call_tool()` path: verified;
  - successful structured MCP result: verified;
  - clean MCP error translation: verified;
  - same-session recovery after tool error: verified;
  - complete MCP suite: 27 passed;
  - full repository collection: 987 tests;
  - full repository regression: 987 passed.
- No production Python source file changed during R6E-C6.
- G3 remains materially advanced rather than complete because the
  remaining MCP tools and later agent-through-MCP execution are still
  pending.
