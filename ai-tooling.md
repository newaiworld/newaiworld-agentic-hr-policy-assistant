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


## 2026-08-19 — S5 MCP Integration R6E-D `get_policy_section` READ Capability

### AI tools used

- ChatGPT was used as an AI engineering assistant throughout R6E-D.
- AI assistance was used to:
  - review the existing S4 exact-section retrieval contract before MCP work;
  - reconcile reviewer feedback with the project governance and frozen
    implementation specification;
  - convert the original informal PS-style plan into the existing R6E
    checkpoint lineage;
  - identify and pin the D2–D6 test-count contract before implementation;
  - inspect the exact `PolicySection` and `get_policy_section` retrieval
    behavior instead of assuming its matching semantics;
  - identify that exact-section lookup is catalogue-backed and does not
    depend on Chroma;
  - design the thin MCP projection and composition boundaries;
  - inspect FastMCP-generated discovery schemas before production
    registration;
  - identify the deliberate collision with the historical one-tool
    registration assertion;
  - design real stdio MCP success, error, and same-session recovery probes;
  - design hermetic subprocess-backed MCP tests suitable for CI;
  - review architecture, regression evidence, and governance wording
    before publication.

### Human verification and engineering controls

AI-generated recommendations were not applied as an unverified batch.

The implementation followed the established project discipline:

`inspect`
→ `implement one capability`
→ `focused test`
→ `real-corpus or real-protocol validation`
→ `regression`
→ `architecture review`
→ `governance`
→ `commit/push only after verification`.

Each checkpoint was stopped and reviewed before the next capability was
introduced.

Notable examples:

- The first D2 test-edit attempt failed because the proposed insertion
  anchor did not exist in the current repository.
  - The edit aborted before writing the test file.
  - The repository was inspected again.
  - A repository-native anchor was selected before retrying.
- FastMCP registration deliberately invalidated the historical
  exactly-one-tool discovery assertion.
  - The production implementation was not changed to preserve the stale
    test.
  - The obsolete global-cardinality assertion was explicitly replaced
    with an exact two-READ-tool contract.
- Two older search discovery tests also contained implicit one-tool
  assumptions.
  - Their search-specific contracts were retained.
  - Only the global addressing assumption was repaired so the tests
    locate `search_policy_documents` by name.
- A documentation EOF hygiene warning was detected by
  `git diff --check` and corrected before continuing.

These corrections were treated as evidence that mechanical AI-generated
patches must remain subordinate to repository inspection and test output.

### R6E-D1 — contract inspection

The existing S4 exact-section retrieval capability was inspected before
MCP implementation.

Verified behavior included:

- domain type: `PolicySection`;
- exact document lookup;
- full section-heading lookup;
- numeric section lookup;
- case-insensitive complete-heading matching;
- unnumbered root-heading lookup;
- exact/case-sensitive `doc_id` behavior;
- missing-document `RetrievalError`;
- missing-section `RetrievalError`;
- ambiguity handling;
- complete normalized section text.

The contract probe directly tested a lower-case document identifier:

`hr-pol-004`

which failed with `RetrievalError`, confirming the exact/case-sensitive
document-ID boundary rather than merely documenting that assumption.

The inspection also established that exact-section lookup is
catalogue-backed and does not require Chroma vector querying.

### Frozen R6E-D test plan

Before implementation, the test progression was pinned:

- published MCP baseline: 27;
- D2 converter: `+4` → 31;
- D3 composition: `+6` → 37;
- D5 registration/discovery: net `+3` → 40;
- D6 live MCP: `+2` → 42.

Published repository baseline:

- 987 tests.

Expected final repository collection:

- `987 + 15 = 1002`.

The observed final counts matched this plan exactly.

### R6E-D2 — `PolicySection` projection

A pure framework-agnostic adapter was added:

`_convert_policy_section(result: PolicySection) -> dict[str, str]`.

Frozen output:

- `title`;
- `section`;
- complete `text`.

The adapter:

- validates the input domain type;
- performs no retrieval;
- performs no normalization;
- performs no section matching;
- performs no truncation;
- performs no Chroma access.

Four focused tests verify:

- exact frozen output shape;
- rejection of the wrong input type;
- preservation of complete text;
- no mutation of the source `PolicySection`.

Verification:

- MCP collection: 31;
- complete MCP regression: 31 passed.

### R6E-D3 — plain-Python exact-section composition

The existing S4 lookup was imported as:

`retrieve_policy_section`.

The public framework-agnostic composition was added:

`get_policy_section(doc_id: str, section: str) -> dict[str, str]`.

The function:

- forwards `doc_id` unchanged;
- forwards `section` unchanged;
- delegates lookup to `rag.retrieve.get_policy_section`;
- projects the returned `PolicySection`;
- does not wrap delegated `TypeError`, `ValueError`, or `RetrievalError`.

Real-corpus acceptance included:

WF1:

- `HR-POL-004`;
- `5.3 International approval`.

WF2:

- `HR-POL-002`;
- `9.1 Three-day request with sufficient balance`.

Six D3 tests were added.

Verification:

- MCP collection: 37;
- complete MCP regression: 37 passed.

### R6E-D4 — composition architecture review

The completed plain-Python capability was reviewed before FastMCP
registration.

Verified boundaries:

- exact lookup logic remains in `rag.retrieve`;
- response projection remains in `mcp/tools_policy.py`;
- no FastMCP registration had leaked into the composition checkpoint;
- the frozen `{title, section, text}` output remained intact;
- full exact-section text remained intact.

No additional implementation change was required.

### R6E-D5 — FastMCP READ registration and discovery

The existing server loader was generalized from a
search-specific loader to:

`_load_policy_tool(name)`.

This avoided duplicating dynamic module loading and callable validation
for every additional project tool.

The production FastMCP server now explicitly registers:

- `search_policy_documents`;
- `get_policy_section`.

Both use:

`ToolAnnotations(readOnlyHint=True)`.

Production discovery for `get_policy_section` exposes:

- required `doc_id`: string;
- required `section`: string.

The generated output schema reflects its `dict[str, str]` return type.

The server was also verified to reuse the framework-agnostic
`mcp/tools_policy.py` implementations rather than implementing retrieval
behavior itself.

The project-local `mcp/` directory remains a non-package and the official
MCP SDK continues to resolve from `site-packages/mcp`.

The historical exactly-one-tool test was deliberately replaced with an
exact two-READ-tool registration contract.

Existing search discovery tests were retained and adapted to locate the
search tool by name rather than depending on global tool cardinality.

Verification:

- MCP collection: 40;
- complete MCP regression: 40 passed.

### R6E-D6 — live stdio MCP execution

Before automated tests were written, production behavior was probed
through the actual MCP stdio protocol boundary.

Successful production invocation:

`ClientSession.call_tool("get_policy_section", ...)`

returned:

- `CallToolResult`;
- `isError=False`;
- direct `structuredContent` containing exactly:
  - `title`;
  - `section`;
  - `text`.

For:

- `HR-POL-004`;
- `5.3 International approval`;

the returned exact-section text length was 308 characters.

The SDK did not wrap this dict return value in a separate `result`
property, so the automated contract was based on the observed SDK
behavior rather than an assumed envelope.

### R6E-D6 error and recovery evidence

A real production request for missing section `99.99` returned:

- `CallToolResult`;
- `isError=True`;
- `structuredContent=None`;
- error content as `TextContent`;
- the lower-layer policy-section-not-found message;
- no traceback.

The same initialized MCP session then successfully executed a valid
`get_policy_section` request.

This demonstrates that an ordinary handled tool error does not poison
the stdio MCP session.

### Hermetic automated live-MCP testing

Two D6 tests were added:

- `test_stdio_client_calls_get_policy_section_through_mcp`;
- `test_stdio_client_get_policy_section_recovers_after_error`.

They use:

- temporary fixture-backed FastMCP subprocesses;
- `StdioServerParameters`;
- `stdio_client`;
- `ClientSession`;
- `ClientSession.call_tool`;
- explicit timeout boundaries.

Fixture-server PIDs are embedded in results and asserted to differ from
the pytest/client PID, proving that the tests cross a subprocess/stdin-
stdout boundary rather than calling the project tool directly.

Verification:

- MCP collection: 42;
- complete MCP regression: 42 passed.

### R6E-D7/D8 — complete regression and architecture review

Full repository verification:

- repository collection: 1002 tests;
- full repository regression: 1002 passed;
- MCP collection: 42 tests;
- MCP regression: 42 passed;
- `python -m pip check`: pass;
- `git diff --check`: pass.

Final technical change scope:

- `mcp/server.py`;
- `mcp/tools_policy.py`;
- `tests/test_mcp.py`.

Architecture review confirmed:

- no retrieval/business logic duplicated in `mcp/server.py`;
- no exact-section matching logic duplicated in `mcp/tools_policy.py`;
- stdio remains the only MCP transport;
- both production tools remain READ-only;
- no direct-call shortcut exists in the D6 live-MCP tests.

### Impact of AI tooling

AI assistance materially reduced the time required to:

- trace the existing retrieval contract;
- reason about MCP/FastMCP generated schemas;
- construct repeatable verification commands;
- design subprocess-boundary tests;
- identify stale invariants as the server evolved from one tool to two;
- structure governance evidence around the actual implementation gates.

The strongest benefit was not code generation alone, but systematic
checkpoint decomposition and evidence collection.

The main observed risk was mechanical patch fragility: AI-proposed file
anchors and assumptions can become stale as the repository evolves.
This was mitigated by inspection-first editing, exact diff review,
focused testing, and stopping immediately when an anchor or invariant
failed.

### Current boundary

R6E-D `get_policy_section` is implemented and verified locally.

It is not yet claimed as published.

Current verified baseline:

- production READ tools: 2;
- MCP collection: 42;
- MCP regression: 42 passed;
- repository collection: 1002;
- full repository regression: 1002 passed;
- dependency health: pass;
- diff hygiene: pass.

G3 is materially advanced but not complete.

Remaining MCP work includes the mock-data READ tools, calculation tools,
confirmation-gated ACTION tools, and later agent-through-MCP execution.

After R6E-D is committed, pushed, and synchronized, the next frozen MCP
capability is:

`lookup_employee_profile(employee_id)`.


### R6E-D publication closure

The verified R6E-D `get_policy_section` implementation, tests, and
governance evidence were published after all local verification gates
completed.

- Commit:
  - `281a5db` — `feat(mcp): add exact policy section read tool`.
- Push:
  - `git push origin main`: successful;
  - remote advanced from `330d072` to `281a5db`.
- Post-push synchronization:
  - `HEAD`: `281a5db`;
  - `main`: `281a5db`;
  - `origin/main`: `281a5db`;
  - `origin/HEAD`: `281a5db`.
- Branch state:
  - local `main` tracks `origin/main` with no ahead/behind divergence.
- Working tree after push:
  - clean.
- Published R6E-D verification:
  - production READ tools: 2;
  - `search_policy_documents`: published;
  - `get_policy_section`: published;
  - both expose `readOnlyHint=True`;
  - real stdio `get_policy_section` invocation: verified;
  - clean MCP error translation: verified;
  - same-session recovery after handled tool error: verified;
  - MCP collection: 42 tests;
  - complete MCP regression: 42 passed;
  - repository collection: 1002 tests;
  - full repository regression: 1002 passed.
- R6E-D status:
  - complete and published.

G3 remains materially advanced rather than complete because the
remaining mock-data READ tools, calculation tools, confirmation-gated
ACTION tools, and later agent-through-MCP execution are still pending.

The next frozen MCP capability is:

`lookup_employee_profile(employee_id)`.

## 2026-08-19 — S5 MCP Integration R6E-E `lookup_employee_profile` READ Capability

### AI tools used

- ChatGPT was used as an AI engineering assistant throughout R6E-E.

- AI assistance was used to:

  - review the published R6E-D baseline before starting the next MCP
    capability;

  - reconcile reviewer feedback with `PROJECT_RULES.md`,
    `IMPLEMENTATION_SPEC.md`, the project instructions, and the frozen
    MCP tool contract;

  - preserve the established
    inspect → implement → focused test → real-data validation →
    regression → review discipline;

  - inspect `mock_data/employees.json` and freeze the
    `lookup_employee_profile(employee_id)` response contract before
    production implementation;

  - freeze the E3–E6 test ledger before implementation;

  - identify the framework-agnostic structured-data boundary;

  - design repository-relative mock-data loading without environment
    variables or cache state;

  - review required-field and type validation;

  - verify exact, case-sensitive employee-ID matching;

  - verify nullable `manager_id` and verbatim `start_date` preservation;

  - design deterministic loader-failure tests;

  - inspect FastMCP registration/discovery behavior before modifying the
    production server;

  - identify exactly two historical global tool-cardinality assertions
    requiring E5 repair;

  - design real production stdio success and error/recovery probes;

  - design CI-safe subprocess-backed live MCP tests;

  - reconcile final MCP and repository test counts;

  - review architecture boundaries before governance closure.

### R6E-E1/E2 — contract inspection and freeze

The capability began with read-only inspection.

The frozen public function contract was:

`lookup_employee_profile(employee_id: str)`

returning exactly:

- `name`;
- `role`;
- `employment_type`;
- `location`;
- `manager_id`;
- `start_date`.

The frozen structured-data rules were:

- employee IDs are exact and case-sensitive;
- IDs are not silently normalized;
- `manager_id` is `str | None`;
- `start_date` remains the source string verbatim;
- unknown employees raise `MockDataError`;
- the unknown-employee message is
  `Employee not found: 'E999'.`;
- malformed structured data becomes `MockDataError`;
- V1 deliberately uses no structured-data cache;
- `mcp/tools_data.py` remains framework-agnostic.

The frozen test ledger was:

- published MCP baseline: 42;
- published repository baseline: 1002;
- E3: +9;
- E4: +3;
- E5: +2;
- E6: +2;
- final expected MCP collection: 58;
- final expected repository collection: 1018.

### R6E-E3 — framework-agnostic employee-data implementation

R6E-E3 added `mcp/tools_data.py`.

The module owns:

- repository-relative employee fixture resolution;
- employee-record validation;
- public profile projection;
- deterministic employee indexing;
- public employee lookup;
- `MockDataError`.

The implementation deliberately contains no FastMCP registration.

Verified behavior included:

- real E001 lookup;
- real E003 lookup;
- nullable E012 `manager_id=None`;
- exact six-field response;
- fresh result projection;
- caller mutation isolation;
- type validation;
- blank-input validation;
- leading/trailing whitespace rejection;
- case-sensitive matching;
- clean unknown-employee failure.

### R6E-E3.3C — test-scaffolding issue and repair

One implementation-process issue was found before publication.

Nine E3 tests had initially been authored against:

`load_project_module(...)`

but no generic helper with that name existed in `tests/test_mcp.py`.

The issue was identified during structural inspection before the final
focused test run.

The repair was deliberately test-only:

- verified `ModuleType` was already imported and used by the established
  loader helpers;

- inspected all nine unresolved call sites verbatim before editing;

- added `TOOLS_DATA_PATH`;

- added explicit `load_project_tools_data()`;

- replaced exactly nine unresolved helper calls;

- preserved the existing explicit per-module loader convention;

- did not introduce a generic loader abstraction;

- normalized EOF hygiene;

- verified one helper definition and exactly nine helper call sites.

The first repair script also exposed a useful validation mistake:
a substring count treated the helper definition and helper calls as the
same thing. The guard stopped before writing.

The corrected repair distinguished:

- one helper definition;
- nine actual test call sites.

Because the edit scripts wrote only after all guards passed, the failed
attempts remained atomic and did not partially modify the repository.

This reinforced two engineering lessons:

1. inspect concrete helper conventions before authoring tests against an
   assumed abstraction;

2. structural assertions should distinguish definitions from call sites
   rather than rely on broad substring counts.

### R6E-E3 verification

After the scaffolding repair:

- focused E3 tests: 9 passed;
- MCP collection: 51;
- repository collection: 1011;
- complete MCP regression: 51 passed;
- `git diff --check`: pass.

### R6E-E4 — structured-data failure coverage

R6E-E4 added exactly three loader-failure tests:

- missing employee file;
- malformed employee JSON;
- duplicate employee IDs.

The existing `_load_employee_index(path=...)` test seam was reused.

No environment-variable configuration or production test hook was added.

Verification:

- focused E4 tests: 3 passed;
- MCP collection: 54;
- repository collection: 1014;
- complete MCP regression: 54 passed.

### R6E-E5 — FastMCP READ registration and discovery

R6E-E5 registered `lookup_employee_profile` through the production
FastMCP server.

The server gained a dedicated data-tool loading path while preserving
the existing policy-tool loader.

Production discovery became exactly:

- `search_policy_documents`;
- `get_policy_section`;
- `lookup_employee_profile`.

All three expose:

`ToolAnnotations(readOnlyHint=True)`.

The generated `lookup_employee_profile` input schema preserves:

- object input;
- required `employee_id`;
- `employee_id` type string.

Exactly two previously frozen global-cardinality assertions were repaired.

Exactly two E5 discovery/registration tests were added.

Verification:

- focused E5 tests: 4 passed, including the two repaired assertions;
- MCP collection: 56;
- repository collection: 1016;
- complete MCP regression: 56 passed.

### R6E-E6 — live stdio MCP execution

Manual production-server probes verified the real protocol path:

`ClientSession`
→ stdio subprocess
→ production `mcp/server.py`
→ FastMCP
→ `lookup_employee_profile`.

Successful E001 invocation returned:

- `isError=False`;
- exact six-field `structuredContent`;
- no contract drift.

Unknown E999 invocation returned:

- `isError=True`;
- `structuredContent=None`;
- clean error text containing
  `Employee not found: 'E999'.`;
- no traceback leakage.

The same initialized MCP session then successfully executed E001,
proving recovery after a handled tool error.

Temporary probes were created outside the repository and removed after
verification.

Two CI-safe live MCP tests were then added using the established
fixture-subprocess pattern with:

- `StdioServerParameters`;
- `stdio_client`;
- `ClientSession`;
- explicit timeouts;
- subprocess PID evidence.

Verification:

- focused E6 tests: 2 passed;
- MCP collection: 58;
- repository collection: 1018;
- complete MCP regression: 58 passed.

### R6E-E7/E8 — regression and architecture review

Final technical review verified:

- exact changed technical files:
  - `mcp/server.py`;
  - `mcp/tools_data.py`;
  - `tests/test_mcp.py`;

- `mcp/tools_data.py` remains framework-agnostic;

- local `mcp/` remains a non-package;

- official MCP SDK namespace remains preserved;

- production READ-tool count: 3;

- all three READ tools expose `readOnlyHint=True`;

- frozen six-field employee-profile contract preserved;

- exact clean unknown-employee error preserved;

- MCP collection: 58;

- MCP regression: 58 passed;

- repository collection: 1018;

- full repository regression: 1018 passed;

- dependency health: pass;

- compile checks: pass;

- `git diff --check`: pass.

### R6E-E8.1 — test-classification correction

A broad grep initially reported:

`employee_profile_unit_tests=10`.

That number was not a real test-ledger change.

The grep also matched the E5 discovery test:

`test_lookup_employee_profile_discovery_preserves_read_contract`.

An exact classification review proved:

- E3 behavior tests: 9;
- E4 loader-failure tests: 3;
- E5 discovery/registration tests: 2;
- E6 live stdio tests: 2;
- total net-new tests: 16.

Therefore:

- 42 + 16 = 58 MCP tests;
- 1002 + 16 = 1018 repository tests.

No actual collection drift occurred.

The lesson is to classify test families using exact test names or
checkpoint-specific patterns rather than broad shared-name prefixes.

### What worked well

- Contract inspection occurred before implementation.

- The test-count ledger was frozen before production work.

- Structured-data business logic remained independent of FastMCP.

- Repository-relative fixture loading avoided runtime configuration
  complexity.

- Real mock-data cases were verified before registration.

- Registration reused the existing framework-agnostic implementation.

- Real production stdio behavior was manually verified before being
  encoded in CI-safe automated tests.

- Fail-closed edit scripts prevented partial repository modification
  when assumptions were wrong.

- Cardinality assumptions were exhaustively inspected before expanding
  the production tool surface.

- Full repository regression and dependency health were completed before
  governance closure.

### What should improve

- Test helpers must be inspected before new tests are authored against
  them.

- Batch-authored tests should receive an earlier structural or focused
  run so repeated scaffolding mistakes surface after the first instance
  rather than after all related tests are written.

- Review scripts should use exact semantic classifications rather than
  broad grep prefixes where test families share a common name.

- macOS `/tmp` canonicalization to `/private/tmp` should be accounted for
  by testing whether a probe is outside the repository rather than
  assuming a canonical `/tmp/...` path.

### Current R6E-E state

R6E-E `lookup_employee_profile` is complete and published.

Published implementation:

- commit: `4b5e561` —
  `feat(mcp): add employee profile read tool`;

- production READ tools: 3;

- MCP collection: 58;

- complete MCP regression: 58 passed;

- repository collection: 1018;

- full repository regression: 1018 passed;

- dependency health: pass;

- compile checks: pass;

- `git diff --check`: pass.

The next frozen MCP capability is:

`lookup_benefits_status(employee_id)`.


### R6E-E publication closure

The verified R6E-E `lookup_employee_profile` implementation, tests, and
governance evidence were published after all local verification gates
completed.

- Commit:
  - `4b5e561` — `feat(mcp): add employee profile read tool`.

- Push:
  - `git push origin main`: successful;
  - remote advanced from `1806db0` to `4b5e561`.

- Post-push synchronization:
  - `HEAD`: `4b5e561`;
  - `main`: `4b5e561`;
  - `origin/main`: `4b5e561`;
  - `origin/HEAD`: `4b5e561`.

- Branch state:
  - local `main` tracks `origin/main` with no ahead/behind divergence.

- Working tree after push:
  - clean.

- Published R6E-E verification:
  - production READ tools: 3;
  - MCP collection: 58;
  - complete MCP regression: 58 passed;
  - repository collection: 1018;
  - full repository regression: 1018 passed;
  - dependency health: pass;
  - compile checks: pass;
  - `git diff --check`: pass.

- R6E-E status:
  - complete and published.

The next frozen MCP capability is:

`lookup_benefits_status(employee_id)`.

## 2026-08-20 — S5 MCP Integration R6E-F0 Reviewer / Compliance Remediation

### Purpose

R6E-F0 was introduced as a bounded compliance checkpoint after
publication of R6E-E and before implementation of
`lookup_benefits_status(employee_id)`.

The checkpoint responded to reviewer concerns by applying the existing
project discipline:

`inspect → verify alleged gap → freeze correction → implement only the
verified gap → focused test → full regression → review`.

No reviewer concern was treated as a defect until repository evidence
proved that the defect actually existed.

### AI tools used

- ChatGPT was used as an AI engineering assistant to:
  - interpret reviewer comments against `PROJECT_RULES.md`,
    `IMPLEMENTATION_SPEC.md`, and committed engineering evidence;
  - distinguish already-satisfied requirements from genuine gaps;
  - design read-only inspection checkpoints;
  - inspect MCP dependency, discovery, annotation, and cardinality
    evidence;
  - design deterministic benefits-policy consistency probes;
  - identify verification-script false positives;
  - freeze and implement the smallest test-only compliance correction;
  - review test counts, file scope, and architecture before governance
    closure.

### Reviewer-gap verification

The review raised concerns about:

- exact pinned MCP dependency evidence;
- `ToolAnnotations` / `readOnlyHint` support;
- annotation propagation through `list_tools()`;
- committed CI discovery assertions;
- current and final MCP tool cardinality;
- READ / CALCULATION / ACTION classification;
- confirmation architecture;
- benefits-data consistency;
- final eight-tool CI coverage.

Inspection proved that most of these requirements were already
satisfied.

Verified existing evidence included:

- `requirements.txt` pins `mcp==1.29.0`;

- the S5 dependency checkpoint is committed in
  `design-and-evaluation.md`;

- the pinned SDK supports `ToolAnnotations` and `readOnlyHint`;

- all three published READ tools are registered with
  `readOnlyHint=True`;

- live `list_tools()` discovery preserves the annotation;

- committed tests already assert discovery, annotation propagation,
  schema preservation, and exact current production cardinality;

- the frozen specification already distinguishes:
  - READ tools;
  - CALCULATION tools;
  - ACTION tools;

- confirmation middleware is already designed to consume discovered
  MCP metadata rather than a hardcoded action-tool registry.

No production correction was required for those concerns.

### Benefits-policy consistency inspection

Before beginning the benefits READ tool, the complete benefits fixture
was checked against employee records and HR-POL-007.

The audit covered all 12 employee/benefits records.

Verified:

- employee records: 12;

- benefits records: 12;

- employees without benefits records: none;

- benefits records without employees: none;

- full-time and part-time employees follow the policy eligibility rule;

- E005, a full-time employee in probation, is recorded as `pending`
  because the frozen as-of date precedes the calculated coverage start;

- E006, a contractor without an exception, is recorded as `ineligible`
  with no coverage start;

- all eligible coverage dates equal the first day of the month following
  30 days of employment;

- all 12 coverage-rule checks passed with zero violations;

- benefit election states are internally consistent:
  - eligible → `enrolled` or `declined`;
  - pending → all `pending`;
  - ineligible → all `not_available`.

No mock-data or policy correction was required.

### Verified compliance gap

The only genuine gap identified by R6E-F0 was:

the frozen final eight-tool S5 MCP contract existed in
`IMPLEMENTATION_SPEC.md` but was not encoded in source-level CI.

The final frozen contract is:

READ / `readOnlyHint=True`:

- `search_policy_documents`;
- `get_policy_section`;
- `lookup_employee_profile`;
- `lookup_benefits_status`.

CALCULATION / `readOnlyHint=True`:

- `check_pto_balance`;
- `check_policy_compliance`.

ACTION / `readOnlyHint=False`:

- `create_mock_hr_ticket`;
- `draft_hr_email`.

### R6E-F0 correction

The correction is deliberately test-only.

Modified:

- `tests/test_mcp.py`.

Not modified:

- `mcp/server.py`;
- `mcp/tools_data.py`;
- `requirements.txt`;
- `IMPLEMENTATION_SPEC.md`;
- `PROJECT_RULES.md`.

Two test-level contracts were added:

- `CURRENT_COMPLETED_MCP_TOOL_NAMES`:
  - `search_policy_documents`;
  - `get_policy_section`;
  - `lookup_employee_profile`;

- `FINAL_REQUIRED_MCP_TOOL_NAMES`:
  - all eight frozen MCP tools.

The existing production-cardinality test now checks `list_tools()`
against `CURRENT_COMPLETED_MCP_TOOL_NAMES`.

A new compliance test verifies that:

- the final contract contains exactly eight names;

- all eight names are unique;

- the current completed tuple is the first three entries of the final
  contract;

- the complete final tuple exactly matches the frozen S5 specification.

This permits incremental implementation without pretending that
unimplemented tools already exist while still protecting the final
eight-tool completion requirement from drift.

### Verification-script issues and lessons

Several inspection failures occurred in review tooling rather than in
production code.

#### Incorrect corpus path

An initial benefits-policy search used:

`corpus/policies_md`

instead of the repository's actual source path:

`corpus/source/policies_md`.

The search therefore returned no evidence even though HR-POL-007
contains the required benefits rules.

Correction:

- inspect repository layout before constructing corpus search paths;
- prefer paths derived from the frozen repository structure rather than
  remembered paths.

#### Pytest invocation mismatch

A review command used:

`pytest --collect-only`

and produced an import failure for `rag`.

The project's verified invocation convention is:

`python -m pytest`.

Re-running through the active project interpreter restored the expected
58-test baseline before the F0 correction.

Lesson:

- use `python -m pytest` consistently for project verification;
- do not substitute the standalone pytest executable during checkpoint
  evidence collection unless executable-path behavior is itself under
  test.

#### Broad substring assignment count

The first F0.4 edit script counted:

`FINAL_REQUIRED_MCP_TOOL_NAMES =`

as a raw substring.

That also matched:

`FINAL_REQUIRED_MCP_TOOL_NAMES ==`

inside an assertion and falsely reported assignment-count drift.

The script was fail-closed and wrote nothing.

Correction:

- use line-anchored structural matching for assignments;
- distinguish syntax-level concepts from broad text substrings.

The corrected guard verified exactly one top-level current-tool
assignment, one final-tool assignment, and one new compliance test.

### R6E-F0 verification

Focused verification:

- final eight-tool contract test: 1 passed;

- current production-cardinality test: 1 passed.

Collection:

- MCP collection advanced exactly:
  - 58 → 59;

- repository collection advanced exactly:
  - 1018 → 1019.

Regression:

- complete MCP regression: 59 passed;

- full repository regression: 1019 passed.

Additional gates:

- dependency health: pass;

- compile checks: pass;

- `git diff --check`: pass;

- production source change guard: pass;

- technical file scope:
  - `tests/test_mcp.py` only.

### What worked well

- Reviewer feedback was treated as a hypothesis to verify rather than a
  reason to modify correct code immediately.

- Existing committed evidence prevented duplicate annotation and
  dependency tests from being added unnecessarily.

- Complete benefits-policy consistency was proven before the benefits
  tool contract was frozen.

- The final-tool compliance fix remained test-only.

- Current runtime cardinality and final S5 completion cardinality are now
  represented separately.

- Fail-closed scripts again prevented partial edits when review logic was
  incorrect.

- Exact test-count baselines exposed collection drift immediately.

### What should improve

- Search paths should be derived from inspected repository structure.

- `python -m pytest` should remain the canonical pytest invocation.

- Verification scripts should prefer AST, anchored regex, or exact test
  names over broad substring matching.

- Compliance remediation should remain bounded so reviewer feedback does
  not delay implementation of the actual remaining MCP capabilities.

### Current R6E-F0 state

R6E-F0 reviewer/compliance remediation is implemented and verified
locally.

It is not yet claimed as published.

Verified baseline:

- current production MCP tools: 3;

- final required MCP tools: 8;

- MCP collection: 59;

- complete MCP regression: 59 passed;

- repository collection: 1019;

- full repository regression: 1019 passed;

- production behavior: unchanged.

Publication remains pending until the test-only correction and
governance evidence are reviewed, committed, pushed, and synchronized.

After R6E-F0 publication closure, the next frozen MCP capability is:

`lookup_benefits_status(employee_id)`.
