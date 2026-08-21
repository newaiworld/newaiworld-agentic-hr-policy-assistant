# Design and Evaluation

## Project Status

S1–S4 are complete and verified. The project is now in S5 — MCP Integration. R6E-C5 FastMCP READ registration, R6E-C6 live invocation of `search_policy_documents(query, k=5)`, R6E-D `get_policy_section(doc_id, section)`, R6E-E `lookup_employee_profile(employee_id)`, and R6E-F0 reviewer/compliance remediation are complete and published.

R6E-F1 `lookup_benefits_status(employee_id)` is complete and published at commit `755768f`. The capability reads only stored `mock_data/benefits.json` state through framework-agnostic `mcp/tools_data.py`, exposes the frozen public response `{elections, eligibility, coverage_start}`, is registered through the existing `_load_data_tool()` path with `readOnlyHint=True`, and is verified through focused behavior tests, loader-failure tests, FastMCP discovery/registration tests, real stdio invocation, same-session error recovery, complete MCP regression, and full repository regression.

R6E-F2 `check_pto_balance(employee_id)` is complete and published at commit `60ec09b`. The capability reads validated stored
state from `mock_data/pto.json` through framework-agnostic
`mcp/tools_data.py`, exposes exactly
`{available_days, accrual_rate, next_accrual_date}`, performs no runtime
entitlement, FTE, or date calculation, is registered through the existing
`_load_data_tool()` path with `readOnlyHint=True`, and has been verified
through behavior, loader-failure, fixture/policy-consistency,
discovery/registration, real stdio, same-session recovery, complete MCP
regression, and full repository regression.


Production discovery now exposes exactly seven completed MCP tools:

- `search_policy_documents`;
- `get_policy_section`;
- `lookup_employee_profile`;
- `lookup_benefits_status`;
- `check_pto_balance`;
- `check_policy_compliance`;
- `create_mock_hr_ticket`.

The current MCP contract therefore contains seven completed tools while the
frozen final S5 contract remains eight tools.

The six READ/CALCULATION capabilities expose `readOnlyHint=True`.

`create_mock_hr_ticket` is the first completed ACTION capability and exposes
`readOnlyHint=False`.

The remaining frozen MCP capability is `draft_hr_email`.

Confirmation is intentionally not implemented inside the
`create_mock_hr_ticket` MCP primitive. Under AD-06 and AD-10, the later
agent/web confirmation middleware owns preview generation,
`pending_confirmation`, server-generated `confirmation_id` binding, explicit
user confirmation, and subsequent ACTION execution.

R6E-F4 `create_mock_hr_ticket` is complete and published at commit
`cf3e3f8`. Production MCP discovery therefore contains seven of the eight
frozen S5 tools. The remaining frozen MCP capability is `draft_hr_email`.
S6–S10 remain pending and are not yet claimed as implemented.

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
| AD-F3-001 | `check_policy_compliance` requires policy-grounded behavior, but the frozen V1 interface is only `topic + employee_id` and the scenario is fixed to `remote_work_international`. | Verify governing policy facts through the real retrieval layer during engineering, then execute deterministic frozen compliance logic at production runtime without RAG, Chroma, embeddings, or policy retrieval. | Improves determinism, latency, evaluation stability, and trace clarity; requires revalidation when relevant corpus/version semantics or HR-POL-004 §4.4, HR-POL-004 §8, or HR-POL-005 §4.5 change. |
| AD-F4-001 | S3 established the ticket lifecycle vocabulary `open`, `pending`, and `closed`, but did not define the initial lifecycle state for the future ticket-creation action. | Persist newly created mock HR tickets with lifecycle status `open`. | New ticket records remain compatible with the existing S3 ticket schema and lifecycle vocabulary; persisted lifecycle state remains distinct from the MCP public action marker `status: "MOCK"`. |
| AD-F4-002 | Existing S3 ticket records contain offset-aware `created_at` timestamps, but no runtime timestamp-generation rule was recorded. | Generate new ticket `created_at` values as offset-aware UTC ISO-8601 timestamps. | Runtime ticket creation is host- and DST-independent, requires no additional dependency, and can use a deterministic internal clock helper in tests. |

### IMPLEMENTATION_SPEC.md v3.4 amendment — 2026-08-21

| Area | Old | New | Reason |
|---|---|---|---|
| ACTION ownership | ACTION tools were described as confirmation-gated without an explicit MCP/orchestration parameter boundary. | MCP ACTION primitives accept business parameters only; confirmation/session state remains in agent/API orchestration under AD-06 and AD-10. | F4 proved the boundary and explicit wording prevents S6 from leaking orchestration state into MCP schemas. |
| Ticket persistence | `create_mock_hr_ticket` was described as an `append-only log`. | Ticket creation preserves existing records and publishes complete validated ticket state atomically. | Matches the implemented F4 persistence mechanism without implying filesystem append I/O. |
| Mutation testing | S5 had no explicit fixture-isolation requirement for state-changing tools. | State-mutating tests use isolated disposable writable state and preserve committed fixtures. | F4 demonstrated real-protocol ACTION testing without modifying authoritative mock data. |
| Mock-data configuration | `MOCK_DATA_DIR` was listed as V1 runtime configuration. | MCP mock-data paths are repository-relative under `PROJECT_ROOT / "mock_data"`. | Production MCP code does not consume `MOCK_DATA_DIR`; retaining it as a runtime knob would be misleading. |
| Acceptance evidence | The spec did not explicitly distinguish implementation, MCP, API, and deployment evidence boundaries. | Evidence must exercise the architectural boundary being claimed. | Prevents lower-level tests from being presented as proof of higher-level rubric behavior. |
| S5 Definition of Done | S5 described only the final eight-tool outcome. | Add incremental per-capability gates while preserving the exact final eight-tool acceptance gate. | Reflects the verified S5 engineering discipline without weakening the final contract. |


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

The capability is implemented, verified, and published. The complete
R6E-D implementation, tests, and governance evidence were committed and
pushed at `281a5db`.

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

R6E-D publication:

- commit: `281a5db` —
  `feat(mcp): add exact policy section read tool`;
- push to `origin/main`: successful;
- remote advanced from `330d072` to `281a5db`;
- `HEAD`, `main`, `origin/main`, and `origin/HEAD` synchronized at
  `281a5db`;
- local `main` tracks `origin/main` without ahead/behind divergence;
- working tree after push: clean.

The published technical change set remains limited to:

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

R6E-D is published.

### S5 — R6E-E `lookup_employee_profile` READ Capability Evidence

R6E-E adds the first frozen mock-data-backed READ tool:

`lookup_employee_profile(employee_id: str) -> {name, role, employment_type, location, manager_id, start_date}`.

The capability is implemented, verified, and published. The complete
R6E-E implementation, tests, and governance evidence were committed and
pushed at `4b5e561`.

#### Architecture boundary

The implementation preserves the separation between structured-data
business logic and MCP registration:

`ClientSession`
→ stdio subprocess
→ FastMCP
→ `lookup_employee_profile`
→ `mcp/tools_data.py`
→ validated `mock_data/employees.json`.

`mcp/tools_data.py` remains framework-agnostic. It imports no MCP SDK
module and contains no FastMCP registration.

The local `mcp/` directory also remains a non-package, preserving the
installed official MCP SDK namespace.

#### Structured-data contract

The employee fixture path is resolved repository-relatively through:

`PROJECT_ROOT / "mock_data" / "employees.json"`.

V1 deliberately uses no cache.

Employee IDs are:

- required strings;
- non-empty;
- exact;
- case-sensitive;
- not silently normalized.

The loader validates required employee fields and types before a record
is exposed through the public tool.

The frozen public response contains exactly:

- `name`;
- `role`;
- `employment_type`;
- `location`;
- `manager_id`;
- `start_date`.

`manager_id` preserves `str | None`.

`start_date` remains the source string verbatim.

The public result is a fresh projection rather than the source record, so
caller mutation does not mutate the underlying fixture object.

#### Failure behavior

Structured-data failures use:

`MockDataError(RuntimeError)`.

Verified failure paths include:

- missing employee data file;
- malformed JSON;
- invalid employee record structure;
- duplicate employee IDs;
- unknown employee ID.

The frozen unknown-employee error is:

`Employee not found: 'E999'.`

The real MCP protocol translates that failure to a handled
`CallToolResult` with:

- `isError=True`;
- `structuredContent=None`;
- clean user-visible error text;
- no traceback leakage.

The same initialized MCP session remains usable after the handled tool
error.

#### Production registration and discovery

Production FastMCP discovery now exposes exactly three READ tools:

- `search_policy_documents`;
- `get_policy_section`;
- `lookup_employee_profile`.

All three advertise:

`readOnlyHint=True`.

`lookup_employee_profile` discovery preserves:

- input schema type: object;
- required argument: `employee_id`;
- `employee_id` type: string.

The server registration reuses the framework-agnostic implementation from
`mcp/tools_data.py`; business logic is not duplicated in `mcp/server.py`.

#### Real-data validation

Representative mock-data validation includes:

- E001:
  - `Alex Rivera`;
  - `Senior Data Analyst`;
  - `full_time`;
  - `SYDNEY_HQ`;
  - manager `E010`;
  - start date `2023-04-17`;

- E003:
  - real fixture lookup verified;

- E012:
  - nullable `manager_id=None` verified.

#### Live MCP validation

Manual real-production stdio probes verified:

- successful `lookup_employee_profile("E001")`;
- exact six-field `structuredContent`;
- unknown `E999` returns a clean MCP error;
- `structuredContent=None` on error;
- no traceback leakage;
- same-session E001 recovery after the handled E999 error.

Automated CI-safe stdio tests reuse the established fixture-subprocess
pattern with:

- real `StdioServerParameters`;
- real `stdio_client`;
- real `ClientSession`;
- explicit timeouts;
- subprocess PID proof distinct from the pytest process.

#### Test ledger

The frozen R6E-E test progression is:

- published MCP baseline: 42;
- published repository baseline: 1002;
- E3 behavior tests: +9;
- E4 loader-failure tests: +3;
- E5 registration/discovery tests: +2;
- E6 live stdio tests: +2;
- total net-new E-series tests: 16;
- final MCP collection: 58;
- final repository collection: 1018.

The E8.1 classification review confirmed the exact split:

- E3: 9;
- E4: 3;
- E5: 2;
- E6: 2.

The earlier broad grep that reported 10 employee-profile tests was only a
classification artifact because it also matched the E5 discovery test.
There was no test-count drift.

#### Verification results

Final local verification:

- exact changed technical files:
  - `mcp/server.py`;
  - `mcp/tools_data.py`;
  - `tests/test_mcp.py`;

- production READ tools: 3;

- complete MCP collection: 58 tests;

- complete MCP regression: 58 passed;

- full repository collection: 1018 tests;

- full repository regression: 1018 passed;

- `python -m pip check`: pass;

- compile checks: pass;

- `git diff --check`: pass.

#### R6E-E publication

- commit: `4b5e561` —
  `feat(mcp): add employee profile read tool`;

- push to `origin/main`: successful;

- remote advanced from `1806db0` to `4b5e561`;

- `HEAD`, `main`, `origin/main`, and `origin/HEAD` synchronized at
  `4b5e561`;

- working tree after push: clean.

#### Current grading boundary

G3 is materially advanced but not complete.

Three production READ tools are now implemented, registered,
discoverable through MCP metadata, and exercised across the real stdio
protocol boundary:

- `search_policy_documents`;
- `get_policy_section`;
- `lookup_employee_profile`.

The remaining mock-data READ tools, calculation tools, confirmation-gated
ACTION tools, and later agent-through-MCP execution are still pending.

R6E-E is now published. The next frozen MCP capability is:

`lookup_benefits_status(employee_id)`.

### S5 — R6E-F0 Reviewer / Compliance Remediation Evidence

R6E-F0 was introduced as a bounded compliance checkpoint before
`lookup_benefits_status(employee_id)` implementation.

The checkpoint followed the established inspect-first sequence:

`inspect → verify alleged gap → freeze correction → implement only the
verified gap → focused test → full regression → architecture review`.

#### Dependency and annotation evidence

R6E-F0 reconfirmed the frozen S5 MCP dependency checkpoint:

- `requirements.txt` pins `mcp==1.29.0`;

- the pinned SDK exposes `ToolAnnotations` and `readOnlyHint`;

- production registration uses
  `ToolAnnotations(readOnlyHint=True)` for all three published READ tools;

- live `list_tools()` discovery preserves `readOnlyHint=True`;

- committed MCP discovery tests already assert annotation propagation
  and current production cardinality;

- no SDK, transport, server-registration, or runtime changes were
  required.

#### Benefits-data consistency evidence

Before freezing the next mock-data READ tool, all 12 benefits records
were cross-checked against employee records and HR-POL-007.

Verified invariants:

- all 12 employees have exactly one benefits record;

- no benefits record references an unknown employee;

- full-time and part-time employee records follow the policy eligibility
  rule;

- the contractor record is ineligible with no coverage start;

- probation does not incorrectly remove eligibility;

- coverage dates follow the first-day-of-the-month-after-30-days rule;

- all 12 coverage/eligibility records pass the deterministic audit;

- election states are internally consistent:
  - eligible → `enrolled` / `declined`;
  - pending → all `pending`;
  - ineligible → all `not_available`.

No benefits fixture or policy change was required.

#### Final eight-tool CI gap

The only verified compliance gap was that the frozen final eight-tool
S5 contract existed in `IMPLEMENTATION_SPEC.md` but was not represented
in source-level CI.

R6E-F0 therefore added test-only contracts:

- `CURRENT_COMPLETED_MCP_TOOL_NAMES`:
  - `search_policy_documents`;
  - `get_policy_section`;
  - `lookup_employee_profile`;

- `FINAL_REQUIRED_MCP_TOOL_NAMES`:
  - `search_policy_documents`;
  - `get_policy_section`;
  - `lookup_employee_profile`;
  - `lookup_benefits_status`;
  - `check_pto_balance`;
  - `check_policy_compliance`;
  - `create_mock_hr_ticket`;
  - `draft_hr_email`.

The existing runtime-discovery test now compares production
`list_tools()` against the current-completed tuple.

A new compliance test verifies:

- exactly eight final tool names;

- all eight names are unique;

- the first three final names equal the current completed contract;

- the final tuple exactly matches the frozen specification.

This preserves incremental implementation while ensuring the final S5
completion target cannot silently drift.

#### R6E-F0 technical scope

Production behavior is unchanged.

Technical modification:

- `tests/test_mcp.py` only.

No changes were made to:

- `mcp/server.py`;

- `mcp/tools_data.py`;

- `requirements.txt`;

- `IMPLEMENTATION_SPEC.md`;

- `PROJECT_RULES.md`.

#### Verified R6E-F0 baseline

- focused final-contract test: pass;

- current production-cardinality test: pass;

- MCP collection: 59 tests;

- complete MCP regression: 59 passed;

- repository collection: 1019 tests;

- full repository regression: 1019 passed;

- dependency health: pass;

- compile checks: pass;

- `git diff --check`: pass.

R6E-F0 is complete and published.

#### R6E-F0 publication

- commit: `c4783d3` —
  `test(mcp): freeze final eight-tool contract`;

- push to `origin/main`: successful;

- remote advanced from `1d369e2` to `c4783d3`;

- `HEAD`, `main`, `origin/main`, and `origin/HEAD` synchronized at
  `c4783d3`;

- working tree after push: clean;

- published MCP collection: 59 tests;

- published MCP regression: 59 passed;

- published repository collection: 1019 tests;

- published full repository regression: 1019 passed;

- production behavior remained unchanged.

R6E-F0 is now published. The next frozen MCP capability is:

`lookup_benefits_status(employee_id)`.

### S5 — R6E-F1 `lookup_benefits_status` READ Capability Evidence

R6E-F1 adds the fourth frozen MCP READ capability:

`lookup_benefits_status(employee_id: str) -> {elections, eligibility, coverage_start}`.

The capability is complete and published at commit `755768f`.

Publication evidence:

- commit: `755768f` —
  `feat(mcp): add benefits status read tool`;

- push to `origin/main`: successful;

- remote advanced from `5d8afc5` to `755768f`;

- synchronized refs:
  `HEAD`, `main`, `origin/main`, and `origin/HEAD` all resolve to
  `755768f`;

- working tree after push: clean.

#### R6E-F1 contract

Input:

- `employee_id`;
- exact case-sensitive string;
- non-empty;
- no leading or trailing whitespace.

Public output:

- `elections`;
- `eligibility`;
- `coverage_start`.

The implementation deliberately does not expose `employee_id` in the public response.

Stored-data semantics:

- source: `mock_data/benefits.json`;
- no runtime employee-data join;
- no policy/RAG recomputation;
- the tool reports frozen stored benefits state only;
- each result returns a fresh top-level dictionary and a fresh nested `elections` dictionary.

Representative verified cases:

- E001:
  - eligibility: `eligible`;
  - coverage start: `2023-06-01`;
  - health support: `enrolled`;
  - professional development: `enrolled`;
  - wellbeing program: `enrolled`;

- E005:
  - eligibility: `pending`;
  - coverage start: `None`;
  - all three elections: `pending`;

- E006:
  - eligibility: `ineligible`;
  - coverage start: `None`;
  - all three elections: `not_available`.

Unknown employee IDs raise a clean `MockDataError`:

`Benefits record not found for employee: 'E999'.`

#### R6E-F1 implementation architecture

`mcp/tools_data.py` remains framework-agnostic and environment-independent.

The benefits implementation adds:

- `BENEFITS_PATH`;
- benefits field and allowed-value constants;
- `_benefits_record_label()`;
- `_validate_benefits_record()`;
- `_project_benefits_status()`;
- `_load_benefits_index()`;
- `lookup_benefits_status()`.

The public lookup calls only the benefits loader and benefits projection path. It does not call:

- `_load_employee_index()`;
- `lookup_employee_profile()`;
- `search_policy_documents()`;
- `get_policy_section()`.

Production registration reuses the existing server abstraction:

`lookup_benefits_status = _load_data_tool("lookup_benefits_status")`.

FastMCP registration preserves the frozen READ classification:

`ToolAnnotations(readOnlyHint=True)`.

The V1 transport remains explicit stdio.

#### R6E-F1 production discovery

Production `list_tools()` now returns exactly four completed READ tools:

1. `search_policy_documents`;
2. `get_policy_section`;
3. `lookup_employee_profile`;
4. `lookup_benefits_status`.

All four expose `readOnlyHint=True`.

The generated `lookup_benefits_status` input schema preserves:

- input type: object;
- `employee_id` type: string;
- required fields: `["employee_id"]`.

The current/final tool contracts are:

- current completed: 4 tools;
- final required: 8 tools.

The current completed tuple remains the exact prefix of the frozen final eight-tool contract.

#### R6E-F1 test progression

The frozen R6E-F1 permanent-test ledger is:

- F1.3 behavior tests: 9;
- F1.4 loader-failure tests: 3;
- F1.5 discovery/registration tests: 2;
- F1.6 real stdio tests: 2;
- total net-new tests: 16.

Behavior coverage verifies:

- exact public projection;
- real E001 status;
- E005 pending status;
- E006 ineligible status;
- non-string input rejection;
- blank input rejection;
- case sensitivity;
- clean unknown-employee errors;
- fresh projection / mutation isolation.

Loader coverage verifies:

- missing benefits file;
- malformed JSON;
- duplicate employee IDs.

Discovery/registration coverage verifies:

- READ annotation and input schema;
- production registration reuses the existing framework-agnostic implementation.

Real stdio coverage verifies:

- successful live MCP invocation;
- separate subprocess execution;
- structured result preservation;
- clean tool error propagation;
- no traceback leakage;
- successful same-session recovery after an E999 failure.

#### R6E-F1 verification baseline

Verified technical evidence:

- production READ tools: 4;
- current MCP contract: 4;
- final MCP contract: 8;
- MCP collection: 75;
- complete MCP regression: 75 passed;
- repository collection: 1035;
- full repository regression: 1035 passed;
- `python -m pip check`: pass;
- compile checks: pass;
- `git diff --check`: pass;
- architecture review: pass.

The functional implementation scope remains exactly:

- `mcp/tools_data.py`;
- `mcp/server.py`;
- `tests/test_mcp.py`.

No dependency, frozen specification, transport, or architecture amendment was required.

#### Current R6E-F1 grading boundary

R6E-F1 materially advances G3 but does not complete S5.

Four READ tools are now implemented, discoverable through MCP metadata, and exercised across the real stdio protocol boundary. The final S5 tool contract remains eight tools, so the calculation and ACTION capabilities remain pending.

R6E-F1 is complete and published.

#### R6E-F1 publication

- commit: `755768f` —
  `feat(mcp): add benefits status read tool`;

- push to `origin/main`: successful;

- remote advanced from `5d8afc5` to `755768f`;

- synchronized refs:
  `HEAD`, `main`, `origin/main`, and `origin/HEAD` all resolve to
  `755768f`;

- published MCP regression: 75 passed;

- published full repository regression: 1035 passed;

- published production READ tools: 4;

- current completed MCP tools: 4;

- final required MCP tools: 8.

R6E-F1 is now published. The next frozen MCP capability is:

`check_pto_balance(employee_id)`.

### S5 — R6E-F2 `check_pto_balance` CALCULATION Capability Evidence

R6E-F2 adds the fifth completed MCP capability:

`check_pto_balance(employee_id: str) -> {available_days, accrual_rate, next_accrual_date}`.

The capability is complete and published at commit `60ec09b`.

Published implementation evidence:

- commit:
  `60ec09b` — `feat(mcp): add pto balance calculation tool`;

- push to `origin/main`:
  successful;

- remote advanced from:
  `101152e` to `60ec09b`;

- synchronized refs:
  - `HEAD`: `60ec09b`;
  - `main`: `60ec09b`;
  - `origin/main`: `60ec09b`;
  - `origin/HEAD`: `60ec09b`;

- working tree after push:
  clean.

#### R6E-F2 contract

Input:

- `employee_id`;
- exact case-sensitive string;
- non-empty;
- no leading or trailing whitespace.

Public output:

- `available_days`;
- `accrual_rate`;
- `next_accrual_date`.

The public response deliberately excludes:

- `employee_id`;
- `accrual_unit`;
- `last_updated`;
- fixture-level metadata.

Authoritative data source:

- `mock_data/pto.json`.

Stored-data semantics:

- `available_days` is read from stored state;
- `accrual_rate` is read from stored state;
- `next_accrual_date` is read from stored state;
- annual entitlement is not recomputed at runtime;
- FTE is not multiplied at runtime;
- `next_accrual_date` is not derived at runtime;
- employee data is not joined at runtime;
- policy/RAG retrieval is not invoked;
- a missing contractor PTO balance is not synthesized.

Representative verified cases:

- E001:
  - `available_days`: `8.0`;
  - `accrual_rate`: `1.6667`;
  - `next_accrual_date`: `2026-09-01`;

- E002:
  - `available_days`: `4.5`;
  - `accrual_rate`: `1.0`;
  - `next_accrual_date`: `2026-09-01`;

- E005:
  - `available_days`: `1.0`;
  - `accrual_rate`: `1.6667`;
  - `next_accrual_date`: `2026-09-01`;

- E008:
  - `available_days`: `3.0`;
  - `accrual_rate`: `0.6667`;
  - `next_accrual_date`: `2026-09-01`.

E006 is a contractor with no PTO record. The frozen runtime behavior is a
clean `MockDataError`:

`PTO balance record not found for employee: 'E006'.`

The tool does not synthesize a zero balance.

#### R6E-F2 implementation architecture

`mcp/tools_data.py` remains framework-agnostic and environment-independent.

The PTO implementation adds:

- `PTO_PATH`;
- PTO schema constants;
- `_pto_record_label()`;
- `_validate_pto_record()`;
- `_project_pto_balance()`;
- `_load_pto_index()`;
- `check_pto_balance()`.

The public runtime path calls only:

- `_load_pto_index()`;
- `_project_pto_balance()`.

It does not call:

- `lookup_employee_profile()`;
- `lookup_benefits_status()`;
- `search_policy_documents()`;
- `get_policy_section()`.

No runtime PTO arithmetic is performed inside
`check_pto_balance()`.

The fixture/policy consistency checks intentionally place policy arithmetic
in tests rather than production runtime.

Production registration reuses the existing server abstraction:

`check_pto_balance = _load_data_tool("check_pto_balance")`.

The tool is registered with:

`ToolAnnotations(readOnlyHint=True)`.

The CALCULATION label describes project-level semantics; `readOnlyHint=True`
describes the MCP side-effect classification.

The V1 transport remains explicit stdio.

#### R6E-F2 production discovery

Production `list_tools()` now returns exactly five completed tools:

1. `search_policy_documents`;
2. `get_policy_section`;
3. `lookup_employee_profile`;
4. `lookup_benefits_status`;
5. `check_pto_balance`.

All five expose:

`readOnlyHint=True`.

The generated `check_pto_balance` input schema preserves:

- input type: object;
- `employee_id` type: string;
- required fields: `["employee_id"]`.

The current/final MCP tool contracts are:

- current completed: 5 tools;
- final required: 8 tools.

The current completed tuple remains the exact prefix of the frozen final
eight-tool contract.

#### R6E-F2 fixture/policy consistency boundary

The frozen PTO fixture stores all three public output fields directly.

The consistency tests verify that:

- full-time stored accrual is consistent with the frozen 20-days-per-year
  rule;
- part-time stored accrual is consistent with recorded FTE;
- the contractor fixture absence for E006 is consistent with the frozen
  contractor PTO rule.

These checks validate fixture consistency but do not convert
`check_pto_balance()` into a policy-calculation engine.

#### R6E-F2 test progression

The frozen R6E-F2 permanent-test ledger is:

- F2.3 behavior/public-contract tests: 10;
- F2.4 loader-failure tests: 3;
- F2.5 fixture/policy-consistency tests: 3;
- F2.6 discovery/registration tests: 2;
- F2.7 real stdio tests: 2;
- total net-new tests: 20.

Behavior coverage verifies:

- exact public schema;
- real E001 balance;
- E002 part-time stored accrual;
- E008 part-time stored accrual;
- E005 probation stored PTO state;
- non-string input rejection;
- blank and padded input rejection;
- case sensitivity;
- clean missing-record errors;
- fresh projection / mutation isolation.

Loader coverage verifies:

- missing PTO file;
- malformed JSON;
- duplicate employee IDs.

Fixture/policy consistency verifies:

- full-time accrual consistency;
- part-time FTE consistency;
- contractor absence consistency.

Discovery/registration coverage verifies:

- generated input schema;
- `readOnlyHint=True`;
- registration reuses the framework-agnostic production implementation;
- the completed MCP contract advances from four to five tools while the
  frozen final contract remains eight.

Real stdio coverage verifies:

- successful subprocess invocation;
- structured response preservation;
- separate-process execution;
- clean tool error propagation;
- no traceback leakage;
- successful recovery in the same initialized MCP session after an E999
  error.

#### R6E-F2 verification baseline

Verified technical evidence:

- production tools: 5;
- current completed MCP tools: 5;
- final required MCP tools: 8;
- net-new R6E-F2 tests: 20;
- MCP collection: 95;
- complete MCP regression: 95 passed;
- repository collection: 1055;
- full repository regression: 1055 passed;
- `python -m pip check`: pass;
- compile checks: pass;
- `git diff --check`: pass;
- architecture review: pass.

The functional implementation scope remains exactly:

- `mcp/tools_data.py`;
- `mcp/server.py`;
- `tests/test_mcp.py`.

No dependency, frozen specification, fixture, or transport amendment was
required.

#### Current R6E-F2 grading boundary

R6E-F2 materially advances the MCP-tool completion objective but does not
complete S5.

Five of the frozen eight MCP tools are now implemented and discoverable.
The remaining frozen capabilities are:

1. `check_policy_compliance`;
2. `create_mock_hr_ticket`;
3. `draft_hr_email`.

R6E-F2 is complete and published.

#### R6E-F2 publication

- implementation commit:
  `60ec09b` — `feat(mcp): add pto balance calculation tool`;

- remote transition:
  `101152e` → `60ec09b`;

- synchronized refs:
  - `HEAD`: `60ec09b`;
  - `main`: `60ec09b`;
  - `origin/main`: `60ec09b`;
  - `origin/HEAD`: `60ec09b`;

- published MCP regression:
  95 passed;

- published full repository regression:
  1055 passed;

- published production MCP tools:
  5;

- current completed MCP tools:
  5;

- final required MCP tools:
  8;

- net-new R6E-F2 tests:
  20.

R6E-F2 is now published. The next frozen MCP capability is:

`check_policy_compliance`.

### S5 — R6E-F3 `check_policy_compliance` CALCULATION Capability Evidence

R6E-F3 adds the sixth completed MCP capability:

`check_policy_compliance(topic: str, employee_id: str) -> {compliant, reasons, policy_refs}`.

The capability is complete and published at commit `5987cdc`.

Published implementation:

- commit:
  `5987cdc` — `feat(mcp): add policy compliance calculation tool`;
- push to `origin/main`:
  successful;
- remote transition:
  `8bd2962` → `5987cdc`;
- synchronized refs:
  - `HEAD`: `5987cdc`;
  - `main`: `5987cdc`;
  - `origin/main`: `5987cdc`;
  - `origin/HEAD`: `5987cdc`;
- post-push working tree:
  clean.

#### R6E-F3 contract

Inputs:

- `topic`;
- `employee_id`.

Both inputs are:

- exact case-sensitive strings;
- non-empty;
- rejected if whitespace-only;
- rejected if padded with leading or trailing whitespace.

The frozen V1 topic is:

`remote_work_international`.

Unsupported topics raise a clean `MockDataError`.

The employee identifier is validated against authoritative employee mock
data. Unknown or case-mismatched employee identifiers raise a clean
`MockDataError`.

Public output:

- `compliant`;
- `reasons`;
- `policy_refs`.

For the frozen E003 international-remote-work scenario, the public result is:

- `compliant`:
  `False`;

- `reasons`:
  1. the six-week international remote-work proposal exceeds the standard
     30-calendar-day limit and therefore requires formal exception review;
  2. international remote work also requires the applicable approvals,
     Information Security review, and overseas-access controls before
     approval;

- `policy_refs`:
  - `HR-POL-004 §4.4`;
  - `HR-POL-004 §8`;
  - `HR-POL-005 §4.5`.

The result is projected freshly for every call so callers cannot mutate
shared production state.

#### R6E-F3 policy-evidence mapping

The compliance result is explicitly grounded as follows.

Reason 1:

> A six-week international remote-work proposal exceeds the standard
> 30-calendar-day limit and requires formal exception review.

Grounding:

- `HR-POL-004 §4.4`;
- `HR-POL-004 §8`.

Reason 2:

> International remote work also requires the applicable approvals,
> Information Security review, and overseas-access controls before approval.

Grounding:

- `HR-POL-004 §8`;
- `HR-POL-005 §4.5`.

This mapping was verified through the real policy retrieval path during
engineering.

The production tool does not perform policy retrieval at runtime.

#### R6E-F3 architecture decision

The relevant architecture decision is recorded in the project Architecture
Decision Log as `AD-F3-001`.

The decision separates engineering-time policy verification from runtime
execution.

Engineering-time responsibility:

- query the real policy retrieval layer;
- verify the compliance facts against the authoritative corpus;
- verify the exact reason-to-policy-reference mapping.

Production-runtime responsibility:

- validate the frozen topic;
- validate employee identity;
- return the deterministic frozen compliance projection.

The production runtime deliberately does not invoke:

- `search_policy_documents()`;
- `get_policy_section()`;
- Chroma;
- embeddings;
- runtime policy retrieval;
- environment-dependent configuration.

This decision was selected because the V1 contract represents one frozen,
deterministic scenario rather than a general-purpose compliance engine.

Benefits:

- deterministic behavior;
- lower runtime latency;
- stable evaluation results;
- simpler and more transparent tool traces;
- no hidden duplicate retrieval inside a tool call;
- clearer separation between policy evidence verification and tool
  execution.

Trade-off:

the frozen compliance constants must be revalidated whenever the governing
policy evidence or corpus semantics change.

Mandatory revalidation triggers include:

- a relevant corpus/version semantic change;
- modification of `HR-POL-004 §4.4`;
- modification of `HR-POL-004 §8`;
- modification of `HR-POL-005 §4.5`.

A future generalized compliance capability would require a separate design
decision and is not silently implied by this V1 implementation.

#### R6E-F3 implementation architecture

`mcp/tools_data.py` remains framework-agnostic and
environment-independent.

The F3 production implementation adds:

- `_SUPPORTED_COMPLIANCE_TOPIC`;
- `_REMOTE_WORK_INTERNATIONAL_REASONS`;
- `_REMOTE_WORK_INTERNATIONAL_POLICY_REFS`;
- `_project_policy_compliance()`;
- `check_policy_compliance()`.

The public runtime path uses:

- `_load_employee_index()`;
- `_project_policy_compliance()`.

Employee data is used only to validate identity.

The frozen compliance result deliberately does not branch on employee
profile attributes.

The implementation contains:

- no MCP framework import;
- no RAG import;
- no Chroma import;
- no embedding-model import;
- no environment read;
- no runtime RAG;
- no runtime retrieval.

Production registration reuses the existing generic server abstraction:

`check_policy_compliance = _load_data_tool("check_policy_compliance")`.

The tool is registered with:

`ToolAnnotations(readOnlyHint=True)`.

The CALCULATION label describes project-level semantics.

`readOnlyHint=True` describes the MCP side-effect classification: the tool
does not mutate stored or external state.

The frozen V1 transport remains explicit stdio.

#### R6E-F3 production discovery

Production `list_tools()` now returns exactly six completed tools:

1. `search_policy_documents`;
2. `get_policy_section`;
3. `lookup_employee_profile`;
4. `lookup_benefits_status`;
5. `check_pto_balance`;
6. `check_policy_compliance`.

All six expose:

`readOnlyHint=True`.

The generated `check_policy_compliance` input schema preserves:

- input type:
  object;
- `topic`:
  string;
- `employee_id`:
  string;
- required:
  `["topic", "employee_id"]`.

The current/final MCP contracts are:

- current completed:
  6 tools;
- final required:
  8 tools.

The current six-tool contract remains the exact prefix of the frozen
eight-tool contract.

#### R6E-F3 test progression

The frozen R6E-F3 permanent-test ledger is:

- behavior/public-contract tests:
  13;
- architecture tests:
  2;
- discovery/registration tests:
  2;
- real stdio tests:
  2;
- total net-new tests:
  19.

Behavior coverage verifies:

- frozen E003 result;
- exact public schema;
- boolean `compliant`;
- frozen reasons;
- frozen policy references;
- fresh projection / mutation isolation;
- non-string topic rejection;
- blank or padded topic rejection;
- unsupported-topic rejection;
- non-string employee identifier rejection;
- blank or padded employee identifier rejection;
- case sensitivity;
- clean unknown-employee errors.

Architecture coverage verifies:

- no runtime policy/RAG retrieval dependency;
- framework independence;
- environment independence.

Discovery/registration coverage verifies:

- exact generated two-argument schema;
- both arguments required;
- `readOnlyHint=True`;
- registration reuses the existing framework-agnostic production
  implementation;
- current completed tool contract advances from five to six;
- frozen final contract remains eight tools.

Real stdio coverage verifies:

- real MCP subprocess invocation;
- initialized `ClientSession`;
- successful structured compliance result;
- subprocess execution independently verified without changing the frozen
  public response schema;
- clean error propagation for an invalid employee;
- no traceback leakage;
- successful valid call through the same initialized session after the
  preceding error.

#### R6E-F3 verification baseline

Verified technical evidence:

- production MCP tools:
  6;
- current completed MCP tools:
  6;
- final required MCP tools:
  8;
- net-new R6E-F3 tests:
  19;
- MCP collection:
  114;
- complete MCP regression:
  114 passed;
- repository collection:
  1074;
- complete repository regression:
  1074 passed;
- `python -m pip check`:
  pass;
- compile checks:
  pass;
- `git diff --check`:
  pass;
- architecture review:
  pass.

The functional implementation scope remains exactly:

- `mcp/tools_data.py`;
- `mcp/server.py`;
- `tests/test_mcp.py`.

No dependency, frozen specification, fixture, or transport amendment was
required.

#### R6E-F3 engineering review findings

Several engineering and review findings materially improved the F3
implementation process.

Policy-grounding reconciliation:

- reviewer feedback required explicit reason-to-policy-reference mapping;
- the final evidence maps each compliance reason to the governing policy
  sections rather than merely listing references globally.

Architecture clarification:

- policy facts are verified through retrieval during engineering;
- runtime execution remains deterministic and does not hide another
  retrieval step inside the compliance tool.

Current-tool contract semantics:

- the existing test name referring only to completed READ tools became
  semantically inaccurate when CALCULATION became the sixth capability;
- it was renamed to refer to completed MCP tools without adding a duplicate
  cardinality test.

Stdio subprocess evidence:

- the frozen response schema could not safely carry a synthetic PID marker;
- separate-process execution was therefore proven using a temporary PID
  file outside the public tool response.

Shell execution discipline:

- an interactive `exit 1` inside a pasted guard terminated the user's shell;
- subsequent guards were executed as child-shell scripts so failure exits
  only the temporary verification process.

Structural-review false positive:

- raw textual occurrence counting conflated symbol definitions with valid
  symbol references;
- the guard was corrected to use AST-based top-level definition counts,
  load-reference counts, and explicit helper-call inspection;
- production code and tests did not change, so the previously verified
  114 / 1074 regression evidence remained authoritative.

Governance preservation finding:

- the prior `PROJECT_STATUS.md` Next Action section mixed operational next
  steps with historical F2 publication evidence;
- replacing the section removed that evidence;
- the preservation guard caught the loss;
- the exact F2/F1 historical publication block was recovered from the
  pre-edit backup and migrated into a dedicated historical section.

#### Current R6E-F3 grading boundary

R6E-F3 materially advances the MCP-tool completion objective but does not
complete S5.

Six of the frozen eight MCP tools are now implemented and discoverable.

The remaining frozen capabilities are:

1. `create_mock_hr_ticket`;
2. `draft_hr_email`.

Both remaining capabilities are ACTION tools and require confirmation-aware
workflow semantics.

R6E-F3 is complete and published.

#### R6E-F3 publication

Published implementation:

- commit:
  `5987cdc` — `feat(mcp): add policy compliance calculation tool`;
- push to `origin/main`:
  successful;
- remote transition:
  `8bd2962` → `5987cdc`;
- synchronized refs:
  - `HEAD`: `5987cdc`;
  - `main`: `5987cdc`;
  - `origin/main`: `5987cdc`;
  - `origin/HEAD`: `5987cdc`;
- post-push working tree:
  clean.

Published R6E-F3 verification:

- production MCP tools:
  6;
- current completed MCP tools:
  6;
- final required MCP tools:
  8;
- net-new R6E-F3 tests:
  19;
- MCP collection:
  114;
- complete MCP regression:
  114 passed;
- repository collection:
  1074;
- complete repository regression:
  1074 passed;
- dependency health:
  pass;
- compile checks:
  pass;
- architecture review:
  pass;
- `git diff --check`:
  pass.

R6E-F3 is now published.

The next frozen MCP capability is:

`create_mock_hr_ticket`.

Do not begin `draft_hr_email`, agent orchestration, or broader
confirmation-workflow integration until `create_mock_hr_ticket` is fully
implemented, verified, governed, and published.
## R6E-F4 — `create_mock_hr_ticket` ACTION Capability

R6E-F4 implements and locally verifies the seventh frozen MCP capability:

`create_mock_hr_ticket(employee_id, category, summary)`.

The project-level semantic classification is ACTION.

The MCP side-effect classification is:

`readOnlyHint=False`.

### R6E-F4 public contract

Inputs:

- `employee_id`: exact non-empty string;
- `category`: exact non-empty string;
- `summary`: exact non-empty string.

Leading or trailing whitespace is rejected rather than silently normalized.

The public result contains exactly:

- `ticket_id`;
- `status`.

For a successful mock action:

- `ticket_id` is allocated sequentially from authoritative ticket state;
- public `status` is `"MOCK"`.

The public `"MOCK"` marker is intentionally distinct from the persisted
ticket lifecycle state.

### R6E-F4 persisted ticket state

A successfully created ticket persists:

- `ticket_id`;
- `employee_id`;
- `category`;
- `summary`;
- lifecycle `status: "open"`;
- offset-aware UTC `created_at`;
- `mock: true`.

The authoritative allocator advances only as part of the validated state
transition.

The implementation preserves the existing S3 ticket lifecycle vocabulary.

The lifecycle and timestamp decisions are recorded as:

- `AD-F4-001`;
- `AD-F4-002`.

### R6E-F4 implementation architecture

`mcp/tools_data.py` remains framework-agnostic.

The production ACTION path is composed from:

- `_load_ticket_state()`;
- `_build_ticket_state_transition()`;
- `_utc_now_iso()`;
- `_write_ticket_state()`;
- `create_mock_hr_ticket()`.

Ticket creation validates authoritative employee and ticket state before
publication.

The state transition is constructed in memory before persistence.

Persistence uses a sibling temporary file followed by `os.replace()` so
publication is atomic at the target-file boundary.

Validation failures occur before filesystem side effects.

Publication failure preserves the authoritative target and removes residual
temporary state.

### R6E-F4 confirmation boundary

`create_mock_hr_ticket` is the production MCP ACTION primitive.

It does not implement user confirmation internally.

This is intentional and consistent with AD-06 and AD-10.

The later agent/web confirmation layer is responsible for:

- generating the action preview;
- storing `pending_confirmation`;
- generating and binding a `confirmation_id`;
- waiting for explicit user confirmation;
- validating the matching confirmation;
- executing the ACTION only after that gate succeeds.

This separation prevents confirmation policy from being duplicated inside
individual MCP ACTION implementations.

### R6E-F4 MCP registration

The production server reuses the existing framework-agnostic implementation
through:

`create_mock_hr_ticket = _load_data_tool("create_mock_hr_ticket")`.

The tool is registered through FastMCP with:

`ToolAnnotations(readOnlyHint=False)`.

Production discovery therefore contains seven completed tools.

The frozen final S5 contract remains eight tools.

The only remaining frozen MCP capability is:

`draft_hr_email`.

### R6E-F4 real stdio verification

Permanent integration tests exercise the ACTION through the real MCP stdio
boundary.

The success path verifies:

- real subprocess-backed stdio transport;
- initialized MCP client session;
- discovery of `readOnlyHint=False`;
- successful ACTION invocation;
- exact structured public response;
- sequential ticket allocation;
- persisted mock ticket state.

Writable state is redirected to an isolated temporary fixture.

The repository production ticket fixture remains unchanged.

The error/recovery path verifies that an invalid ACTION call does not poison
the initialized MCP session and that a subsequent valid READ succeeds through
the same session without causing another ticket mutation.

### R6E-F4 verification baseline

Verified local evidence:

- production MCP tools: 7;
- current completed MCP tools: 7;
- final required MCP tools: 8;
- net-new R6E-F4 tests: 33;
- ACTION-focused tests: 9 passed;
- real stdio ACTION tests: 2 passed;
- MCP collection: 147;
- complete MCP regression: 147 passed;
- complete repository regression: 1107 passed;
- dependency health: pass;
- compile checks: pass;
- production ticket fixture integrity: pass;
- residual repository temporary-file check: pass;
- `git diff --check`: pass.

The repository production ticket baseline remains:

- ticket count: 4;
- next ticket number: 1005;
- ticket identifiers:
  `TKT-1001` through `TKT-1004`.

R6E-F4 is complete and published.

Published implementation commit:

`cf3e3f8` — `feat(mcp): add mock HR ticket action tool`.

Post-push synchronization verified that `HEAD`, `main`, `origin/main`, and
`origin/HEAD` all resolve to
`cf3e3f8dc32ececd33409240ec30b6c21d571e7a`.

The working tree was clean after implementation publication.

The next frozen MCP capability is:

`draft_hr_email`.
