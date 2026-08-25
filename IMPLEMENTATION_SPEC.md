# ============================================================
# IMPLEMENTATION_SPEC.md — Build Specification (v3.5, frozen)
# Read PROJECT_RULES.md first — it governs this file.
# Amendments: record old → new + reason in the decision log
# (design-and-evaluation.md), same commit ("spec:").
# ============================================================

## 1. STACK (frozen)
- Python 3.11, venv, requirements.txt with EXACT pins only
  (package==x.y.z, via pip freeze; regenerate on any dependency
  change — "pinned" means reproducible, not "recent")
- FastAPI + Uvicorn — one app, one process
- UI: single static index.html (fetch + tiny markdown render)
- MCP: official `mcp` SDK (FastMCP), stdio subprocess in-service.
  STDIO IS THE ONLY TRANSPORT IN V1. The HTTP split-service
  option is documented as future work in design-and-evaluation.md
  — not built, not env-configurable.
- Vector store: Chroma. Persist dir gitignored; built during the
  DEPLOY BUILD STEP (§6), loaded at startup.
- Embeddings: BAAI/bge-small-en-v1.5 via sentence-transformers
  (local, deterministic, 384-dim, 512-token context).
  WHY bge-small and not all-MiniLM-L6-v2 (AD-09): MiniLM
  truncates inputs at 256 WordPiece tokens, so any chunk longer
  than ~250 words would SILENTLY lose its tail — and policy
  sections end with eligibility/exception/approval clauses.
  bge-small's 512-token context covers our 450-token max chunk
  with headroom; same size class, same sentence-transformers
  API, zero cost difference. Deploy build pre-downloads the
  model so cold starts never fetch it.
- LLM: Groq free tier (llama-3.3-70b-versatile), OpenAI-compatible
  client; fallback = OpenRouter free model (same client, new
  base_url + key)
- Session state: in-memory dict keyed by conversation_id (§8).
  Single-process free tier makes this sufficient; known
  limitation documented in design-and-evaluation.md.
- pytest; GitHub Actions; Render free web service
- Forbidden without spec amendment: LangChain/LangGraph, Docker,
  Postgres, rerankers, auth, paid APIs, React/Node, HTTP MCP
- WHY: fewest moving parts satisfying every G-item; each extra
  framework or transport is a new failure mode.

## 2. ARCHITECTURE (single service — reuse diagram in docs)
browser → FastAPI (/chat, /health, static UI)
        → agent/orchestrator.py (plan + tool loop + trace)
          → MCP client (stdio) → mcp/server.py (8 tools)
            → rag/ (Chroma index)    [policy tools]
            → mock_data/*.json       [data tools]
        → LLM via agent/llm.py — the ONLY file calling the API
The agent NEVER touches rag/ or mock_data/ directly — only via
MCP tools. That separation is what G3 grades.

## 3. REPO LAYOUT (create exactly this on day 1)
app/            main.py (FastAPI), static/index.html
agent/          orchestrator.py, llm.py, prompts.py, trace.py
mcp/            server.py, tools_policy.py, tools_data.py
rag/            ingest.py, chunk.py, embed.py, store.py, retrieve.py
corpus/         version.json
                source/policies_md/*.md
                source/policies_pdf/*.pdf
                processed/chunks.json        (committed, deterministic)
mock_data/      employees.json, pto.json, benefits.json, tickets.json
evaluation/     eval_set.jsonl, run_eval.py, results/
tests/          test_chunk.py, test_retrieve.py, test_mcp.py, test_app.py
logs/           .gitkeep only — runtime traces/eval logs live here
                (gitignored: logs/* except .gitkeep). Curated demo
                traces are EXPORTED to evaluation/results/ before
                they are needed as evidence.
.github/workflows/ci.yml
chroma_db/      (FULLY GENERATED, gitignored: Chroma store +
                 index_metadata.json. Source of truth =
                 corpus/version.json + chunks.json. Safe to
                 delete and rebuild at any time — never commit.)
requirements.txt, .env.example, .gitignore, README.md
design-and-evaluation.md, ai-tooling.md, deployed.md   (repo ROOT —
                the brief lists them by name; graders look there)
PROJECT_RULES.md, IMPLEMENTATION_SPEC.md, PROJECT_STATUS.md,
SUBMISSION_CHECKLIST.md
RULE: no file outside this tree without a reason in the README.

## 4. DATA SPEC
- Corpus: 12–15 policy docs, ~40–60 pages, ≥2 formats (.md+.pdf);
  topics: PTO, holidays, remote work, expenses, data security,
  benefits, onboarding, equipment, leave, conduct
- Numbers (accrual rates, thresholds) MUST be consistent across
  docs — eval gold answers depend on them
- WF1 DESIGN NOTE: tax/approval/location rules for international
  remote work live INSIDE the remote-work policy doc; data
  security requirements live in the data-security doc. WF1
  retrieval therefore spans exactly 2–3 well-separated docs —
  enough to prove multi-document retrieval, not enough to
  scatter.
- corpus/version.json:
    {"version": "1.0", "created": "YYYY-MM-DD",
     "documents": [
       {"doc_id": "HR-POL-001", "title": "...", "format": "md",
        "doc_version": "1.0", "effective_date": "2026-01-01"},
       ... ]}
  Bump the top-level "version" on ANY corpus change — it feeds
  the index rebuild check (§6). Per-document doc_version and
  effective_date model real HR document lifecycle.
- Mock data: 10–12 employees incl. part-time, contractor, new
  hire in probation; PTO balances, benefits elections, offices,
  a few existing tickets. One office is the company HQ; WF1/WF2
  demo employees (E001, E003) are full-time, mid-tenure.
- Obviously fake: names like "Alex Rivera", @example.com emails

## 5. MCP TOOL CONTRACT (8 tools; fix schemas before coding)
Rubric minimum = 5 tools; we implement 8 for headroom, and the
demo exercises ≥5 of them (G3).

DEPENDENCY CHECKPOINT (S5 gate, do FIRST in S5):
  Verify the pinned `mcp` SDK version supports tool annotations
  (ToolAnnotations / readOnlyHint) in @mcp.tool(). If NOT
  supported: STOP and propose a spec amendment (rules A5) —
  never silently hardcode safety metadata.
  CHECKPOINT EVIDENCE (commit to repo or decision log):
    - mcp package name + exact pinned version
    - one example tool declaration showing the annotation
    - list_tools() output showing annotations present
    - CI assertion result (test_mcp.py discovery test)

TOOL DISCOVERY (this is the MCP learning objective — L3):
  1. At agent startup, the MCP client connects over stdio and
     calls list_tools() on the server.
  2. The returned tool schemas are converted into the LLM
     client's tool-calling format and injected into the agent's
     planning context.
  3. The LLM selects ONLY from discovered tools. Hardcoding a
     tool list anywhere in agent/ is a rule violation (G3).
  4. CI asserts list_tools() returns all 8 expected names.

TOOL CLASSIFICATION — implemented via MCP tool ANNOTATIONS,
not a custom registry:
  READ        readOnlyHint=true   — search_policy_documents,
              get_policy_section, lookup_employee_profile,
              lookup_benefits_status
  CALCULATION readOnlyHint=true   — check_pto_balance,
              check_policy_compliance
  ACTION      readOnlyHint=false  — create_mock_hr_ticket,
              draft_hr_email  (ALWAYS confirmation-gated, G6)
The agent's confirmation middleware reads annotations from the
DISCOVERED tool metadata: readOnlyHint=false → require
confirmation. Any future action tool inherits the gate
automatically; no hardcoded tool names anywhere.

RAG-backed (READ):
  search_policy_documents(query: str, k: int = 5)
      -> [{doc_id, title, section, snippet, score}]
  get_policy_section(doc_id: str, section: str)
      -> {title, section, text}
Mock-data backed (READ):
  lookup_employee_profile(employee_id)
      -> {name, role, employment_type, location, manager_id,
          start_date}
  lookup_benefits_status(employee_id)
      -> {elections, eligibility, coverage_start}
Mock-data backed (CALCULATION):
  check_pto_balance(employee_id)
      -> {available_days, accrual_rate, next_accrual_date}
  check_policy_compliance(topic, employee_id)
      -> {compliant, reasons, policy_refs}
Mock actions (ACTION, confirmation-gated):
  create_mock_hr_ticket(employee_id, category, summary)
      -> {ticket_id, status: "MOCK"}
         Creates exactly one new mock ticket while preserving
         existing ticket records; persisted ticket state is
         published atomically.
  draft_hr_email(to_role, subject, context)
      -> {draft_text, note: "MOCK — not sent"}

ACTION BOUNDARY:
- ACTION MCP tools accept business parameters only.
- They MUST NOT accept confirmed, confirmation_id,
  conversation_id, pending_confirmation, preview, or equivalent
  agent/API/session state.
- MCP ACTION implementations do not determine whether user
  confirmation has occurred.
- The agent/API orchestration layer derives the confirmation
  requirement from discovered readOnlyHint=false metadata and
  completes the preview + confirmation-id gate before dispatch.

STATE-MUTATING ACTION SAFETY:
- Automated tests MUST use isolated disposable writable state;
  committed mock_data fixtures MUST remain unchanged.
- File-backed mutation MUST validate complete replacement state
  before publication.
- Publication MUST be atomic so a failed write does not expose
  partial authoritative state.
- Temporary publication artifacts SHOULD be removed after
  publication failure.

RULES: tools return plain JSON dicts; bad input raises a clean
error message; tools never read env vars directly.

## 6. RAG CONTRACT
- LIFECYCLE: the generated Chroma policy index is a deployment
  BUILD ARTIFACT. Production indexing uses the canonical
  process-separated lifecycle:
    1. install the exact pinned dependencies;
    2. run `python -m rag.index build`;
    3. allow the build process to exit;
    4. run `python -m rag.index publish` in a fresh process;
    5. verify the published policy collection contains exactly
       400 chunks;
    6. verify the published metadata is current for the corpus
       version, embedding model/dimension, target chunk size,
       and overlap configuration.
  The build phase uses the production embedding path, so model
  loading/download occurs while constructing the validated
  hidden index; no duplicate deployment-only model loader is
  required.
- PRODUCTION RUNTIME INDEX LOGIC:
    1. never parse, chunk, embed, build, or publish the policy
       index during application startup or request handling;
    2. load the published build artifact through the production
       Chroma store APIs;
    3. report the index as ready only when the expected policy
       collection is available, contains exactly 400 chunks,
       and `index_metadata.json` is current for the active
       corpus and frozen embedding/chunk configuration;
    4. otherwise report the index as degraded. A missing or
       stale production index requires a new explicit
       build/publish deployment cycle; production runtime does
       not rebuild it.
- LOCAL DEVELOPMENT: after a corpus version or frozen index
  configuration change, explicitly run
  `python -m rag.index build` followed by
  `python -m rag.index publish`. Automatic startup rebuilding,
  an `index:"building"` runtime state, and `/chat` warming-up
  behavior are not part of V1.
  WHY: S8 verified the process-separated hidden-build and safe
  publication lifecycle. Preparing and validating the index
  before runtime gives a simpler invariant, prevents free-tier
  cold starts from performing parse/embed work, and causes
  stale generated state to be repaired through an explicit
  reproducible build rather than runtime mutation.
- chunks.json DETERMINISM (canonical form):
    encoding:   UTF-8, LF newlines only (no CRLF)
    json:       json.dumps(chunks, sort_keys=True,
                ensure_ascii=False), single trailing newline
    content:    keys sorted; no timestamps; chunk text
                whitespace-normalized (collapse runs of
                whitespace); NO floats — scores are computed at
                query time and never stored here
  The CI determinism test compares this canonical form
  byte-for-byte. (Exact pins — §1 — pin the parser too, removing
  the main drift source; canonicalization removes the rest.)
- chunks.json roles: (a) human-reviewable pipeline output,
  (b) debugging aid, (c) CI determinism test.
- Parse markdown + PDF (pypdf); strip boilerplate; keep headings
- Chunk: heading-aware; long sections split to a TARGET of ~350
  tokens, HARD MAX 450 tokens, 10–15% overlap; never merge two
  sections into one chunk; fully deterministic. (450 max stays
  inside the embedding model's 512-token context — see AD-09.)
- Metadata per chunk: {doc_id, title, section_path,
  source_format, snippet} — this metadata IS the citation (G2)
- Embed locally during ingestion AND at query time
- Retrieve: top-k=5 + optional doc-type filter; multi-doc
  questions get a query-rewrite step (split into sub-questions,
  retrieve for each)
- Prompt instructs: cite [doc_id §section] after each claim;
  missing evidence → say so + suggest escalation
- Guardrail: low similarity or off-topic → polite refusal +
  "contact HR" (never improvise policy)

## 7. AGENT CONTRACT
- STARTUP: connect MCP client → list_tools() → convert schemas
  → inject into planning context (§5). If discovery fails, start
  in degraded mode: tool answers disabled, /health reports
  mcp:"degraded" — never pretend tools exist.
- TIMEOUTS: MCP tool call = 10s max; LLM call = 30s max. A
  timeout is a failure mode, handled like "MCP down" (below) —
  caught, logged to trace, never a hang or stack trace.
- Loop (max 6 iterations): read message → LLM decides: answer |
  call tool | ask clarifying question → execute via MCP client →
  log to trace → repeat
- MAX-ITERATION EXHAUSTION: if the loop hits iteration 6 without
  concluding — stop tool execution, synthesize the best answer
  from evidence gathered so far, state plainly in the answer
  that the task could not be fully completed, suggest
  escalation/rephrasing, and record termination:
  "max_iterations" in the trace. Never silently truncate.
- Intent shortcuts allowed (e.g. "PTO" → check_pto_balance
  first), but the LLM makes final tool choices from the
  discovered set
- Trace item: {step, tool, arguments, result_summary, sources[],
  decision} — returned by /chat and rendered in the UI (G5)
- PROMPT VERSIONING: agent/prompts.py defines PROMPT_VERSION
  (e.g. "1.0"). Every trace and every eval result row records
  it. Any prompt edit bumps the version and adds a decision-log
  entry. Without this you cannot explain eval-score changes.
- Failure modes (each covered by a test; pass rate is reported
  as failure-recovery accuracy, §11):
  unknown employee   → "I couldn't find that employee ID"
  MCP down/timeout   → catch; answer from policy only; flag
                       degraded mode in trace + /health
  no retrieval hits  → refuse + suggest escalation
  ambiguous request  → ask ONE clarifying question
- Safety: ACTION-class tools (readOnlyHint=false) show a preview
  and wait for explicit user confirmation before executing (G6)
- Sensitive topics (harassment, discrimination, medical) →
  retrieve conduct policy + ALWAYS escalate; never adjudicate

## 8. API CONTRACT
POST /chat
  req:  {message: str,
         conversation_id?: str,    # server generates if absent
         employee_id?: str,
         confirmed?: bool,
         confirmation_id?: str}    # required when confirmed=true
  res:  {conversation_id: str,
         answer: str,
         citations: [{doc_id, title, section, snippet}],
         trace: [trace item],
         pending_confirmation?: {confirmation_id, tool,
                                 arguments, preview}}
  SESSION STATE: server keeps an in-memory store keyed by
  conversation_id: {history, pending_confirmation}. When the
  agent proposes an ACTION-class tool, /chat returns
  pending_confirmation with a server-generated confirmation_id
  and stores it; the UI shows the preview; the user's "confirm"
  sends {conversation_id, confirmed: true, confirmation_id};
  the agent executes ONLY if the id matches the stored pending
  action — a confirm can no longer fire detached from its
  preview. No crypto, no persistence.
  KNOWN LIMITATION (document as future work): the store is
  in-memory, so a server restart drops pending actions; the
  user simply re-asks. Acceptable in v1 — all actions are mock;
  ticket creation preserves existing ticket records.
GET /health
  res:  {status: "ok", mcp: "connected"|"degraded",
         index: "ready"|"building", index_chunks: int,
         corpus_version: str, llm: "ok"|"degraded"}
UI: chat pane + citation chips + collapsible trace pane +
one-click buttons for WF1/WF2 (§10) + confirm dialog.

## 9. ENV VARS (.env.example documents every one)
LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, CHROMA_DIR, CORPUS_DIR

Mock-data MCP tools resolve structured V1 data repository-
relatively from PROJECT_ROOT / "mock_data"; mock-data paths are
not runtime environment configuration.

(Transport is fixed stdio — no MCP_TRANSPORT / MCP_SERVER_URL
in v1; the HTTP split is future-work documentation only.)

## 10. DEMO WORKFLOW CONTRACT (fixes WHAT the demo proves)
Two frozen workflows. The UI's one-click buttons send exactly
these inputs; the eval harness includes both as tasks; the
video narrates these exact tool sequences (G4, G10).

WF1 — REMOTE WORK ELIGIBILITY
  Input: "I'm employee E003. Can I work remotely from overseas
         for six weeks?"
  Expected MCP call sequence:
    1. lookup_employee_profile(employee_id="E003")
    2. search_policy_documents(query=<remote work incl.
       location/approval rules + data security>, k=5)
    3. check_policy_compliance(topic="remote_work_international",
       employee_id="E003")
    4. (optional) draft_hr_email(to_role="manager", ...)
       — ACTION class → preview → user confirms → executes
  Expected output: grounded conditional answer citing the
  remote-work policy (duration limit, approval chain, location
  rules) + data-security policy (device/VPN requirements), with
  cited next steps. Retrieval spans exactly 2 docs (§4 note).
  Presenter narrates: each tool name, its arguments, what it
  returned, which citations were retrieved, and the final
  answer's basis.

WF2 — PTO REQUEST GUIDANCE
  Input: "I'm employee E001. Can I take 3 days of PTO next
         week?"
  Expected MCP call sequence:
    1. lookup_employee_profile(employee_id="E001")
    2. check_pto_balance(employee_id="E001")
    3. search_policy_documents(query=<PTO policy + approval
       requirements>, k=5)
    4. create_mock_hr_ticket(employee_id="E001", category="PTO",
       summary=...) OR draft_hr_email(...)
       — ACTION class → preview → user confirms → executes
  Expected output: balance check result, cited PTO policy,
  manager-approval requirement, and the mock action completed
  AFTER confirmation.
  Presenter narrates: balance lookup, policy citations, the
  confirmation gate, and the mock action's result.

Together WF1+WF2 exercise 6 MCP tools (≥5 required, G3) and
both safety gates (G6). Demo employee IDs exist in mock_data
and are chosen so results are deterministic.

## 11. EVALUATION SPEC
- evaluation/eval_set.jsonl: 24 items, 5 categories —
  simple policy Q (8), multi-doc Q (5), tool-requiring task (6,
  incl. WF1+WF2), ambiguous → clarification (3),
  out-of-scope → refusal (2);
  each with gold answer + gold tool sequence + gold doc_ids
- RUN METADATA: every evaluation/results/ file records
  {generation_model, judge_model, judge_prompt_version,
  llm_base_url, PROMPT_VERSION, corpus_version, embedding_model,
  k, timestamp} — results without configuration are not
  reproducible (L5)
- JUDGE INDEPENDENCE: where possible the judge model differs
  from the generation model (different family or provider).
  If the same free-tier model must judge its own output, that
  self-judge bias is documented as a limitation in
  design-and-evaluation.md — never hidden.
- ONE HARNESS RUN produces all metrics — the cost is the 24
  queries, not the scoring. Metrics split:

  REQUIRED (each named in the rubric, G8):
    retrieval recall@k (fraction of gold doc_ids in top-k;
      PRIMARY metric for the ablation)
    groundedness + citation accuracy (LLM-judge, temp 0)
    tool-selection accuracy (vs gold tool sequences)
    workflow completion rate
    escalation/clarification accuracy (rule-based on the 5
      ambiguous/refusal items)
    action-safety pass rate (no ACTION tool fires unconfirmed)
    failure-recovery accuracy (scripted §7 failure modes)
    latency p50/p95, cold vs warm separately
    ablation: k ∈ {3, 5, 8} → recall@k + groundedness

  OPTIONAL (first to cut under time pressure; not G-items):
    human usefulness review (1–5 per item, mean + bias note)
    exact/partial match on short gold answers
    LLM-judged context relevance
- Determinism: temperature 0, fixed seed; results committed to
  evaluation/results/ (real numbers, not placeholders)

## 12. DEFINITION OF DONE + MINIMUM TEST SET

ACCEPTANCE EVIDENCE:
Verify a capability at the architectural boundary being claimed:
  direct Python call       → implementation behaviour
  FastMCP list_tools()     → discovery/schema/annotations
  MCP stdio call_tool()    → protocol invocation
  HTTP/TestClient          → application API behaviour
  deployed URL             → deployment behaviour

Evidence from a lower boundary MUST NOT substitute for a
higher-boundary rubric claim.

"pytest green" means AT LEAST these tests exist and pass:
  test_chunk.py     same corpus version → byte-identical
                    canonical chunks.json (determinism)
  test_retrieve.py  5 known questions → expected doc_id in top-5
  test_mcp.py       list_tools() returns all 8 tool names;
                    unknown employee_id → clean error, not crash
  test_app.py       /health returns status ok; /chat returns
                    citations for a known question; an ACTION
                    tool without confirmed=true + matching
                    confirmation_id does NOT execute

S1 repo+env: fresh clone → venv → pip install → pytest green;
   all four governance files at root
S2 corpus: 12+ docs in corpus/source/, 2 formats, all 10 topics,
   numbers consistent, version.json with full metadata committed
S3 mock data: 10+ employees incl. part-time, contractor, new
   hire; WF1/WF2 demo employees (E001, E003) present
S4 RAG: ingest → canonical chunks.json committed; CLI query
   returns cited chunks; retrieval spot-check passes; no chunk
   exceeds 450 tokens
S5 MCP:
   INCREMENTAL — implement, register, and verify one capability
   before advancing; schema matches §5; READ/CALCULATION tools
   expose readOnlyHint=true; ACTION tools expose
   readOnlyHint=false; bad input produces a clean
   protocol-visible error; mutation tests use isolated
   disposable fixture state; protocol claims are verified
   through MCP rather than inferred only from Python calls.

   FINAL — SDK annotation checkpoint evidence exists;
   list_tools() returns exactly all 8 frozen tool names with
   correct schemas/annotations; every tool is callable through
   MCP; stdio discovery/invocation evidence exists; committed
   fixtures remain unchanged; full regression is green.
S6 agent CLI: WF1 and WF2 run end-to-end (§10) with readable
   trace; tools came from discovery, not a hardcoded list;
   max-iteration exhaustion behaves per §7
S7 web: /chat + /health OK locally; UI shows answer, citations,
   trace, confirmation flow; WF1/WF2 buttons work
S8 deploy+CI: Render build uses the canonical
   `python -m rag.index build` → fresh-process
   `python -m rag.index publish` lifecycle and verifies a
   current 400-chunk published index; push → CI green
   (incl. chunks determinism + MCP discovery tests) → deploy;
   production runtime performs no index rebuild; hosted health,
   WF1/WF2, confirmation-gated ACTION behavior, and measured
   cold-start timing are recorded in deployed.md
S9 eval: 24 gold Qs; all REQUIRED metrics + run metadata +
   ablation written into design-and-evaluation.md
S10 video+submit: SUBMISSION_CHECKLIST.md fully checked
RULE: never start step N+1 with step N red. Fix or simplify —
never skip.

## 13. IMPLEMENTATION PRIORITY (when time runs short)
Priority 1 — G1–G7: the working deployed system (RAG + agent +
  MCP + web + CI). This is the product; without it nothing else
  is gradeable.
Priority 2 — G8 REQUIRED evaluation metrics + ablation (§11).
Priority 3 — OPTIONAL metrics, human review, UI polish.
NEVER sacrifice at any pressure level: the MCP tool path (G3),
citations (G2), the minimum test set (G7), and the confirmation
gate (G6). A smaller correct system always outscores a larger
unverifiable one.

## 14. DO-NOT-BUILD LIST (concrete; re-read when tempted)
- No login/accounts, no real email sending, no real database
- No streaming tokens, no multi-language UI, no voice
- No reranker, no hybrid BM25, no LangGraph, no multi-agent
- No HTTP MCP transport, no second MCP server (document both as
  future work; build neither)
- No persistent session store (in-memory is the v1 contract)
- No confirmation-token crypto (id match only — §8)
- No auxiliary tooling that serves no G-item (e.g. a status
  dashboard generator — PROJECT_STATUS.md is hand-edited)
- No separate ADR files (decision log lives in
  design-and-evaluation.md — §15)
- Max 15 corpus docs / 12 employees — eval consistency > volume
- If it doesn't serve a G-item, it does not exist

## 15. DOCUMENTATION & DECISION LOG
- The four required docs stay at repo ROOT — the brief lists
  them by name and graders will look there.
- README.md MUST contain, in this order:
    1. Problem statement        2. Architecture diagram
    3. Setup instructions       4. Environment variables
    5. Running locally          6. Running tests
    7. Deployed URL + cold-start note   8. Known limitations
- Record architecture decisions and spec amendments as AD-style
  entries (context → decision → consequence) in a table inside
  design-and-evaluation.md. Seed entries:
  AD-01  stdio-only MCP transport (HTTP = future work)
  AD-02  committed chunks.json + version-triggered full
         re-ingest (amends v2.1's partial rebuild)
  AD-03  in-memory session store keyed by conversation_id
  AD-04  evidence-based rules + inspect-before-change (rules v3)
  AD-05  process-separated build/publish in the deploy build
         step; production runtime validates the baked index and
         never rebuilds it
  AD-06  tool classification via MCP annotations
         (readOnlyHint) instead of a custom registry
  AD-07  S5 dependency checkpoint: verify pinned mcp SDK
         supports tool annotations before building tools
  AD-08  chunks.json canonical serialization (UTF-8, LF, sorted
         keys, no timestamps/floats, normalized whitespace)
  AD-09  embedding model = bge-small-en-v1.5 (512-token ctx)
         because MiniLM's 256-token truncation would silently
         drop policy-section tails; chunks sized 350/450 to fit
  AD-10  confirmation_id binds each confirm to its preview
         (id match only; restart-loss remains a documented
         limitation)
