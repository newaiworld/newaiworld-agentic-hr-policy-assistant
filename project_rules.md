# ============================================================
# PROJECT_RULES.md — Operating Constitution (v3.1)
# Project: Agentic HR Policy Assistant (solo build, one week)
# Companions: IMPLEMENTATION_SPEC.md (what we build),
#             PROJECT_STATUS.md (where we are now),
#             SUBMISSION_CHECKLIST.md (submission admin)
#
# PRECEDENCE: this file governs HOW we work and WHY.
# The spec governs WHAT we build. If they conflict, this file
# wins. If a rule blocks progress, stop and ask the human —
# never silently override.
# ============================================================

## 0. DECISION HIERARCHY
When priorities conflict, decide in this order:
1. Safety, integrity, correctness — never compromised.
2. Grading non-negotiables (§3) — they define "done"; in this
   project they ARE the learning path.
3. Depth of understanding — never accept output you cannot
   explain (§2 test).
4. Simplicity and reliability — boring-and-working beats clever.
5. Implementation convenience — considered last, sacrificed first.
Never optimize for speed if it reduces correctness, evaluation
quality, understanding, or reliability.
(2 vs 3 rarely conflict. If they do: deliver the minimum
implementation that satisfies the G-item, then explain and
document the concept BEFORE starting the next task.
"Understand it later" is not permission to accumulate
unexplained code.)

## 1. MISSION
Build and deploy, in one week, a working HR assistant that:
(a) answers company-policy questions with citations (RAG), and
(b) completes two multi-step HR workflows by calling MCP tools
over mock employee data.
The project is graded against a rubric AND serves as proof of
learning. Working + understood beats impressive + fragile.

## 2. LEARNING OBJECTIVES (why this project exists)
L1. RAG — parsing, chunking, embedding, retrieval, citation,
    guardrails.
L2. AGENTS — the loop: intent → plan → tool selection →
    execution → synthesis, with an operational trace.
L3. MCP — server, tool schemas, client discovery, transport
    choice, error handling.
L4. EVALUATION — gold sets; answer-quality, retrieval,
    agent-behavior, and latency metrics; one ablation.
L5. GOVERNANCE — secrets hygiene, green CI, accurate docs,
    reproducibility by a stranger.
L6. DEPLOYMENT — reproducible environments, configuration
    management, CI/CD, and production deployment on a
    free-tier host.
TEST: for every file you can answer "what does this do and why
is it here?" If not, pause and learn it before moving on.
Unexplained code is a rule violation.

## 3. GRADING NON-NEGOTIABLES (the rubric, as outcomes)
G1.  A public free-tier URL runs the full system.
G2.  Policy answers are grounded and cited (doc, section, snippet).
G3.  The agent calls MCP tools THROUGH the MCP layer — never via
     direct function calls. Every tool in the spec is implemented;
     the demo exercises at least five of them live.
G4.  Two multi-step workflows complete end-to-end in the demo.
G5.  An operational trace shows tool names, arguments, outputs,
     and sources — never hidden chain-of-thought.
G6.  Write-style actions are mock and confirmation-gated.
G7.  CI runs tests (including an MCP tool test) and deploys only
     on green.
G8.  Evaluation covers answer quality, retrieval quality, agent
     behavior, latency, and one ablation, against a gold set.
G9.  Docs exist and are accurate: README, design-and-evaluation,
     ai-tooling, deployed, plus evaluation/, mock_data/, mcp/.
G10. A 7–10 minute demo video shows both workflows live on the
     deployed URL, narrating every tool call, and meets every
     item in SUBMISSION_CHECKLIST.md.

## 4. MUST / SHOULD / AVOID
MUST — a violation means the project is broken:
- MUST keep secrets in environment variables; never commit keys.
- MUST keep main always-runnable; tests green before advancing.
- MUST cite every policy claim; refuse what the corpus cannot
  support — never invent policy.
- MUST log every AI-tooling session in ai-tooling.md the same day.
- MUST be able to explain every line you submit.
- MUST distinguish implemented-and-verified features from
  written-but-untested code, in every doc and every claim.
- MUST support technical claims with evidence: tests, logs,
  metrics, screenshots, or the deployed URL.
SHOULD — deviation needs a written note in design-and-evaluation.md:
- SHOULD build one step at a time, in the S1→S10 order.
- SHOULD prefer the boring option when two approaches both work.
- SHOULD add a test when you add a capability.
- SHOULD ask one clarifying question when a request is ambiguous
  (applies to you AND to the agent you are building).
AVOID — forbidden without explicit human approval + documentation:
- AVOID frameworks, databases, or services not in the spec.
- AVOID features the rubric does not grade (auth, streaming,
  multi-user, elaborate UI).
- AVOID letting an AI assistant refactor outside the current step.
- AVOID "temporary" shortcuts that bypass MCP, citations, or tests.

## 5. AI ASSISTANT BEHAVIOR CONTRACT
(Pasted into ChatGPT / Claude Code / Cursor / Copilot. When you
are that assistant, you MUST:)
A1. Follow this constitution and IMPLEMENTATION_SPEC.md exactly.
A2. Work on ONE step at a time. Before any implementation
    recommendation or code change, state the current phase (read
    PROJECT_STATUS.md) and the G-item the work serves. If a
    request is outside the current phase and serves no G-item,
    challenge it: name the gap and ask before proceeding.
    (Example: "add authentication" → outside S1–S10, serves no
    G-item → decline and explain.)
A3. Before generating code, say what you will build and why in
    2–3 sentences; afterwards, explain anything non-obvious.
A4. Generate the smallest change that satisfies the current
    step's Definition of Done — never "the whole app".
A5. Never invent dependencies, files, or tools outside the spec.
    If the spec genuinely lacks something, propose a spec
    amendment; do not act unilaterally.
A6. Never expose or fabricate chain-of-thought in the product;
    the trace contains operational facts only.
A7. If instructions conflict, surface the conflict and ask —
    do not guess.
A8. Teach progressively: explain unfamiliar concepts the first
    time they are used, maintain professional engineering
    terminology, and explain trade-offs when a decision matters.
    Calibrate to a trainee AI engineer — never oversimplify at
    the cost of accuracy.
A9. Never silently change architecture, folder structure, APIs,
    tool contracts, or evaluation methodology. If a change seems
    needed, explain the impact and propose a spec amendment.
    Example violation: "fixing" an MCP issue by replacing MCP
    calls with direct Python calls — it works, but fails G3.
A10. Inspect before you change: read the existing files and
     understand the current state before generating edits.
     Never assume an implementation is missing — duplicate
     architectures are a rule violation.
A11. At the beginning of each AI session, inspect
     PROJECT_RULES.md, IMPLEMENTATION_SPEC.md, and
     PROJECT_STATUS.md. Do not make implementation decisions
     from memory alone — the files are the project state,
     not your recollection of it.

## 6. CHANGE GOVERNANCE
- This constitution is STABLE: changes only by deliberate human
  decision, committed with prefix "rules:".
- IMPLEMENTATION_SPEC.md is AMENDABLE: record old → new + reason
  in the decision log (design-and-evaluation.md), update the
  spec in the same commit ("spec:").
- PROJECT_STATUS.md is STATE: updated freely at every session
  close ("status:" commits); it records, it never governs.
- SUBMISSION_CHECKLIST.md is ADMINISTRATIVE: changes only if the
  course brief changes.
- Code commits SHOULD use: feat:, fix:, test:, docs:, refactor:.
- No undocumented drift: if code and spec disagree, one is wrong
  — fix before continuing.

## 7. WORKING AGREEMENT (solo cadence)
- One step at a time; a step ends at its Definition of Done
  (spec §12) — green, or not done.
- Timebox: stuck > 90 minutes → simplify or ask for help.
  Never silently skip a requirement.
- Daily close: commit, push, update ai-tooling.md, and update
  PROJECT_STATUS.md (phase, blockers, tomorrow's first task).
- The rubric is the backlog: before building anything, name the
  G-item it serves. If none — do not build it.
