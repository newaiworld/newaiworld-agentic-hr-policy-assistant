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
