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
