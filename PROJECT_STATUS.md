# PROJECT_STATUS.md — Project Command Centre

Project: Agentic HR Policy Assistant  
Company: Promote Health Analytics Pty Ltd  
Current phase: S4 — RAG Pipeline  
Overall completion: ████░░░░░░ 40%  
Last updated: 2026-08-05  

## Phase Progress

- S1 Foundation — complete
- S2 Policy Corpus — complete
- S3 Mock Data — complete
- S4 RAG — next
- S5 MCP — not started
- S6 Agent — not started
- S7 Web — not started
- S8 Deployment and CI — not started
- S9 Evaluation — not started
- S10 Demo and submission — not started

## S3 Completion Summary

### Files created

- `mock_data/employees.json`
- `mock_data/pto.json`
- `mock_data/benefits.json`
- `mock_data/tickets.json`

### Population

- 12 synthetic employees
- 2 part-time employees
- 1 contractor
- 1 probationary employee
- 1 employee on leave
- 4 work locations
- valid manager hierarchy

### Frozen workflow readiness

#### WF1 — International remote work

- E003 exists
- E003 is full-time and active
- E003 location is `SYDNEY_HQ`
- E003 manager is E010
- ready for policy-compliance evaluation against HR-POL-004 and HR-POL-005

#### WF2 — PTO request

- E001 exists
- E001 is full-time and active
- E001 has 8.0 available PTO days
- E001 manager is E010
- no existing E001 PTO ticket
- next mock ticket ID is `TKT-1005`

### Edge cases included

- E002: part-time, 0.6 FTE, accrual 1.0 day/month
- E005: probation, start date 2026-07-15, benefits pending
- E006: contractor, PTO-ineligible, benefits-ineligible
- E007: low PTO balance of 1.5 days
- E008: part-time, 0.4 FTE, accrual 0.6667 days/month
- E009: employment status `leave` with retained PTO balance

### Validation completed

- JSON syntax: pass
- record-count validation: pass
- referential integrity: pass
- manager-reference validation: pass
- manager-cycle validation: pass
- policy consistency: pass
- benefits-date validation: pass
- part-time accrual validation: pass
- ticket sequencing: pass
- synthetic-data safety: pass
- legacy company-name check: pass

## Current Objective

Begin S4:

1. parse Markdown and PDF policy sources;
2. normalize text;
3. implement heading-aware chunking;
4. count tokens using `BAAI/bge-small-en-v1.5`;
5. generate deterministic `corpus/processed/chunks.json`;
6. embed chunks;
7. persist Chroma;
8. implement retrieval and citation tests.

## Current Risks

| Risk | Probability | Mitigation |
|---|---|---|
| Chunk boundaries split operative rules | medium | heading-aware chunking and exact tokenizer checks |
| PDF and Markdown parsing differ | medium | normalize both sources and compare metadata |
| Chroma rebuild is non-deterministic | medium | deterministic chunk IDs and version stamp |
| Retrieval misses workflow-critical rules | medium | add WF1/WF2 retrieval tests |
| Tool contracts drift from data schema | low | preserve frozen S3 fields and controlled vocabulary |

## Next Commit

`feat(mock-data): add synthetic HR datasets for S3 workflows`
