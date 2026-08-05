---
doc_id: "HR-POL-013"
title: "HR Requests and Case Management Procedure"
document_type: "policy"
version: "1.2"
effective_date: "2026-01-01"
owner: "People and Culture"
status: "active"
applies_to:
  - "full_time"
  - "part_time"
  - "contractor"
  - "probation"
keywords:
  - "HR request"
  - "HR ticket"
  - "case management"
  - "escalation"
  - "mock action"
---

# HR Requests and Case Management Procedure

**Company:** Promote Health Analytics Pty Ltd

## 1. Purpose

This procedure defines how Promote Health Analytics Pty Ltd receives, records, routes, escalates, and closes HR requests and cases. It also defines the safe use of mock ticket and draft-message actions in the agentic demonstration.

## 2. Scope

This procedure applies to ordinary HR questions, policy exceptions, benefits queries, leave support, remote-work reviews, conduct concerns, and mock HR actions.

## 3. Definitions

### 3.1 Request

A question or service need that can normally be resolved through policy guidance or an administrative action.

### 3.2 Case

A matter requiring documented review, judgement, coordination, sensitivity, or escalation.

### 3.3 Mock action

A demonstration action that creates a mock ticket or draft text but does not perform a real external action.

### 3.4 Explicit confirmation

A user response that confirms the specific previewed mock action.

## 4. Policy Requirements

### 4.1 Minimum request information

A request should include the employee ID where relevant, request category, summary, relevant dates, and desired outcome.

The system must not invent a missing employee ID or silently substitute another employee.

### 4.2 Policy evidence

Policy guidance must identify the authoritative document and section supporting each material claim.

Where policy evidence is absent or insufficient, the response must state the limitation and recommend People and Culture review.

### 4.3 Sensitive cases

Harassment, discrimination, retaliation, medical, safety, and other sensitive matters must be routed for human review.

The system must not determine credibility, legal liability, diagnosis, or disciplinary outcome.

### 4.4 Data minimisation

A request should contain only the information necessary to provide guidance or route the matter.

Sensitive details should not be repeated unnecessarily in summaries or traces.

## 5. Procedures or Application

### 5.1 Request intake

Identify whether the request needs policy retrieval, employee data, a calculation, clarification, escalation, or a mock action.

If essential information is ambiguous, ask one concise clarifying question.

### 5.2 Policy and data checks

Retrieve authoritative policy evidence and, where relevant, retrieve employee-specific mock data through MCP tools.

The agent must not directly access the RAG index or mock files outside the MCP tool path.

### 5.3 Preview of a mock action

Before creating a mock ticket or draft message, present the tool name, arguments, and human-readable preview.

The preview must clearly state that the action is mock and does not send a real email or create a real external case.

### 5.4 Confirmation for mock actions

Mock HR actions require an explicit user confirmation before execution.

Confirmation must match the pending action and its confirmation identifier.

A missing, stale, or mismatched confirmation must not execute the action.

### 5.5 Closure and trace

The final response should state the policy basis, employee-specific facts used, action result, and any remaining next step.

The operational trace records tool names, arguments, result summaries, sources, and decisions without exposing hidden chain-of-thought.

### 5.6 Case classification and priority

Requests should be classified by subject, urgency, sensitivity, and assigned responsible team. A routine balance question differs from a policy exception, a security incident, or a sensitive workplace concern.

Urgency should be based on a specific deadline, safety risk, payroll or benefits impact, access failure, or operational consequence rather than the requester's preference alone.

### 5.7 Information quality

The case summary should distinguish confirmed facts, requester statements, retrieved records, policy evidence, and unresolved questions. It must not present an allegation or assumption as a verified fact.

Where a correction is made, the updated record should preserve enough context to understand what changed and why.

### 5.8 Handover and closure

A case transferred to another owner should include the category, relevant policy references, actions completed, outstanding questions, and requested next step. Sensitive details should be minimised.

A request may be closed when the answer or mock action is complete, the matter has been handed to the responsible human owner, or the requester does not provide required information after an appropriate follow-up.

## 6. Exceptions and Escalation

- A request may be answered without employee data where the user asks only for general policy information.
- A sensitive issue may be escalated without a complete narrative where requesting more detail would be unnecessary or intrusive.
- A mock action remains optional and must not be executed merely because the user asked the underlying policy question.

## 7. Responsibilities

### 7.1 Requesters

Provide accurate information and confirm only actions they understand.

### 7.2 People and Culture

Review escalated or exceptional matters.

### 7.3 Agent and MCP tools

Use discovered tools, preserve the confirmation gate, and produce grounded operational traces.

## 8. Decision Rules

- If an employee ID is required and missing, ask one clarifying question.
- If the employee ID is unknown, return a clean not-found message.
- If policy evidence is missing, do not invent a policy answer.
- If a tool is classified as an action, preview it and wait for a matching explicit confirmation.
- If a matter is sensitive, escalate to People and Culture without adjudicating it.

## 9. Examples

### 9.1 PTO ticket preview

The agent checks E001's profile and balance, retrieves HR-POL-002, and presents a mock ticket preview. The ticket is created only after matching confirmation.

### 9.2 Remote-work draft

The agent reviews E003, retrieves HR-POL-004 and HR-POL-005, explains that six weeks exceeds the standard limit, and may preview a draft manager email.

### 9.3 Unknown employee

A lookup for E999 returns a clean not-found result rather than an exception trace.

## 10. Frequently Asked Questions

### 10.1 Does asking for a ticket create one immediately?

No. A preview and explicit matching confirmation are required.

### 10.2 Can a real email be sent?

No. The demonstration tool produces mock draft text only.

### 10.3 What happens if policy evidence is missing?

The system states the limitation and recommends People and Culture review.

### 10.4 Will the trace show private reasoning?

No. It shows operational facts such as tools, arguments, summaries, sources, and decisions.

## 11. Related Documents

- **HR-POL-002 Paid Time Off Policy** - PTO guidance and mock request actions.
- **HR-POL-004 Remote and Flexible Work Policy** - remote-work review and approvals.
- **HR-POL-011 Workplace Conduct Policy** - sensitive-case escalation.

## 12. Version History

| Version | Effective date | Summary |
|---|---|---|
| 1.2 | 2026-08-05 | Final S2 release: synchronized metadata, standardized source schemas, and multi-extractor PDF validation. |
| 1.1 | 2026-08-05 | Clean S2 release: corrected company name and strengthened corpus/PDF quality controls. |
| 1.0 | 2026-01-01 | Initial synthetic policy for the Agentic HR Policy Assistant project. |

> Synthetic policy notice: This document was created for an educational agentic AI project. It is not legal advice and does not reproduce a real employer policy.
