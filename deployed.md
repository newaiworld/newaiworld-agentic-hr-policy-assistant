# Deployment

## Status

Deployed and hosted on Google Cloud Run.

The final S10 submission/demo release is running from published source
`aef0ddab770e17a7750971f365dd54f204517930`.

## Application URL

https://agentic-hr-policy-assistant-ykwvm3nhfq-ts.a.run.app

## Release Identity

- Source commit: `aef0ddab770e17a7750971f365dd54f204517930`
- GitHub Actions CI run: `33494287703` — success
- Cloud Run revision: `agentic-hr-policy-assistant-00020-kuv`
- Immutable image digest: `sha256:3802b5da57611fb007e9f9061c12a3adb2a7b8463925c51141cd107b42185763`
- Deployment region: `australia-southeast1`
- Deployment platform: `linux/amd64`
- Production traffic: `100%` on the validated revision

The Cloud Run deployment is bound to the immutable image digest rather
than relying on a mutable image tag.

## Health Endpoint

`https://agentic-hr-policy-assistant-ykwvm3nhfq-ts.a.run.app/health`

Hosted acceptance returned HTTP 200 with:

```json
{
  "status": "ok",
  "mcp": "connected",
  "index": "ready",
  "index_chunks": 400,
  "corpus_version": "1.2",
  "llm": "ok"
}
```

A repeated warm-runtime health request also returned HTTP 200.

## Hosted Workflow Acceptance

The deployed release passed both frozen demonstration workflows.

### WF1 — Remote Work Eligibility

The hosted WF1 remote-work workflow passed its acceptance gate using the
deployed agent, MCP runtime, policy retrieval, employee context, policy
compliance checking, and grounded citations.

### WF2 — PTO Request Guidance

The hosted WF2 workflow passed its action-proposal and confirmation
acceptance gates.

The confirmation-gated `draft_hr_email` action:

- executed only after the matching confirmation;
- executed exactly once;
- used the arguments bound to the pending confirmation snapshot;
- cleared the pending confirmation after execution; and
- rejected replay of the same confirmation with HTTP 409.

The generated HR email remained a mock artifact and was not sent.

## Runtime Acceptance

Final S10 production acceptance confirmed:

- Cloud Run Ready condition: PASS
- traffic to qualified revision: 100%
- MCP runtime: connected
- policy index: ready
- policy index chunks: 400
- corpus version: 1.2
- LLM runtime: ok
- hosted root endpoint: HTTP 200
- hosted health endpoint: HTTP 200
- WF1 production acceptance: PASS
- WF1 required compliance evidence: PASS
- WF1 grounded citations `HR-POL-004 §4.4` and `HR-POL-004 §5.3`: PASS
- WF2 production pre-confirmation acceptance: PASS
- WF2 confirmation-bound action execution: PASS
- HR-POL-002 grounding retained after confirmation: PASS
- pending confirmation cleared after execution: PASS
- `MOCK — not sent` safeguard retained: PASS

## Cold-Start Evidence

A true request-driven scale-from-zero measurement was performed after an
idle period.

Cloud Run logs classified the new instance start as:

`AUTOSCALING`

The measurement showed:

- client-observed request latency: **10.032 seconds**
- instance start to server process: **6.149 seconds**
- instance start to application ready: **8.927 seconds**
- instance start to Uvicorn ready: **8.935 seconds**
- instance start to successful startup probe: **8.935 seconds**

The single cold-start request returned HTTP 200 and the complete healthy
runtime contract.

This measurement demonstrates that the production container can scale
from zero and become application-ready without rebuilding the policy
index or downloading the embedding model at runtime.

## Deployment Architecture Notes

The deployment image builds and publishes the Chroma policy index during
the container build lifecycle. The generated local development index is
excluded from the Docker build context.

The runtime container uses the packaged policy index and offline model
artifacts. Runtime startup therefore does not perform policy parsing,
embedding generation, or index publication.

The release image was built for `linux/amd64`. This replaced the
previous failed image whose Cloud Run startup terminated with an
`exec format error` before the application process started.

## Known Limitations

- The application uses an in-memory conversation/session store.
  Conversation state is therefore not durable across instance
  replacement or process restart.
- The deployment is intentionally a single-service demonstration
  architecture rather than a horizontally distributed production HR
  platform.
- HR action tools create mock artifacts only; they do not modify a real
  HR information system or send real email.
- Cold-start latency is materially higher than warm health-check latency
  because the service can scale to zero.
- Deployment success is only one part of the submission evidence; final
  qualification also includes S9 evaluation, CI, and S10 production
  workflow acceptance.

## S10 Final Submission Qualification

The final candidate was first deployed at zero production traffic, validated
through `/health`, WF1, WF2 pre-confirmation, and WF2 confirmation, and then
promoted unchanged to 100% production traffic.

Final qualification evidence:

- local agent regression: 131 passed;
- complete local repository regression: 1435 passed;
- dependency consistency: PASS;
- GitHub Actions CI run `33494287703`: success;
- immutable candidate image verified by digest;
- candidate WF1: PASS;
- candidate WF2 confirmation workflow: PASS;
- production WF1: PASS;
- production WF2 confirmation workflow: PASS.

No application, RAG, MCP, corpus, dependency, or deployment-architecture
changes were made after the validated candidate was promoted.
