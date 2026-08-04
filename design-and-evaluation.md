# Design and Evaluation

## Project Status

The project is currently in S1 — Foundation. Architecture and evaluation results will be added and verified progressively.

## Architecture Decision Log

| ID | Context | Decision | Consequence |
|---|---|---|---|
| AD-01 | The project must remain free-tier compatible and simple to deploy. | Use stdio-only MCP transport in v1. | The MCP server runs as a subprocess within the single deployed service. |
