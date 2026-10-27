# Dataset Doctor: Agentic Workflows & Collaboration Protocol

## 1. Overview

Dataset Doctor is built to support both human developers and autonomous AI coding agents (such as Antigravity, Claude, and Cursor). This document establishes the working contract, boundaries, and operational guidelines for AI agents interacting with the platform.

---

## 2. Agent Operational Roles

### 2.1. The Data Quality Copilot
- **Function**: Operates via MCP or CLI to inspect uploaded datasets, surface silent failure risks (e.g. pre-split balancing, hidden sentinels, ambiguous dates), and explain trade-offs to the user.
- **Contract**: Never mutates data silently. The copilot formulates hypotheses, presents measured cross-validation evidence, and requires explicit user approval before applying destructive changes.

### 2.2. The Autonomous QA & Benchmark Agent
- **Function**: Executes regression suites, planted anomaly verification, and benchmark gates (`python -m doctor.bench --all`).
- **Contract**: An agent must never loosen acceptance thresholds to achieve a passing gate. If a gate fails, the agent must trace the root cause, fix the detection/remediation logic, and re-run all earlier gates to prevent regressions.

---

## 3. Engineering Guidelines for AI Coding Agents

When pair-programming or extending the Dataset Doctor codebase, agents must strictly follow these rules:

1. **Preserve What Works**:
   - Do not replace the existing codebase with greenfield templates or alternative architectural patterns.
   - Respect the four-part module decomposition: `detect.py`, `advise.py`, `fix.py`, `explain.py`.
2. **Never Weaken Core Guarantees**:
   - `split.py` is the single source of truth for train/test splitting.
   - The test set is completely immutable—never drop test rows, never resample test labels, and never learn transformation parameters from test data.
   - Keep resampling (SMOTE, oversampling) inside the training pipeline only.
3. **Evidence-Based Validation**:
   - Never claim a test or benchmark passed without executing the command and verifying the terminal output.
   - When modifying issue modules, always run:
     ```bash
     pytest tests/
     python -m doctor.bench --all
     ```
4. **Transparent Audit Logging**:
   - Record gate passes, benchmark metrics, and design decisions in `docs/gate_log.md` and `docs/notes.md`.

---

## 4. LLM Explanation Engine Integration

Dataset Doctor includes an optional plain-English explanation layer:
- **Default State**: Off. All native explanations are deterministic string templates grounded in calculated metrics (`explain.py`).
- **Optional LLM Enhancer**: When enabled with a user-provided API key, the explanation engine receives a structured JSON context extracted directly from `Issue.evidence` and `Recommendation.measured`.
- **Strict Anti-Hallucination Prompting**: Prompts mandate that the LLM cite exact numeric metrics from the context. Any statement unsupported by calculated evidence is rejected.
