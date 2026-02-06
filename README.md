# Self-Improving Coding Agent

A production-grade autonomous coding agent that diagnoses failing tests, generates patches via LLM-driven reasoning, and iteratively refines its fixes until the test suite passes. The system is built on a stateful graph-based execution loop (LangGraph), augmented with retrieval-augmented generation (RAG), persistent episodic memory (Mem0), a knowledge graph (Neo4j), and full observability via OpenTelemetry.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [High-Level Data Flow](#high-level-data-flow)
  - [Execution Graph](#execution-graph)
  - [Component Breakdown](#component-breakdown)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Agent](#running-the-agent)
  - [Analysis-Only Mode](#analysis-only-mode)
  - [Repository Indexing](#repository-indexing)
  - [Docker Sandbox Execution](#docker-sandbox-execution)
  - [REST API Server](#rest-api-server)
  - [Evaluation Harness](#evaluation-harness)
  - [Health Check](#health-check)
  - [Run History](#run-history)
- [Patch Application Workflow](#patch-application-workflow)
- [Memory and Knowledge Graph](#memory-and-knowledge-graph)
  - [Mem0 Episodic Memory](#mem0-episodic-memory)
  - [Neo4j Graph Store](#neo4j-graph-store)
  - [Fix Cards](#fix-cards)
- [Observability](#observability)
- [Testing](#testing)
- [Security Considerations](#security-considerations)
- [License](#license)

---

## Overview

The Self-Improving Coding Agent automates the diagnosis and resolution of software defects. Given a goal (e.g., "Fix failing tests"), the agent executes a multi-step reasoning loop that:

1. **Reproduces** the failure by running the project's test suite.
2. **Plans** a repair strategy using a fast LLM.
3. **Retrieves** relevant code context through vector search (FAISS or pgvector) and lexical recall (ripgrep), combined with past fix memories from Mem0.
4. **Audits** the codebase for SQL injection risks, typos, and code quality issues using rule-based static analysis and LLM-powered review.
5. **Generates** a unified diff patch using a strong LLM, informed by test output, retrieved context, and safety check results.
6. **Applies** the patch to an isolated workspace with multi-strategy fallback (standard apply, recount, realignment, 3-way merge, fuzz matching).
7. **Validates** the patch through static analysis (Ruff, Bandit) and re-executes the test suite.
8. **Reflects** on the outcome using an LLM, producing structured lessons learned.
9. **Persists** the episode to Mem0 and Neo4j as a structured fix card for cross-repository recall.
10. **Decides** whether to iterate, halt on success, or terminate on reaching the maximum iteration count.

The agent supports an `--until-green` mode that continues iterating until the test suite passes, bounded by a configurable maximum iteration count.

---

## Architecture

### High-Level Data Flow

```
                          +------------------+
                          |   CLI / REST API  |
                          +--------+---------+
                                   |
                                   v
                     +-------------+-------------+
                     |     LangGraph State Graph  |
                     |  (Stateful Execution Loop) |
                     +---+-----+-----+-----+----+
                         |     |     |     |
              +----------+  +--+--+  |  +--+----------+
              |             |     |  |  |              |
              v             v     v  v  v              v
        +---------+   +-------+ +------+  +--------+ +----------+
        | LiteLLM |   |  RAG  | | Mem0 |  | Neo4j  | |  Tools   |
        | (2-tier)|   | FAISS/| |Memory|  | Graph  | | sandbox, |
        | fast +  |   |pgvec. | |Store |  | Store  | | ruff,    |
        | strong  |   +-------+ +------+  +--------+ | bandit,  |
        +---------+                                   | pytest   |
                                                      +----------+
                                   |
                          +--------+---------+
                          |   Observability   |
                          | OTLP / Prometheus |
                          |   Phoenix / CSV   |
                          +------------------+
```

### Execution Graph

The core agent loop is implemented as a LangGraph `StateGraph` with the following node topology. Each node receives the full graph state, performs its operation, records a step result, and signals the next node via a router.

**Full execution mode:**

```
reproduce --> plan --> retrieve --> audit --> patch --> apply_patch
                                                          |
                                          +-------+-------+
                                          |               |
                                       safety          reflect
                                          |               |
                                       execute      memory_update
                                          |               |
                                     retry_decide      decide
                                       /     \            |
                                  execute   reflect    router --> (next iteration or END)
```

**Analysis-only mode** (no patching or execution):

```
plan --> retrieve --> audit --> reflect --> memory_update --> decide
```

Every edge in the graph passes through a halt check, enabling pause/resume semantics via checkpoints. The `--stop-after` flag allows pausing after any named step, and `--resume-run-id` resumes from the last checkpoint.

### Component Breakdown

| Component | Location | Responsibility |
|---|---|---|
| **Agent Graph** | `agent/graph.py` | LangGraph state machine defining the full execution loop with 13 nodes and conditional routing. |
| **Agent State** | `agent/state.py` | Data models for `RunState`, `RunConfig`, `IterationRecord`, and `StepResult`. |
| **Checkpoints** | `agent/checkpoints.py` | JSON-based checkpoint store for pause/resume across runs. |
| **CLI** | `app/cli.py` | Argument parser and entry point for all commands (`run`, `index`, `eval`, `doctor`, `runs`). |
| **REST API** | `app/main.py` | FastAPI application with endpoints for runs and evaluations (sync and async). |
| **Doctor** | `app/doctor.py` | Diagnostic tool that verifies binaries, Docker services, database connections, and API keys. |
| **LLM Client** | `llm/client.py` | Two-tier LLM routing via LiteLLM -- `fast_complete` for planning/retrieval and `strong_complete` for patching/reflection. |
| **RAG Indexing** | `rag/index_repo.py` | AST-aware Python chunking and text chunking, with FAISS or pgvector storage. |
| **RAG Retrieval** | `rag/retrieve_repo.py` | Hybrid retrieval combining vector similarity search with ripgrep-based lexical recall. |
| **Memory Store** | `memory/store.py` | Mem0-backed episodic memory with structured fix card storage and similarity search. |
| **Graph Store** | `memory/graph_store.py` | Neo4j-backed knowledge graph storing runs, iterations, steps, files, fix cards, and failure signatures. |
| **Sandbox** | `tools/sandbox.py` | Docker container execution with CPU, memory, and network isolation. |
| **Patching** | `tools/patching.py` | Patch generation, validation, workspace isolation, and multi-strategy application with automatic line realignment. |
| **Static Checks** | `tools/static_checks.py` | Ruff (linting) and Bandit (security) execution with sandbox support. |
| **Static Audit** | `tools/static_audit.py` | AST-based SQL injection detection, typo analysis, and LLM-powered code/SQL review. |
| **Test Runner** | `tools/run_tests.py` | Test suite execution with sandbox and local modes. |
| **Eval Runner** | `eval/runner.py` | Evaluation harness running multiple agent loops and computing aggregate metrics. |
| **Eval Metrics** | `eval/metrics.py` | Metrics: pass@1, pass@k, time-to-green, repeated mistake rate. |
| **Storage** | `storage/db.py` | PostgreSQL schema management for runs, evals, and eval_runs tables. |
| **Observability** | `observability/` | OpenTelemetry tracing (OTLP/Phoenix), Prometheus counters/histograms, structured JSON logging, CSV metric export. |

---

## Project Structure

```
self-improving-coding-agent/
|-- agent/
|   |-- graph.py              # LangGraph execution loop (13 nodes)
|   |-- state.py              # RunState, RunConfig, IterationRecord, StepResult
|   |-- checkpoints.py        # JSON checkpoint store for pause/resume
|-- app/
|   |-- cli.py                # CLI entry point (run, index, eval, doctor, runs)
|   |-- main.py               # FastAPI REST API server
|   |-- doctor.py             # Dependency and configuration health checks
|   |-- run_artifacts.py      # Post-run artifact serialization
|-- config/
|   |-- api.env.example       # API server environment template
|   |-- graph.env.example     # Neo4j graph store template
|   |-- infra.env.example     # Docker Compose infrastructure template
|   |-- llm.env.example       # LLM model and routing template
|   |-- memory.env.example    # Mem0 memory backend template
|   |-- observability.env.example  # Tracing and metrics template
|   |-- rag.env.example       # RAG indexing and retrieval template
|-- db/
|   |-- migrations/
|       |-- 001_repo_chunks.sql  # pgvector schema migration
|-- docs/                     # Detailed design documents (9 sections)
|-- eval/
|   |-- runner.py             # Multi-run evaluation harness
|   |-- metrics.py            # pass@1, pass@k, time-to-green, repeated-mistake-rate
|-- llm/
|   |-- client.py             # Two-tier LLM client (fast + strong) via LiteLLM
|   |-- config.py             # LLM configuration resolution (env + CLI overrides)
|-- memory/
|   |-- store.py              # MemoryStore with Mem0 backend
|   |-- mem0_client.py        # Mem0 API client wrapper
|   |-- graph_store.py        # Neo4j graph store (runs, steps, fix cards, files)
|   |-- config.py             # Memory configuration resolution
|   |-- graph_config.py       # Graph store configuration resolution
|-- observability/
|   |-- tracing.py            # OpenTelemetry tracing with OTLP export
|   |-- metrics.py            # Prometheus counters, histograms, CSV export
|   |-- logging.py            # Structured JSON logging
|-- rag/
|   |-- chunking.py           # AST-aware Python chunking + text chunking
|   |-- embeddings.py         # Embedding provider abstraction (LiteLLM, mock)
|   |-- index_repo.py         # Repository indexing (FAISS, pgvector)
|   |-- retrieve_repo.py      # Hybrid retrieval (vector + lexical recall)
|   |-- lexical.py            # ripgrep-based lexical file recall
|   |-- index_meta.py         # Index staleness tracking
|   |-- stores/
|       |-- faiss_store.py    # FAISS vector store
|       |-- pgvector_store.py # pgvector (PostgreSQL) vector store
|-- storage/
|   |-- db.py                 # PostgreSQL connection and schema initialization
|   |-- run_store.py          # Run CRUD operations
|   |-- eval_store.py         # Evaluation CRUD operations
|-- tools/
|   |-- sandbox.py            # Docker sandbox execution engine
|   |-- patching.py           # Patch generation, validation, multi-strategy application
|   |-- static_checks.py      # Ruff and Bandit static analysis
|   |-- static_audit.py       # SQL/code audit with AST analysis + LLM review
|   |-- run_tests.py          # Test suite runner
|-- tests/                    # Unit and integration tests
|-- main.py                   # Entry point
|-- pyproject.toml            # Project metadata and dependencies
|-- docker-compose.yml        # Infrastructure services (Postgres, Neo4j, Phoenix, LiteLLM)
|-- Dockerfile.agent          # Sandbox container image
```

---

## Prerequisites

- **Python** >= 3.11
- **Docker** (for sandbox execution and infrastructure services)
- **Git** (for patch application)
- **ripgrep** (`rg`) -- optional but recommended for lexical RAG recall
- An LLM provider API key (OpenAI, Groq, Anthropic, or any LiteLLM-compatible provider)
- A Mem0 API key for persistent memory
- Neo4j instance for graph memory

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd self-improving-coding-agent

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e ".[dev]"
```

Start the required infrastructure services:

```bash
docker compose up -d
```

This provisions:

| Service | Port | Purpose |
|---|---|---|
| PostgreSQL + pgvector | 5434 | Run storage, evaluation records, pgvector RAG backend |
| Neo4j | 7474 (HTTP), 7687 (Bolt) | Knowledge graph for runs, steps, fix cards |
| Phoenix | 6006 | OpenTelemetry trace visualization |
| LiteLLM Proxy | 4000 | Optional LLM routing proxy |

---

## Configuration

All configuration is driven by environment variables with CLI flag overrides. Template files are located in `config/`:

| Template | Purpose |
|---|---|
| `config/llm.env.example` | LLM provider, model names, temperature, timeout |
| `config/memory.env.example` | Mem0 API key, host, organization, project IDs |
| `config/graph.env.example` | Neo4j URI, credentials, database name |
| `config/rag.env.example` | RAG backend, embedding provider, chunk sizes |
| `config/infra.env.example` | Docker Compose service configuration |
| `config/observability.env.example` | Tracing, metrics port, Phoenix integration |
| `config/api.env.example` | DATABASE_URL for the REST API |

Core environment variables:

```bash
# LLM (required)
export LLM_ENABLED=1
export FAST_MODEL=gpt-4o-mini          # Used for planning, retrieval query rewriting
export STRONG_MODEL=gpt-4o             # Used for patch generation, reflection, audits
export LITELLM_API_KEY=<your-key>

# Memory (required)
export MEMORY_BACKEND=mem0
export MEM0_API_KEY=<your-key>

# Graph (required)
export GRAPH_BACKEND=neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4jpass

# Database (for API and eval persistence)
export DATABASE_URL="postgresql://agent:agentpass@localhost:5434/agentdb"
```

Do not commit secrets. Use `config/*.env.example` as templates and store actual credentials in your shell environment or a secrets manager.

---

## Usage

### Running the Agent

```bash
python main.py run \
  --goal "Fix failing tests" \
  --repo-path /path/to/target/repo \
  --max-iters 5 \
  --until-green \
  --ruff-cmd "ruff check ." \
  --bandit-cmd "bandit -r ." \
  --test-cmd "pytest -q"
```

Key flags:

| Flag | Description |
|---|---|
| `--goal` | Natural language description of the task |
| `--repo-path` | Path to the target repository |
| `--max-iters` | Maximum number of fix iterations |
| `--until-green` | Keep iterating until tests pass (respects `--max-iters`) |
| `--apply` | Write the patch to the target repository (default: workspace only) |
| `--stop-after <step>` | Pause after a named step (e.g., `plan`, `retrieve`, `patch`) |
| `--resume-run-id <id>` | Resume a paused run from its checkpoint |
| `--json` | Output the final run state as JSON |
| `--verbose` | Show detailed step-by-step output |

### Analysis-Only Mode

Run planning, retrieval, static audit, and reflection without generating or applying patches:

```bash
python main.py run \
  --analysis-only \
  --goal "Scan SQL in ml/train_model.py" \
  --repo-path /path/to/repo
```

The static audit includes rule-based checks (SQL injection detection, keyword typo analysis) and an LLM-powered review. The LLM determines whether SQL review, code review, or both are needed based on the stated goal.

### Repository Indexing

Index a repository for RAG retrieval:

```bash
# FAISS backend (default, local files)
python main.py index --repo-path /path/to/repo

# pgvector backend (PostgreSQL)
python main.py index \
  --repo-path /path/to/repo \
  --rag-backend pgvector \
  --rag-pg-dsn "postgresql://agent:agentpass@localhost:5434/agentdb"
```

The indexer uses AST-aware chunking for Python files (splitting on function and class boundaries) and fixed-size text chunking for all other supported file types. Runs automatically index the repository if the RAG index is missing or stale (disable with `--no-auto-index`).

### Docker Sandbox Execution

Build the sandbox image and run tool commands in an isolated container:

```bash
docker build -t self-improving-agent:latest -f Dockerfile.agent .

python main.py run \
  --use-sandbox \
  --repo-path /path/to/repo \
  --sandbox-image self-improving-agent:latest \
  --sandbox-cpus 1.0 \
  --sandbox-memory 1g \
  --sandbox-network none \
  --ruff-cmd "ruff check ." \
  --bandit-cmd "bandit -r ." \
  --test-cmd "pytest -q"
```

The sandbox mounts the repository into the container, enforces CPU and memory limits, and disables network access by default.

### REST API Server

```bash
export DATABASE_URL="postgresql://agent:agentpass@localhost:5434/agentdb"
uvicorn app.main:create_app --factory --reload
```

| Endpoint | Method | Description |
|---|---|---|
| `/runs` | POST | Create a new agent run (sync or async via `run_mode`) |
| `/runs` | GET | List all runs with pagination (`limit`, `offset`) |
| `/runs/{run_id}` | GET | Retrieve a specific run by ID |
| `/evals` | POST | Trigger an evaluation suite (sync or async) |
| `/evals/{eval_id}` | GET | Retrieve evaluation results by ID |

### Evaluation Harness

Run multiple agent loops and compute aggregate metrics:

```bash
python main.py eval \
  --goal "Fix failing tests" \
  --runs 5 \
  --max-iters 3
```

Computed metrics:

- **pass@1** -- Whether the first run succeeded.
- **pass@k** -- Whether any of the k runs succeeded.
- **Average iterations** -- Mean iteration count across runs.
- **Average time-to-green** -- Mean wall-clock time to first passing test suite.
- **Repeated mistake rate** -- Fraction of iterations that generated duplicate patch fingerprints.

### Health Check

Verify that all dependencies and services are correctly configured:

```bash
python main.py doctor
```

This checks: Docker, Git, ripgrep, Postgres connectivity, Neo4j connectivity, API keys, and LLM configuration.

### Run History

List recent runs with status summaries:

```bash
python main.py runs
```

---

## Patch Application Workflow

Generated patches follow a safe, isolated workflow:

1. **Workspace creation** -- The target repository is copied to a temporary workspace directory (excluding `.git`, `.venv`, `node_modules`, and other non-essential directories). A fresh `git init` is run in the workspace.
2. **Patch validation** -- The patch is checked against blocked path prefixes (`.git/`, `.env`, `../`, `/`) and a configurable size limit (default: 200 KB).
3. **Multi-strategy application** -- The agent attempts six fallback strategies in order:
   - Standard `git apply`
   - `git apply --recount` (fixes incorrect hunk line counts)
   - Automatic line realignment (searches for context in the actual file and corrects hunk offsets)
   - `git apply --3way` (index-based conflict resolution)
   - `git apply --recount -C0` (minimal context matching)
   - Unix `patch --fuzz=3` (permissive context matching)
4. **Optional repo application** -- With `--apply`, the patch is also written to the original repository after successful workspace application.
5. **Patch archival** -- All generated patches are saved to `runs/patches/<run_id>.patch`.

---

## Memory and Knowledge Graph

### Mem0 Episodic Memory

Each iteration stores an episode summary and a structured fix card in Mem0. On subsequent runs, the retrieval step queries Mem0 for similar past fixes, enabling cross-repository learning. Metadata is automatically truncated to comply with Mem0 size limits while preserving priority fields (summary, root cause, fix, verification, error signature).

Configuration:

```bash
export MEMORY_BACKEND=mem0
export MEM0_API_KEY=<your-key>
# Optional: MEM0_HOST, MEM0_ORG_ID, MEM0_PROJECT_ID, MEM0_USER_ID, MEM0_AGENT_ID
```

### Neo4j Graph Store

The graph store persists the full execution topology:

- **Run** nodes linked to **Iteration** nodes via `HAS_ITERATION`
- **Iteration** nodes linked to **Step** nodes via `HAS_STEP`
- **Iteration** nodes linked to **File** nodes via `RETRIEVED_FILE`
- **Run** nodes linked to **FixCard** nodes via `HAS_FIX_CARD`
- **FixCard** nodes linked to **File** nodes via `CHANGED_FILE`
- **Iteration/FixCard** nodes linked to **FailureSignature** and **BugType** nodes

This enables graph queries such as: "Which files are most frequently involved in failures?" or "What fix patterns resolved this error signature?"

Configuration:

```bash
export GRAPH_BACKEND=neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4jpass
```

### Fix Cards

After each iteration, the agent generates a structured fix card via the strong LLM containing:

| Field | Description |
|---|---|
| `summary` | One-line description of what was fixed |
| `root_cause` | Identified root cause of the failure |
| `fix` | Description of the applied fix |
| `verification` | How the fix was verified |
| `error_signature` | First line of stderr or the last error message |
| `files_changed` | List of files modified by the patch |
| `debug_steps` | Reasoning steps taken during diagnosis |

Fix cards are stored in both Mem0 (for similarity search) and Neo4j (for graph-based queries).

---

## Observability

The agent is instrumented with three observability pillars:

**Tracing (OpenTelemetry)**

Every graph node and LLM call is wrapped in an OpenTelemetry span. Traces are exported via OTLP to Phoenix or any compatible collector.

```bash
export TRACING_ENABLED=1
export TRACING_DEFAULT_PHOENIX=1    # Auto-configure Phoenix at localhost:6006
# Or set OTEL_EXPORTER_OTLP_ENDPOINT for a custom collector
```

**Metrics (Prometheus)**

Prometheus counters and histograms are exposed on a configurable HTTP port:

- `agent_runs_total` -- Total agent runs
- `agent_iterations_total` -- Total iterations across all runs
- `agent_steps_total` -- Total steps across all iterations
- `agent_run_duration_seconds` -- Run duration histogram

```bash
export METRICS_PORT=8000
```

**Logging**

Structured JSON logging captures step-level events with `run_id`, `step`, `iteration`, and `status` fields.

**CSV Export**

Per-run metrics can be appended to a CSV file for offline analysis:

```bash
export METRICS_CSV_PATH=metrics.csv
```

---

## Testing

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_graph.py
```

The test suite covers the agent graph, LLM client, memory store, graph store, RAG (FAISS and pgvector), patching, sandbox execution, and observability modules.

---

## Security Considerations

- **Sandbox isolation** -- All tool commands (linting, security scanning, test execution) can be run inside a Docker container with restricted CPU, memory, and network access.
- **Patch validation** -- Patches are validated against blocked path prefixes to prevent writes to `.git/`, `.env`, parent directories, or absolute paths.
- **Patch size limits** -- A configurable maximum patch size (default: 200 KB) prevents excessively large modifications.
- **Workspace isolation** -- Patches are applied to a temporary workspace by default. The `--apply` flag must be explicitly set to write changes to the target repository.
- **Secrets hygiene** -- No secrets are stored in configuration files. All credentials are read from environment variables. The `config/*.env.example` files serve as templates only.

---

## License

See the project repository for license information.
