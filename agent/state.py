from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StepResult:
    name: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: Optional[str] = None

    def finish(self) -> None:
        self.ended_at = utc_now_iso()


@dataclass
class IterationRecord:
    index: int
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: Optional[str] = None
    steps: List[StepResult] = field(default_factory=list)

    def add_step(self, name: str, status: str, details: Optional[Dict[str, Any]] = None) -> StepResult:
        step = StepResult(name=name, status=status, details=details or {})
        self.steps.append(step)
        return step

    def finish(self) -> None:
        self.ended_at = utc_now_iso()


@dataclass
class RunState:
    run_id: str
    goal: str
    status: str
    created_at: str = field(default_factory=utc_now_iso)
    completed_at: Optional[str] = None
    iterations: List[IterationRecord] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(cls, goal: str, run_id: Optional[str] = None) -> "RunState":
        return cls(run_id=run_id or str(uuid4()), goal=goal, status="running")

    def complete(self, status: str = "completed") -> None:
        self.status = status
        self.completed_at = utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    goal: str
    run_id: Optional[str] = None
    max_iters: int = 1
    until_green: bool = False
    dry_run: bool = False
    analysis_only: bool = False
    ruff_cmd: Optional[List[str] | str] = None
    bandit_cmd: Optional[List[str] | str] = None
    test_cmd: Optional[List[str] | str] = None
    resume_run_id: Optional[str] = None
    checkpoint_dir: str = ".checkpoints"
    max_retries: int = 1
    force_retry_once: bool = False
    stop_after_step: Optional[str] = None
    use_sandbox: bool = False
    sandbox_image: str = "self-improving-agent:latest"
    sandbox_cpus: float = 1.0
    sandbox_memory: str = "1g"
    sandbox_network: str = "none"
    sandbox_user: Optional[str] = None
    repo_path: str = "."
    llm_enabled: Optional[bool] = None
    fast_model: Optional[str] = None
    strong_model: Optional[str] = None
    litellm_base_url: Optional[str] = None
    litellm_api_key: Optional[str] = None
    fast_temperature: Optional[float] = None
    strong_temperature: Optional[float] = None
    litellm_timeout_s: Optional[int] = None
    rag_backend: Optional[str] = None
    rag_top_k: Optional[int] = None
    rag_index_dir: Optional[str] = None
    rag_pg_dsn: Optional[str] = None
    auto_index: bool = False
    rag_chunk_size: Optional[int] = None
    rag_chunk_overlap: Optional[int] = None
    rag_lexical_top_k: Optional[int] = None
    rag_vector_top_k: Optional[int] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_batch_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    memory_backend: Optional[str] = None
    mem0_api_key: Optional[str] = None
    mem0_host: Optional[str] = None
    mem0_org_id: Optional[str] = None
    mem0_project_id: Optional[str] = None
    mem0_user_id: Optional[str] = None
    mem0_agent_id: Optional[str] = None
    mem0_app_id: Optional[str] = None
    memory_top_k: Optional[int] = None
    memory_fail_fast: Optional[bool] = None
    graph_backend: Optional[str] = None
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: Optional[str] = None
    graph_fail_fast: Optional[bool] = None
    apply_to_repo: bool = False
    workspace_root: str = ".workspaces"
    patch_output_dir: str = "runs/patches"
    patch_max_bytes: int = 200_000
    patch_blocked_prefixes: Optional[List[str]] = None