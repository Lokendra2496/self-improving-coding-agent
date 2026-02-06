from pathlib import Path

from observability.metrics import write_metrics_csv
from observability.tracing import configure_tracing


def test_configure_tracing_noop() -> None:
    configure_tracing()


def test_write_metrics_csv(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    write_metrics_csv("run-1", {"counters": {"runs_total": 1}, "timings": {}}, path=str(path))
    assert path.exists()
