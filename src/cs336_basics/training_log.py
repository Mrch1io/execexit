from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO
from uuid import uuid4

RunStatus = Literal["completed", "interrupted", "failed"]


class ExperimentLogger:
    """Append structured run configuration and loss-curve events to JSONL."""

    def __init__(
        self,
        path: str | Path,
        *,
        experiment_name: str,
        config: Mapping[str, Any],
    ) -> None:
        if not experiment_name.strip():
            raise ValueError("experiment_name must not be empty")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid4().hex
        self.experiment_name = experiment_name
        self._stream: TextIO = self.path.open("a", encoding="utf-8", buffering=1)
        self._closed = False
        self._write(
            {
                "event": "run_start",
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "timestamp_utc": _timestamp_utc(),
                "config": dict(config),
            }
        )

    def log_metrics(
        self,
        *,
        step: int,
        wall_clock_seconds: float,
        split: Literal["train", "validation"],
        loss: float,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if wall_clock_seconds < 0:
            raise ValueError(
                f"wall_clock_seconds must be non-negative, got {wall_clock_seconds}"
            )

        record: dict[str, Any] = {
            "event": "metrics",
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp_utc": _timestamp_utc(),
            "step": step,
            "wall_clock_seconds": wall_clock_seconds,
            "split": split,
            "loss": _json_metric(loss),
        }
        if metrics is not None:
            record["metrics"] = {
                name: _json_metric(value) for name, value in metrics.items()
            }
        self._write(record)

    def close(
        self,
        *,
        status: RunStatus,
        step: int,
        wall_clock_seconds: float,
    ) -> None:
        if self._closed:
            return
        self._write(
            {
                "event": "run_end",
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "timestamp_utc": _timestamp_utc(),
                "status": status,
                "step": step,
                "wall_clock_seconds": wall_clock_seconds,
            }
        )
        self._stream.close()
        self._closed = True

    def _write(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed experiment logger")
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        )
        self._stream.write(serialized + "\n")
        self._stream.flush()


def _json_metric(value: float) -> float | str:
    numeric_value = float(value)
    if math.isfinite(numeric_value):
        return numeric_value
    return str(numeric_value)


def _timestamp_utc() -> str:
    return datetime.now(UTC).isoformat()
