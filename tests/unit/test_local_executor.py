"""Tests for the local executor — runs experiments, captures failures gracefully."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from iterate.adapters.compute.local import LocalExecutor
from iterate.adapters.data.tabular import load_csv
from iterate.schemas.experiment import Candidate
from iterate.targets.model import ModelTarget

if TYPE_CHECKING:
    from pathlib import Path


def _target(tmp_path: Path) -> ModelTarget:
    n = 120
    frame = pd.DataFrame(
        {
            "num": [i % 10 for i in range(n)],
            "cat": (["a", "b", "c"] * (n // 3 + 1))[:n],
            "churn": [1 if (i % 10) >= 6 else 0 for i in range(n)],
        }
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return ModelTarget(load_csv(path, target="churn"), metric="f1")


def test_baseline_succeeds_and_is_timed(tmp_path: Path) -> None:
    result = LocalExecutor().execute(_target(tmp_path))
    assert result.succeeded
    assert result.metrics is not None
    assert result.experiment_id == "baseline"
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0


def test_good_candidate_succeeds(tmp_path: Path) -> None:
    candidate = Candidate(
        description="xgboost",
        changes={"model": "xgboost.XGBClassifier", "params": {"n_estimators": 10}},
        rationale="try boosted trees",
    )
    result = LocalExecutor().execute(_target(tmp_path), candidate)
    assert result.succeeded
    assert result.experiment_id == candidate.id


def test_fit_time_failure_is_captured_not_raised(tmp_path: Path) -> None:
    # max_iter=-1 is rejected by sklearn at fit time -> must come back as a failed result.
    candidate = Candidate(
        description="invalid max_iter",
        changes={"params": {"max_iter": -1}},
        rationale="broken proposal",
    )
    result = LocalExecutor().execute(_target(tmp_path), candidate)
    assert not result.succeeded
    assert result.error is not None
    assert result.metrics is None
    assert result.experiment_id == candidate.id
    assert result.duration_seconds is not None


def test_disallowed_model_failure_is_captured(tmp_path: Path) -> None:
    candidate = Candidate(
        description="off-list model",
        changes={"model": "numpy.array", "params": {}},
        rationale="should be rejected by the factory",
    )
    result = LocalExecutor().execute(_target(tmp_path), candidate)
    assert not result.succeeded
    assert result.error is not None
    assert "allowed library" in result.error
