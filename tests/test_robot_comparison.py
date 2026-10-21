import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def metrics():
    path = Path(__file__).resolve().parents[1] / "scripts/summarize_robot_comparison.py"
    spec = importlib.util.spec_from_file_location("comparison_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_units_are_not_mixed(metrics):
    a = np.zeros((2, 32, 14))
    b = a.copy()
    b[..., metrics.JOINTS] = .1
    b[..., [6, 13]] = .002
    result = metrics.physical_metrics(a, b)
    assert result["by_unit"]["joints_rad"]["rmse"] == pytest.approx(.1)
    assert result["by_unit"]["grippers_mm"]["rmse"] == pytest.approx(2)
    assert "rmse" not in result
    assert result["per_dimension"][6]["unit"] == "m"


def test_entire_chunk_not_just_prefix(metrics):
    a = np.zeros((1, 32, 14))
    b = a.copy()
    b[:, 31, 0] = 2
    result = metrics.error_metrics(a, b)
    assert result["max_abs"] == 2
    assert result["rmse"] == pytest.approx(2/np.sqrt(32*14))
    assert result["per_dimension"][0]["rmse"] == pytest.approx(2/np.sqrt(32))


def test_nonfinite_actions_rejected(metrics):
    a = np.zeros((32, 14))
    b = a.copy()
    b[0, 0] = np.nan
    with pytest.raises(ValueError):
        metrics.error_metrics(a, b)
