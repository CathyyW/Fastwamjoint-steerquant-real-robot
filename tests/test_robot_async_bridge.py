import numpy as np
import pytest

from fastwam_steerquant.robot_async_client import W4A8Async50PolicyClient, authkey_from_config
from fastwam_steerquant.robot_async_server import serve_connection


def diagnostics():
    return dict(backend="cutlass_w4a8", activation_bits=8, rotation="none", sparse_enabled=False,
                deployment_sha256="expected", task="pack", action_coordinates="physical",
                kernel_audit=dict(verified_sites=600, native_calls=6000, cuda_kernel_verified=True, tile64_kernels=6000),
                runtime_settings=dict(w4a8_tile="64", cuda_graph=True, fuse_block=False, fused_modules=0),
                cuda_graph=True, summary=dict(denoise_calls=10, finite=True, graph_replays=10))


@pytest.mark.parametrize("key,value", [("backend", "reference"), ("activation_bits", 4),
    ("rotation", "rht"), ("sparse_enabled", True), ("deployment_sha256", "other"),
    ("task", "stack"), ("action_coordinates", "normalized")])
def test_wrong_backend_or_task_rejected(key, value):
    client = W4A8Async50PolicyClient.__new__(W4A8Async50PolicyClient)
    client.expected_sha256, client.expected_task = "expected", "pack"
    info = diagnostics()
    client.validate_diagnostics(info)
    info[key] = value
    with pytest.raises(RuntimeError):
        client.validate_diagnostics(info, require_summary=False)


@pytest.mark.parametrize("section,key,value", [("kernel_audit", "verified_sites", 599),
    ("kernel_audit", "native_calls", 5999), ("kernel_audit", "cuda_kernel_verified", False),
    ("summary", "graph_replays", 0), ("summary", "finite", False), ("summary", "denoise_calls", 9)])
def test_no_silent_fallback(section, key, value):
    client = W4A8Async50PolicyClient.__new__(W4A8Async50PolicyClient)
    client.expected_sha256, client.expected_task = "expected", "pack"
    info = diagnostics()
    info[section][key] = value
    with pytest.raises(RuntimeError):
        client.validate_diagnostics(info)


def test_authkey_is_required_and_not_literal_placeholder(monkeypatch):
    config = {"policy_server": {"authkey": "${oc.env:FASTWAM_AUTHKEY}"}}
    monkeypatch.delenv("FASTWAM_AUTHKEY", raising=False)
    with pytest.raises(ValueError):
        authkey_from_config(config)
    monkeypatch.setenv("FASTWAM_AUTHKEY", "test-only")
    assert authkey_from_config(config) == b"test-only"


def test_protocol_reset_and_physical_actions_are_not_transformed():
    class Policy:
        observation = None
        summary = None

        def infer(self, observation):
            assert observation == {"frame": 7}
            return np.full((32, 14), 0.123, np.float32)

        def diagnostics(self):
            return {"ok": True, **diagnostics()}

    class Connection:
        def __init__(self):
            self.requests = iter([None, {"op": "get_action"}, {"op": "infer", "obs": {"frame": 7}},
                                  {"op": "reset"}, {"op": "get_action"}, {"op": "shutdown"}])
            self.responses = []

        def recv(self):
            return next(self.requests)

        def send(self, response):
            self.responses.append(response)

    conn = Connection()
    assert serve_connection(Policy(), conn)
    assert [x["ok"] for x in conn.responses] == [False, False, True, True, False, True]
    np.testing.assert_array_equal(conn.responses[2]["action"], np.full((32, 14), 0.123, np.float32))


def test_legacy_entry_delegates_to_portable_client():
    import runpy
    from pathlib import Path
    from fastwam_steerquant.real_robot.run import main
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts/deploy_real_w4a8_async50.py"))
    assert module["main"] is main
