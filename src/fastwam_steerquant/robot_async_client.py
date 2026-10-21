"""Hardware-free W4A8 RPC client; consumed by the existing async50 runtime."""
from multiprocessing.connection import Client
import os

import numpy as np


def authkey_from_config(config):
    value = config["policy_server"]["authkey"]
    if value in ("${FASTWAM_AUTHKEY}", "${oc.env:FASTWAM_AUTHKEY}"):
        value = os.environ.get("FASTWAM_AUTHKEY", "")
    if not value:
        raise ValueError("Set FASTWAM_AUTHKEY identically in the server/client environment.")
    return value.encode() if isinstance(value, str) else value


class W4A8Async50PolicyClient:
    """One connection, one producer; diagnostics RPC remains inside timing budget."""
    def __init__(self, host, port, authkey, *, expected_sha256, expected_task):
        self.expected_sha256 = expected_sha256
        self.expected_task = expected_task
        self.last_diagnostics = None
        self.conn = Client((host, int(port)), authkey=authkey.encode() if isinstance(authkey, str) else authkey)

    def _request(self, payload):
        self.conn.send(dict(payload))
        if not self.conn.poll(10.0):
            raise TimeoutError("W4A8 policy RPC response exceeded 10 seconds")
        response = self.conn.recv()
        if not isinstance(response, dict) or not response.get("ok", False):
            raise RuntimeError(f"W4A8 RPC failed: {response}")
        return response

    def validate_diagnostics(self, info, *, require_summary=True):
        if (info.get("backend") != "cutlass_w4a8" or info.get("activation_bits") != 8
                or info.get("rotation") != "none" or info.get("sparse_enabled") is not False
                or info.get("deployment_sha256") != self.expected_sha256
                or info.get("task") != self.expected_task
                or info.get("action_coordinates") != "physical"):
            raise RuntimeError("Server W4A8 deployment identity/coordinate contract mismatch")
        proof = info.get("kernel_audit") or {}
        if proof.get("verified_sites") != 600 or proof.get("native_calls") != 6000 or not proof.get("cuda_kernel_verified"):
            raise RuntimeError("Server lacks a complete native W4A8 forward audit")
        settings = info.get("runtime_settings") or {}
        if (settings.get("w4a8_tile") != "64" or settings.get("cuda_graph") is not True
                or settings.get("fuse_block") is not False or settings.get("fused_modules") != 0
                or proof.get("tile64_kernels") != 6000):
            raise RuntimeError("Server has not verified the tile64/Graph/no-extra-fusion profile")
        if require_summary:
            summary = info.get("summary") or {}
            if summary.get("denoise_calls") != 10 or summary.get("finite") is not True:
                raise RuntimeError("W4A8 inference diagnostics failed")
            if info.get("cuda_graph") and summary.get("graph_replays") != 10:
                raise RuntimeError("Expected ten live graph replays; no eager fallback allowed")

    def infer(self, observation):
        action = np.asarray(self._request({"op": "infer", "obs": dict(observation)})["action"], dtype=np.float32)
        if action.shape != (32, 14) or not np.isfinite(action).all():
            raise ValueError("Expected finite physical-coordinate [32,14] action chunk")
        info = self._request({"op": "diagnostics"})
        self.validate_diagnostics(info)
        self.last_diagnostics = info
        return action

    def reset(self):
        self._request({"op": "reset"})

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
