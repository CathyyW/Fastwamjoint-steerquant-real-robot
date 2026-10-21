"""Packed W4A4/RHT service policy; distinct native kernel, no W4A8 profile."""
from collections import Counter
import os
from pathlib import Path

import numpy as np
import torch

from .robot_async_server import PackedRobotPolicy, prepare_adapter
from .policy import QuantizedPolicy
from .runtime import WAMQuantLinear
from .rollout_graph import LiveJointDiTGraph


class PackedW4A4RobotPolicy(PackedRobotPolicy):
    def __init__(self, config):
        self.config = config
        w, q = config["wam"], config["wam"]["quantization"]
        if os.environ.get("FASTWAM_RUNTIME_PROFILE"):
            raise ValueError("W4A4 must not inherit the W4A8 runtime profile")
        if os.environ.get("FASTWAM_W4A4_TILE", "auto") != "auto":
            raise ValueError("This W4A4 profile requires its native auto tile selection")
        os.environ["FASTWAM_W4A4_TILE"] = "auto"
        os.environ["FASTWAM_W4A4_EXTENSION"] = q["extension"]
        self.adapter, self.contract = prepare_adapter(config)
        self.policy = QuantizedPolicy(w["ckpt"], self.adapter, device=w["device"],
                                      construct_device="cpu", cuda_graph=True, fuse_block=False)
        from .kernels.w4a4.loader import load_extension
        self.extension = load_extension()
        if Path(self.extension.__file__).resolve() != Path(q["extension"]).resolve():
            raise RuntimeError("A different W4A4 extension was already loaded")
        self.policy.runtime_settings["w4a4_tile"] = "auto"
        self.policy.runtime_settings.pop("w4a8_tile", None)
        self.quant_layers = {n: m for n, m in self.policy.model.named_modules() if isinstance(m, WAMQuantLinear)}
        self.observation = self.summary = None
        self.kernel_audit = {}
        self.assert_native()

    def assert_native(self):
        settings = self.policy.runtime_settings
        if (len(self.quant_layers) != 600 or settings["cuda_graph"] is not True
                or settings["fuse_block"] is not False or self.policy.fused_modules != 0):
            raise RuntimeError("Requires 600 W4A4/RHT targets, Graph, no extra fusion")
        for name, m in self.quant_layers.items():
            if (m.backend != "cutlass_w4a4" or m.rotation != "rht" or m.activation_bits != 4
                    or m.weight.dtype != torch.uint8 or m.weight.device.type != "cuda"
                    or m.supports_fused_adaln or m.supports_fused_gate_residual
                    or any(getattr(m, k) is None or getattr(m, k).dtype != torch.float32 for k in
                           ("weight_scales", "input_scale", "gamma_gains", "clipping_ranges", "input_rotation_signs"))):
                raise RuntimeError(f"W4A4/RHT packed native invariant failed: {name}")

    def warmup(self, observation):
        from .kernels.w4a4 import wam_ops
        self.policy.model._rollout_graph.remove()
        counts, handles = Counter(), []
        native_calls = 0
        original = wam_ops.symmetric_scheduled_stream_rht_linear

        def counted(*a, **kw):
            nonlocal native_calls
            native_calls += 1
            return original(*a, **kw)

        try:
            for name, module in self.quant_layers.items():
                def hook(m, a, result, name=name):
                    counts[name] += 1
                handles.append(module.register_forward_hook(hook))
            wam_ops.symmetric_scheduled_stream_rht_linear = counted
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                eager = self.infer(observation)
            kernels = Counter(e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA)
            gemms = {k: v for k, v in kernels.items() if "GemmWithEpilogueVisitor" in k}
            producers = {k: v for k, v in kernels.items() if "rht" in k.lower() and "quantize" in k.lower()}
            if (set(counts) != set(self.quant_layers) or set(counts.values()) != {10}
                    or native_calls != 6000 or sum(gemms.values()) != 6000 or sum(producers.values()) != 6000):
                sample = [(name[:220], count) for name, count in kernels.most_common(12)]
                raise RuntimeError(f"Incomplete W4A4 audit: native={native_calls}, GEMM={sum(gemms.values())}, "
                                   f"RHT={sum(producers.values())}, CUDA_events={sum(kernels.values())}, sample={sample}")
            self.kernel_audit = dict(verified_sites=600, native_calls=native_calls, cuda_kernel_verified=True,
                                    rht_producer_kernels=sum(producers.values()), cutlass_kernels=sum(gemms.values()),
                                    extension=self.extension.__file__, producer_names=list(producers),
                                    cutlass_names=list(gemms))
        finally:
            wam_ops.symmetric_scheduled_stream_rht_linear = original
            for handle in handles:
                handle.remove()
        LiveJointDiTGraph(self.policy.model, num_calls=10).install()
        actual = self.infer(observation)
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(eager), rtol=1e-3, atol=1e-3)
        self.kernel_audit["graph_eager_output_match"] = True
        self.infer(observation)
        self.observation = None

    def diagnostics(self):
        q = self.config["wam"]["quantization"]
        return dict(ok=True, backend="cutlass_w4a4", activation_bits=4, rotation="rht", sparse_enabled=False,
                    task=q["task"], action_coordinates="physical", deployment_sha256=q["deployment_sha256"],
                    checkpoint=self.config["wam"]["ckpt"], cuda_graph=True,
                    runtime_settings=self.policy.runtime_settings, kernel_audit=self.kernel_audit, summary=self.summary)
