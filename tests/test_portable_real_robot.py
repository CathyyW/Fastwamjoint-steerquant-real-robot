"""Mock-only public client contracts. No ROS, SDK, CAN or model is started."""
from contextlib import ExitStack
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fastwam_steerquant.real_robot import actions, config, observations, passive_can, run
from fastwam_steerquant.real_robot.runtime import AsyncJointRTCRuntime
from fastwam_steerquant.robot_async_client import W4A8Async50PolicyClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["pack", "stack"])
def cfg(request):
    return config.load_config(ROOT / f"configs/robot_{request.param}_w4a8_live.example.yaml")


def pose(cfg):
    return np.asarray(cfg["robot"]["left_init_position"] + cfg["robot"]["right_init_position"], dtype=np.float32)


def diagnostics(cfg):
    return dict(ok=True, backend="cutlass_w4a8", activation_bits=8, rotation="none", sparse_enabled=False,
        deployment_sha256=cfg["policy"]["deployment_sha256"], task=cfg["policy"]["task"], action_coordinates="physical",
        kernel_audit=dict(verified_sites=600, native_calls=6000, cuda_kernel_verified=True, tile64_kernels=6000),
        runtime_settings=dict(w4a8_tile="64", cuda_graph=True, fuse_block=False, fused_modules=0),
        cuda_graph=True, control_hz=40, latency_window_ms=775,
        summary=dict(denoise_calls=10, finite=True, graph_replays=10))


def rpc_client(cfg):
    client = W4A8Async50PolicyClient.__new__(W4A8Async50PolicyClient)
    client.expected_task = cfg["policy"]["task"]
    client.expected_sha256 = cfg["policy"]["deployment_sha256"]
    client.conn = Mock()
    client.conn.poll.return_value = True
    return client


def readers(cfg):
    pixels = {key: np.full((480, 640, 3), [9, 20, 31], np.uint8) for key in observations.CAMERAS}
    state, images = Mock(), Mock()
    state.get.return_value = pose(cfg)
    images.get.return_value = (pixels, {key: 10. for key in pixels})
    return state, images


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "8UC3"])
def test_native_ros_decode_keeps_channels_and_row_padding(encoding):
    msg = SimpleNamespace(encoding=encoding, height=2, width=1, step=4,
                          data=bytes([1, 2, 3, 99, 4, 5, 6, 99]))
    decoded = observations.ros_image_native(msg)
    np.testing.assert_array_equal(decoded, [[[1, 2, 3]], [[4, 5, 6]]])
    assert decoded.flags.c_contiguous
    msg.encoding = "mono8"
    with pytest.raises(ValueError):
        observations.ros_image_native(msg)


def test_three_cameras_state_and_rpc_native_declaration(cfg):
    state, images = readers(cfg)
    obs = observations.ObservationSource(cfg, state, images).read_observation(0)
    np.testing.assert_array_equal(obs["state"], pose(cfg))
    assert set(obs["images"]) == set(observations.CAMERAS)
    assert obs["images"]["cam_high"][0, 0].tolist() == [9, 20, 31]
    assert obs["image_channel_order"] == "native"
    client = rpc_client(cfg)
    client.conn.recv.side_effect = [dict(ok=True, action=np.zeros((32, 14))), diagnostics(cfg)]
    assert client.infer(obs).shape == (32, 14)
    request = client.conn.send.call_args_list[0].args[0]
    assert request["op"] == "infer" and request["obs"]["image_channel_order"] == "native"


@pytest.mark.parametrize("bad_state", [np.zeros(13), np.full(14, np.nan)])
def test_invalid_state_rejected(cfg, bad_state):
    state, images = readers(cfg)
    state.get.return_value = bad_state
    with pytest.raises(ValueError, match="14-dimensional"):
        observations.ObservationSource(cfg, state, images).read_observation(0)


@pytest.mark.parametrize("bad_chunk", [np.zeros((31, 14)), np.zeros((32, 13)), np.full((32, 14), np.nan)])
def test_invalid_rpc_action_rejected(cfg, bad_chunk):
    client = rpc_client(cfg)
    client.conn.recv.return_value = dict(ok=True, action=bad_chunk)
    with pytest.raises(ValueError, match=r"\[32,14\]"):
        client.infer({"image_channel_order": "native"})


@pytest.mark.parametrize("field,value", [("task", "wrong"), ("deployment_sha256", "0" * 64),
    ("control_hz", 50), ("latency_window_ms", 620), ("cuda_graph", False)])
def test_server_task_sha_and_profile_rejected(cfg, field, value):
    client = rpc_client(cfg)
    info = diagnostics(cfg)
    config.validate_server(client, info, require_summary=True)
    info[field] = value
    with pytest.raises(RuntimeError):
        config.validate_server(client, info, require_summary=True)


def test_cross_task_configuration_rejected(cfg):
    cfg["policy"]["task"] = "stack" if cfg["policy"]["task"] == "pack" else "pack"
    with pytest.raises(ValueError, match="SHA256"):
        config.validate_config(cfg)


def test_authkey_comes_only_from_shared_environment(cfg, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(config, "W4A8Async50PolicyClient", factory)
    monkeypatch.delenv("FASTWAM_AUTHKEY", raising=False)
    with pytest.raises(ValueError, match="FASTWAM_AUTHKEY"):
        config.make_client(cfg)
    monkeypatch.setenv("FASTWAM_AUTHKEY", "mock-only-key")
    config.make_client(cfg)
    assert factory.call_args.args == ("127.0.0.1", 18765, b"mock-only-key")
    assert factory.call_args.kwargs == dict(expected_task=cfg["policy"]["task"], expected_sha256=cfg["policy"]["deployment_sha256"])


def test_default_shadow_runs_only_memory_sink(cfg, monkeypatch):
    state, images = readers(cfg)
    client = rpc_client(cfg)
    def reply():
        request = client.conn.send.call_args.args[0]
        return dict(ok=True, action=np.tile(pose(cfg), (32, 1))) if request["op"] == "infer" else diagnostics(cfg)
    client.conn.recv.side_effect = reply
    monkeypatch.setattr(run, "make_client", Mock(return_value=client))
    monkeypatch.setattr(run, "PassivePiperState", Mock(return_value=state))
    monkeypatch.setattr(run, "LiveImages", Mock(return_value=images))
    controller = Mock(side_effect=AssertionError("Must never connect SDK in shadow"))
    executing = Mock(side_effect=AssertionError("Must never construct execution sink in shadow"))
    monkeypatch.setattr(actions, "DualPiperController", controller)
    monkeypatch.setattr(run, "ExecutingActionSink", executing)
    monkeypatch.setattr("builtins.input", Mock(side_effect=AssertionError("No motion prompt in shadow")))
    args = run.parse_args(["--config", "unused", "--max-steps", "4"])
    assert not args.execute_actions and not args.move_to_initial
    report = run.run(args, cfg)
    assert report["steps"] == 4 and report["action_execution_elapsed_s"] is None
    controller.assert_not_called()
    executing.assert_not_called()
    state.close.assert_called_once()
    images.close.assert_called_once()
    client.conn.close.assert_called_once()


@pytest.mark.parametrize("flags", [["--move-to-initial"], ["--execute-actions", "--yes"], ["--max-steps", "0"]])
def test_unsafe_cli_flags_rejected(flags):
    with pytest.raises(SystemExit):
        run.parse_args(["--config", "unused", *flags])


@pytest.mark.parametrize("initialize", [False, True])
def test_confirmation_precedes_connect_enable_and_initial_move(cfg, monkeypatch, initialize):
    events = []
    sink = Mock()
    sink.move_initial.side_effect = lambda: events.append("initial")
    sink.close.side_effect = lambda: events.append("close")
    monkeypatch.setattr(run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: events.append("confirm") or "EXECUTE")
    monkeypatch.setattr(run, "ExecutingActionSink", lambda *args: events.append("connect-enable") or sink)
    flags = ["--execute-actions"] + (["--move-to-initial"] if initialize else [])
    args = run.parse_args(["--config", "unused", *flags])
    with ExitStack() as stack:
        run.prepare_action_sink(args, cfg, Mock(), stack)
        events.append("policy")
    assert events == ["confirm", "connect-enable"] + (["initial"] if initialize else []) + ["policy", "close"]


@pytest.mark.parametrize("tty,answer", [(False, "EXECUTE"), (True, ""), (True, "no")])
def test_no_confirmation_means_no_sdk_construction(cfg, monkeypatch, tty, answer):
    monkeypatch.setattr(run.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr("builtins.input", lambda _: answer)
    constructor = Mock()
    monkeypatch.setattr(run, "ExecutingActionSink", constructor)
    args = run.parse_args(["--config", "unused", "--execute-actions", "--move-to-initial"])
    with ExitStack() as stack, pytest.raises(RuntimeError):
        run.prepare_action_sink(args, cfg, Mock(), stack)
    constructor.assert_not_called()


def test_governor_projection_slew_and_rejected_command_state(cfg):
    initial = pose(cfg)
    governor = actions.build_action_governor(cfg, initial)
    raw = initial.copy()
    raw[0], raw[6], raw[13] = 100, 1, -1
    for t in range(100):
        previous = governor.previous_action.copy()
        shaped = governor.shape(raw, t=t)
        assert np.all(np.abs(shaped - previous) <= governor.max_step_delta + 2e-7)
        assert np.all(shaped >= governor.lower_bounds) and np.all(shaped <= governor.upper_bounds)
        governor.commit(raw_action=raw, shaped_action=shaped, t=t)
    assert shaped[13] == 0
    np.testing.assert_allclose(shaped[0], governor.upper_bounds[0] - .005, atol=1e-6)
    before = governor.previous_action.copy()
    sink = Mock()
    sink.send_action.side_effect = RuntimeError("tracking error")
    runtime = AsyncJointRTCRuntime(policy=Mock(), observation_source=Mock(), action_sink=sink, action_governor=governor)
    with pytest.raises(RuntimeError, match="tracking"):
        runtime._shape_and_send_action(0, raw)
    np.testing.assert_array_equal(governor.previous_action, before)


def test_final_sink_bounds_step_gripper_tracking_and_sdk_rounding(cfg, monkeypatch):
    state, _ = readers(cfg)
    robot = Mock()
    monkeypatch.setattr(actions, "DualPiperController", Mock(return_value=robot))
    sink = actions.ExecutingActionSink(cfg, state)
    sink.send_action(0, pose(cfg))
    assert robot.left_arm.move_js.call_count == 1
    bad = pose(cfg)
    bad[13] = -.000794
    with pytest.raises(RuntimeError, match="bounds"):
        sink.send_action(1, bad)
    bad = pose(cfg)
    bad[0] += .3
    with pytest.raises(RuntimeError, match="jump"):
        sink.send_action(1, bad)
    bad = pose(cfg)
    bad[6] -= .02
    with pytest.raises(RuntimeError, match="Gripper jump"):
        sink.send_action(1, bad)
    state.get.return_value = pose(cfg) + .5
    with pytest.raises(RuntimeError, match="tracking"):
        sink.send_action(1, pose(cfg))
    assert robot.left_arm.move_js.call_count == 1
    gripper = Mock()
    sink._grip(gripper, np.float32(.1), 6)
    gripper.move_gripper_m.assert_called_once_with(value=.1, force=1.)


def test_can_decode_and_fresh_14_dim_state_are_receive_only(cfg, monkeypatch):
    sockets = [Mock(), Mock()]
    factory = Mock(side_effect=sockets)
    monkeypatch.setattr(passive_can.socket, "socket", factory)
    monkeypatch.setattr(passive_can.threading, "Thread", Mock())
    reader = passive_can.PassivePiperState(cfg["robot"])
    sockets[0].bind.assert_called_once_with((cfg["robot"]["left_channel"],))
    np.testing.assert_allclose(reader.decode(0x2A5, struct.pack(">ii", -90000, 180000)), [-np.pi / 2, np.pi])
    np.testing.assert_allclose(reader.decode(0x2A8, struct.pack(">ihBB", 88600, 1000, 0x40, 0)), [.0886])
    with pytest.raises(ValueError, match="mode"):
        reader.decode(0x2A8, struct.pack(">ihBB", 88600, 1000, 0x40, 1))
    with pytest.raises(TimeoutError):
        reader.get(timeout_s=0)
    reader.frames = {(arm, ident): (time.monotonic(), np.full(1 if ident == 0x2A8 else 2, arm))
                     for arm in (0, 1) for ident in (0x2A5, 0x2A6, 0x2A7, 0x2A8)}
    np.testing.assert_array_equal(reader.get(timeout_s=0), [0] * 7 + [1] * 7)
    reader.frames = {k: (stamp - 1, value) for k, (stamp, value) in reader.frames.items()}
    with pytest.raises(TimeoutError):
        reader.get(timeout_s=0)
    reader.close()
    for sock in sockets:
        sock.send.assert_not_called()
        sock.sendall.assert_not_called()


def test_camera_age_skew_and_callback_keep_native(cfg):
    camera = observations.LiveImages.__new__(observations.LiveImages)
    camera.config = cfg["cameras"]
    camera.lock, camera.frames, camera.error = threading.Lock(), {}, None
    camera.rospy = Mock()
    camera.rospy.is_shutdown.return_value = False
    camera.rospy.Time.now.return_value.to_sec.return_value = 100.
    msg = SimpleNamespace(encoding="bgr8", height=1, width=1, step=3, data=bytes([4, 5, 6]),
        header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 99.99)))
    for key in observations.CAMERAS:
        camera._receive(msg, key)
    images, stamps = camera.get()
    assert images["cam_high"].tolist() == [[[4, 5, 6]]]
    camera.frames["cam_high"] = (99., images["cam_high"])
    with pytest.raises(TimeoutError):
        camera.get(timeout_s=.001)


@pytest.mark.parametrize("section,key,value", [("runtime", "control_hz", 50), ("runtime", "action_dim", 13),
    ("runtime", "action_wait_timeout_s", float("nan")), ("robot", "max_action_delta", 0),
    ("safety", "gripper_max_step_delta", .03), ("cameras", "max_age_s", float("inf"))])
def test_invalid_config_fails_before_hardware(cfg, section, key, value):
    cfg[section][key] = value
    with pytest.raises(ValueError):
        config.validate_config(cfg)


def test_native_declaration_is_accepted_by_server_adapter_without_channel_swap(cfg):
    from fastwam_steerquant.adapters.real_robot import RealRobotAdapter
    adapter = RealRobotAdapter.__new__(RealRobotAdapter)
    adapter.task = cfg["policy"]
    adapter.config = {}
    adapter.processor = object()
    adapter.context = adapter.context_mask = object()
    preprocessing = Mock()
    preprocessing.WAMObservation.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
    adapter.preprocessing = preprocessing
    state, images = readers(cfg)
    obs = observations.ObservationSource(cfg, state, images).read_observation(0)
    obs["task"] = cfg["policy"]["task"]
    model = SimpleNamespace(device="cpu", torch_dtype="mock-only")
    result = adapter.infer_kwargs(model, obs, seed=42)
    received = preprocessing.preprocess_real_images.call_args.args[0]
    for camera in observations.CAMERAS:
        np.testing.assert_array_equal(getattr(received, camera), obs["images"][camera])
    assert result["action_horizon"] == 32 and result["num_inference_steps"] == 10


def test_profile_is_checked_after_every_inference(cfg, monkeypatch):
    client = rpc_client(cfg)
    monkeypatch.setenv("FASTWAM_AUTHKEY", "mock-only-key")
    monkeypatch.setattr(config, "W4A8Async50PolicyClient", Mock(return_value=client))
    assert config.make_client(cfg) is client
    wrong = diagnostics(cfg)
    wrong["control_hz"] = 50
    client.conn.recv.side_effect = [dict(ok=True, action=np.zeros((32, 14))), wrong]
    with pytest.raises(RuntimeError, match="40 Hz"):
        client.infer({"image_channel_order": "native"})


def test_all_modules_and_help_import_without_hardware_or_private_sources():
    code = '''
import builtins, importlib, pkgutil, sys
original = builtins.__import__
forbidden = {"qi", "rospy", "sensor_msgs", "cv_bridge", "pyAgxArm", "piper_sdk", "h5py"}
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in forbidden:
        raise AssertionError("Unexpected external import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import fastwam_steerquant.real_robot as package
for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
    importlib.import_module(module.name)
from fastwam_steerquant.real_robot.run import parse_args
try:
    parse_args(["--help"])
except SystemExit as exc:
    assert exc.code == 0
assert not any(name.split(".")[0] in forbidden for name in sys.modules)
'''
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)
