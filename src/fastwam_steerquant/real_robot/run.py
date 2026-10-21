"""Run the portable pack/stack W4A8 client (default: passive shadow at 40 Hz)."""
import argparse
from contextlib import ExitStack
import json
import sys

from .actions import ExecutingActionSink, ShadowActionSink, build_action_governor
from .config import load_config, make_client, validate_server
from .observations import LiveImages, ObservationSource
from .passive_can import PassivePiperState
from .runtime import AsyncJointRTCRuntime, AsyncJointRTCRuntimeConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Portable live example YAML")
    parser.add_argument("--host", help="Override model server host")
    parser.add_argument("--port", type=int, help="Override model server port")
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--execute-actions", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable robot only after an interactive terminal confirmation")
    parser.add_argument("--move-to-initial", action=argparse.BooleanOptionalAction, default=False,
                        help="Move to YAML initial pose after confirmation; requires --execute-actions")
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.move_to_initial and not args.execute_actions:
        parser.error("--move-to-initial requires --execute-actions")
    if args.port is not None and not 0 < args.port < 65536:
        parser.error("--port must be in 1..65535")
    return args


def prepare_action_sink(args, config, state, stack):
    if args.move_to_initial and not args.execute_actions:
        raise ValueError("--move-to-initial requires --execute-actions")
    if not args.execute_actions:
        return ShadowActionSink()
    # No SDK construction, connect, enable, or initialization before this gate.
    if not sys.stdin.isatty():
        raise RuntimeError("Real motion requires confirmation in an interactive terminal")
    initialization = "including moving to the configured initial pose" if args.move_to_initial else "without initial-pose movement"
    answer = input(f"Enable both arms and execute real actions ({initialization})? Type EXECUTE: ")
    if answer != "EXECUTE":
        raise RuntimeError("Robot motion was not confirmed")
    sink = ExecutingActionSink(config, state)
    stack.callback(sink.close)
    if args.move_to_initial:
        sink.move_initial()
    return sink


def run(args, config):
    with ExitStack() as stack:
        client = stack.enter_context(make_client(config, host=args.host, port=args.port))
        client.reset()
        validate_server(client, client._request({"op": "diagnostics"}))
        state = PassivePiperState(config["robot"])
        stack.callback(state.close)
        images = LiveImages(config)
        stack.callback(images.close)
        images.get(timeout_s=config["cameras"]["startup_timeout_s"])
        source = ObservationSource(config, state, images)
        initial = source.read_observation(0)["state"]
        # Validate shaping and the observed start state before any enable.
        build_action_governor(config, initial)
        print("SteerQuant W4A8: 40 Hz, [32,14]; " +
              ("REAL EXECUTION requested" if args.execute_actions else "SHADOW: no enable, no motion commands"), flush=True)
        sink = prepare_action_sink(args, config, state, stack)
        initial = sink.previous if args.execute_actions else source.read_observation(0)["state"]
        governor = build_action_governor(config, initial)
        runtime = AsyncJointRTCRuntime(policy=client, observation_source=source, action_sink=sink,
            action_governor=governor, config=AsyncJointRTCRuntimeConfig(
                **config["runtime"], max_steps=args.max_steps,
                prompt=config["policy"]["prompt"], stop_on_spacebar=True))
        # The runtime always joins its sole producer before this stack closes
        # observation readers and the RPC connection. No files are written.
        try:
            report = runtime.run()
        except KeyboardInterrupt:
            report = runtime.report()
        print(json.dumps(report, allow_nan=False), flush=True)
        return report


def main(argv=None):
    args = parse_args(argv)
    return run(args, load_config(args.config))


if __name__ == "__main__":
    main()
