# Copyright (c) 2026 Rockylix. MIT license; see LICENSE.
"""Experiment shaping, final command checks, and separate shadow/SDK sinks."""
import time

import numpy as np

from .buffer import RecordingActionSink
from .controller import DualPiperController
from .runtime import ContinuousActionGovernor, RetryableActionSafetyError


def per_dimension(config, joint_key, gripper_key):
    values = np.full(14, float(config[joint_key]), dtype=np.float32)
    values[[6, 13]] = float(config[gripper_key])
    return values


def build_action_governor(config, initial_action):
    cfg, robot = config["safety"], config["robot"]
    limits = per_dimension(cfg, "joint_max_step_delta", "gripper_max_step_delta")
    if (np.max(limits) > robot["max_action_delta"] + 1e-7
            or np.max(limits[[6, 13]]) > robot["max_gripper_delta"] + 1e-7
            or np.mean(limits) > robot["action_safety_threshold"]):
        raise ValueError("Action shaping step limits exceed final sink safety limits")
    return ContinuousActionGovernor(
        initial_action=initial_action,
        lower_bounds=cfg["lower_bounds"], upper_bounds=cfg["upper_bounds"],
        max_step_delta=limits,
        bound_margin=per_dimension(cfg, "joint_bound_margin", "gripper_bound_margin"),
        raw_action_lower_bounds=cfg["raw_action_lower_bounds"],
        raw_action_upper_bounds=cfg["raw_action_upper_bounds"],
        raw_action_bound_tolerance=per_dimension(
            cfg, "joint_raw_action_bound_tolerance", "gripper_raw_action_bound_tolerance"))


class ShadowActionSink(RecordingActionSink):
    """Memory-only sink; never constructs an SDK controller or transmits CAN."""
    execute_actions = False


class ExecutingActionSink(RecordingActionSink):
    execute_actions = True

    def __init__(self, config, state):
        super().__init__()
        self.config, self.state = config, state
        self.lower = np.asarray(config["safety"]["lower_bounds"], dtype=np.float32)
        self.upper = np.asarray(config["safety"]["upper_bounds"], dtype=np.float32)
        self.previous = np.asarray(state.get(), dtype=np.float32)
        self._check_bounds(self.previous)
        self.robot = DualPiperController(config)
        self.robot.connect()

    def _grip(self, gripper, width, index):
        # Preserve float32 upper-bound rounding guard for the SDK's float check.
        cfg = self.config["safety"]
        width = min(float(cfg["upper_bounds"][index]),
                    max(float(cfg["lower_bounds"][index]), float(width)))
        force = self.config["robot"]["gripper_force"]
        if callable(getattr(gripper, "move_gripper_m", None)):
            gripper.move_gripper_m(value=width, force=force)
        else:
            gripper.move_gripper(width=width, force=force)

    def _command(self, action):
        self.robot.left_arm.move_js(action[:6].tolist())
        self._grip(self.robot.left_gripper, action[6], 6)
        self.robot.right_arm.move_js(action[7:13].tolist())
        self._grip(self.robot.right_gripper, action[13], 13)

    def _check_bounds(self, action):
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Invalid robot action; expected finite [14]")
        invalid = np.flatnonzero((action < self.lower) | (action > self.upper))
        if invalid.size:
            raise RetryableActionSafetyError(
                f"Action outside Piper joint/gripper bounds at dimensions {invalid.tolist()}")

    def move_initial(self):
        robot = self.config["robot"]
        target = np.asarray(robot["left_init_position"] + robot["right_init_position"], dtype=np.float32)
        self._check_bounds(target)
        current = self.state.get()
        for action in np.linspace(current, target, robot["initial_steps"]):
            self.send_action(-1, action)
            time.sleep(robot["initial_period_s"])
        self.records.clear()

    def send_action(self, t, action):
        action = np.asarray(action, dtype=np.float32)
        self._check_bounds(action)
        current = self.state.get(timeout_s=0)
        cfg = self.config["robot"]
        delta = np.abs(action - self.previous)
        if np.mean(delta) > cfg["action_safety_threshold"]:
            raise RetryableActionSafetyError("Action jump exceeds baseline safety threshold")
        if np.max(delta) > cfg["max_action_delta"] + 1e-7:
            raise RetryableActionSafetyError("Maximum action jump exceeds safety threshold")
        if np.max(delta[[6, 13]]) > cfg["max_gripper_delta"] + 1e-7:
            raise RetryableActionSafetyError("Gripper jump exceeds safety threshold")
        if np.mean(np.abs(current - self.previous)) > cfg["state_safety_threshold"]:
            raise RuntimeError("Robot tracking error exceeds baseline safety threshold")
        self._command(action)
        self.previous = action.copy()
        super().send_action(t, action)

    def close(self):
        self.robot.close()
