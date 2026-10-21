# Copyright (c) 2026 Rockylix. MIT license; see LICENSE.
"""Minimal dual Piper SDK connection; called only after terminal confirmation."""
import time


class DualPiperController:
    def __init__(self, config):
        self.config = config["robot"]
        self.left_arm = self.right_arm = None
        self.left_gripper = self.right_gripper = None

    def connect(self):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        try:
            for side in ("left", "right"):
                cfg = create_agx_arm_config(robot="piper", comm="can",
                    channel=self.config[f"{side}_channel"], bitrate=self.config["bitrate"])
                arm = AgxArmFactory.create_arm(cfg)
                setattr(self, f"{side}_arm", arm)
                gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
                setattr(self, f"{side}_gripper", gripper)
                arm.connect()
            time.sleep(.5)
            if not all(arm.is_ok() for arm in (self.left_arm, self.right_arm)):
                raise RuntimeError("Robot connection status check failed")
            for arm in (self.left_arm, self.right_arm):
                arm.set_flange_vel_acc_limits(**{k: self.config[k] for k in (
                    "max_linear_vel", "max_angular_vel", "max_linear_acc", "max_angular_acc")}, timeout=1.0)
                arm.set_speed_percent(self.config["speed_pct"])
                for _ in range(5):
                    if arm.enable():
                        break
                    time.sleep(.5)
                else:
                    raise RuntimeError("Robot arm enable timeout")
        except BaseException:
            self.close()
            raise

    def close(self):
        for arm in (self.left_arm, self.right_arm):
            if arm is not None and callable(getattr(arm, "disconnect", None)):
                arm.disconnect()
