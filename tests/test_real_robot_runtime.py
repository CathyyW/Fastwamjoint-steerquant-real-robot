# Copyright (c) 2026 Rockylix. MIT license; see src/fastwam_steerquant/real_robot/LICENSE.
# Adapted from piper-cosmos3-edge d984db2; see docs/PORTABLE_REAL_ROBOT_CLIENT.md.
import os
import pty
import threading
import time
import unittest

import numpy as np

from fastwam_steerquant.real_robot.runtime import (
    AsyncJointRTCRuntime,
    AsyncJointRTCRuntimeConfig,
    ContinuousActionGovernor,
    RetryableActionSafetyError,
)
from fastwam_steerquant.real_robot.buffer import RecordingActionSink


class _ObservationSource:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def read_observation(self, t):
        self.calls.append((int(t), threading.current_thread().name))
        return {
            "images": {},
            "state": np.zeros(14, dtype=np.float32),
            "prompt": "test",
        }


class _DelayedPolicy:
    def __init__(self, delays, chunk_size=16) -> None:
        self.delays = list(delays)
        self.chunk_size = int(chunk_size)
        self.calls: list[str] = []

    def infer(self, obs):
        self.calls.append(threading.current_thread().name)
        index = len(self.calls) - 1
        delay = self.delays[min(index, len(self.delays) - 1)]
        time.sleep(delay)
        return np.full(
            (self.chunk_size, 14),
            float(index) * 0.01,
            dtype=np.float32,
        )


class _ThreadRecordingSink(RecordingActionSink):
    def __init__(self) -> None:
        super().__init__()
        self.threads: list[str] = []

    def send_action(self, t, action):
        self.threads.append(threading.current_thread().name)
        super().send_action(t, action)


class _RejectPositiveFirstJointSink(RecordingActionSink):
    def send_action(self, t, action):
        if float(action[0]) > 0.5:
            raise RetryableActionSafetyError("first joint is unsafe")
        super().send_action(t, action)


class AsyncJointRTCRuntimeTest(unittest.TestCase):
    @staticmethod
    def _governor(initial_action=None) -> ContinuousActionGovernor:
        initial = (
            np.zeros(14, dtype=np.float32)
            if initial_action is None
            else np.asarray(initial_action, dtype=np.float32)
        )
        lower = np.full(14, -2.0, dtype=np.float32)
        upper = np.full(14, 2.0, dtype=np.float32)
        lower[[6, 13]] = 0.0
        upper[[6, 13]] = 0.1
        max_step = np.full(14, 0.08, dtype=np.float32)
        max_step[[6, 13]] = 0.02
        margin = np.full(14, 0.01, dtype=np.float32)
        margin[[6, 13]] = 0.0
        raw_lower = np.full(14, -2.2, dtype=np.float32)
        raw_upper = np.full(14, 2.2, dtype=np.float32)
        raw_lower[[6, 13]] = [-0.0118, -0.002]
        raw_upper[[6, 13]] = [0.1748, 0.0984]
        raw_tolerance = np.full(14, 0.02, dtype=np.float32)
        raw_tolerance[[6, 13]] = 0.005
        return ContinuousActionGovernor(
            initial_action=initial,
            lower_bounds=lower,
            upper_bounds=upper,
            max_step_delta=max_step,
            bound_margin=margin,
            raw_action_lower_bounds=raw_lower,
            raw_action_upper_bounds=raw_upper,
            raw_action_bound_tolerance=raw_tolerance,
        )

    def test_projects_training_valid_gripper_target_to_hardware_bound(self) -> None:
        initial = np.zeros(14, dtype=np.float32)
        initial[6] = 0.09
        governor = self._governor(initial)
        raw_action = initial.copy()
        raw_action[6] = 0.105575

        shaped = governor.shape(raw_action, t=223)

        self.assertAlmostEqual(float(shaped[6]), 0.1, places=6)
        governor.commit(raw_action=raw_action, shaped_action=shaped, t=223)
        self.assertEqual(governor.bound_projected_steps, 1)

    def test_projects_gripper_target_outside_training_distribution(self) -> None:
        initial = np.zeros(14, dtype=np.float32)
        initial[6] = 0.09
        governor = self._governor(initial)
        raw_action = np.zeros(14, dtype=np.float32)
        raw_action[6] = 0.18
        raw_action[:6] = initial[:6]
        raw_action[13] = initial[13]

        shaped = governor.shape(raw_action, t=223)
        governor.commit(raw_action=raw_action, shaped_action=shaped, t=223)

        self.assertAlmostEqual(float(shaped[6]), 0.1, places=6)
        report = governor.report()
        self.assertEqual(report["raw_out_of_distribution_steps"], 1)
        self.assertEqual(report["raw_ood_dimension_counts"][6], 1)

    def test_projects_large_joint_distribution_excursion_without_rejecting(
        self,
    ) -> None:
        governor = self._governor()
        raw_action = np.zeros(14, dtype=np.float32)
        raw_action[0] = 2.3

        shaped = governor.shape(raw_action, t=4)
        governor.commit(raw_action=raw_action, shaped_action=shaped, t=4)

        self.assertAlmostEqual(float(shaped[0]), 0.08, places=6)
        self.assertEqual(governor.report()["raw_out_of_distribution_steps"], 1)

    def test_rejects_non_finite_action(self) -> None:
        governor = self._governor()
        raw_action = np.zeros(14, dtype=np.float32)
        raw_action[6] = 0.18
        raw_action[0] = np.nan

        with self.assertRaisesRegex(ValueError, "only finite values"):
            governor.shape(raw_action, t=223)

    def test_projects_small_raw_extrapolation_observed_in_sync_runtime(
        self,
    ) -> None:
        governor = self._governor()
        raw_action = np.zeros(14, dtype=np.float32)
        raw_action[13] = -0.002244881

        shaped = governor.shape(raw_action, t=10)

        self.assertEqual(float(shaped[13]), 0.0)

    def test_projects_negative_gripper_value_from_async_failure(self) -> None:
        governor = self._governor()
        raw_action = np.zeros(14, dtype=np.float32)
        raw_action[6] = -0.019778

        shaped = governor.shape(raw_action, t=267)
        governor.commit(raw_action=raw_action, shaped_action=shaped, t=267)

        self.assertEqual(float(shaped[6]), 0.0)
        report = governor.report()
        self.assertEqual(report["raw_out_of_distribution_steps"], 1)
        self.assertEqual(report["raw_ood_dimension_counts"][6], 1)

    def test_producer_infers_off_control_thread_and_runs_multiple_chunks(self) -> None:
        policy = _DelayedPolicy([0.002], chunk_size=16)
        source = _ObservationSource()
        sink = _ThreadRecordingSink()
        runtime = AsyncJointRTCRuntime(
            policy=policy,
            observation_source=source,
            action_sink=sink,
            config=AsyncJointRTCRuntimeConfig(
                chunk_size=16,
                control_hz=100.0,
                max_steps=20,
                startup_timeout_s=1.0,
                action_wait_timeout_s=0.05,
                producer_join_timeout_s=1.0,
                min_latency_headroom_steps=1.0,
                sleep=True,
            ),
        )

        report = runtime.run()

        self.assertEqual(report["mode"], "async_single_producer")
        self.assertEqual(report["steps"], 20)
        self.assertGreaterEqual(report["num_inferences"], 2)
        self.assertEqual(report["starved_waits"], 0)
        self.assertTrue(all(name == "wam-joint-producer" for name in policy.calls))
        self.assertTrue(
            all(name == "wam-joint-producer" for _, name in source.calls)
        )
        self.assertTrue(all(name != "wam-joint-producer" for name in sink.threads))
        trace = runtime.action_trace_records()
        self.assertEqual(len(trace), 20)
        self.assertEqual(
            [record.control_t for record in trace],
            list(range(20)),
        )
        self.assertTrue(all(record.cursors for record in trace))
        self.assertTrue(
            all(
                len(record.cursors)
                == len(record.counter_local_indices)
                == len(record.wall_time_local_indices)
                == len(record.weights)
                for record in trace
            )
        )
        inference_records = runtime.inference_chunk_records()
        self.assertTrue(
            all(
                record.inference_finished_monotonic_ns
                >= record.inference_started_monotonic_ns
                >= record.observation_finished_monotonic_ns
                >= record.observation_started_monotonic_ns
                for record in inference_records
            )
        )

    def test_spacebar_listener_requests_stop_without_enter(self) -> None:
        runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0]),
            observation_source=_ObservationSource(),
            action_sink=RecordingActionSink(),
        )
        master_fd, slave_fd = pty.openpty()
        thread = threading.Thread(
            target=runtime._spacebar_listener_loop,
            args=(slave_fd,),
        )
        try:
            thread.start()
            time.sleep(0.05)
            os.write(master_fd, b" ")
            thread.join(timeout=1.0)
        finally:
            runtime.request_stop("test_cleanup")
            thread.join(timeout=1.0)
            os.close(master_fd)
            os.close(slave_fd)

        self.assertFalse(thread.is_alive())
        self.assertTrue(runtime.stop_event.is_set())
        self.assertEqual(runtime.report()["stop_reason"], "spacebar")

    def test_requested_stop_reports_first_action_to_end_elapsed_time(self) -> None:
        policy = _DelayedPolicy([0.001], chunk_size=64)
        sink = RecordingActionSink()
        runtime = AsyncJointRTCRuntime(
            policy=policy,
            observation_source=_ObservationSource(),
            action_sink=sink,
            config=AsyncJointRTCRuntimeConfig(
                chunk_size=64,
                control_hz=100.0,
                max_steps=100,
                startup_timeout_s=1.0,
                producer_join_timeout_s=1.0,
                sleep=True,
            ),
        )
        result: dict[str, object] = {}

        def run_runtime() -> None:
            result.update(runtime.run())

        thread = threading.Thread(target=run_runtime)
        thread.start()
        deadline = time.monotonic() + 1.0
        while not sink.records and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(sink.records)
        time.sleep(0.02)
        runtime.request_stop("spacebar")
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result["stop_reason"], "spacebar")
        self.assertLess(int(result["steps"]), 100)
        self.assertGreaterEqual(
            float(result["action_execution_elapsed_s"]),
            0.015,
        )

    def test_producer_failure_is_propagated_before_control_starts(self) -> None:
        class BadPolicy:
            def infer(self, obs):
                return np.zeros((32, 13), dtype=np.float32)

        runtime = AsyncJointRTCRuntime(
            policy=BadPolicy(),
            observation_source=_ObservationSource(),
            action_sink=RecordingActionSink(),
            config=AsyncJointRTCRuntimeConfig(
                max_steps=1,
                startup_timeout_s=1.0,
                producer_join_timeout_s=1.0,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            runtime.run()

    def test_rejects_steady_latency_without_chunk_headroom(self) -> None:
        runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0, 0.03], chunk_size=4),
            observation_source=_ObservationSource(),
            action_sink=RecordingActionSink(),
            config=AsyncJointRTCRuntimeConfig(
                chunk_size=4,
                control_hz=100.0,
                max_steps=8,
                startup_timeout_s=1.0,
                action_wait_timeout_s=0.1,
                producer_join_timeout_s=1.0,
                min_latency_headroom_steps=0.0,
                sleep=True,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            runtime.run()
        self.assertIsNotNone(runtime.producer_error)
        self.assertIn(
            "cannot safely sustain",
            str(runtime.producer_error),
        )

    def test_startup_latency_does_not_consume_first_chunk_horizon(self) -> None:
        runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.06, 0.005], chunk_size=4),
            observation_source=_ObservationSource(),
            action_sink=RecordingActionSink(),
            config=AsyncJointRTCRuntimeConfig(
                chunk_size=4,
                control_hz=100.0,
                max_steps=4,
                startup_timeout_s=1.0,
                action_wait_timeout_s=0.1,
                producer_join_timeout_s=1.0,
                min_latency_headroom_steps=0.0,
                sleep=True,
            ),
        )

        report = runtime.run()

        self.assertEqual(report["steps"], 4)
        self.assertGreater(report["inference_latency_s"]["max"], 0.04)

    def test_latency_budget_uses_two_consecutive_samples(self) -> None:
        runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0]),
            observation_source=_ObservationSource(),
            action_sink=RecordingActionSink(),
            config=AsyncJointRTCRuntimeConfig(
                chunk_size=32,
                control_hz=30.0,
                min_latency_headroom_steps=1.0,
            ),
        )

        # The real trace's 469 ms + 526 ms window occupies 29.85 control
        # steps and is safe even though incorrectly doubling 526 ms gives 31.56.
        runtime._validate_latency_window(
            previous_latency_s=0.469,
            current_latency_s=0.526,
        )

        with self.assertRaisesRegex(RuntimeError, "31.56 steps > 31.00"):
            runtime._validate_latency_window(
                previous_latency_s=0.526,
                current_latency_s=0.526,
            )

    def test_safety_retry_uses_newly_arrived_latest_chunk_only(self) -> None:
        sink = _RejectPositiveFirstJointSink()
        runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0]),
            observation_source=_ObservationSource(),
            action_sink=sink,
            config=AsyncJointRTCRuntimeConfig(
                fresh_chunk_grace_s=0.1,
                action_wait_timeout_s=0.1,
                producer_join_timeout_s=1.0,
                sleep=False,
            ),
        )
        old_chunk = np.ones((32, 14), dtype=np.float32)
        runtime.buffer.enqueue(old_chunk, cursor=0)

        def enqueue_fresh_chunk() -> None:
            time.sleep(0.01)
            runtime.buffer.enqueue(
                np.zeros((32, 14), dtype=np.float32),
                cursor=3,
            )
            runtime.chunk_ready_event.set()

        thread = threading.Thread(target=enqueue_fresh_chunk)
        thread.start()
        action = runtime._select_and_send_action(3)
        thread.join()

        self.assertTrue(np.array_equal(action, np.zeros(14, dtype=np.float32)))
        self.assertEqual(len(sink.records), 1)
        self.assertEqual(runtime.safety_retry_attempts, 1)
        self.assertEqual(runtime.safety_retry_successes, 1)
        self.assertEqual(runtime.safety_retry_timeouts, 0)

    def test_safety_retry_preserves_stop_without_new_chunk(self) -> None:
        runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0]),
            observation_source=_ObservationSource(),
            action_sink=_RejectPositiveFirstJointSink(),
            config=AsyncJointRTCRuntimeConfig(
                fresh_chunk_grace_s=0.005,
                action_wait_timeout_s=0.1,
                producer_join_timeout_s=1.0,
                sleep=False,
            ),
        )
        runtime.buffer.enqueue(np.ones((32, 14), dtype=np.float32), cursor=0)

        with self.assertRaisesRegex(
            RetryableActionSafetyError,
            "first joint is unsafe",
        ):
            runtime._select_and_send_action(3)

        self.assertEqual(runtime.safety_retry_attempts, 1)
        self.assertEqual(runtime.safety_retry_successes, 0)
        self.assertEqual(runtime.safety_retry_timeouts, 1)

    def test_safety_retry_skips_rejected_fresh_chunk_until_safe_one(self) -> None:
        sink = _RejectPositiveFirstJointSink()
        runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0]),
            observation_source=_ObservationSource(),
            action_sink=sink,
            config=AsyncJointRTCRuntimeConfig(
                fresh_chunk_grace_s=0.1,
                action_wait_timeout_s=0.1,
                producer_join_timeout_s=1.0,
                sleep=False,
            ),
        )
        runtime.buffer.enqueue(np.ones((32, 14), dtype=np.float32), cursor=0)

        def enqueue_two_fresh_chunks() -> None:
            time.sleep(0.01)
            runtime.buffer.enqueue(
                np.full((32, 14), 0.8, dtype=np.float32),
                cursor=3,
            )
            runtime.chunk_ready_event.set()
            time.sleep(0.01)
            runtime.buffer.enqueue(
                np.zeros((32, 14), dtype=np.float32),
                cursor=5,
            )
            runtime.chunk_ready_event.set()

        thread = threading.Thread(target=enqueue_two_fresh_chunks)
        thread.start()
        action = runtime._select_and_send_action(5)
        thread.join()

        self.assertTrue(np.array_equal(action, np.zeros(14, dtype=np.float32)))
        self.assertEqual(len(sink.records), 1)
        self.assertEqual(runtime.safety_retry_attempts, 1)
        self.assertEqual(runtime.safety_retry_rejections, 1)
        self.assertEqual(runtime.safety_retry_successes, 1)
        self.assertEqual(runtime.safety_retry_timeouts, 0)

    def test_governor_projects_small_bound_overshoot_and_limits_slew(self) -> None:
        initial = np.zeros(14, dtype=np.float32)
        initial[3] = 1.98
        governor = self._governor(initial)
        raw_action = initial.copy()
        raw_action[0] = 0.40
        raw_action[3] = 2.005
        raw_action[6] = 0.08

        shaped = governor.shape(raw_action, t=10)

        self.assertAlmostEqual(float(shaped[0]), 0.08, places=6)
        self.assertAlmostEqual(float(shaped[3]), 1.99, places=6)
        self.assertAlmostEqual(float(shaped[6]), 0.02, places=6)
        governor.commit(raw_action=raw_action, shaped_action=shaped, t=10)
        report = governor.report()
        self.assertEqual(report["shaped_steps"], 1)
        self.assertEqual(report["slew_limited_steps"], 1)
        self.assertEqual(report["bound_projected_steps"], 1)
        self.assertLessEqual(float(report["max_sent_delta"]), 0.080001)

    def test_governor_is_optional_and_only_changes_attached_runtime(self) -> None:
        raw_chunk = np.full((32, 14), 0.4, dtype=np.float32)
        raw_chunk[:, [6, 13]] = 0.04

        plain_sink = RecordingActionSink()
        plain_runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0]),
            observation_source=_ObservationSource(),
            action_sink=plain_sink,
            config=AsyncJointRTCRuntimeConfig(sleep=False),
        )
        plain_runtime.buffer.enqueue(raw_chunk, cursor=0)
        plain_action = plain_runtime._select_and_send_action(0)

        shaped_sink = RecordingActionSink()
        shaped_runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0]),
            observation_source=_ObservationSource(),
            action_sink=shaped_sink,
            config=AsyncJointRTCRuntimeConfig(sleep=False),
            action_governor=self._governor(),
        )
        shaped_runtime.buffer.enqueue(raw_chunk, cursor=0)
        shaped_action = shaped_runtime._select_and_send_action(0)

        self.assertTrue(np.allclose(plain_action[:6], 0.4))
        self.assertTrue(np.allclose(plain_action[7:13], 0.4))
        self.assertEqual(float(plain_action[6]), float(np.float32(0.04)))
        self.assertEqual(float(plain_action[13]), float(np.float32(0.04)))
        self.assertTrue(np.allclose(shaped_action[:6], 0.08))
        self.assertTrue(np.allclose(shaped_action[7:13], 0.08))
        self.assertAlmostEqual(float(shaped_action[6]), 0.02, places=6)
        self.assertAlmostEqual(float(shaped_action[13]), 0.02, places=6)

    def test_successful_send_publishes_next_pending_control_step(self) -> None:
        runtime = AsyncJointRTCRuntime(
            policy=_DelayedPolicy([0.0]),
            observation_source=_ObservationSource(),
            action_sink=RecordingActionSink(),
            config=AsyncJointRTCRuntimeConfig(sleep=False),
        )
        runtime.buffer.enqueue(
            np.zeros((32, 14), dtype=np.float32),
            cursor=0,
        )

        runtime._select_and_send_action(3)

        self.assertEqual(runtime.buffer.get_control_time(), 4)


if __name__ == "__main__":
    unittest.main()
