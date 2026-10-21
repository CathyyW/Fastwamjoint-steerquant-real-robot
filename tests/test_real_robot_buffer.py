# Copyright (c) 2026 Rockylix. MIT license; see src/fastwam_steerquant/real_robot/LICENSE.
import unittest
import numpy as np
from fastwam_steerquant.real_robot.buffer import RealTimeChunkingBuffer

class AsyncBufferTest(unittest.TestCase):
    def test_buffer_fuses_overlapping_chunks_with_recency_weight(self) -> None:
        buffer = RealTimeChunkingBuffer(chunk_size=3, exp_weight_factor=1.0)
        buffer.enqueue(np.ones((3, 2), dtype=np.float32), cursor=0)
        buffer.enqueue(np.full((3, 2), 3.0, dtype=np.float32), cursor=1)

        action = buffer.get_action(1)

        weights = np.exp(np.array([0.0, 1.0], dtype=np.float32))
        expected = (1.0 * weights[0] + 3.0 * weights[1]) / weights.sum()
        self.assertTrue(np.allclose(action, np.full(2, expected, dtype=np.float32)))


    def test_buffer_smoothsteps_new_overlapping_chunk_from_arrival(self) -> None:
        buffer = RealTimeChunkingBuffer(
            chunk_size=32,
            exp_weight_factor=0.2,
            arrival_smoothstep_steps=5,
        )
        buffer.set_control_time(0)
        buffer.enqueue(np.zeros((32, 2), dtype=np.float32), cursor=0)
        buffer.set_control_time(20)
        buffer.enqueue(np.ones((32, 2), dtype=np.float32), cursor=14)

        actual = [float(buffer.get_action(t)[0]) for t in range(20, 25)]

        progress = np.arange(1, 6, dtype=np.float32) / 5.0
        ramps = progress * progress * (3.0 - 2.0 * progress)
        new_rank_weight = float(np.exp(np.float32(0.2)))
        expected = new_rank_weight * ramps / (1.0 + new_rank_weight * ramps)
        self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=1e-6))
        self.assertTrue(all(left < right for left, right in zip(actual, actual[1:])))


    def test_buffer_does_not_reduce_only_available_chunk(self) -> None:
        buffer = RealTimeChunkingBuffer(
            chunk_size=32,
            exp_weight_factor=0.2,
            arrival_smoothstep_steps=5,
        )
        buffer.set_control_time(20)
        buffer.enqueue(np.full((32, 2), 3.0, dtype=np.float32), cursor=14)

        action = buffer.get_action(20)

        self.assertTrue(np.array_equal(action, np.full(2, 3.0, dtype=np.float32)))


    def test_buffer_latest_only_bypasses_arrival_ramp(self) -> None:
        buffer = RealTimeChunkingBuffer(
            chunk_size=32,
            exp_weight_factor=0.2,
            arrival_smoothstep_steps=5,
        )
        buffer.set_control_time(0)
        buffer.enqueue(np.zeros((32, 2), dtype=np.float32), cursor=0)
        buffer.set_control_time(20)
        buffer.enqueue(np.ones((32, 2), dtype=np.float32), cursor=14)

        action, _, cursors = buffer.get_action_snapshot(20, latest_only=True)

        self.assertEqual(cursors, (14,))
        self.assertTrue(np.array_equal(action, np.ones(2, dtype=np.float32)))


    def test_buffer_exposes_atomic_fusion_diagnostics(self) -> None:
        buffer = RealTimeChunkingBuffer(
            chunk_size=4,
            exp_weight_factor=0.2,
            arrival_smoothstep_steps=2,
        )
        buffer.set_control_time(0)
        buffer.enqueue(np.zeros((4, 2), dtype=np.float32), cursor=0)
        buffer.set_control_time(2)
        buffer.enqueue(np.ones((4, 2), dtype=np.float32), cursor=1)

        snapshot = buffer.get_action_fusion_snapshot(2)

        self.assertEqual(snapshot.cursors, (0, 1))
        self.assertEqual(snapshot.arrival_control_t, (0, 2))
        self.assertEqual(snapshot.local_indices, (2, 1))
        self.assertEqual(snapshot.candidate_actions.shape, (2, 2))
        self.assertAlmostEqual(sum(snapshot.weights), 1.0, places=6)
        self.assertIsNotNone(snapshot.action)
