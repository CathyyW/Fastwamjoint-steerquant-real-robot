# Copyright (c) 2026 Rockylix. MIT license; see LICENSE.
"""Receive-only SocketCAN feedback; no SDK and no CAN transmit API."""
import select
import socket
import struct
import threading
import time

import numpy as np


class PassivePiperState:
    """Read unsolicited Piper joint/gripper feedback; never transmit CAN frames."""
    FRAME = struct.Struct("=IB3x8s")

    def __init__(self, robot):
        self.max_age_s = float(robot["state_max_age_s"])
        self.frames = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.sockets = []
        self.thread = None
        try:
            for side in ("left", "right"):
                conn = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                self.sockets.append(conn)
                conn.bind((robot[f"{side}_channel"],))
                conn.setblocking(False)
            self.thread = threading.Thread(target=self._receive, name="passive-piper-state", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def decode(can_id, data):
        if len(data) != 8:
            raise ValueError("Piper feedback must contain 8 bytes")
        if can_id in (0x2A5, 0x2A6, 0x2A7):
            return np.deg2rad(np.asarray(struct.unpack(">ii", data), dtype=np.float64) / 1000.)
        if can_id == 0x2A8:
            if data[7] != 0:
                raise ValueError("Expected Piper gripper width feedback, got a nonzero mode")
            return np.asarray([struct.unpack(">i", data[:4])[0] * 1e-6])
        raise ValueError(f"Unexpected feedback CAN ID {can_id:#x}")

    def _receive(self):
        try:
            while not self.stop.is_set():
                ready, _, _ = select.select(self.sockets, [], [], 0.05)
                for conn in ready:
                    raw = conn.recv(self.FRAME.size)
                    can_id, size, data = self.FRAME.unpack(raw)
                    if can_id not in (0x2A5, 0x2A6, 0x2A7, 0x2A8):
                        continue
                    value = self.decode(can_id, data[:size])
                    with self.lock:
                        self.frames[(self.sockets.index(conn), can_id)] = (time.monotonic(), value)
        except BaseException as exc:
            self.error = exc

    def get(self, timeout_s=2.0):
        deadline = time.monotonic() + timeout_s
        while True:
            if self.error is not None:
                raise RuntimeError("Passive CAN feedback reader failed") from self.error
            with self.lock:
                frames = dict(self.frames)
            if len(frames) == 8 and time.monotonic() - min(v[0] for v in frames.values()) < self.max_age_s:
                return np.concatenate([frames[(arm, ident)][1] for arm in (0, 1)
                                       for ident in (0x2A5, 0x2A6, 0x2A7, 0x2A8)]).astype(np.float32)
            if time.monotonic() >= deadline:
                raise TimeoutError("Missing or stale passive CAN joint/gripper feedback; no state fallback")
            time.sleep(0.005)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        for conn in self.sockets:
            conn.close()
