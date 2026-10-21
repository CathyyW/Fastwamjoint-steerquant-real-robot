# Copyright (c) 2026 Rockylix. MIT license; see LICENSE.
"""Three synchronized native-channel images and passive 14-dimensional state."""
import threading
import time

import numpy as np

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def ros_image_native(msg):
    """Equivalent to cv_bridge passthrough for supported 8-bit three-channel images.

    Honor row padding; never reorder channels, resize, or infer a color encoding.
    Reject encodings that would require inventing channels or dropping alpha.
    """
    if msg.encoding not in ("rgb8", "bgr8", "8UC3"):
        raise ValueError(f"Expected native three-channel uint8 image, got {msg.encoding}")
    if msg.height <= 0 or msg.width <= 0 or msg.step < msg.width * 3:
        raise ValueError("Invalid camera image dimensions or row stride")
    data = (np.asarray(msg.data, dtype=np.uint8) if isinstance(msg.data, (list, tuple))
            else np.frombuffer(msg.data, dtype=np.uint8))
    if data.size != msg.height * msg.step:
        raise ValueError("Camera image data length does not match height * step")
    return np.ascontiguousarray(data.reshape(msg.height, msg.step)[:, :msg.width * 3]
                               .reshape(msg.height, msg.width, 3))


class LiveImages:
    def __init__(self, config):
        # Importing this module or --help must never load or start ROS.
        import rospy
        from sensor_msgs.msg import Image
        self.rospy = rospy
        self.config = config["cameras"]
        self.lock = threading.Lock()
        self.frames = {}
        self.error = None
        self.subscribers = []
        rospy.init_node("steerquant_client", anonymous=True, disable_signals=True)
        try:
            for key in CAMERAS:
                self.subscribers.append(rospy.Subscriber(
                    self.config["topics"][key], Image, self._receive,
                    callback_args=key, queue_size=1, buff_size=4 * 1024 * 1024))
        except BaseException:
            self.close()
            raise

    def _receive(self, msg, key):
        try:
            pixels = ros_image_native(msg)
            with self.lock:
                self.frames[key] = (msg.header.stamp.to_sec(), pixels)
        except Exception as exc:
            self.error = exc

    def get(self, timeout_s=2.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self.rospy.is_shutdown():
            if self.error:
                raise RuntimeError("ROS image decoding failed") from self.error
            with self.lock:
                frames = dict(self.frames)
            if set(frames) == set(CAMERAS):
                stamps = [v[0] for v in frames.values()]
                age = self.rospy.Time.now().to_sec() - min(stamps)
                if (0 <= age < self.config["max_age_s"]
                        and max(stamps) - min(stamps) <= self.config["max_skew_s"]):
                    return ({k: frames[k][1] for k in CAMERAS},
                            {k: frames[k][0] for k in CAMERAS})
            time.sleep(.005)
        raise TimeoutError("Three fresh synchronized cameras unavailable")

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.rospy.signal_shutdown("SteerQuant client finished")


class ObservationSource:
    def __init__(self, config, state, images):
        self.config, self.state, self.images = config, state, images

    def read_observation(self, t):
        images, stamps = self.images.get()
        if set(images) != set(CAMERAS):
            raise ValueError("Expected all three named cameras")
        for name in CAMERAS:
            image = np.asarray(images[name])
            if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError(f"{name} must be uint8 [480,640,3]")
        state = np.asarray(self.state.get(), dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 14-dimensional observation state")
        return {"images": images, "state": state,
                "prompt": self.config["policy"]["prompt"],
                "image_channel_order": "native",
                "_runtime_timing": {"camera_timestamps_s": stamps}}
