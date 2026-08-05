"""ZMQ camera client for VLA inference.

Connects to the camera ZMQ server, subscribes to the four robot cameras, and
exposes them under the VLA naming convention, resized to 640x480 (WxH) and in
RGB, so they line up with the LeRobot training dataset.

Source topic -> VLA camera name:
    stereo/left  -> cam_left_high
    stereo/right -> cam_right_high
    hand_left    -> cam_left_wrist
    hand_right   -> cam_right_wrist

Design note (real-time / 30fps):
    The receive thread does NOT decode. It only drains the socket and keeps the
    latest raw JPEG bytes per camera, which is very cheap and lets it keep up
    with the publisher. Decoding + resizing happens lazily in get()/
    get_observation(), so only the frames you actually feed to the policy get
    decoded -- not the whole stream. This removes the single-thread decode
    bottleneck that caps fps when the stereo (high) frames are large.

Examples:
    # preview all four renamed/resized streams in windows
    ./venv/bin/python vla_image_client.py --connect tcp://192.168.123.164:5555

    # headless: print the TRUE per-camera publish rate, no windows
    ./venv/bin/python vla_image_client.py --connect tcp://192.168.123.164:5555 --no-display
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
import zmq


# Source topic (published by the server) -> VLA camera name (matches LeRobot).
TOPIC_TO_VLA: dict[str, str] = {
    "stereo/left":  "cam_left_high",
    "stereo/right": "cam_right_high",
    "hand_left":    "cam_left_wrist",
    "hand_right":   "cam_right_wrist",
}

# Resize every frame to (width, height). Matches resize_width / resize_height
# from convert_unitree_json_to_lerobot.py. cv2.resize wants (W, H); the
# resulting numpy array is (H, W, 3) -> (480, 640, 3).
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
RESIZE = (RESIZE_WIDTH, RESIZE_HEIGHT)


class VLAImageClient:
    """Subscriber that yields resized, renamed RGB frames for a VLA policy.

    Keeps only the latest *raw JPEG* per camera (cheap receive thread) and
    decodes on demand.

    Args:
        connect:   server address, e.g. ``tcp://192.168.123.164:5555``.
        topic_map: source-topic -> vla-name mapping (defaults to TOPIC_TO_VLA).
        resize:    (width, height) to resize to, or None to keep native size.
        rgb:       if True (default) frames are RGB; otherwise BGR.
        rcvhwm:    receive high-water mark (newest-frames buffer depth).
    """

    def __init__(
        self,
        connect: str,
        topic_map: dict[str, str] = TOPIC_TO_VLA,
        resize: tuple[int, int] | None = RESIZE,
        rgb: bool = True,
        rcvhwm: int = 8,
    ):
        self._topic_map = dict(topic_map)
        self._resize = resize
        self._rgb = rgb

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(connect)
        for src in self._topic_map:
            self._sock.setsockopt_string(zmq.SUBSCRIBE, src)
        # Only ever hold the latest frames; drop backlog for real-time use.
        self._sock.setsockopt(zmq.RCVHWM, rcvhwm)

        self._raw: dict[str, bytes] = {}     # latest JPEG bytes, keyed by VLA name
        self._meta: dict[str, dict] = {}
        self._recv_counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # Hot path: just drain the socket and stash the newest raw bytes.
        # No imdecode / resize here -> the thread keeps up with the publisher.
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=100)):
                continue
            topic_b, meta_b, jpg = self._sock.recv_multipart()
            vla_name = self._topic_map.get(topic_b.decode("utf-8"))
            if vla_name is None:
                continue  # ignore depth / unmapped topics
            with self._lock:
                self._raw[vla_name] = jpg
                self._meta[vla_name] = json.loads(meta_b.decode("utf-8"))
                self._recv_counts[vla_name] += 1

    def _decode(self, jpg: bytes) -> np.ndarray | None:
        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if frame is None:
            return None
        if self._resize is not None:
            frame = cv2.resize(frame, self._resize, interpolation=cv2.INTER_AREA)
        if self._rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    @property
    def camera_names(self) -> list[str]:
        """VLA camera names in a stable order."""
        return list(self._topic_map.values())

    def get(self, vla_name: str) -> np.ndarray | None:
        """Decode the latest frame for one VLA camera, or None if not seen."""
        with self._lock:
            jpg = self._raw.get(vla_name)
        return None if jpg is None else self._decode(jpg)

    def get_observation(self) -> dict[str, np.ndarray] | None:
        """All four cameras as ``{vla_name: HxWx3 uint8 RGB}``.

        Returns None until every camera has produced at least one frame.
        Decoding happens here, so this is the only place CPU is spent on pixels.
        """
        names = list(self._topic_map.values())
        with self._lock:
            if not all(n in self._raw for n in names):
                return None
            raws = {n: self._raw[n] for n in names}  # bytes are immutable -> safe to use outside lock
        return {n: self._decode(j) for n, j in raws.items()}

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        """Block until all four cameras have a frame, else raise TimeoutError."""
        deadline = time.time() + timeout_s
        wanted = set(self._topic_map.values())
        while time.time() < deadline:
            with self._lock:
                have = set(self._raw.keys())
            if wanted <= have:
                return
            time.sleep(0.05)
        with self._lock:
            have = set(self._raw.keys())
        missing = ", ".join(sorted(wanted - have)) or "<none>"
        raise TimeoutError(f"Timed out waiting for cameras: {missing}")

    def pop_counts(self) -> dict[str, int]:
        """Snapshot + reset per-camera received-frame counters (= server rate)."""
        with self._lock:
            c = dict(self._recv_counts)
            self._recv_counts.clear()
            return c

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close(linger=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp://192.168.123.164:5555",
                    help="camera server address")
    ap.add_argument("--no-display", action="store_true",
                    help="don't open windows, just measure the publish rate")
    ap.add_argument("--display-fps", type=float, default=30.0,
                    help="cap preview decode/draw rate (Hz)")
    args = ap.parse_args()

    client = VLAImageClient(args.connect)
    print(f"[vla-client] connected to {args.connect}")
    print(f"[vla-client] cameras: {', '.join(client.camera_names)}")
    print(f"[vla-client] resizing every frame to {RESIZE_WIDTH}x{RESIZE_HEIGHT} (WxH)")
    print("[vla-client] waiting for all four cameras...")
    client.wait_ready(timeout_s=15.0)
    print("[vla-client] all cameras ready  (fps below = TRUE publish rate)")

    last_report = time.time()
    frame_period = 1.0 / args.display_fps if args.display_fps > 0 else 0.0
    next_frame = time.time()

    try:
        while True:
            now = time.time()

            if not args.no_display and now >= next_frame:
                obs = client.get_observation()
                if obs is not None:
                    for name, frame in obs.items():
                        cv2.imshow(name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                next_frame = now + frame_period

            if now - last_report >= 2.0:
                counts = client.pop_counts()
                rates = ", ".join(
                    f"{k}={v / (now - last_report):.1f}fps" for k, v in counts.items()
                )
                print(f"[vla-client] {rates}")
                last_report = now

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        client.close()


if __name__ == "__main__":
    main()