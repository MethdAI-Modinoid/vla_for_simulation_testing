"""ZMQ camera client for VLA data collection.

Connects to the camera ZMQ server, subscribes to the robot cameras, and exposes
them under the data-collection naming convention, resized to 640x480 (WxH) and
in RGB so they line up with the LeRobot training dataset.

For data collection we only use three of the four published cameras:

    Source topic  -> dataset image key
    stereo/right  -> ego_view    (main ego view)
    hand_left     -> ego_left    (left wrist)
    hand_right    -> ego_right   (right wrist)

(``stereo/left`` / ``cam_left_high`` is intentionally dropped.)

Two wire protocols are accepted, detected per message by part count:

    3 parts  ``[topic, meta, jpg]``  -- the real-robot camera server. Topics are
             the ``VLA_DATA_TOPIC_MAP`` keys above and the JPEG is BGR.
    1 part   msgpack ``{"timestamps": {...}, "images": {...}}`` -- the
             ``SensorServer`` protocol used by both MuJoCo sims (see
             gr00t_wbc.control.sensor.sensor_server). Its image keys are
             already the dataset keys and its frames are RGB.

Accepting both is what lets the same exporter record from the robot rig and
from a sim without a flag. Note the socket therefore subscribes to "" and
filters topics in software -- a prefix subscription would silently drop every
msgpack message, since those start with msgpack bytes rather than a topic.

Design note (real-time / 30fps):
    The receive thread does NOT decode. It only drains the socket and keeps the
    latest raw JPEG bytes per camera, which is very cheap and lets it keep up
    with the publisher. Decoding + resizing happens lazily in read()/get(), so
    only the frames actually fed to the policy get decoded -- not the whole
    stream. This removes the single-thread decode bottleneck that caps fps when
    the stereo (high) frames are large.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


# Source topic (published by the server) -> dataset image key (matches the
# feature names registered in gr00t_wbc.data.utils.get_dataset_features).
VLA_DATA_TOPIC_MAP: Dict[str, str] = {
    "stereo/right": "ego_view",   # main ego view  (cam_right_high)
    "hand_left":    "ego_left",   # left wrist      (cam_left_wrist)
    "hand_right":   "ego_right",  # right wrist     (cam_right_wrist)
}

# Resize every frame to (width, height). cv2.resize wants (W, H); the resulting
# numpy array is (H, W, 3) -> (480, 640, 3), matching RS_VIEW_CAMERA_* dims.
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
RESIZE = (RESIZE_WIDTH, RESIZE_HEIGHT)


class VLAImageClient:
    """Subscriber that yields resized, renamed RGB frames for VLA data collection.

    Keeps only the latest *raw JPEG* per camera (cheap receive thread) and
    decodes on demand.

    Args:
        connect:   server address, e.g. ``tcp://192.168.123.164:5555``.
        topic_map: source-topic -> dataset-key mapping (defaults to
                   VLA_DATA_TOPIC_MAP).
        resize:    (width, height) to resize to, or None to keep native size.
        rgb:       if True (default) frames are RGB; otherwise BGR.
        rcvhwm:    receive high-water mark (newest-frames buffer depth).
    """

    def __init__(
        self,
        connect: str,
        topic_map: Dict[str, str] = VLA_DATA_TOPIC_MAP,
        resize: Optional[tuple[int, int]] = RESIZE,
        rgb: bool = True,
        rcvhwm: int = 8,
    ):
        self._topic_map = dict(topic_map)
        self._wanted_keys = set(self._topic_map.values())
        self._resize = resize
        self._rgb = rgb

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        # Socket options only take effect if set before connect().
        # Subscribe to everything and filter topics in _loop: a prefix
        # subscription would drop every single-part msgpack message.
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        # Only ever hold the latest frames; drop backlog for real-time use.
        self._sock.setsockopt(zmq.RCVHWM, rcvhwm)
        self._sock.connect(connect)

        # Latest *undecoded* payload per dataset key, as (kind, payload).
        self._raw: Dict[str, Tuple[str, Any]] = {}
        self._meta: Dict[str, dict] = {}
        self._recv_time: Dict[str, float] = {}  # wall-clock time the frame arrived
        self._recv_counts: Dict[str, int] = defaultdict(int)
        self._total_recv = 0                    # every msg off the socket, pre-filter
        self._seen_topics: set = set()          # raw topic strings actually published
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
            parts = self._sock.recv_multipart()
            with self._lock:
                self._total_recv += 1
            if len(parts) == 3:
                self._handle_multipart(parts)
            elif len(parts) == 1:
                self._handle_msgpack(parts[0])
            else:
                # Neither protocol -- record it so debug_status() surfaces it.
                with self._lock:
                    self._seen_topics.add(f"<{len(parts)}-part message>")

    def _handle_multipart(self, parts: list) -> None:
        """Real-robot protocol: ``[topic, meta, jpg]``, one camera per message."""
        topic_b, meta_b, jpg = parts
        topic = topic_b.decode("utf-8", errors="replace")
        with self._lock:
            self._seen_topics.add(topic)
        key = self._topic_map.get(topic)
        if key is None:
            return  # ignore depth / unmapped topics (e.g. stereo/left)
        with self._lock:
            self._raw[key] = ("jpg_bgr", jpg)
            self._meta[key] = json.loads(meta_b.decode("utf-8"))
            self._recv_time[key] = time.time()
            self._recv_counts[key] += 1

    def _handle_msgpack(self, packed: bytes) -> None:
        """SensorServer protocol: one msgpack dict carrying every camera.

        Keys are already dataset keys and frames are already RGB, so no topic
        mapping and no colour conversion are applied. Values are either a
        base64 JPEG string (ImageUtils.encode_image) or a bare ndarray.
        """
        try:
            msg = msgpack.unpackb(packed, object_hook=mnp.decode)
        except Exception:
            with self._lock:
                self._seen_topics.add("<undecodable 1-part message>")
            return
        if not isinstance(msg, dict) or not isinstance(msg.get("images"), dict):
            with self._lock:
                self._seen_topics.add("<1-part message, not an image dict>")
            return
        timestamps = msg.get("timestamps") or {}
        arrival = time.time()
        with self._lock:
            for key, value in msg["images"].items():
                self._seen_topics.add(key)
                if key not in self._wanted_keys:
                    continue  # e.g. a sim publishing extra debug cameras
                self._raw[key] = (
                    "b64_rgb" if isinstance(value, str) else "array_rgb",
                    value,
                )
                self._meta[key] = {}
                # Prefer the publisher's capture time: the exporter diffs these
                # against the proprio timestamp to record img_state_delta.
                self._recv_time[key] = timestamps.get(key, arrival)
                self._recv_counts[key] += 1

    def _decode(self, payload: Tuple[str, Any]) -> Optional[np.ndarray]:
        """Decode one stashed payload to the requested colour order.

        ``jpg_bgr`` frames come off the wire as BGR (the robot rig encodes BGR);
        the msgpack kinds are already RGB. Both converge on self._rgb here.
        """
        kind, value = payload
        if kind == "array_rgb":
            frame = value
            native_rgb = True
        else:
            raw = base64.b64decode(value) if kind == "b64_rgb" else value
            arr = np.frombuffer(raw, dtype=np.uint8)
            # imdecode just reverses the JPEG; it does not reinterpret colour,
            # so an RGB-encoded frame decodes back to RGB.
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            native_rgb = kind == "b64_rgb"
        if frame is None:
            return None
        if self._resize is not None and (frame.shape[1], frame.shape[0]) != self._resize:
            frame = cv2.resize(frame, self._resize, interpolation=cv2.INTER_AREA)
        if self._rgb and not native_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif not self._rgb and native_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    @property
    def camera_names(self) -> list[str]:
        """Dataset image keys in a stable order."""
        return list(self._topic_map.values())

    def get(self, key: str) -> Optional[np.ndarray]:
        """Decode the latest frame for one camera, or None if not seen."""
        with self._lock:
            payload = self._raw.get(key)
        return None if payload is None else self._decode(payload)

    def read(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Latest frame per camera in the data-exporter message format.

        Returns ``{"images": {key: HxWx3 uint8 RGB}, "timestamps": {key: t}}``
        once every camera has produced at least one frame, else ``None``. This
        mirrors ComposedCameraClientSensor.read() so it is a drop-in image
        source for the data exporter.
        """
        names = list(self._topic_map.values())
        with self._lock:
            if not all(n in self._raw for n in names):
                return None
            # The receive thread replaces entries rather than mutating them, so
            # this snapshot stays valid while we decode outside the lock.
            raws = {n: self._raw[n] for n in names}
            timestamps = {n: self._recv_time[n] for n in names}
        images = {n: self._decode(p) for n, p in raws.items()}
        return {"images": images, "timestamps": timestamps}

    def debug_status(self) -> Dict[str, Any]:
        """Snapshot for diagnosing why frames aren't arriving.

        - ``total_msgs`` 0 -> nothing on the socket: server down / wrong
          host:port / firewall.
        - ``total_msgs`` > 0 but ``missing_keys`` non-empty -> the server is
          publishing, but not (all of) the topics this client expects. Compare
          ``seen_topics`` against ``expected_topics`` to spot a name mismatch or
          a camera that isn't streaming.
        """
        with self._lock:
            return {
                "total_msgs": self._total_recv,
                "seen_topics": sorted(self._seen_topics),
                "expected_topics": sorted(self._topic_map.keys()),
                "received_keys": sorted(self._raw.keys()),
                "missing_keys": [k for k in self._topic_map.values() if k not in self._raw],
            }

    def pop_counts(self) -> Dict[str, int]:
        """Snapshot + reset per-camera received-frame counters (= server rate)."""
        with self._lock:
            c = dict(self._recv_counts)
            self._recv_counts.clear()
            return c

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close(linger=0)
