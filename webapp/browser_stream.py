"""
OpenCV-compatible synchronous frame reader fed asynchronously by browser clients.
Enables 0-MB pre-upload playback of heavy phone/desktop videos across all scan speeds.
"""

import cv2
import time
import threading
import numpy as np


class BrowserFrameCapture:
    """
    VideoCapture-compatible adapter backed by on-demand browser <video> decoding.
    Avoids uploading multi-GB video files upfront: the browser keeps the file local and
    streams only the lightweight ~480p JPEG frames needed at 1x-40x scan speeds.
    """

    def __init__(self, slot_id: int, filename: str, duration_sec: float, fps: float = 15.0, on_demand=None):
        self.slot_id = int(slot_id)
        self.filename = str(filename or f"camera_{slot_id + 1}.mp4")
        self.duration_sec = max(0.5, float(duration_sec))
        self.fps = max(5.0, min(30.0, float(fps)))
        self.total_frames = max(1, int(round(self.duration_sec * self.fps)))
        self._on_demand = on_demand

        self._cond = threading.Condition()
        self._opened = True
        self._released = False
        self._eof = False

        self.cursor_pos = 0
        self.demand_seq = 0
        self.demand_sec = 0.0
        self._delivered_seq = -1
        self._delivered_frame = None

    def isOpened(self):
        return bool(self._opened and not self._released)

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.cursor_pos)
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return float((self.cursor_pos / max(1.0, self.fps)) * 1000.0)
        return 0.0

    def set(self, prop_id, value):
        if prop_id != cv2.CAP_PROP_POS_FRAMES:
            return False
        notify_args = None
        with self._cond:
            if self._released or not self._opened:
                return False
            target = max(0, int(round(value)))
            self.cursor_pos = target
            if target >= self.total_frames:
                self._eof = True
                self._cond.notify_all()
                return True
            self._eof = False
            self.demand_sec = min(max(0.0, self.duration_sec - 0.02), target / max(1.0, self.fps))
            self.demand_seq += 1
            notify_args = (self.slot_id, self.demand_seq, self.demand_sec, self.fps)
            self._cond.notify_all()
        if notify_args and self._on_demand:
            self._on_demand(*notify_args)
        return True

    def grab(self):
        with self._cond:
            if self._released or not self._opened or self._eof:
                return False
            self.cursor_pos += 1
            if self.cursor_pos >= self.total_frames:
                self._eof = True
                self._cond.notify_all()
                return False
            return True

    def read(self):
        notify_args = None
        with self._cond:
            if self._released or not self._opened or self._eof or self.cursor_pos >= self.total_frames:
                return False, None
            if self.demand_seq == 0 or self._delivered_seq >= self.demand_seq:
                self.demand_sec = min(max(0.0, self.duration_sec - 0.02), self.cursor_pos / max(1.0, self.fps))
                self.demand_seq += 1
                notify_args = (self.slot_id, self.demand_seq, self.demand_sec, self.fps)
            target_seq = self.demand_seq

        if notify_args and self._on_demand:
            self._on_demand(*notify_args)

        deadline = time.time() + 25.0
        last_ping = time.time()
        while True:
            re_notify = None
            with self._cond:
                while (
                    not self._released
                    and self._opened
                    and not self._eof
                    and self._delivered_seq < target_seq
                ):
                    now = time.time()
                    rem = deadline - now
                    if rem <= 0:
                        return False, None
                    if (now - last_ping) >= 3.0 and self._on_demand:
                        last_ping = now
                        re_notify = (self.slot_id, target_seq, self.demand_sec, self.fps)
                        break
                    self._cond.wait(timeout=min(0.20, rem))
                if re_notify is None:
                    if self._released or not self._opened or self._eof or self._delivered_frame is None:
                        return False, None
                    frame = self._delivered_frame
                    self.cursor_pos += 1
                    return True, frame
            if re_notify and self._on_demand:
                self._on_demand(*re_notify)

    def deliver_jpeg(self, seq: int, raw_bytes: bytes) -> bool:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        with self._cond:
            if self._released:
                return False
            if int(seq) >= self.demand_seq or self.demand_seq == 0:
                self._delivered_frame = img
                self._delivered_seq = max(self._delivered_seq, int(seq))
                self._cond.notify_all()
            return True

    def signal_eof(self):
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def pending_demand(self):
        with self._cond:
            if (
                not self._released
                and self._opened
                and not self._eof
                and self.demand_seq > self._delivered_seq
                and self.demand_seq > 0
            ):
                return {"slot_id": self.slot_id, "seq": self.demand_seq, "sec": self.demand_sec, "fps": self.fps}
            return None

    def release(self):
        with self._cond:
            self._released = True
            self._opened = False
            self._cond.notify_all()
