"""Progress tracking for blocking stages of the live WHAM/GMR pipeline."""

import threading
import time


class PipelineProgress:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._last = {}
        self._busy = {}

    def mark(self, stage, frame_id):
        with self._lock:
            self._last[stage] = (frame_id, self._clock())

    def begin(self, stage, frame_id):
        with self._lock:
            self._busy[stage] = (frame_id, self._clock())

    def finish(self, stage, frame_id):
        with self._lock:
            self._last[stage] = (frame_id, self._clock())
            self._busy.pop(stage, None)

    def snapshot(self):
        with self._lock:
            now = self._clock()
            last = {stage: {"frame": frame, "age_s": round(now - at, 1)}
                    for stage, (frame, at) in self._last.items()}
            busy = {stage: {"frame": frame, "elapsed_s": round(now - at, 1)}
                    for stage, (frame, at) in self._busy.items()}
            return last, busy

    def stalled(self, timeout_s):
        with self._lock:
            now = self._clock()
            return [(stage, frame, now - at)
                    for stage, (frame, at) in self._busy.items()
                    if now - at >= timeout_s]
