"""A bounded webcam queue that replaces stale frames before a slow consumer wakes."""

from queue import Queue


class LatestFrameQueue(Queue):
    def __init__(self):
        super().__init__(maxsize=1)
        self.dropped = 0

    def put(self, item, block=True, timeout=None):
        with self.not_full:
            if self._qsize():
                # Never replace a stop marker with a late producer frame.
                if self.queue[0] is None and item is not None:
                    return
                self._get()
                self.dropped += 1
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
