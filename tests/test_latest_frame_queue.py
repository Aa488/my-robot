import unittest

from scripts.latest_frame_queue import LatestFrameQueue


class LatestFrameQueueTests(unittest.TestCase):
    def test_slow_consumer_receives_the_newest_frame(self):
        queue = LatestFrameQueue()
        for frame in range(100):
            queue.put(frame)
        self.assertEqual(queue.get_nowait(), 99)
        self.assertEqual(queue.dropped, 99)

    def test_stop_marker_survives_late_frame(self):
        queue = LatestFrameQueue()
        queue.put(1)
        queue.put(None)
        queue.put(2)
        self.assertIsNone(queue.get_nowait())


if __name__ == "__main__":
    unittest.main()
