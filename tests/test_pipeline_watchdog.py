import unittest

from scripts.pipeline_watchdog import PipelineProgress


class PipelineProgressTests(unittest.TestCase):
    def test_idle_stage_does_not_trigger_restart(self):
        now = [0.0]
        progress = PipelineProgress(clock=lambda: now[0])
        progress.mark("input", 40)
        now[0] = 30.0
        self.assertEqual(progress.stalled(15.0), [])

    def test_blocked_detector_is_reported_with_frame_and_duration(self):
        now = [0.0]
        progress = PipelineProgress(clock=lambda: now[0])
        progress.begin("detect", 41)
        now[0] = 15.1
        self.assertEqual(progress.stalled(15.0), [("detect", 41, 15.1)])
        progress.finish("detect", 41)
        self.assertEqual(progress.stalled(15.0), [])
        last, busy = progress.snapshot()
        self.assertEqual(last["detect"]["frame"], 41)
        self.assertEqual(busy, {})


if __name__ == "__main__":
    unittest.main()
