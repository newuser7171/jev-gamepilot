"""ADB capture: no leaked screencap children, no silently dead stream thread."""
import subprocess
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from adapters.phone_adapter import AdbController


class _FakeProc:
    """Mimics the slice of Popen that _exec_out touches."""

    def __init__(self):
        self.returncode = None
        self.kill_calls = 0
        self.communicate_calls = 0

    def communicate(self, timeout=None):
        self.communicate_calls += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="adb", timeout=timeout)
        return b"", b""

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def make_controller():
    ctrl = AdbController.__new__(AdbController)
    ctrl.adb_bin = "adb"
    ctrl.device_serial = "test-device"
    ctrl.screen_width = 1080
    ctrl.screen_height = 2400
    ctrl.is_landscape = False
    ctrl._frame_lock = threading.Lock()
    ctrl._frame_id = 0
    ctrl._latest_frame = None
    ctrl._new_frame_event = threading.Event()
    ctrl._streaming = False
    ctrl._stream_thread = None
    return ctrl


class ExecOutTests(unittest.TestCase):
    def test_timeout_kills_the_child_once(self):
        ctrl = make_controller()
        proc = _FakeProc()
        with patch("adapters.phone_adapter.subprocess.Popen", return_value=proc):
            self.assertIsNone(ctrl._exec_out(["screencap"], 0.01))
        self.assertEqual(proc.kill_calls, 1)
        self.assertIsNotNone(proc.poll())

    def test_os_error_returns_none(self):
        ctrl = make_controller()
        with patch("adapters.phone_adapter.subprocess.Popen", side_effect=OSError("adb missing")):
            self.assertIsNone(ctrl._exec_out(["screencap"], 0.01))


class CaptureFallbackTests(unittest.TestCase):
    def test_png_fallback_decodes_frame(self):
        ctrl = make_controller()
        source = np.zeros((2400, 1080, 3), dtype=np.uint8)
        ok, png = cv2.imencode(".png", source)
        self.assertTrue(ok)

        with patch.object(ctrl, "_exec_out", side_effect=[None, png.tobytes()]):
            frame = ctrl.capture_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, source.shape)
        self.assertEqual(ctrl._frame_id, 1)

    def test_both_paths_fail_returns_none(self):
        ctrl = make_controller()
        with patch.object(ctrl, "_exec_out", return_value=None):
            self.assertIsNone(ctrl.capture_frame())


class StreamWorkerTests(unittest.TestCase):
    def test_worker_survives_one_capture_exception(self):
        ctrl = make_controller()
        calls = []

        def flaky_capture():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("usb glitch")
            ctrl._streaming = False
            return None

        with patch.object(ctrl, "capture_frame", side_effect=flaky_capture):
            ctrl.start_background_stream()
            ctrl._stream_thread.join(timeout=3.0)

        self.assertEqual(len(calls), 2)
        self.assertFalse(ctrl._stream_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
