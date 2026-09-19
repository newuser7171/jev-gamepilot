"""
Vision Engine for Jev-GamePilot.
Captures screen regions at high frame rates using mss and provides window detection.
"""

import time
from typing import Dict, Optional, Tuple
import cv2
import mss
import numpy as np
import pygetwindow as gw


class VisionEngine:
    def __init__(self, target_region: Optional[Dict[str, int]] = None):
        """
        Initialize the vision engine.
        target_region: dict with keys 'left', 'top', 'width', 'height'
        """
        self.sct = mss.mss()
        self.region = target_region or self._get_default_region()
        self.last_frame: Optional[np.ndarray] = None
        self.last_capture_time: float = 0.0
        self.fps: float = 0.0

    def _get_default_region(self) -> Dict[str, int]:
        """Default to primary monitor center zone."""
        mon = self.sct.monitors[1]
        w = int(mon["width"] * 0.6)
        h = int(mon["height"] * 0.5)
        left = int((mon["width"] - w) / 2)
        top = int((mon["height"] - h) / 2)
        return {"left": left, "top": top, "width": w, "height": h}

    def set_region(self, left: int, top: int, width: int, height: int):
        """Set explicit capture bounding box."""
        self.region = {
            "left": max(0, int(left)),
            "top": max(0, int(top)),
            "width": max(100, int(width)),
            "height": max(100, int(height)),
        }

    def snap_to_window(self, window_title_keyword: str) -> bool:
        """
        Searches for an open window containing the keyword and sets region to it.
        """
        try:
            windows = gw.getAllWindows()
            for win in windows:
                if (
                    win.title
                    and window_title_keyword.lower() in win.title.lower()
                ):
                    if win.width > 100 and win.height > 100:
                        self.set_region(
                            win.left, win.top, win.width, win.height
                        )
                        return True
        except Exception as e:
            print(f"[VisionEngine] Window snap error: {e}")
        return False

    def list_windows(self) -> list[str]:
        """List titles of all visible windows."""
        try:
            return [
                w.title
                for w in gw.getAllWindows()
                if w.title.strip() and w.width > 200 and w.height > 200
            ]
        except Exception:
            return []

    def capture_frame(self) -> np.ndarray:
        """
        Captures a single frame from the configured region.
        Returns BGR numpy array.
        """
        t0 = time.perf_counter()
        raw = self.sct.grab(self.region)
        # raw is BGRA, convert to BGR
        frame = np.array(raw)[:, :, :3]

        dt = time.perf_counter() - t0
        self.fps = 1.0 / dt if dt > 0 else 60.0
        self.last_frame = frame
        self.last_capture_time = time.time()
        return frame

    def close(self):
        self.sct.close()
