"""
Input Controller for Jev-GamePilot.
Executes rapid, non-blocking keystrokes for gaming actions with multi-tier emergency stop fail-safes.
"""

import ctypes
import threading
import time
from typing import Optional
import pyautogui

# Enable PyAutoGUI emergency stop if mouse is flicked to corner
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001

# Windows Virtual Key Codes
VK_SPACE = 0x20
VK_UP = 0x26
VK_DOWN = 0x28
VK_LEFT = 0x25
VK_RIGHT = 0x27
KEYEVENTF_KEYUP = 0x0002


def _send_key_down(vk_code: int):
    scan = ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)
    ctypes.windll.user32.keybd_event(vk_code, scan, 0, 0)


def _send_key_up(vk_code: int):
    scan = ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)
    ctypes.windll.user32.keybd_event(vk_code, scan, KEYEVENTF_KEYUP, 0)


class InputController:
    def __init__(self):
        try:
            user32 = ctypes.windll.user32
            hdesk = user32.OpenDesktopW("default", 0, False, 0x10000000)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

        self.is_enabled = False
        self.is_ducking = False
        self.last_action_time = 0.0
        self.duck_release_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def enable(self):
        """Arm the controller for automated inputs."""
        with self._lock:
            self.is_enabled = True
        print("[InputController] ARMED. Press ESC to disable.")

    def disable(self):
        """Emergency stop: immediately release any held keys and disarm."""
        with self._lock:
            self.is_enabled = False
            self._release_all()
        print("[InputController] DISARMED / EMERGENCY STOPPED.")

    def _release_all(self):
        try:
            if self.duck_release_timer:
                self.duck_release_timer.cancel()
            _send_key_up(VK_DOWN)
            _send_key_up(VK_SPACE)
            _send_key_up(VK_UP)
            _send_key_up(VK_LEFT)
            _send_key_up(VK_RIGHT)
            self.is_ducking = False
        except Exception as e:
            print(f"[InputController] Release error: {e}")

    def trigger_jump(self, duration_sec: float = 0.14):
        """Executes a crisp jump."""
        if not self.is_enabled:
            return

        def worker():
            with self._lock:
                if not self.is_enabled:
                    return
                # Release duck if active
                if self.is_ducking:
                    _send_key_up(VK_DOWN)
                    self.is_ducking = False

                _send_key_down(VK_SPACE)
                time.sleep(duration_sec)
                _send_key_up(VK_SPACE)

        threading.Thread(target=worker, daemon=True).start()

    def trigger_duck(self, hold_sec: float = 0.35):
        """Executes a duck slide."""
        if not self.is_enabled:
            return

        with self._lock:
            if not self.is_enabled:
                return
            if not self.is_ducking:
                _send_key_down(VK_DOWN)
                self.is_ducking = True

            # Cancel previous release timer if already ducking
            if self.duck_release_timer:
                self.duck_release_timer.cancel()

            def release():
                with self._lock:
                    if self.is_ducking:
                        _send_key_up(VK_DOWN)
                        self.is_ducking = False

            self.duck_release_timer = threading.Timer(hold_sec, release)
            self.duck_release_timer.daemon = True
            self.duck_release_timer.start()

    def trigger_fast_fall(self):
        """Ducks briefly while airborne to snap the Dino back to the ground."""
        if not self.is_enabled:
            return

        def worker():
            with self._lock:
                if not self.is_enabled:
                    return
                _send_key_down(VK_DOWN)
                time.sleep(0.04)
                _send_key_up(VK_DOWN)

        threading.Thread(target=worker, daemon=True).start()

    def trigger_restart(self):
        """Restarts the game after a Game Over."""
        if not self.is_enabled:
            return
        self.trigger_jump(duration_sec=0.1)

    def trigger_dodge_left(self):
        if not self.is_enabled:
            return
        _send_key_down(VK_LEFT)
        time.sleep(0.05)
        _send_key_up(VK_LEFT)

    def trigger_dodge_right(self):
        if not self.is_enabled:
            return
        _send_key_down(VK_RIGHT)
        time.sleep(0.05)
        _send_key_up(VK_RIGHT)
