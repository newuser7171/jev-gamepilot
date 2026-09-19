"""
Universal Input Controller for Jev-GamePilot.
Dispatches arbitrary keyboard keys and mouse clicks across any game window
with hardware scan codes and multi-tier emergency stop fail-safes.
"""

import ctypes
import threading
import time
from typing import Optional, Tuple
import pyautogui
from profile_manager import GameAction

# PyAutoGUI safety configuration
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001

KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


def resolve_vk(key_name: str) -> Optional[int]:
    """Resolves human-readable key name to Windows Virtual Key code."""
    k = key_name.lower().strip()
    special = {
        "space": 0x20,
        "enter": 0x0D,
        "return": 0x0D,
        "up": 0x26,
        "down": 0x28,
        "left": 0x25,
        "right": 0x27,
        "shift": 0x10,
        "ctrl": 0x11,
        "tab": 0x09,
        "esc": 0x1B,
    }
    if k in special:
        return special[k]
    if len(k) == 1 and k.isalnum():
        return ord(k.upper())
    return None


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
        self.held_keys = set()
        self.duck_release_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def enable(self):
        with self._lock:
            self.is_enabled = True
        print("[InputController] ARMED. Press ESC to disable.")

    def disable(self):
        with self._lock:
            self.is_enabled = False
            self._release_all()
        print("[InputController] DISARMED / EMERGENCY STOPPED.")

    def _release_all(self):
        try:
            if self.duck_release_timer:
                self.duck_release_timer.cancel()
            for vk in list(self.held_keys):
                _send_key_up(vk)
            self.held_keys.clear()
        except Exception as e:
            print(f"[InputController] Release error: {e}")

    def press_key(self, vk_code: int, duration_sec: float = 0.10):
        """Sends a non-blocking keypress with hardware scan code."""
        if not self.is_enabled:
            return

        def worker():
            with self._lock:
                if not self.is_enabled:
                    return
                _send_key_down(vk_code)
                self.held_keys.add(vk_code)

            time.sleep(duration_sec)

            with self._lock:
                _send_key_up(vk_code)
                self.held_keys.discard(vk_code)

        threading.Thread(target=worker, daemon=True).start()

    def click_at(self, screen_x: int, screen_y: int):
        """Clicks directly at screen coordinates (for aim trainers/clickers)."""
        if not self.is_enabled:
            return

        try:
            pyautogui.click(screen_x, screen_y)
        except Exception as e:
            print(f"[InputController] Click error: {e}")

    def dispatch_action(
        self,
        action: GameAction,
        target_coords: Optional[Tuple[int, int]] = None,
        viewport_offset: Optional[Tuple[int, int]] = None,
    ):
        """Universal dispatcher that executes any GameAction."""
        if not self.is_enabled:
            return

        # Mouse click actions
        if action.is_mouse_click and target_coords:
            cx, cy = target_coords
            if viewport_offset:
                cx += viewport_offset[0]
                cy += viewport_offset[1]
            self.click_at(cx, cy)
            return

        # Keyboard actions
        if action.key.lower() in ["none", "wait", ""]:
            return

        vk = resolve_vk(action.key)
        if vk is not None:
            self.press_key(vk, duration_sec=action.duration_sec)

    # Backward compatibility helpers for Dino runner
    def trigger_jump(self, duration_sec: float = 0.14):
        self.press_key(0x20, duration_sec=duration_sec)

    def trigger_duck(self, hold_sec: float = 0.35):
        self.press_key(0x28, duration_sec=hold_sec)

    def trigger_fast_fall(self):
        self.press_key(0x28, duration_sec=0.04)

    def trigger_restart(self):
        self.press_key(0x20, duration_sec=0.10)
