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
        "backspace": 0x08,
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
            hdesk = user32.OpenInputDesktop(0, False, 0x10000000)
            if not hdesk:
                hdesk = user32.OpenDesktopW("default", 0, False, 0x10000000)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

        self.is_enabled = False
        self.held_keys = set()
        self.duck_release_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._card_slot_idx = 0
        self._sol_col_idx = 0

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

    def press_key_sync(self, vk_code: int, duration_sec: float = 0.10):
        """Sends a blocking synchronous keypress."""
        if not self.is_enabled:
            return
        with self._lock:
            _send_key_down(vk_code)
            self.held_keys.add(vk_code)
        time.sleep(duration_sec)
        with self._lock:
            _send_key_up(vk_code)
            self.held_keys.discard(vk_code)

    def click_at(self, screen_x: int, screen_y: int):
        """Clicks directly at screen coordinates (for aim trainers/clickers)."""
        if not self.is_enabled:
            return

        try:
            pyautogui.click(int(screen_x), int(screen_y))
        except Exception as e:
            print(f"[InputController] Click error: {e}")

    def right_click_at(self, screen_x: int, screen_y: int):
        """Right-clicks directly at screen coordinates."""
        if not self.is_enabled:
            return

        try:
            pyautogui.rightClick(int(screen_x), int(screen_y))
        except Exception as e:
            print(f"[InputController] Right click error: {e}")

    def double_click_at(self, screen_x: int, screen_y: int):
        """Double-clicks directly at screen coordinates."""
        if not self.is_enabled:
            return

        try:
            pyautogui.doubleClick(int(screen_x), int(screen_y))
        except Exception as e:
            print(f"[InputController] Double click error: {e}")

    def drag_to(self, x1: int, y1: int, x2: int, y2: int, duration_sec: float = 0.22):
        """Smoothly drags mouse from (x1, y1) to (x2, y2) for PC card games & inventory management."""
        if not self.is_enabled:
            return

        mouse_pressed = False
        try:
            pyautogui.moveTo(int(x1), int(y1))
            time.sleep(0.02)
            pyautogui.mouseDown()
            mouse_pressed = True
            time.sleep(0.04)
            pyautogui.moveTo(int(x2), int(y2), duration=duration_sec)
            time.sleep(0.04)
        except Exception as e:
            print(f"[InputController] Drag error: {e}")
        finally:
            if mouse_pressed:
                try:
                    pyautogui.mouseUp()
                except Exception:
                    # The corner failsafe can also block mouseUp; always release.
                    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

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

    def dispatch_pc_action(
        self,
        action_name: str,
        target_coords: Optional[Tuple[int, int]] = None,
        viewport_offset: Tuple[int, int] = (0, 0),
        viewport_size: Tuple[int, int] = (1920, 1080),
    ):
        """
        High-level dispatcher for PC games.
        Converts relative viewport ratios into absolute screen pixels and executes native mouse/keyboard gestures.
        """
        if not self.is_enabled:
            return

        act = action_name.lower().strip()
        vx, vy = viewport_offset
        vw, vh = viewport_size

        # 1. Balatro (Steam / PC)
        if "balatro_play_hand" in act or "play_hand" in act:
            # Enter key or click Play Hand button at lower-left
            self.press_key(0x0D, duration_sec=0.08)
            self.click_at(vx + int(vw * 0.38), vy + int(vh * 0.76))
        elif "balatro_discard" in act or "discard" in act:
            # Backspace key or click Discard button at lower-right
            self.press_key(0x08, duration_sec=0.08)
            self.click_at(vx + int(vw * 0.62), vy + int(vh * 0.76))
        elif "cash_out" in act or "next_round" in act:
            self.click_at(vx + int(vw * 0.50), vy + int(vh * 0.50))
        elif "select_card" in act:
            slot_ratios = [0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72]
            self._card_slot_idx = (self._card_slot_idx + 1) % len(slot_ratios)
            cx = vx + int(vw * slot_ratios[self._card_slot_idx])
            cy = vy + int(vh * 0.90)
            self.click_at(cx, cy)

        # 2. Slay the Spire (Steam / PC)
        elif "play_card_enemy" in act or "attack_enemy" in act:
            start_x = vx + int(vw * 0.50)
            start_y = vy + int(vh * 0.92)
            dest_x = vx + int(vw * 0.75)
            dest_y = vy + int(vh * 0.45)
            self.drag_to(start_x, start_y, dest_x, dest_y, duration_sec=0.24)
        elif "play_card_self" in act or "defend" in act:
            start_x = vx + int(vw * 0.50)
            start_y = vy + int(vh * 0.92)
            dest_x = vx + int(vw * 0.25)
            dest_y = vy + int(vh * 0.55)
            self.drag_to(start_x, start_y, dest_x, dest_y, duration_sec=0.22)
        elif "end_turn" in act:
            # Press 'E' hotkey or click End Turn button
            self.press_key(ord('E'), duration_sec=0.08)
            self.click_at(vx + int(vw * 0.88), vy + int(vh * 0.50))

        # 3. Hearthstone / MTG Arena / Master Duel (PC)
        elif "play_card" in act:
            start_x = vx + int(vw * 0.50)
            start_y = vy + int(vh * 0.94)
            if "left" in act:
                dest_x = vx + int(vw * 0.35)
            elif "right" in act:
                dest_x = vx + int(vw * 0.65)
            else:
                dest_x = vx + int(vw * 0.50)
            dest_y = vy + int(vh * 0.56)
            self.drag_to(start_x, start_y, dest_x, dest_y, duration_sec=0.25)
        elif "attack_face" in act:
            fx = vx + int(vw * 0.50)
            fy = vy + int(vh * 0.62)
            tx = vx + int(vw * 0.50)
            ty = vy + int(vh * 0.18)
            self.drag_to(fx, fy, tx, ty, duration_sec=0.22)
        elif "attack_minion" in act or "trade_minion" in act:
            fx = vx + int(vw * 0.50)
            fy = vy + int(vh * 0.62)
            tx = vx + int(vw * 0.50)
            ty = vy + int(vh * 0.38)
            self.drag_to(fx, fy, tx, ty, duration_sec=0.22)
        elif "hero_power" in act:
            self.click_at(vx + int(vw * 0.64), vy + int(vh * 0.76))

        # 4. Solitaire & Classic Card Puzzles (PC)
        elif "sweep_all_columns" in act:
            col_ratios = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            for r in col_ratios:
                self.click_at(vx + int(vw * r), vy + int(vh * 0.55))
                time.sleep(0.04)
        elif "tap_col" in act or "tap_tableau_column" in act:
            col_ratios = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            self._sol_col_idx = (self._sol_col_idx + 1) % len(col_ratios)
            cx = vx + int(vw * col_ratios[self._sol_col_idx])
            self.click_at(cx, vy + int(vh * 0.45))
            time.sleep(0.03)
            self.click_at(cx, vy + int(vh * 0.65))
        elif "draw_stock" in act:
            self.click_at(vx + int(vw * 0.08), vy + int(vh * 0.20))
        elif "tap_waste_card" in act:
            self.click_at(vx + int(vw * 0.08), vy + int(vh * 0.45))
        elif "auto_complete" in act:
            self.click_at(vx + int(vw * 0.50), vy + int(vh * 0.88))

        # 5. Aim Lab / Target Clickers
        elif "click_target" in act or "target" in act:
            if target_coords:
                self.click_at(vx + target_coords[0], vy + target_coords[1])
            else:
                self.click_at(vx + vw // 2, vy + vh // 2)

        # 6. Roblox / Minecraft Movement
        elif "move_forward" in act or "walk" in act:
            self.press_key(ord('W'), duration_sec=0.20)
        elif "move_backward" in act:
            self.press_key(ord('S'), duration_sec=0.20)
        elif "move_left" in act or "dodge_left" in act:
            self.press_key(ord('A'), duration_sec=0.15)
        elif "move_right" in act or "dodge_right" in act:
            self.press_key(ord('D'), duration_sec=0.15)
        elif "jump" in act:
            self.press_key(0x20, duration_sec=0.12)
        elif "duck" in act or "slide" in act:
            self.press_key(0x28, duration_sec=0.25)
        elif "use_place" in act:
            self.right_click_at(vx + vw // 2, vy + vh // 2)
        elif "interact" in act or "use" in act:
            self.press_key(ord('E'), duration_sec=0.10)
        elif "attack_mine" in act:
            self.click_at(vx + vw // 2, vy + vh // 2)
        elif act == "accelerate":
            self.press_key(0x26, duration_sec=0.30)
        elif act == "turn_left":
            self.press_key(0x25, duration_sec=0.12)
        elif act == "turn_right":
            self.press_key(0x27, duration_sec=0.12)
        elif act == "brake":
            self.press_key(0x28, duration_sec=0.08)

        # 7. Fallback to generic key or target click
        else:
            vk = resolve_vk(act)
            if vk is not None:
                self.press_key(vk, duration_sec=0.10)
            elif target_coords:
                self.click_at(vx + target_coords[0], vy + target_coords[1])

    # Backward compatibility helpers for Dino runner
    def trigger_jump(self, duration_sec: float = 0.14):
        self.press_key(0x20, duration_sec=duration_sec)

    def trigger_duck(self, hold_sec: float = 0.35):
        self.press_key(0x28, duration_sec=hold_sec)

    def trigger_fast_fall(self):
        self.press_key(0x28, duration_sec=0.04)

    def trigger_restart(self):
        self.press_key(0x20, duration_sec=0.10)
