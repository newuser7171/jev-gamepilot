"""Offline regression tests: no real keyboard or mouse events are sent."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("controller_under_test", ROOT / "input_controller.py")
controller = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"pyautogui": Mock()}):
    spec.loader.exec_module(controller)


class InputTests(unittest.TestCase):
    def setUp(self):
        # Avoid desktop attachment as well as input injection.
        with patch.object(controller.ctypes, "windll", Mock(), create=True):
            self.ctrl = controller.InputController()
        self.ctrl.is_enabled = True
        self.ctrl.press_key = Mock()
        self.ctrl.click_at = Mock()
        self.ctrl.right_click_at = Mock()

    def test_racing_actions(self):
        for name, key, duration in [("accelerate", 0x26, .30), ("turn_left", 0x25, .12),
                                    ("turn_right", 0x27, .12), ("brake", 0x28, .08)]:
            with self.subTest(action=name):
                self.ctrl.press_key.reset_mock()
                self.ctrl.dispatch_pc_action(name)
                self.ctrl.press_key.assert_called_once_with(key, duration_sec=duration)

    def test_place_uses_right_click_at_viewport_center(self):
        self.ctrl.dispatch_pc_action("use_place", viewport_offset=(100, 50), viewport_size=(800, 600))
        self.ctrl.right_click_at.assert_called_once_with(500, 350)
        self.ctrl.press_key.assert_not_called()

    def test_backspace_binding(self):
        self.assertEqual(controller.resolve_vk("backspace"), 0x08)

    def test_disarmed_dispatch_does_nothing(self):
        self.ctrl.is_enabled = False
        self.ctrl.dispatch_pc_action("accelerate")
        self.ctrl.dispatch_pc_action("use_place")
        self.ctrl.press_key.assert_not_called()
        self.ctrl.right_click_at.assert_not_called()

    def test_drag_releases_mouse_when_movement_fails(self):
        mouse = Mock()
        mouse.moveTo.side_effect = [None, RuntimeError("movement failed")]
        with patch.object(controller, "pyautogui", mouse), patch.object(controller.time, "sleep"):
            self.ctrl.drag_to(10, 20, 30, 40)
        mouse.mouseDown.assert_called_once()
        mouse.mouseUp.assert_called_once()

    def test_drag_failsafe_still_releases_native_mouse_button(self):
        mouse = Mock()
        mouse.mouseUp.side_effect = RuntimeError("corner failsafe")
        native = Mock()
        with patch.object(controller, "pyautogui", mouse), \
                patch.object(controller.time, "sleep"), \
                patch.object(controller.ctypes, "windll", native, create=True):
            self.ctrl.drag_to(10, 20, 30, 40)
        native.user32.mouse_event.assert_called_once_with(0x0004, 0, 0, 0, 0)

    def test_successful_drag_releases_once(self):
        mouse = Mock()
        with patch.object(controller, "pyautogui", mouse), patch.object(controller.time, "sleep"):
            self.ctrl.drag_to(10, 20, 30, 40)
        mouse.mouseDown.assert_called_once()
        mouse.mouseUp.assert_called_once()

    def test_interact_still_uses_e(self):
        self.ctrl.dispatch_pc_action("interact")
        self.ctrl.press_key.assert_called_once_with(ord("E"), duration_sec=.10)


if __name__ == "__main__":
    unittest.main()
