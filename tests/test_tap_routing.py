"""Exercise physical dispatch boundaries without sending any inputs."""
import subprocess
import unittest
from unittest.mock import Mock, patch

from adapters.phone_adapter import AdbController
from profile_manager import DEFAULT_PROFILES
from universal_brain import UniversalBrain
from universal_vision import UniversalEntity, UniversalSceneState
from test_input_controller import controller


class TapRoutingTests(unittest.TestCase):
    def setUp(self):
        self.phone = AdbController.__new__(AdbController)
        self.phone.screen_width = 1080
        self.phone.screen_height = 2400
        self.phone.is_landscape = False
        self.phone.adb_bin = "adb"
        self.phone.device_serial = "test-device"

    def test_phone_solitaire_specific_taps(self):
        for action, method, args in [
            ("tap_waste_card", "tap_solitaire_waste", ()),
            ("tap_col_1", "tap_solitaire_column", (0,)),
            ("tap_col_7", "tap_solitaire_column", (6,)),
            ("tap_foundation", "tap_solitaire_foundation", (1,)),
        ]:
            with self.subTest(action=action), patch.object(self.phone, method) as specific, \
                    patch.object(self.phone, "tap") as generic:
                self.phone.dispatch_action(action)
                if method == "tap_solitaire_column":
                    specific.assert_called_once_with(*args, y_ratio=.55)
                else:
                    specific.assert_called_once_with(*args)
                generic.assert_not_called()

    def test_phone_jump_tap_and_double_jump_are_not_swipes(self):
        for action, method in [("jump_tap", "tap"), ("double_jump", "double_tap")]:
            with self.subTest(action=action), patch.object(self.phone, method) as tap, \
                    patch.object(self.phone, "swipe_up") as swipe:
                self.phone.dispatch_action(action)
                tap.assert_called_once()
                swipe.assert_not_called()

    def test_adb_failure_is_reported(self):
        result = subprocess.CompletedProcess([], 1, "", "device offline")
        with patch("adapters.phone_adapter.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "device offline"):
                self.phone.tap(120, 300)

    def test_adb_tap_uses_selected_device(self):
        with patch("adapters.phone_adapter.subprocess.run", return_value=Mock(returncode=0)) as run:
            self.phone.tap(120, 300)
        self.assertEqual(run.call_args.args[0], ["adb", "-s", "test-device", "shell", "input tap 120 300"])

    def test_pc_named_columns_use_requested_column(self):
        ctrl = controller.InputController.__new__(controller.InputController)
        ctrl.is_enabled = True
        ctrl._sol_col_idx = 0
        ctrl.click_at = Mock()
        with patch.object(controller.time, "sleep"):
            for column, x in [(1, 200), (7, 800), (3, 400)]:
                ctrl.click_at.reset_mock()
                ctrl.dispatch_pc_action(f"tap_col_{column}", viewport_size=(1000, 1000))
                self.assertEqual(ctrl.click_at.call_args_list[0].args, (x, 450))

    def test_pc_profile_keyboard_bindings_and_idle_actions(self):
        ctrl = controller.InputController.__new__(controller.InputController)
        ctrl.is_enabled = True
        ctrl.press_key = Mock()
        ctrl.click_at = Mock()
        for profile in DEFAULT_PROFILES:
            if profile.id.startswith("mobile_"):
                continue
            for action in profile.actions:
                if action.is_mouse_click or action.key.startswith("mouse_"):
                    continue
                with self.subTest(profile=profile.id, action=action.name):
                    ctrl.press_key.reset_mock()
                    ctrl.click_at.reset_mock()
                    ctrl.dispatch_pc_action(action.name, target_coords=(50, 60), action_binding=action)
                    vk = controller.resolve_vk(action.key)
                    if vk is not None:
                        ctrl.press_key.assert_called_once_with(vk, duration_sec=action.duration_sec)
                    else:
                        ctrl.press_key.assert_not_called()
                    ctrl.click_at.assert_not_called()

    def test_clicker_reflex_emits_declared_click_action(self):
        brain = UniversalBrain.__new__(UniversalBrain)
        target = UniversalEntity(10, 20, 30, 40, click_x=25, click_y=40)
        scene = UniversalSceneState(best_target=target, targets=[target, target])
        for profile in DEFAULT_PROFILES:
            if profile.category == "clicker" and "fruit" not in profile.id:
                with self.subTest(profile=profile.id):
                    result = brain._evaluate_intelligent_reflex(profile, scene)
                    self.assertIn(result["action"], [a.name for a in profile.actions])
                    if result["action"] == "click_target":
                        self.assertEqual(result["target_coords"], (25, 40))

    def test_all_mobile_profile_actions_reach_input_transport(self):
        idle = {"wait", "maintain_course", "maintain_heading", "glide"}
        for landscape in (False, True):
            self.phone.is_landscape = landscape
            self.phone.screen_width, self.phone.screen_height = ((2400, 1080) if landscape else (1080, 2400))
            for profile in DEFAULT_PROFILES:
                if not (profile.id.startswith("mobile_") or profile.id in {"runner_3lane", "flappy_tap"}):
                    continue
                for action in profile.actions:
                    with self.subTest(profile=profile.id, action=action.name, landscape=landscape), \
                            patch.object(self.phone, "_run_shell") as transport, \
                            patch("adapters.phone_adapter.time.sleep"):
                        self.phone.dispatch_action(action.name)
                        if action.name in idle:
                            transport.assert_not_called()
                        else:
                            self.assertTrue(transport.called, "Action sent no touch command")

    def test_all_pc_profile_actions_reach_input_methods(self):
        ctrl = controller.InputController.__new__(controller.InputController)
        ctrl.is_enabled = True
        ctrl._card_slot_idx = ctrl._sol_col_idx = 0
        methods = ["click_at", "right_click_at", "press_key", "drag_to"]
        for method in methods:
            setattr(ctrl, method, Mock())
        for profile in DEFAULT_PROFILES:
            if profile.id.startswith("mobile_") or profile.id == "chess_copilot":
                continue  # Chess requires the separate board/move adapter.
            for action in profile.actions:
                for method in methods:
                    getattr(ctrl, method).reset_mock()
                with self.subTest(profile=profile.id, action=action.name), patch.object(controller.time, "sleep"):
                    ctrl.dispatch_pc_action(action.name, target_coords=(20, 30), action_binding=action)
                    sent = any(getattr(ctrl, method).called for method in methods)
                    self.assertEqual(sent, action.key != "none")
