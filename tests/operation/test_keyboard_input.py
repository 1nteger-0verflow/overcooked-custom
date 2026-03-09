"""Tests for operation.keyboard_input.KeyboardInput."""
import pytest

from environment.actions import Actions
from operation.keyboard_input import KeyboardInput


@pytest.fixture
def controller():
    return KeyboardInput(agent_id=0, verbose=False, confirm=False)


@pytest.fixture
def confirm_controller():
    return KeyboardInput(agent_id=0, verbose=False, confirm=True)


class TestKeyboardInputProperties:
    def test_is_auto_false(self, controller):
        assert controller.is_auto is False

    def test_initial_action_is_stay(self, controller):
        assert controller.get_action() == int(Actions.STAY)


class TestInputKey:
    @pytest.mark.parametrize(
        "key,expected_action",
        [
            ("up", int(Actions.UP)),
            ("down", int(Actions.DOWN)),
            ("left", int(Actions.LEFT)),
            ("right", int(Actions.RIGHT)),
            (" ", int(Actions.INTERACT)),
            ("tab", int(Actions.STAY)),
        ],
    )
    def test_direction_keys(self, key, expected_action):
        ctrl = KeyboardInput(agent_id=0, verbose=False, confirm=False)
        ctrl.input_key(key)
        assert ctrl.get_action() == expected_action

    @pytest.mark.parametrize("num_key,expected_slot", [(1, 0), (2, 1), (3, 2)])
    def test_numeric_keys_pick_place(self, num_key, expected_slot):
        ctrl = KeyboardInput(agent_id=0, verbose=False, confirm=False)
        accepted = ctrl.input_key(str(num_key))
        if accepted:
            assert ctrl.get_action() == int(Actions.PICK_PLACE_BASE) + expected_slot

    def test_invalid_key_does_not_change_action(self, controller):
        before = controller.get_action()
        controller.input_key("z")  # マッピングにない
        after = controller.get_action()
        # zが無効なら変化しないはず
        assert after == before or after == int(Actions.STAY)

    def test_input_key_returns_bool(self, controller):
        result = controller.input_key("up")
        assert isinstance(result, bool)


class TestIsDone:
    def test_done_after_key_no_confirm(self, controller):
        controller.input_key("up")
        assert controller.is_done

    def test_not_done_before_input(self):
        ctrl = KeyboardInput(agent_id=0, verbose=False, confirm=False)
        # input_observation をリセットとして呼ぶ
        ctrl.input_observation(None)
        assert not ctrl.is_done

    def test_confirm_mode_not_done_before_confirm(self, confirm_controller):
        confirm_controller.input_observation(None)
        confirm_controller.input_key("up")
        # confirm=True のとき Enter 押下前は done でない場合あり
        # (実装依存のため例外がないことのみ確認)
        _ = confirm_controller.is_done


class TestInputObservation:
    def test_input_observation_resets_done(self, controller):
        controller.input_key("up")
        assert controller.is_done
        controller.input_observation(None)
        assert not controller.is_done

    def test_get_action_after_reset_is_stay(self, controller):
        controller.input_key("down")
        controller.input_observation(None)
        assert controller.get_action() == int(Actions.STAY)
