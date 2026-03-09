"""Tests for environment.actions."""
import chex
import jax
import jax.numpy as jnp
import pytest

from environment.actions import Actions, ActionType


class TestActionsEnumValues:
    def test_right_is_zero(self):
        assert Actions.RIGHT == 0

    def test_stay_is_four(self):
        assert Actions.STAY == 4

    def test_interact_is_five(self):
        assert Actions.INTERACT == 5

    def test_pick_place_base_is_six(self):
        assert Actions.PICK_PLACE_BASE == 6

    def test_action_type_enum_values(self):
        assert ActionType.MOVE == 0
        assert ActionType.NOP == 1
        assert ActionType.INTERACTION == 2
        assert ActionType.PICK_PLACE == 3


class TestDeclareActionSet:
    """declare_action_set の形状・値テスト."""

    def test_single_capacity(self):
        result = Actions.declare_action_set([3])
        # max(capacity) + PICK_PLACE_BASE = 3 + 6 = 9 → arange(9) = [0..8]
        chex.assert_shape(result, (9,))
        assert int(result[0]) == 0
        assert int(result[-1]) == 8

    def test_multi_agent_capacity(self):
        result = Actions.declare_action_set([2, 5])
        # max=5, so arange(11)
        chex.assert_shape(result, (11,))

    def test_returns_jax_array(self):
        result = Actions.declare_action_set([3])
        chex.assert_type(result, jnp.int32)


class TestActionType:
    """action_type の分岐網羅テスト."""

    @pytest.mark.parametrize(
        "action,expected_type,expected_idx",
        [
            (-1, ActionType.NOP, -1),           # 無効アクション
            (0, ActionType.MOVE, -1),            # RIGHT
            (1, ActionType.MOVE, -1),            # DOWN
            (2, ActionType.MOVE, -1),            # LEFT
            (3, ActionType.MOVE, -1),            # UP
            (4, ActionType.NOP, -1),             # STAY
            (5, ActionType.INTERACTION, -1),     # INTERACT
            (6, ActionType.PICK_PLACE, 0),       # PICK_PLACE_BASE
            (7, ActionType.PICK_PLACE, 1),       # PICK_PLACE_BASE + 1
            (9, ActionType.PICK_PLACE, 3),       # PICK_PLACE_BASE + 3
        ],
    )
    def test_action_type_branch(self, action, expected_type, expected_idx):
        act_type, idx = Actions.action_type(action)
        assert int(act_type) == int(expected_type)
        assert int(idx) == expected_idx

    def test_action_type_jit_compatible(self):
        """jax.jit でラップしても同一結果を返す."""
        fn = jax.jit(Actions.action_type)
        act_type, idx = fn(5)
        assert int(act_type) == int(ActionType.INTERACTION)


class TestActionToDirection:
    """action_to_direction の方向ベクトルテスト."""

    @pytest.mark.parametrize(
        "action,expected",
        [
            (Actions.RIGHT, [0, +1]),
            (Actions.DOWN, [+1, 0]),
            (Actions.LEFT, [0, -1]),
            (Actions.UP, [-1, 0]),
            (Actions.STAY, [0, 0]),
        ],
    )
    def test_direction_vector(self, action, expected):
        result = Actions.action_to_direction(action)
        chex.assert_shape(result, (2,))
        assert list(map(int, result)) == expected

    def test_direction_returns_array(self):
        result = Actions.action_to_direction(Actions.UP)
        assert isinstance(result, jax.Array)


class TestActionToString:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, "RIGHT"),
            (1, "DOWN"),
            (2, "LEFT"),
            (3, "UP"),
            (4, "STAY"),
            (5, "INTERACT"),
            (6, "PICK_PLACE_0"),
            (8, "PICK_PLACE_2"),
        ],
    )
    def test_action_to_string(self, value, expected):
        assert Actions.action_to_string(value) == expected
