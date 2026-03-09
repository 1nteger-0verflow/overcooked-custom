"""Tests for operation.random_input.RandomInput.

Note: RandomInput は Python 標準の random.choice を使用しており JAX PRNG 非対応。
      そのため jax.jit 互換性テストは対象外とし、動作の正確性のみ検証する。
"""

import pytest

from operation.random_input import RandomInput


@pytest.fixture
def controller():
    return RandomInput(agent_id=0, num_actions=9)


class TestRandomInputProperties:
    def test_is_auto_true(self, controller):
        assert controller.is_auto is True

    def test_is_done_true(self, controller):
        # RandomInput は常に即座に行動を選択できる
        assert controller.is_done is True


class TestGetAction:
    @pytest.mark.parametrize("num_actions", [6, 9, 12])
    def test_action_in_valid_range(self, num_actions):
        ctrl = RandomInput(agent_id=0, num_actions=num_actions)
        for _ in range(50):
            action = ctrl.get_action()
            assert 0 <= action < num_actions

    def test_get_action_returns_int(self, controller):
        action = controller.get_action()
        assert isinstance(action, int)

    def test_single_action_always_zero(self):
        ctrl = RandomInput(agent_id=0, num_actions=1)
        assert ctrl.get_action() == 0

    def test_multiple_calls_vary(self):
        """num_actions が十分大きければ複数回呼んで複数の値が出る (統計的)."""
        ctrl = RandomInput(agent_id=0, num_actions=100)
        actions = {ctrl.get_action() for _ in range(200)}
        assert len(actions) > 1

    def test_agent_id_does_not_affect_range(self):
        for agent_id in range(3):
            ctrl = RandomInput(agent_id=agent_id, num_actions=9)
            action = ctrl.get_action()
            assert 0 <= action < 9
