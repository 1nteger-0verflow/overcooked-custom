"""Tests for environment.agent."""

import chex
import jax
import jax.numpy as jnp
import pytest

from environment.actions import Actions
from environment.agent import Agent


def _make_agent(
    pos: list[list[int]], direction: list[list[int]], capacity: int = 3, height: int = 8, width: int = 10
) -> Agent:
    """単一エージェントの Agent PyTree を構築するヘルパー."""
    num_agents = len(pos)
    return Agent(
        pos=jnp.array(pos, dtype=jnp.int32),
        dir=jnp.array(direction, dtype=jnp.int32),
        capacity=jnp.array([capacity] * num_agents, dtype=jnp.int32),
        inventory=jnp.zeros((num_agents, capacity), dtype=jnp.int32),
        view_sizes=jnp.zeros((num_agents, 2), dtype=jnp.int32),
        grid_observed_step=jnp.full((num_agents, height, width), -1, dtype=jnp.int32),
    )


class TestNumAgents:
    def test_single_agent(self):
        agent = _make_agent([[2, 3]], [[-1, 0]])
        assert agent.num_agents == 1

    def test_two_agents(self):
        agent = _make_agent([[2, 3], [4, 5]], [[-1, 0], [1, 0]])
        assert agent.num_agents == 2


class TestGetFwdPos:
    @pytest.mark.parametrize(
        ("pos", "direction", "expected_fwd"),
        [
            ([[2, 3]], [[-1, 0]], [[1, 3]]),  # UP
            ([[2, 3]], [[1, 0]], [[3, 3]]),  # DOWN
            ([[2, 3]], [[0, -1]], [[2, 2]]),  # LEFT
            ([[2, 3]], [[0, 1]], [[2, 4]]),  # RIGHT
        ],
    )
    def test_forward_position(self, pos, direction, expected_fwd):
        agent = _make_agent(pos, direction)
        fwd = agent.get_fwd_pos()
        chex.assert_shape(fwd, (1, 2))
        assert list(map(int, fwd[0])) == expected_fwd[0]


class TestMoveInBounds:
    """move_in_bounds の境界クリップテスト."""

    def test_normal_move(self):
        # pos=(2,3), dir=DOWN=(1,0) → new_pos=(3,3)
        # move_in_bounds は単一エージェントを想定 (pos が 1-dim)
        single_agent = Agent(
            pos=jnp.array([2, 3], dtype=jnp.int32),
            dir=jnp.array([1, 0], dtype=jnp.int32),
            capacity=jnp.array([3], dtype=jnp.int32),
            inventory=jnp.zeros((1, 3), dtype=jnp.int32),
            view_sizes=jnp.zeros((1, 2), dtype=jnp.int32),
            grid_observed_step=jnp.full((1, 8, 10), -1, dtype=jnp.int32),
        )
        result = single_agent.move_in_bounds(jnp.array([1, 0]), height=8, width=10)
        chex.assert_shape(result, (2,))
        assert list(map(int, result)) == [3, 3]

    def test_clip_at_top(self):
        # pos=(0,3), dir=UP=(-1,0) → new_pos clipped to (0,3)
        single_agent = Agent(
            pos=jnp.array([0, 3], dtype=jnp.int32),
            dir=jnp.array([-1, 0], dtype=jnp.int32),
            capacity=jnp.array([3], dtype=jnp.int32),
            inventory=jnp.zeros((1, 3), dtype=jnp.int32),
            view_sizes=jnp.zeros((1, 2), dtype=jnp.int32),
            grid_observed_step=jnp.full((1, 8, 10), -1, dtype=jnp.int32),
        )
        result = single_agent.move_in_bounds(jnp.array([-1, 0]), height=8, width=10)
        assert list(map(int, result)) == [0, 3]

    def test_clip_at_right_boundary(self):
        # pos=(3, 9), dir=RIGHT=(0,+1) → clipped to (3,9)
        single_agent = Agent(
            pos=jnp.array([3, 9], dtype=jnp.int32),
            dir=jnp.array([0, 1], dtype=jnp.int32),
            capacity=jnp.array([3], dtype=jnp.int32),
            inventory=jnp.zeros((1, 3), dtype=jnp.int32),
            view_sizes=jnp.zeros((1, 2), dtype=jnp.int32),
            grid_observed_step=jnp.full((1, 8, 10), -1, dtype=jnp.int32),
        )
        result = single_agent.move_in_bounds(jnp.array([0, 1]), height=8, width=10)
        assert list(map(int, result)) == [3, 9]


class TestComputeViewBox:
    """compute_view_box の視野範囲計算テスト."""

    def test_zero_view_size_returns_full_grid(self):
        # view_sizes=[0,0] のとき視野制限なし → 全グリッドを観測
        agent = Agent(
            pos=jnp.array([[2, 3]], dtype=jnp.int32),
            dir=jnp.array([[-1, 0]], dtype=jnp.int32),
            capacity=jnp.array([3], dtype=jnp.int32),
            inventory=jnp.zeros((1, 3), dtype=jnp.int32),
            view_sizes=jnp.array([[0, 0]], dtype=jnp.int32),
            grid_observed_step=jnp.full((1, 8, 10), -1, dtype=jnp.int32),
        )
        boxes = agent.compute_view_box(8, 10)
        chex.assert_shape(boxes, (1, 4))
        x_min, x_max, y_min, y_max = map(int, boxes[0])
        # 全グリッド (x_min=0, x_max=10, y_min=0, y_max=8) に近い範囲
        assert x_min == 3  # pos[1] + 0
        assert x_max == 4  # pos[1] + 0 + 1
        assert y_min == 2  # pos[0] + 0
        assert y_max == 3  # pos[0] + 0 + 1

    def test_view_box_clipped_at_grid_boundary(self):
        # pos=(0,0)、UP方向で前方視野3 → y_min がクリップされる
        agent = Agent(
            pos=jnp.array([[0, 0]], dtype=jnp.int32),
            dir=jnp.array([[-1, 0]], dtype=jnp.int32),
            capacity=jnp.array([3], dtype=jnp.int32),
            inventory=jnp.zeros((1, 3), dtype=jnp.int32),
            view_sizes=jnp.array([[3, 1]], dtype=jnp.int32),
            grid_observed_step=jnp.full((1, 8, 10), -1, dtype=jnp.int32),
        )
        boxes = agent.compute_view_box(8, 10)
        x_min, _x_max, y_min, _y_max = map(int, boxes[0])
        assert y_min == 0  # -3 がクリップされて 0
        assert x_min == 0  # -1 がクリップされて 0

    def test_view_box_shape(self):
        agent = _make_agent([[2, 3], [5, 6]], [[-1, 0], [1, 0]])
        boxes = agent.compute_view_box(8, 10)
        chex.assert_shape(boxes, (2, 4))

    def test_view_box_jit_compatible(self):
        agent = _make_agent([[2, 3]], [[-1, 0]])
        fn = jax.jit(agent.compute_view_box, static_argnums=(0, 1))
        boxes = fn(8, 10)
        chex.assert_shape(boxes, (1, 4))


class TestUpdateObservedGrid:
    def test_update_marks_observed_cells(self):
        agent = Agent(
            pos=jnp.array([[2, 3]], dtype=jnp.int32),
            dir=jnp.array([[-1, 0]], dtype=jnp.int32),
            capacity=jnp.array([3], dtype=jnp.int32),
            inventory=jnp.zeros((1, 3), dtype=jnp.int32),
            view_sizes=jnp.array([[0, 0]], dtype=jnp.int32),
            grid_observed_step=jnp.full((1, 8, 10), -1, dtype=jnp.int32),
        )
        updated = agent.update_observed_grid(jnp.array(5), height=8, width=10)
        # エージェントの位置 (2,3) は観測済みになるはず
        assert int(updated.grid_observed_step[0, 2, 3]) == 5

    def test_update_returns_agent_pytree(self):
        agent = _make_agent([[2, 3]], [[-1, 0]])
        updated = agent.update_observed_grid(jnp.array(1), height=8, width=10)
        assert isinstance(updated, Agent)


class TestAgentStr:
    """Agent.__str__ の文字列出力テスト (L99-111)."""

    def test_str_returns_string(self):
        agent = _make_agent([[2, 3]], [[-1, 0]])
        assert isinstance(str(agent), str)

    def test_str_contains_agent_label(self):
        agent = _make_agent([[2, 3]], [[-1, 0]])
        assert "agent0" in str(agent)

    def test_str_two_agents_both_labeled(self):
        agent = _make_agent([[2, 3], [5, 6]], [[-1, 0], [1, 0]])
        result = str(agent)
        assert "agent0" in result
        assert "agent1" in result

    def test_str_contains_pos_info(self):
        agent = _make_agent([[2, 3]], [[-1, 0]])
        assert "pos" in str(agent)


class TestNumActions:
    """Agent.num_actions プロパティのテスト."""

    def test_num_actions_equals_capacity_plus_base(self):
        agent = _make_agent([[2, 3]], [[-1, 0]], capacity=3)
        # num_actions は配列 (per-agent)
        assert int(agent.num_actions[0]) == 3 + Actions.PICK_PLACE_BASE

    def test_num_actions_scales_with_capacity(self):
        agent_cap2 = _make_agent([[2, 3]], [[-1, 0]], capacity=2)
        agent_cap4 = _make_agent([[2, 3]], [[-1, 0]], capacity=4)
        assert int(agent_cap4.num_actions[0]) > int(agent_cap2.num_actions[0])

    def test_num_actions_two_agents(self):
        agent = _make_agent([[2, 3], [4, 5]], [[-1, 0], [1, 0]], capacity=3)
        chex.assert_shape(agent.num_actions, (2,))
        assert int(agent.num_actions[0]) == int(agent.num_actions[1])
