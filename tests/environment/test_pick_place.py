"""Tests for environment.pick_place — reward branches."""

import jax
import jax.numpy as jnp
import pytest

from config import EnvParameterConfig, RewardConfig
from environment.agent import Agent
from environment.dynamic_object import DynamicObject
from environment.pick_place import pick_and_place
from environment.state import Channel

# compact layout の POT 位置 (WCP   ATcW の P = row 2, col 2)
_POT_ROW, _POT_COL = 2, 2
_COMPACT_HEIGHT, _COMPACT_WIDTH = 8, 10


def _single_agent(row: int, col: int, dir_r: int, dir_c: int, inventory: list[int]) -> Agent:
    """pick_and_place に渡す単一エージェントを作成する."""
    capacity = len(inventory)
    inv = jnp.array(inventory, dtype=jnp.int32)
    return Agent(
        pos=jnp.array([row, col], dtype=jnp.int32),
        dir=jnp.array([dir_r, dir_c], dtype=jnp.int32),
        capacity=jnp.array(capacity, dtype=jnp.int32),
        inventory=inv,
        view_sizes=jnp.array([0, 0], dtype=jnp.int32),
        grid_observed_step=jnp.full((_COMPACT_HEIGHT, _COMPACT_WIDTH), -1, dtype=jnp.int32),
    )


def _state_with_two_ing0_in_pot(compact_state):
    """ポットに ingredient_0 が 2 個入った State (3 個目を入れると _start_cooking が起動)."""
    ing_obj = 2 * int(DynamicObject.BASE_INGREDIENT)
    new_grid = compact_state.grid.at[_POT_ROW, _POT_COL, Channel.obj].set(ing_obj)
    new_grid = new_grid.at[_POT_ROW, _POT_COL, Channel.extra].set(0)
    return compact_state.replace(grid=new_grid)


class TestCookingDurationClip:
    """_start_cooking で cooking_duration が clip(min=1) されること."""

    def test_cooking_duration_is_at_least_one_when_range_is_tiny(self, compact_state):
        """cooking_duration_range が極小でも調理時間 >= 1 になること.

        conftest menu: recipe [0,0,0], duration=2
        coeff=0.01 → floor(2 * 0.01) = 0 → clip(min=1) → 1
        """
        state = _state_with_two_ing0_in_pot(compact_state)
        ing0 = int(DynamicObject.ingredient(0))
        # agent: (2, 3) 向き左 (dir=[0,-1]) → fwd_pos = (2, 2) = POT
        agent = _single_agent(2, 3, 0, -1, [ing0, 0, 0])

        param = EnvParameterConfig(cooking_duration_range=(0.01, 0.01))
        key = jax.random.PRNGKey(0)

        new_state, _, _, _, _ = pick_and_place(state, agent, key, jnp.array(0), RewardConfig(), param)

        pot_extra = int(new_state.grid[_POT_ROW, _POT_COL, Channel.extra])
        assert pot_extra >= 1, f"cooking_duration={pot_extra} < 1 (clip が効いていない)"

    def test_cooking_duration_is_positive_with_normal_range(self, compact_state):
        """通常の cooking_duration_range でも duration >= 1 になること."""
        state = _state_with_two_ing0_in_pot(compact_state)
        ing0 = int(DynamicObject.ingredient(0))
        agent = _single_agent(2, 3, 0, -1, [ing0, 0, 0])

        param = EnvParameterConfig(cooking_duration_range=(1.0, 1.0))
        key = jax.random.PRNGKey(0)

        new_state, _, _, _, _ = pick_and_place(state, agent, key, jnp.array(0), RewardConfig(), param)

        pot_extra = int(new_state.grid[_POT_ROW, _POT_COL, Channel.extra])
        assert pot_extra >= 1


class TestCookingReward:
    """注文済み/未注文の料理を調理したときの報酬分岐."""

    def _dish_encoding(self) -> int:
        """recipe [0,0,0] の完成品エンコーディング (count=0)."""
        ing_obj = 3 * int(DynamicObject.BASE_INGREDIENT)
        return int(DynamicObject.set_count(ing_obj | DynamicObject.COOKED | DynamicObject.PLATE, 0))

    def test_ordered_dish_gives_pot_start_cooking_reward(self, compact_state):
        """ordered_menu に含まれるレシピを調理開始したとき pot_start_cooking 報酬が返ること."""
        state = _state_with_two_ing0_in_pot(compact_state)
        # ordered_menu[0, 0] に完成品を設定 → is_ordered = True
        dish = self._dish_encoding()
        new_ordered = state.customer.ordered_menu.at[0, 0].set(dish)
        state = state.replace(customer=state.customer.replace(ordered_menu=new_ordered))

        ing0 = int(DynamicObject.ingredient(0))
        agent = _single_agent(2, 3, 0, -1, [ing0, 0, 0])
        reward_cfg = RewardConfig()
        param = EnvParameterConfig(cooking_duration_range=(1.0, 1.0))

        _, _, _, shaped_reward, _ = pick_and_place(state, agent, jax.random.PRNGKey(0), jnp.array(0), reward_cfg, param)

        assert float(shaped_reward) == pytest.approx(reward_cfg.shaped_reward.pot_start_cooking)

    def test_unordered_dish_gives_erroneous_cooking_penalty(self, compact_state):
        """ordered_menu が空のとき erroneous_cooking ペナルティが返ること."""
        state = _state_with_two_ing0_in_pot(compact_state)
        # ordered_menu を全て EMPTY に (デフォルト: リセット直後は全 0)
        new_ordered = jnp.zeros_like(state.customer.ordered_menu)
        state = state.replace(customer=state.customer.replace(ordered_menu=new_ordered))

        ing0 = int(DynamicObject.ingredient(0))
        agent = _single_agent(2, 3, 0, -1, [ing0, 0, 0])
        reward_cfg = RewardConfig()
        param = EnvParameterConfig(cooking_duration_range=(1.0, 1.0))

        _, _, _, shaped_reward, _ = pick_and_place(state, agent, jax.random.PRNGKey(0), jnp.array(0), reward_cfg, param)

        assert float(shaped_reward) == pytest.approx(-reward_cfg.penalty.erroneous_cooking)
