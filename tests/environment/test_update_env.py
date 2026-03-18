"""Tests for environment.update_env."""

import jax
import jax.numpy as jnp
import pytest

from config import EnvParameterConfig
from environment.dynamic_object import DynamicObject
from environment.state import Channel
from environment.static_object import StaticObject
from environment.update_env import progress_cooking


def _find_first_pot(state) -> tuple[int, int]:
    """グリッド内の最初の POT セルの (row, col) を返す."""
    pot_mask = state.grid[:, :, Channel.env] == StaticObject.POT
    pos = jnp.argwhere(pot_mask, size=1)[0]
    return int(pos[0]), int(pos[1])


class TestProgressCookingVolumeClip:
    """progress_cooking で volume が常に >= 1 にクリップされること."""

    @pytest.fixture
    def pot_pos(self, compact_state):
        return _find_first_pot(compact_state)

    @pytest.fixture
    def about_to_finish_state(self, compact_state, pot_pos):
        """ポットに recipe [0,0,0] の食材が入り、あと 1 ステップで調理完了する State."""
        r, c = pot_pos
        # 3 * BASE_INGREDIENT = ingredient_0 が 3 個 → conftest menu の recipe [0,0,0]
        ing_obj = 3 * int(DynamicObject.BASE_INGREDIENT)
        new_grid = compact_state.grid.at[r, c, Channel.obj].set(ing_obj)
        new_grid = new_grid.at[r, c, Channel.extra].set(1)  # あと 1 ステップで完了
        return compact_state.replace(grid=new_grid)

    def test_volume_clipped_to_at_least_one(self, about_to_finish_state, pot_pos):
        """volume_range が極小でも調理完了後の volume >= 1 になること.

        conftest menu: recipe [0,0,0], volume=4
        coeff=0.1 → floor(4 * 0.1) = 0 → clip(min=1) → 1
        """
        r, c = pot_pos
        param = EnvParameterConfig(volume_range=(0.1, 0.1))
        key = jax.random.PRNGKey(0)

        new_state = progress_cooking(about_to_finish_state, param, key)

        cooked_obj = new_state.grid[r, c, Channel.obj]
        assert bool((cooked_obj & DynamicObject.COOKED) != 0), "調理が完了していない"
        volume = int(DynamicObject.get_count(cooked_obj))
        assert volume >= 1, f"volume={volume} は 1 未満 (clip が効いていない)"

    def test_volume_normal_range_is_positive(self, about_to_finish_state, pot_pos):
        """通常の volume_range (1.0, 1.0) でも volume >= 1 になること."""
        r, c = pot_pos
        param = EnvParameterConfig(volume_range=(1.0, 1.0))
        key = jax.random.PRNGKey(0)

        new_state = progress_cooking(about_to_finish_state, param, key)
        cooked_obj = new_state.grid[r, c, Channel.obj]
        volume = int(DynamicObject.get_count(cooked_obj))
        assert volume >= 1

    def test_cooking_in_progress_does_not_set_cooked_flag(self, compact_state, pot_pos):
        """extra > 1 のとき (調理中) COOKED フラグが立たないこと."""
        r, c = pot_pos
        ing_obj = 3 * int(DynamicObject.BASE_INGREDIENT)
        new_grid = compact_state.grid.at[r, c, Channel.obj].set(ing_obj)
        new_grid = new_grid.at[r, c, Channel.extra].set(5)
        state = compact_state.replace(grid=new_grid)

        param = EnvParameterConfig(volume_range=(1.0, 1.0))
        key = jax.random.PRNGKey(0)
        new_state = progress_cooking(state, param, key)

        cooked_obj = new_state.grid[r, c, Channel.obj]
        assert bool((cooked_obj & DynamicObject.COOKED) == 0), "調理中なのに COOKED が立った"
        assert int(new_state.grid[r, c, Channel.extra]) == 4, "extra がデクリメントされていない"
