"""Tests for environment.reset_env.Initializer."""

import chex
import jax
import jax.numpy as jnp
import pytest

from environment.state import State


class TestInitializerMinimal:
    """最小構成 (1エージェント) での初期化テスト."""

    def test_initialize_returns_state(self, minimal_env, prng_key):
        _, state = minimal_env.reset(prng_key)
        assert isinstance(state, State)

    def test_grid_shape(self, minimal_state, minimal_env):
        h, w = minimal_env.height, minimal_env.width
        chex.assert_shape(minimal_state.grid, (h, w, 3))

    def test_agent_pos_shape(self, minimal_state):
        pos = minimal_state.agents.pos
        chex.assert_shape(pos, (1, 2))

    def test_agent_inventory_shape(self, minimal_state):
        inv = minimal_state.agents.inventory
        # 1エージェント、capacity=3
        chex.assert_shape(inv, (1, 3))

    def test_initial_time_is_zero(self, minimal_state):
        assert int(minimal_state.time) == 0

    def test_agent_direction_is_up(self, minimal_state):
        # 初期向きは UP=(-1,0)
        dir_ = minimal_state.agents.dir[0]
        assert list(map(int, dir_)) == [-1, 0]

    def test_prev_actions_is_stay(self, minimal_state):
        from environment.actions import Actions

        prev = minimal_state.prev_actions
        chex.assert_shape(prev, (1,))
        assert int(prev[0]) == int(Actions.STAY)


class TestInitializerCompact:
    """2エージェント・フル機能での初期化テスト."""

    def test_grid_shape(self, compact_state, compact_env):
        h, w = compact_env.height, compact_env.width
        chex.assert_shape(compact_state.grid, (h, w, 3))

    def test_two_agents(self, compact_state):
        chex.assert_shape(compact_state.agents.pos, (2, 2))

    def test_agent_pos_within_bounds(self, compact_state, compact_env):
        for i in range(compact_env.num_agents):
            y, x = map(int, compact_state.agents.pos[i])
            assert 0 <= y < compact_env.height
            assert 0 <= x < compact_env.width

    def test_customer_status_all_empty(self, compact_state):
        from environment.customer import CustomerStatus

        statuses = compact_state.customer.status
        assert all(int(s) == int(CustomerStatus.empty) for s in statuses)

    def test_inventory_starts_empty(self, compact_state):
        inv = compact_state.agents.inventory
        assert jnp.all(inv == 0)

    def test_grid_static_layer_nonzero(self, compact_state):
        # Channel 0 (env) には壁などが入っていて全ゼロではない
        env_layer = compact_state.grid[:, :, 0]
        assert jnp.any(env_layer > 0)

    def test_deterministic_with_same_key(self, compact_env, prng_key):
        _, state1 = compact_env.reset(prng_key)
        _, state2 = compact_env.reset(prng_key)
        chex.assert_trees_all_equal(state1.agents.pos, state2.agents.pos)

    def test_different_keys_may_differ_with_random_position(self, compact_config):
        """random_agent_position=True のとき鍵が違えば位置が変わりうる."""
        from environment.overcooked import OvercookedCustom

        env = OvercookedCustom(compact_config, random_agent_position=True)
        key1 = jax.random.PRNGKey(1)
        key2 = jax.random.PRNGKey(999)
        _, state1 = env.reset(key1)
        _, state2 = env.reset(key2)
        # 必ずしも異なるとは限らないが例外なく実行できることを確認
        assert isinstance(state1, State)
        assert isinstance(state2, State)


class TestInitializerObsShape:
    """reset() が返す obs の形状テスト."""

    def test_obs_shape_minimal(self, minimal_env, prng_key):
        obs, _ = minimal_env.reset(prng_key)
        expected = minimal_env.obs_shape
        chex.assert_shape(obs, expected)

    def test_obs_shape_compact(self, compact_env, prng_key):
        obs, _ = compact_env.reset(prng_key)
        expected = compact_env.obs_shape
        chex.assert_shape(obs, expected)
