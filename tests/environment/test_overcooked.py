"""Integration tests for environment.overcooked.OvercookedCustom.

- gymnasium.utils.env_checker は JAX-native インターフェース (action_space・observation_space 未実装) に
  非互換のため適用外。代わりに chex + pytest でインターフェース・形状・型を網羅的に検証する。
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from environment.actions import Actions
from environment.overcooked import OvercookedCustom
from environment.reward import RewardType
from environment.state import State


# ---------------------------------------------------------------------------
# プロパティテスト
# ---------------------------------------------------------------------------
class TestProperties:
    def test_name(self, minimal_env):
        assert minimal_env.name == "Overcooked custom"

    def test_num_agents_minimal(self, minimal_env):
        assert minimal_env.num_agents == 1

    def test_num_agents_compact(self, compact_env):
        assert compact_env.num_agents == 2

    def test_num_actions_equals_action_set_length(self, minimal_env):
        assert minimal_env.num_actions == len(minimal_env.action_set)

    def test_num_actions_is_positive(self, minimal_env):
        assert minimal_env.num_actions > 0

    def test_obs_shape_is_4tuple(self, minimal_env):
        assert len(minimal_env.obs_shape) == 4

    def test_obs_shape_first_dim_is_num_agents(self, minimal_env):
        assert minimal_env.obs_shape[0] == minimal_env.num_agents

    def test_max_steps(self, minimal_env, minimal_config):
        assert minimal_env.max_steps == int(minimal_config.schedule.terminal_time)


# ---------------------------------------------------------------------------
# reset() テスト
# ---------------------------------------------------------------------------
class TestReset:
    def test_obs_shape_matches_env(self, minimal_env, prng_key):
        obs, _ = minimal_env.reset(prng_key)
        chex.assert_shape(obs, minimal_env.obs_shape)

    def test_state_is_state_type(self, minimal_env, prng_key):
        _, state = minimal_env.reset(prng_key)
        assert isinstance(state, State)

    def test_initial_time_zero(self, minimal_env, prng_key):
        _, state = minimal_env.reset(prng_key)
        assert int(state.time) == 0

    def test_reset_is_deterministic(self, minimal_env, prng_key):
        obs1, _ = minimal_env.reset(prng_key)
        obs2, _ = minimal_env.reset(prng_key)
        chex.assert_trees_all_equal(obs1, obs2)

    def test_obs_dtype_is_float_or_int(self, minimal_env, prng_key):
        obs, _ = minimal_env.reset(prng_key)
        assert obs.dtype in (jnp.float32, jnp.int32, jnp.float16)


# ---------------------------------------------------------------------------
# step_env() テスト
# ---------------------------------------------------------------------------
class TestStepEnv:
    @pytest.fixture
    def step_result_minimal(self, minimal_env, minimal_state, prng_key):
        actions = jnp.array([int(Actions.STAY)])
        key = jax.random.PRNGKey(1)
        return minimal_env.step_env(minimal_state, actions, key)

    @pytest.fixture
    def step_result_compact(self, compact_env, compact_state):
        actions = jnp.array([int(Actions.STAY), int(Actions.STAY)])
        key = jax.random.PRNGKey(2)
        return compact_env.step_env(compact_state, actions, key)

    def test_obs_shape(self, step_result_minimal, minimal_env):
        obs, *_ = step_result_minimal
        chex.assert_shape(obs, minimal_env.obs_shape)

    def test_state_type(self, step_result_minimal):
        _, state, *_ = step_result_minimal
        assert isinstance(state, State)

    def test_time_incremented(self, minimal_env, minimal_state):
        actions = jnp.array([int(Actions.STAY)])
        key = jax.random.PRNGKey(3)
        _, new_state, *_ = minimal_env.step_env(minimal_state, actions, key)
        assert int(new_state.time) == int(minimal_state.time) + 1

    def test_rewards_shape_minimal(self, step_result_minimal, minimal_env):
        _, _, rewards, shaped_rewards, _, _ = step_result_minimal
        chex.assert_shape(rewards, (minimal_env.num_agents,))
        chex.assert_shape(shaped_rewards, (minimal_env.num_agents,))

    def test_rewards_shape_compact(self, step_result_compact, compact_env):
        _, _, rewards, shaped_rewards, _, _ = step_result_compact
        chex.assert_shape(rewards, (compact_env.num_agents,))
        chex.assert_shape(shaped_rewards, (compact_env.num_agents,))

    def test_done_is_scalar(self, step_result_minimal):
        *_, done = step_result_minimal
        assert done.ndim == 0

    def test_done_false_at_start(self, step_result_minimal):
        *_, done = step_result_minimal
        assert not bool(done)

    def test_reward_type_is_valid(self, step_result_minimal):
        import jax.numpy as jnp

        _, _, _, _, reward_type, _ = step_result_minimal
        valid = {int(r) for r in RewardType}
        assert int(jnp.squeeze(reward_type)) in valid

    @pytest.mark.parametrize(
        "action", [Actions.RIGHT, Actions.DOWN, Actions.LEFT, Actions.UP, Actions.STAY, Actions.INTERACT]
    )
    def test_all_basic_actions_run(self, minimal_env, minimal_state, action):
        actions = jnp.array([int(action)])
        key = jax.random.PRNGKey(10 + int(action))
        obs, state, rewards, shaped, rtype, done = minimal_env.step_env(minimal_state, actions, key)
        chex.assert_shape(obs, minimal_env.obs_shape)
        assert isinstance(state, State)

    def test_step_jit_compiled(self, minimal_env, minimal_state):
        """step_env は @jax.jit 装飾済みで JIT 実行されること."""
        actions = jnp.array([int(Actions.STAY)])
        key = jax.random.PRNGKey(0)
        # 2回呼んでも例外が出ないことを確認 (JITキャッシュ)
        minimal_env.step_env(minimal_state, actions, key)
        minimal_env.step_env(minimal_state, actions, key)


# ---------------------------------------------------------------------------
# update_timestep() テスト
# ---------------------------------------------------------------------------
class TestUpdateTimestep:
    def test_increments_time(self, minimal_env, minimal_state):
        new_state, done = minimal_env.update_timestep(minimal_state)
        assert int(new_state.time) == int(minimal_state.time) + 1

    def test_done_before_terminal_time(self, minimal_env, minimal_state):
        _, done = minimal_env.update_timestep(minimal_state)
        assert not bool(done)

    def test_done_at_terminal_time(self, minimal_env, minimal_config):
        from omegaconf import OmegaConf

        terminal = int(minimal_config.schedule.terminal_time)
        _, state = minimal_env.reset(jax.random.PRNGKey(0))
        state = state.replace(time=jnp.array(terminal - 1))
        _, done = minimal_env.update_timestep(state)
        assert bool(done)


# ---------------------------------------------------------------------------
# 複数ステップのロールアウトテスト
# ---------------------------------------------------------------------------
class TestRollout:
    def test_short_rollout_does_not_raise(self, minimal_env, prng_key):
        obs, state = minimal_env.reset(prng_key)
        key = prng_key
        for step in range(5):
            key, subkey = jax.random.split(key)
            actions = jnp.array([int(Actions.STAY)])
            obs, state, rewards, shaped, rtype, done = minimal_env.step_env(state, actions, subkey)
        chex.assert_shape(obs, minimal_env.obs_shape)

    def test_rollout_state_time_increases(self, minimal_env, prng_key):
        _, state = minimal_env.reset(prng_key)
        key = prng_key
        for _ in range(3):
            key, subkey = jax.random.split(key)
            actions = jnp.array([int(Actions.STAY)])
            _, state, *_ = minimal_env.step_env(state, actions, subkey)
        assert int(state.time) == 3
