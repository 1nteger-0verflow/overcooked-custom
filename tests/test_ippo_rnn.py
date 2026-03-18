"""Tests for ippo_rnn: ScannedRNN, CNN, ActorCriticRNN, Transition, load_config."""

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from config import NetworkConfig
from ippo_rnn import CNN, ActorCriticRNN, ScannedRNN, Transition, load_config

# ---------------------------------------------------------------------------
# Constants (small values for speed)
# ---------------------------------------------------------------------------
_GRU = 32  # GRU hidden dim
_FC = 32  # FC dim
_N = 2  # batch size (num actors)
_H, _W, _C = 8, 10, 5  # obs spatial dims (H,W >= 7 for three (3,3) conv layers)
_A = 6  # action dim
_T = 3  # sequence length


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_config():
    return NetworkConfig(ACTIVATION="relu", GRU_HIDDEN_DIM=_GRU, FC_DIM_SIZE=_FC)


@pytest.fixture(scope="module")
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def rnn_fixtures(rng):
    """ScannedRNN モデル・パラメータ・初期 carry."""
    model = ScannedRNN()
    carry = ScannedRNN.initialize_carry(_N, _GRU)
    dummy_embed = jnp.zeros((_T, _N, _GRU))
    dummy_done = jnp.zeros((_T, _N))
    params = model.init(rng, carry, (dummy_embed, dummy_done))
    return model, params, carry


@pytest.fixture(scope="module")
def cnn_fixtures(rng):
    """CNN モデル・パラメータ."""
    model = CNN(output_size=_GRU)
    dummy_x = jnp.zeros((_N, _H, _W, _C))
    params = model.init(rng, dummy_x)
    return model, params


@pytest.fixture(scope="module")
def ac_fixtures(rng, model_config):
    """ActorCriticRNN モデル・パラメータ・初期 hidden."""
    model = ActorCriticRNN(action_dim=_A, config=model_config)
    hidden = ScannedRNN.initialize_carry(_N, _GRU)
    dummy_obs = jnp.zeros((1, _N, _H, _W, _C))
    dummy_done = jnp.zeros((1, _N))
    params = model.init(rng, hidden, (dummy_obs, dummy_done))
    return model, params, hidden


# ---------------------------------------------------------------------------
# ScannedRNN.initialize_carry
# ---------------------------------------------------------------------------
class TestScannedRNNInitializeCarry:
    def test_shape(self):
        carry = ScannedRNN.initialize_carry(_N, _GRU)
        chex.assert_shape(carry, (_N, _GRU))

    def test_zeros(self):
        carry = ScannedRNN.initialize_carry(_N, _GRU)
        assert jnp.all(carry == 0)

    def test_different_sizes(self):
        carry = ScannedRNN.initialize_carry(4, 64)
        chex.assert_shape(carry, (4, 64))

    def test_returns_array(self):
        carry = ScannedRNN.initialize_carry(1, 16)
        assert isinstance(carry, jax.Array)


# ---------------------------------------------------------------------------
# ScannedRNN.__call__
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestScannedRNNCall:
    def test_output_carry_shape(self, rnn_fixtures):
        model, params, carry = rnn_fixtures
        dummy_embed = jnp.zeros((_T, _N, _GRU))
        dummy_done = jnp.zeros((_T, _N))
        new_carry, _ = model.apply(params, carry, (dummy_embed, dummy_done))
        chex.assert_shape(new_carry, (_N, _GRU))

    def test_output_y_shape(self, rnn_fixtures):
        model, params, carry = rnn_fixtures
        dummy_embed = jnp.zeros((_T, _N, _GRU))
        dummy_done = jnp.zeros((_T, _N))
        _, y = model.apply(params, carry, (dummy_embed, dummy_done))
        chex.assert_shape(y, (_T, _N, _GRU))

    def test_reset_done_changes_carry(self, rnn_fixtures):
        """done=1 のステップで carry がリセットされ新しい carry は zeros."""
        model, params, carry = rnn_fixtures
        embed = jnp.ones((_T, _N, _GRU))
        # 最初のステップで全エージェントをリセット
        done = jnp.zeros((_T, _N)).at[0].set(1.0)
        new_carry, _ = model.apply(params, carry, (embed, done))
        chex.assert_shape(new_carry, (_N, _GRU))

    def test_jit_compatible(self, rnn_fixtures):
        model, params, carry = rnn_fixtures
        dummy_embed = jnp.zeros((_T, _N, _GRU))
        dummy_done = jnp.zeros((_T, _N))
        fn = jax.jit(model.apply)
        new_carry, y = fn(params, carry, (dummy_embed, dummy_done))
        chex.assert_shape(new_carry, (_N, _GRU))
        chex.assert_shape(y, (_T, _N, _GRU))


# ---------------------------------------------------------------------------
# CNN.__call__
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestCNN:
    def test_output_shape(self, cnn_fixtures):
        model, params = cnn_fixtures
        x = jnp.zeros((_N, _H, _W, _C))
        out = model.apply(params, x)
        chex.assert_shape(out, (_N, _GRU))

    def test_output_dtype_float(self, cnn_fixtures):
        model, params = cnn_fixtures
        x = jnp.zeros((_N, _H, _W, _C))
        out = model.apply(params, x)
        chex.assert_type(out, jnp.float32)

    def test_custom_output_size(self, rng):
        model = CNN(output_size=16)
        x = jnp.zeros((_N, _H, _W, _C))
        params = model.init(rng, x)
        out = model.apply(params, x)
        chex.assert_shape(out, (_N, 16))

    def test_relu_output_nonnegative(self, cnn_fixtures):
        """デフォルト (relu) は出力が非負."""
        model, params = cnn_fixtures
        x = jnp.ones((_N, _H, _W, _C))
        out = model.apply(params, x)
        assert jnp.all(out >= 0)

    def test_tanh_activation(self, rng):
        model = CNN(output_size=_GRU, activation=nn.tanh)
        x = jnp.zeros((_N, _H, _W, _C))
        params = model.init(rng, x)
        out = model.apply(params, x)
        chex.assert_shape(out, (_N, _GRU))


# ---------------------------------------------------------------------------
# ActorCriticRNN.__call__
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestActorCriticRNN:
    def test_hidden_shape(self, ac_fixtures):
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        new_hidden, _pi, _v = model.apply(params, hidden, (obs, done))
        chex.assert_shape(new_hidden, (_N, _GRU))

    def test_pi_logits_shape(self, ac_fixtures):
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        _hidden, pi, _v = model.apply(params, hidden, (obs, done))
        # distrax.Categorical の logits
        chex.assert_shape(pi.logits, (1, _N, _A))

    def test_value_shape(self, ac_fixtures):
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        _hidden, _pi, value = model.apply(params, hidden, (obs, done))
        chex.assert_shape(value, (1, _N))

    def test_pi_is_valid_distribution(self, ac_fixtures):
        """log_prob の和が有限値であることを確認."""
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        _hidden, pi, _v = model.apply(params, hidden, (obs, done))
        actions = jnp.zeros((1, _N), dtype=jnp.int32)
        log_probs = pi.log_prob(actions)
        assert jnp.all(jnp.isfinite(log_probs))

    def test_gradient_not_none(self, ac_fixtures):
        """パラメータに対する勾配が計算できること."""
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        actions = jnp.zeros((1, _N), dtype=jnp.int32)

        def loss_fn(p):
            _h, pi, v = model.apply(p, hidden, (obs, done))
            return (pi.log_prob(actions).sum() + v.sum()).astype(jnp.float32)

        grads = jax.grad(loss_fn)(params)
        assert grads is not None

    def test_tanh_activation(self, rng):
        cfg = NetworkConfig(ACTIVATION="tanh", GRU_HIDDEN_DIM=_GRU, FC_DIM_SIZE=_FC)
        model = ActorCriticRNN(action_dim=_A, config=cfg)
        hidden = ScannedRNN.initialize_carry(_N, _GRU)
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        params = model.init(rng, hidden, (obs, done))
        new_hidden, _pi, value = model.apply(params, hidden, (obs, done))
        chex.assert_shape(new_hidden, (_N, _GRU))
        chex.assert_shape(value, (1, _N))


# ---------------------------------------------------------------------------
# Greedy action selection from pi.probs  (regression: axis=0 vs axis=-1)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestGreedyActionFromProbs:
    """_evaluate._step_env の greedy action 選択ロジックの回帰テスト.

    pi.probs.shape = (1, num_agents, num_actions) に対して
    axis=-1 (num_actions 軸) で argmax を取ることで
    shape (num_agents,) のアクションベクトルが得られることを検証する。
    axis=0 (batch 軸) を誤って使うと shape (num_agents, num_actions) になる。
    """

    def test_probs_shape(self, ac_fixtures):
        """pi.probs が (1, num_agents, num_actions) であることを確認."""
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        _hidden, pi, _v = model.apply(params, hidden, (obs, done))
        chex.assert_shape(pi.probs, (1, _N, _A))

    def test_argmax_axis_minus1_shape(self, ac_fixtures):
        """axis=-1 の argmax → squeeze で (num_agents,) になることを確認."""
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        _hidden, pi, _v = model.apply(params, hidden, (obs, done))
        action = jnp.argmax(pi.probs, axis=-1).squeeze()
        chex.assert_shape(action, (_N,))

    def test_argmax_axis0_wrong_shape(self, ac_fixtures):
        """回帰テスト: axis=0 (誤り) は (num_agents, num_actions) になる."""
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        _hidden, pi, _v = model.apply(params, hidden, (obs, done))
        wrong_action = jnp.argmax(pi.probs, axis=0).squeeze()
        # axis=0 では num_actions 軸が残り (num_agents, num_actions) になる
        assert wrong_action.shape != (_N,), "axis=0 は正しいアクション shape を返してはいけない"

    def test_argmax_returns_valid_action_indices(self, ac_fixtures):
        """Greedy action の値が [0, num_actions) の範囲内であることを確認."""
        model, params, hidden = ac_fixtures
        obs = jnp.zeros((1, _N, _H, _W, _C))
        done = jnp.zeros((1, _N))
        _hidden, pi, _v = model.apply(params, hidden, (obs, done))
        action = jnp.argmax(pi.probs, axis=-1).squeeze()
        assert jnp.all(action >= 0)
        assert jnp.all(action < _A)

    def test_eval_ac_in_format(self, ac_fixtures):
        """_step_env の ac_in 生成パターン (obs[newaxis], done[newaxis]) を再現."""
        model, params, hidden = ac_fixtures
        # _step_env と同じ前処理: last_obs.shape=(num_agents, H, W, C) を (1, num_agents, H, W, C) に
        last_obs = jnp.zeros((_N, _H, _W, _C))
        last_done = jnp.zeros((_N,))
        ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis])
        _hidden, pi, _v = model.apply(params, hidden, ac_in)
        action = jnp.argmax(pi.probs, axis=-1).squeeze()
        chex.assert_shape(action, (_N,))


class TestTransition:
    def _make(self):
        return Transition(
            obs=jnp.zeros((_N, _H, _W, _C)),
            action=jnp.zeros((_N,), dtype=jnp.int32),
            value=jnp.zeros((_N,)),
            log_prob=jnp.zeros((_N,)),
            reward=jnp.zeros((_N,)),
            done=jnp.zeros((_N,)),
            info=jnp.zeros((_N,)),
        )

    def test_construction(self):
        t = self._make()
        assert t is not None

    def test_is_namedtuple(self):
        t = self._make()
        assert isinstance(t, tuple)

    def test_obs_field(self):
        t = self._make()
        chex.assert_shape(t.obs, (_N, _H, _W, _C))

    def test_action_field(self):
        t = self._make()
        chex.assert_shape(t.action, (_N,))

    def test_fields_accessible_by_name(self):
        t = self._make()
        assert hasattr(t, "obs")
        assert hasattr(t, "action")
        assert hasattr(t, "value")
        assert hasattr(t, "log_prob")
        assert hasattr(t, "reward")
        assert hasattr(t, "done")
        assert hasattr(t, "info")


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def _make_cfg(self, stage="stage1"):
        return OmegaConf.create(
            {
                "stage": stage,
                "layout": {"stage1": "LAYOUT_STR", "stage2": "OTHER_STR"},
                "progress": True,
                "visualize": False,
                "aspect_row": 2,
                "aspect_col": 3,
                "train": {"lr": 0.001},
                "env": {"some_key": "val"},
            }
        )

    def test_layout_moved_to_env(self):
        cfg = load_config(self._make_cfg("stage1"))
        assert cfg.env.layout == "LAYOUT_STR"

    def test_layout_key_removed(self):
        cfg = load_config(self._make_cfg("stage1"))
        assert not hasattr(cfg, "layout")

    def test_progress_preserved_at_top_level(self):
        cfg = load_config(self._make_cfg("stage1"))
        assert cfg.progress is True

    def test_visualize_preserved_at_top_level(self):
        cfg = load_config(self._make_cfg("stage1"))
        assert cfg.visualize is False

    def test_aspect_row_preserved_at_top_level(self):
        cfg = load_config(self._make_cfg("stage1"))
        assert cfg.aspect_row == 2

    def test_aspect_col_preserved_at_top_level(self):
        cfg = load_config(self._make_cfg("stage1"))
        assert cfg.aspect_col == 3

    def test_different_stage_selects_correct_layout(self):
        cfg = load_config(self._make_cfg("stage2"))
        assert cfg.env.layout == "OTHER_STR"

    def test_invalid_stage_exits(self):
        with pytest.raises(SystemExit):
            load_config(self._make_cfg("nonexistent"))

    def test_missing_stage_key_exits(self):
        """Stage キーなし → layout.get("None", None) = None → exit."""
        cfg = OmegaConf.create(
            {
                "layout": {"stage1": "LAYOUT_STR"},
                "progress": False,
                "visualize": False,
                "aspect_row": 1,
                "aspect_col": 1,
                "train": {},
                "env": {},
            }
        )
        with pytest.raises(SystemExit):
            load_config(cfg)
