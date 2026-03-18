import dataclasses
import functools
import sys
from collections import abc
from pathlib import Path
from typing import NamedTuple

import absl.logging
import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint as ocp
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from orbax.checkpoint.checkpoint_managers import preservation_policy
from tqdm import tqdm

from config import AppConfig, NetworkConfig, TrainConfig, app_config_from_omegaconf, network_config_from_train
from environment.overcooked import OvercookedCustom
from environment.reward import RewardType
from environment.state import State as EnvState
from visualize.visualizer import OvercookedCustomVisualizer


class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry: jax.Array, x: tuple[jax.Array, jax.Array]):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x  # x = (embedding, done)のtuple
        # ins: (NUM_ACTOR, hidden_dim)

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(resets[:, jnp.newaxis], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.key(0), (batch_size, hidden_size))


class CNN(nn.Module):
    # observationからRNNの隠れ層に変換
    output_size: int = 64
    activation: abc.Callable[[jax.Array], jax.Array] = nn.relu

    @nn.compact
    def __call__(self, x: jax.Array, _train: bool = False):
        x = nn.Conv(features=128, kernel_size=(1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)

        x = nn.Conv(features=128, kernel_size=(1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)

        x = nn.Conv(features=8, kernel_size=(1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)

        x = nn.Conv(features=16, kernel_size=(3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)

        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)

        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=self.output_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        return self.activation(x)


class ActorCriticRNN(nn.Module):
    action_dim: int
    config: NetworkConfig

    # https://stackoverflow.com/questions/79658104/how-to-type-hint-flax-linen-module-applys-output-correctly
    # https://github.com/google/flax/pull/4783
    # によると、flax.linenでの修正予定はない
    @nn.compact
    def __call__(self, hidden: jax.Array, x: tuple[jax.Array, jax.Array]):
        obs, dones = x
        embedding = obs
        activation = nn.relu if self.config.ACTIVATION == "relu" else nn.tanh

        embed_model = CNN(output_size=self.config.GRU_HIDDEN_DIM, activation=activation)
        embedding = jax.vmap(embed_model)(embedding)
        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        # hidden: (NUM_ACTORS, hidden_dim)
        # rnn_in.embedding: (1, NUM_ACTORS, hidden_dim)
        # rnn_in.dones: (1, NUM_ACTORS)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config.FC_DIM_SIZE, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        # 方策
        pi = distrax.Categorical(logits=actor_mean)

        # 状態価値(の推論)
        critic = nn.Dense(self.config.FC_DIM_SIZE, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    # env0.agent0, env0.agent1,... , env1.agent0, env1.agent1, ... の順に格納する
    # 元の状態
    obs: jnp.ndarray
    # 選択した行動
    action: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    # 行動結果
    reward: jnp.ndarray
    done: jnp.ndarray
    info: jnp.ndarray


_RunnerState = tuple[TrainState, EnvState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def _adjust_row_col(r: int, c: int, num_envs: int) -> tuple[int, int]:
    for rows in range(1, num_envs):
        for cols in range(1, int(rows * c / r) + 1):
            if rows * cols >= num_envs:
                return (rows, cols)
    return (1, num_envs)


def _create_lr_schedule(train_config: TrainConfig, num_minibatches: int) -> optax.Schedule:
    base_learning_rate = train_config.LR
    lr_warmup = train_config.LR_WARMUP
    total_update_steps = train_config.NUM_TRAINING_STEPS * train_config.NUM_UPDATE_EPOCHS * num_minibatches
    warmup_steps = int(lr_warmup * total_update_steps)
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=base_learning_rate, transition_steps=warmup_steps)
    cosine_steps = max(total_update_steps - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=cosine_steps)
    return optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])


def _create_optimizer(train_config: TrainConfig, lr_schedule: optax.Schedule) -> optax.GradientTransformation:
    if train_config.ANNEAL_LR:
        return optax.chain(optax.clip_by_global_norm(train_config.MAX_GRAD_NORM), optax.adam(lr_schedule, eps=1e-5))
    lr = train_config.LR
    return optax.chain(optax.clip_by_global_norm(train_config.MAX_GRAD_NORM), optax.adam(lr, eps=1e-5))


def _create_ent_schedule(train_config: TrainConfig, num_minibatches: int) -> optax.Schedule:
    if train_config.ANNEAL_ENT:
        total_opt_steps = train_config.NUM_TRAINING_STEPS * train_config.NUM_UPDATE_EPOCHS * num_minibatches
        return optax.linear_schedule(
            init_value=train_config.ENT_COEF,
            end_value=train_config.ENT_END,
            transition_steps=int(total_opt_steps * train_config.ENT_COOLDOWN),
        )
    return optax.constant_schedule(train_config.ENT_COEF)


def _calculate_gae(
    rollout_buffer: Transition, last_val: jax.Array, gamma: float, gae_lambda: float
) -> tuple[jax.Array, jax.Array]:
    def _get_advantages(gae_and_next_value: tuple[jax.Array, jax.Array], transition: Transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (transition.done, transition.value, transition.reward)
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages, (jnp.zeros_like(last_val), last_val), rollout_buffer, reverse=True, unroll=16
    )
    return advantages, advantages + rollout_buffer.value


def _initialize_network_params(
    network: ActorCriticRNN, rng: jax.Array, train_config: TrainConfig, env: OvercookedCustom
) -> dict:
    init_x = (jnp.zeros((1, train_config.NUM_ENVS, *env.obs_shape[1:])), jnp.zeros((1, train_config.NUM_ENVS)))
    init_hstate = ScannedRNN.initialize_carry(train_config.NUM_ENVS, train_config.GRU_HIDDEN_DIM)
    return network.init(rng, init_hstate, init_x)


def _save_metrics(metrics: dict[str, jax.Array], seed_idx: int, model_dir: str) -> None:
    save_dir = Path(model_dir) / f"metrics_{seed_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)
    keys = [
        "update_step",
        "eval_score",
        "original_reward",
        "shaped_reward",
        "combined_reward",
        "loss_total",
        "loss_value",
        "loss_actor",
        "entropy",
        "approx_kl",
        "adv_std",
        "clipfrac",
        "grad_norm",
        "skip_update_rate",
        "lr",
        "anneal_factor",
        "ent_coef",
        "env_step",
    ]
    with open(save_dir / "metrics.csv", "w") as f:
        for key in keys:
            out = f"{key}," + ",".join([str(x) for x in metrics[key]])
            print(out, file=f)
    fig = plt.figure()
    ax = fig.subplots()
    steps = metrics["update_step"]
    for label in keys[1:10]:
        ax.clear()
        ax.plot(steps, metrics[label], label=label)
        ax.set_title(label)
        ax.set_xlabel("update_step")
        fig.savefig(save_dir / f"{label}.png")


def make_train(app_config: AppConfig):
    train_config = app_config.train
    env = OvercookedCustom(app_config.env, random_agent_position=train_config.RANDOM_AGENT_POS)
    num_actors = env.num_agents * train_config.NUM_ENVS
    assert num_actors % train_config.MINIBATCH_SIZE == 0, (
        f"MINIBATCH_SIZE({train_config.MINIBATCH_SIZE}) must devide NUM_ACTORS({num_actors})"
    )
    num_minibatches = num_actors // train_config.MINIBATCH_SIZE
    # ・保存先は絶対パスで指定しなければならない
    model_dir = Path(HydraConfig.get().runtime.output_dir).parent.as_posix()
    # 学習のHORIZONステップ目以降はshaped_rewardの重みが0
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=train_config.REW_SHAPING_HORIZON
    )
    # チェックポイントの保存用設定
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # https://github.com/google/flax/discussions/3130
    absl.logging.set_verbosity(absl.logging.WARNING)
    # https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html#multiple-item-checkpointing
    options = ocp.CheckpointManagerOptions(
        create=True,
        save_interval_steps=train_config.CHECKPOINT_INTERVAL_STEP,
        preservation_policy=preservation_policy.BestN(
            n=train_config.CHECKPOINT_KEEP, get_metric_fn=lambda m: m["eval_score"], reverse=False
        ),
    )
    checkpoint_manager = ocp.CheckpointManager(model_dir, options=options, metadata=dataclasses.asdict(train_config))
    # 進捗表示
    if app_config.progress:
        progress_bar = tqdm(total=train_config.NUM_TRAINING_STEPS)
    # 学習中の可視化
    viz = OvercookedCustomVisualizer()
    viz_rows: int | None = None
    viz_cols: int | None = None
    if app_config.visualize:
        viz_rows, viz_cols = _adjust_row_col(app_config.aspect_row, app_config.aspect_col, train_config.NUM_ENVS)
        viz.show()
    eval_state_seq = []  # チェックポイント保存時に１周評価した結果をgif保存するため

    lr_schedule = _create_lr_schedule(train_config, num_minibatches)
    ent_schedule = _create_ent_schedule(train_config, num_minibatches)

    def save_checkpoint(train_state: TrainState, metric: dict[str, jax.Array], step: int):
        # CheckpointManager.saveのstep数はintでなければならないのでcallbackで実装(update_stepはjitのint32[]でNG)
        checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(train_state.params), obs_shape=ocp.args.StandardSave(env.obs_shape)
            ),
            metrics=metric,
        )
        checkpoint_manager.wait_until_finished()
        # 進捗を更新
        if app_config.progress:
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    k: f"{int(metric[k] * 1000) / 1000:.3f}"
                    for k in ["combined_reward", "shaped_reward", "original_reward"]
                }
            )

    def visualize_state(states: EnvState):
        if app_config.visualize:
            viz.render_multi(
                states, viz_rows, viz_cols, title=f"{states.time[0]} / {env.max_steps} step", caption="caption"
            )

    def train(rng: jax.Array, seed_idx: int):
        # NUM_SEEDS並列に実行
        network = ActorCriticRNN(env.num_actions, config=network_config_from_train(train_config))

        # COLLECT TRAJECTORIES
        def _env_step(last_runner_state: _RunnerState, _: None):
            # 現在の方策でnetworkが状態から各actorの行動を出力し、その行動で環境を1step進める
            (
                train_state,  # env_stepでは更新しない(パラメータを参照するのみ)
                last_env_state,  # NUM_ENVS個の環境
                last_obs,  # (NUM_ENVS, num_agents, height, width, channel)
                last_done,  # (NUM_ACTORS, )
                update_step,  # 学習のstepなのでovercookedの操作をしても更新しない
                hstate,  # (ENV_ACTORS, GRU_HIDDEN_DIM)
                rng,
            ) = last_runner_state

            # 方策から行動を選択するのはactorごとに行うので、(NUM_ACTORS, ...)のshapeにする
            # obs_batch: (NUM_ACTORS, height, width, info_layers)
            obs_batch = last_obs.reshape(num_actors, *env.obs_shape[1:])
            # ac_in.shape:
            # (1, NUM_ACTORS, height, width, info_layer),
            # (1, NUM_ACTORS)
            ac_in = (obs_batch[jnp.newaxis, :], last_done[jnp.newaxis, :])
            # value はcriticの評価値
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            rng, _rng = jax.random.split(rng)
            # SELECT ACTION
            # env0.agent0, env0.agent1, ..., env1.agent0, env1.agent1, ... の順
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            # NUM_ENVS個の環境に対してnum_agent分の行動を入力して、1step実行させる
            # jax.debug.print("action: {}", action, ordered=True)
            # jax.debug.print("log_prob: {}", log_prob, ordered=True)
            # jax.debug.print("value: {}", value, ordered=True)
            env_act = action.reshape((train_config.NUM_ENVS, env.num_agents))
            # env_act:         agent0, agent1, ...
            #          env0  [ [act00, act01, ...],
            #          env1  [ [act10, act11, ...],
            #            :

            # 乱数の更新
            rng, _rng = jax.random.split(rng)
            # 並行環境にそれぞれ違う乱数を適用し、様々な状態が現れるようにする
            rng_step = jax.random.split(_rng, train_config.NUM_ENVS)

            # STEP ENV
            new_obsv, new_env_state, original_reward, shaped_rewards, _, done = jax.vmap(
                env.step_env, in_axes=(0, 0, 0)
            )(last_env_state, env_act, rng_step)
            # debug時の注意：new_env_state.grid.shapeが(NUM_ENVS, height, width, 3)になっているため、
            #               printするときState.__str__のgrid[y,x,ch]!=EMPTY の判定でエラーになる
            # jax.debug.print(
            #    "step: {}\nact: {}\npos: {}",
            #    new_env_state.time,
            #    env_act,
            #    new_env_state.agents.pos,
            # )
            # jax.debug.print("env step: {}", new_env_state.time)

            anneal_factor = rew_shaping_anneal(update_step)
            combined_reward = original_reward + anneal_factor * shaped_rewards
            info = {}
            info["original_reward"] = original_reward
            info["shaped_reward"] = shaped_rewards
            info["anneal_factor"] = jnp.full_like(shaped_rewards, anneal_factor)
            info["combined_reward"] = combined_reward

            info = jax.tree_util.tree_map(lambda x: x.reshape(num_actors), info)

            # ------------------------------------------------------------------------
            # auto-reset(環境がterminateしたとき、次ステップは初期化した状態から開始する)
            # ------------------------------------------------------------------------
            rng, _reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(_reset_rng, train_config.NUM_ENVS)

            def _maybe_reset(done_i: jax.Array, key_i: jax.Array, obs_i: jax.Array, state_i: EnvState):
                def _reset(_: None):
                    return env.reset(key_i)

                def _keep(_: None):
                    return obs_i, state_i

                return jax.lax.cond(done_i, _reset, _keep, operand=None)

            new_obsv, new_env_state = jax.vmap(_maybe_reset)(done, reset_keys, new_obsv, new_env_state)
            # doneはenvにつき1つなのでagent数分複製して(NUM_ACTORS,)の形にする
            done_batch = jnp.repeat(done, env.num_agents)

            new_runner_state = (
                train_state,  # ネットワークは学習していないので更新しない
                new_env_state,  # 1STEPS進んだ状態
                new_obsv,
                done_batch,
                update_step,  # _env_stepでは更新しない
                hstate,
                rng,
            )

            # 1stepの遷移内容
            transition = Transition(
                # 元の状態
                obs=obs_batch,
                # 選択した行動
                action=action.squeeze(),
                value=value.squeeze(),
                log_prob=log_prob.squeeze(),
                # 行動結果
                reward=combined_reward.reshape(num_actors),
                done=done_batch,
                info=info,
            )
            jax.debug.callback(visualize_state, new_env_state)
            # jax.debug.print("transition: {}", transition)
            return new_runner_state, transition

        def _loss_fn(
            params: dict[str, jax.Array],
            init_hstate: jax.Array,
            rollout_buffer: Transition,
            gae: jax.Array,
            targets: jax.Array,
            ent_coef: jax.Array,
        ):
            # 方策勾配法では方策の更新によって得られるデータが変わっていくため、損失の値そのものには意味がない
            # https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

            # RERUN NETWORK
            _, pi, value = network.apply(params, init_hstate, (rollout_buffer.obs, rollout_buffer.done))

            # CALCULATE VALUE LOSS
            value_pred_clipped = rollout_buffer.value + (value - rollout_buffer.value).clip(
                -train_config.CLIP_EPS, train_config.CLIP_EPS
            )
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # CALCULATE ACTOR LOSS
            log_prob = pi.log_prob(rollout_buffer.action)
            log_prob = log_prob.squeeze()
            old_log_prob = rollout_buffer.log_prob.squeeze()
            log_ratio_raw = log_prob - old_log_prob  # (T, B)
            # approx KL（SpinningUp等で使われる近似）
            approx_kl = (-log_ratio_raw).mean()
            approx_kl = jnp.nan_to_num(approx_kl, nan=0.0, posinf=1e9, neginf=-1e9)

            # clipfrac（ratio を exp せず log 空間で判定：オーバーフロー回避）
            log_clip_hi = jnp.log1p(train_config.CLIP_EPS)  # log(1+eps)
            log_clip_lo = jnp.log1p(-train_config.CLIP_EPS)  # log(1-eps)
            clipfrac = ((log_ratio_raw > log_clip_hi) | (log_ratio_raw < log_clip_lo)).mean()
            clipfrac = jnp.nan_to_num(clipfrac, nan=0.0, posinf=1.0, neginf=0.0)
            # surrogate に使う ratio は安定のため clipして exp
            log_ratio = jnp.clip(log_ratio_raw, -10.0, 10.0)
            ratio = jnp.exp(log_ratio)
            # ratio = jnp.exp(log_prob - rollout_buffer.log_prob)

            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = jnp.clip(ratio, 1.0 - train_config.CLIP_EPS, 1.0 + train_config.CLIP_EPS) * gae
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = loss_actor + train_config.VF_COEF * value_loss - ent_coef * entropy
            return total_loss, (value_loss, loss_actor, entropy, approx_kl, clipfrac)

        # UPDATE NETWORK
        def _update_epoch(
            epoch_update_state: tuple[TrainState, jax.Array, Transition, jax.Array, jax.Array, jax.Array], _: None
        ):
            def _update_minibatch(
                train_state: TrainState, batch_info: tuple[jax.Array, Transition, jax.Array, jax.Array]
            ):
                init_hstate, rollout_buffer, advantages, targets = batch_info
                init_hstate = init_hstate.squeeze(axis=0)

                ent_coef = ent_schedule(train_state.step)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_value, aux), grads = grad_fn(
                    train_state.params, init_hstate, rollout_buffer, advantages, targets, ent_coef
                )

                # grads を finite に強制
                safe_grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)

                # loss/grads が finite のときだけ更新（NaN汚染を防ぐ）
                grads_finite = jax.tree_util.tree_reduce(
                    lambda a, b: a & b, jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), safe_grads)
                )
                is_finite = jnp.isfinite(loss_value) & grads_finite

                def _apply(ts: TrainState):
                    return ts.apply_gradients(grads=safe_grads)

                def _skip(ts: TrainState):
                    return ts

                # 追加：更新が起きているかの即死チェック用ログ
                grad_norm = optax.global_norm(safe_grads)
                lr = lr_schedule(train_state.step)
                skip_update = 1.0 - is_finite.astype(jnp.float32)

                # aux の後ろに追加（既存の index を壊さない）
                aux = (*aux, grad_norm, lr, skip_update)

                train_state = jax.lax.cond(is_finite, _apply, _skip, train_state)

                # ログに出すlossもfinite化
                loss_value_safe = jnp.nan_to_num(loss_value, nan=0.0, posinf=0.0, neginf=0.0)
                total_loss = (loss_value_safe, aux)
                return train_state, total_loss

            train_state, init_hstate, rollout_buffer, advantages, targets, rng = epoch_update_state

            # エージェント単位の履歴をシャッフルしてminibatchを作成する
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, num_actors)

            init_hstate = jnp.reshape(init_hstate, (1, num_actors, -1))
            batch = (init_hstate, rollout_buffer, advantages.squeeze(), targets.squeeze())

            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], num_minibatches, -1, *list(x.shape[2:])]), 1, 0),
                shuffled_batch,
            )
            # jax.debug.print("batch: {}", batch)
            # jax.debug.print("shuffled_batch: {}", shuffled_batch)
            # jax.debug.print("minibatches: {}", minibatches)

            train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
            epoch_update_state = (train_state, init_hstate.squeeze(), rollout_buffer, advantages, targets, rng)
            return epoch_update_state, total_loss

        def _update_step(runner_state: _RunnerState, _: None):  # jax.lax.scanに渡すため未使用の引数が必要
            stepwise_initial_hstate = runner_state[5]  # (NUM_ACTORS, GRU_HIDDEN_DIM)
            #################################################
            # 現在の方策に従って行動し、学習データを収集する
            #################################################
            # NUM_ENVS個の環境をそれぞれTIMESTEPS分更新する
            # runner_state: TIMESTEPS更新後の状態
            # rollout_buffer: TIMESTEPS分の状態遷移のリスト(rollout buffer) (TIMESTEPS, NUM_ACTORS)
            runner_state, rollout_buffer = jax.lax.scan(_env_step, runner_state, None, train_config.TIMESTEPS)
            # jax.debug.print("rollout: {}", rollout_buffer)
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = runner_state
            # last_obs_batch: (NUM_ACTORS, height, width, info_layers)
            last_obs_batch = last_obs.reshape(num_actors, *env.obs_shape[1:])

            #################################################
            # CALCULATE ADVANTAGE
            #################################################
            # ac_in.shape:
            # (1, NUM_ACTORS, height, width, info_layers),
            # (1, NUM_ACTORS)
            ac_in = (last_obs_batch[jnp.newaxis, :], last_done[jnp.newaxis, :])
            # last_val は critic の評価値
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()  # (NUM_ACTORS, )

            advantages, targets = _calculate_gae(rollout_buffer, last_val, train_config.GAMMA, train_config.GAE_LAMBDA)
            # advantages: 行動価値 - 状態価値
            # targets: 行動価値
            # advantages, targets: (TIMESTEPS, NUM_ACTORS)
            # jax.debug.print("advantages: {}", advantages)
            # jax.debug.print("targets: {}", targets)

            #################################################
            # パラメータの更新
            #################################################
            # env_stepに従って隠し状態が更新されるが、学習はそれとは別に行うので
            # 回す前の状態を保存していた
            update_state = (train_state, stepwise_initial_hstate, rollout_buffer, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, train_config.NUM_UPDATE_EPOCHS)
            train_state = update_state[0]
            metric = rollout_buffer.info
            rng = update_state[-1]
            loss_value_mb = loss_info[0]  # (NUM_UPDATE_EPOCHS, NUM_MINIBATCHES)
            (
                value_loss_mb,
                actor_loss_mb,
                entropy_mb,
                approx_kl_mb,
                clipfrac_mb,
                grad_norm_mb,
                lr_mb,
                skip_update_mb,
            ) = loss_info[1]
            #################################################

            # 結果の記録
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = (update_step + 1) * train_config.TIMESTEPS * train_config.NUM_ENVS
            metric["ent_coef"] = ent_schedule(train_state.step)
            metric["loss_total"] = loss_value_mb.mean()
            metric["loss_value"] = value_loss_mb.mean()
            metric["loss_actor"] = actor_loss_mb.mean()
            metric["entropy"] = entropy_mb.mean()
            metric["approx_kl"] = approx_kl_mb.mean()
            metric["clipfrac"] = clipfrac_mb.mean()
            metric["grad_norm"] = grad_norm_mb.mean()
            metric["lr"] = lr_mb.mean()
            metric["skip_update_rate"] = skip_update_mb.mean()
            metric["adv_std"] = advantages.std()

            update_step = update_step + 1
            # モデルの評価(同じ初期状態から1周したときの報酬)
            is_test_step = (update_step > 1) & (update_step % train_config.CHECKPOINT_INTERVAL_STEP == 0)
            metric["eval_score"] = jax.lax.cond(
                is_test_step, _evaluate, lambda ts, _: 0.0, train_state.params, update_step
            )

            # (参考) https://github.com/luchris429/purejaxrl/issues/13#issuecomment-1823925382
            jax.debug.callback(save_checkpoint, train_state, metric, update_step)

            runner_state = (
                train_state,  # 更新した方策で次stepのenv_stepを行いrollout_bufferを作成する
                env_state,  # TIMESTEPS進んだ状態から次stepのenv_stepでの収集を再開する
                last_obs,  # 最後の状態から行動選択を再開する
                last_done,  # terminal_timeになったら初期状態に戻す
                update_step,
                hstate,  # last_doneがTrueになったら初期化される
                rng,
            )
            return runner_state, metric

        def _initialize_runner_state(rng: jax.Array):
            # ネットワークの初期化
            rng, initialize_rng = jax.random.split(rng)
            init_network_params = _initialize_network_params(network, initialize_rng, train_config, env)
            tx = _create_optimizer(train_config, lr_schedule)
            init_train_state = TrainState.create(apply_fn=network.apply, params=init_network_params, tx=tx)

            # 環境の初期化
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, train_config.NUM_ENVS)
            init_obsv, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            init_done = jnp.zeros((num_actors), dtype=bool)
            init_step = 0
            # ActorCritic学習の初期隠れ状態  (NUM_ACTORS = NUM_ENVS * num_agents, hidden_dim)
            init_hstate = ScannedRNN.initialize_carry(num_actors, train_config.GRU_HIDDEN_DIM)

            rng, init_rng = jax.random.split(rng)
            return (
                init_train_state,  # ネットワークパラメータ初期値
                init_env_state,  # 初期化した環境 (NUM_ENVS,)
                init_obsv,  # 初期状態の観測値 (NUM_ENVS,)
                init_done,  # is_terminal初期値 (NUM_ACTORS,)
                init_step,  # 学習ステップ数初期値(=0)
                init_hstate,  # RNN隠れ状態 (NUM_ACTORS, GRU_HIDDEN_DIM)
                init_rng,
            )

        def save_eval_results(state, done, step, task_rewards):
            eval_state_seq.append(state)
            if done:
                result_save_dir = Path(train_config.MODEL_DIR) / "eval"
                result_save_dir.mkdir(parents=True, exist_ok=True)
                viz.animate(eval_state_seq, str(result_save_dir / f"eval_{step}.gif"))
                with open(result_save_dir / f"rewards_{step}.csv", "w") as f:
                    for i, t in enumerate(RewardType):
                        print(f"{t.name},{task_rewards[i]}", file=f)
                    print(f"original,{task_rewards[-1]}", file=f)
                eval_state_seq.clear()

        def _evaluate(params, step):
            # 評価環境の初期化
            eval_rng = jax.random.PRNGKey(train_config.EVAL_SEED)
            env_rng, init_rng = jax.random.split(eval_rng)
            init_step = 0
            init_obsv, init_env_state = env.reset(env_rng)
            init_hstate = ScannedRNN.initialize_carry(env.num_agents, train_config.GRU_HIDDEN_DIM)
            done = jnp.repeat(False, env.num_agents)
            score = jnp.zeros(env.num_agents)
            task_rewards = jnp.zeros((len(RewardType) + 1, env.num_agents))
            init_eval_state = (init_step, init_env_state, init_obsv, init_hstate, init_rng, score, task_rewards, done)

            def _continue(eval_state):
                return ~eval_state[-1][0]

            def _step_env(eval_state):
                (env_step, last_env_state, last_obs, hstate, rng, score, task_rewards, last_done) = eval_state
                rng, _rng = jax.random.split(rng)
                ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis])
                hstate, pi, value = network.apply(params, hstate, ac_in)
                action = jnp.argmax(pi.probs, axis=-1).squeeze()
                obs, env_state, original_reward, shaped_rewards, reward_types, done = env.step_env(
                    last_env_state, action, _rng
                )
                done = env_step + 1 == env.max_steps

                # 報酬をタスクごとに集計
                def _accum_reward(reward_array, x):
                    reward_type, reward, agent_idx = x
                    cur_value = reward_array[reward_type, agent_idx]
                    return reward_array.at[reward_type, agent_idx].set(cur_value + reward), None

                task_rewards, _ = jax.lax.scan(
                    _accum_reward, task_rewards, (reward_types, shaped_rewards, jnp.arange(env.num_agents))
                )
                task_rewards = task_rewards.at[-1].set(task_rewards[-1] + original_reward)

                jax.debug.callback(save_eval_results, env_state, done, step, task_rewards)

                return (
                    env_step + 1,
                    env_state,
                    obs,
                    hstate,
                    rng,
                    score + original_reward + shaped_rewards,
                    task_rewards,
                    jnp.repeat(done, env.num_agents),
                )

            terminal_eval_state = jax.lax.while_loop(_continue, _step_env, init_eval_state)
            score = terminal_eval_state[5]

            return jnp.sum(score)

        ###################################################
        # train の処理本体
        ###################################################
        init_runner_state = _initialize_runner_state(rng)
        # TRAIN LOOP
        final_runner_state, metric = jax.lax.scan(
            _update_step, init_runner_state, None, train_config.NUM_TRAINING_STEPS
        )
        jax.debug.callback(_save_metrics, metric, seed_idx, model_dir)
        return {"runner_state": final_runner_state, "metrics": metric}

    return train


def load_config(config: DictConfig) -> DictConfig:
    from omegaconf import open_dict

    layout = config.layout.get(str(config.get("stage", None)), None)
    if layout is None:
        print("select one of stages by stage=(stage_name)")
        print(list(config.layout.keys()))
        sys.exit()
    with open_dict(config):
        config.env["layout"] = layout
        del config.layout
    return config


@hydra.main(version_base=None, config_path="../config", config_name="ippo_rnn")
def main(config: DictConfig):
    config = load_config(config)
    app_config = app_config_from_omegaconf(config)

    num_seeds = app_config.train.NUM_SEEDS
    with jax.disable_jit(False):
        rng = jax.random.key(app_config.train.SEED)
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(app_config))
        out = jax.vmap(train_jit)(rngs, jnp.arange(num_seeds))


if __name__ == "__main__":
    main()
