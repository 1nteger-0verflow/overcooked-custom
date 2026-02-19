import functools
from pathlib import Path
from typing import Any, Callable, NamedTuple

import absl.logging
import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import yaml
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from environment.overcooked import OvercookedCustom
from visualize.visualizer import OvercookedCustomVisualizer


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x  # x = (embedding, done)のtuple
        # ins: (NUM_ACTOR, hidden_dim)

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            new_carry,
            rnn_state,
        )
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
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class ActorCriticRNN(nn.Module):
    action_dim: int
    config: DictConfig

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = obs
        if self.config.ACTIVATION == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config.GRU_HIDDEN_DIM,
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)
        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        # hidden: (NUM_ACTORS, hidden_dim)
        # rnn_in.embedding: (1, NUM_ACTORS, hidden_dim)
        # rnn_in.dones: (1, NUM_ACTORS)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config.FC_DIM_SIZE,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)

        # 方策
        pi = distrax.Categorical(logits=actor_mean)

        # 状態価値(の推論)
        critic = nn.Dense(
            self.config.FC_DIM_SIZE,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

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


def make_train(merged_config: DictConfig):
    env_config = merged_config.env
    config = merged_config.train
    env = OvercookedCustom(env_config, config.RANDOM_AGENT_POS)
    with open_dict(config):
        config.NUM_ACTORS = env.num_agents * config.NUM_ENVS
        assert config.NUM_ACTORS % config.MINIBATCH_SIZE == 0, (
            f"MINIBATCH_SIZE({config.MINIBATCH_SIZE}) must devide NUM_ACTORS({config.NUM_ACTORS})"
        )
        config.NUM_MINIBATCHES = config.NUM_ACTORS // config.MINIBATCH_SIZE
        # ・保存先は絶対パスで指定しなければならない
        config.MODEL_DIR = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        ).parent.as_posix()
    # 学習のHORIZONステップ目以降はshaped_rewardの重みが0
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config.REW_SHAPING_HORIZON
    )
    # チェックポイントの保存用設定
    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    # https://github.com/google/flax/discussions/3130
    absl.logging.set_verbosity(absl.logging.WARNING)
    # https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html#multiple-item-checkpointing
    options = ocp.CheckpointManagerOptions(
        create=True, save_interval_steps=config.CHECKPOINT_INTERVAL_STEP
    )
    checkpoint_manager = ocp.CheckpointManager(
        config.MODEL_DIR,
        options=options,
        metadata=config,
    )
    # 進捗表示
    if config.progress:
        progress_bar = tqdm(total=config.NUM_TRAINING_STEPS)
    # 学習中の可視化
    if config.visualize:
        r = config.aspect_row  # 環境を並べる際の列と行の比率
        c = config.aspect_col  # 環境を並べる際の列と行の比率

        def _adjust_row_col(r, c):
            # 縦：横が大体r:cになるような並べ方を探索
            for rows in range(1, config.NUM_ENVS):
                for cols in range(1, int(rows * c / r) + 1):
                    if config.NUM_ENVS <= rows * cols:
                        return (rows, cols)
            return (1, config.NUM_ENVS)

        rows, cols = _adjust_row_col(r, c)
        with open_dict(config):
            config.viz_rows = rows
            config.viz_cols = cols
        viz = OvercookedCustomVisualizer()
        viz.show()

    def initialize_network_params(network: ActorCriticRNN, rng: jax.Array):
        # ネットワークのパラメータ初期化は環境ごとに行う（エージェントの区別なし）
        # shape: (1,NUM_ENV, height, width, channel), (1, NUM_ENVS)
        init_x = (
            jnp.zeros((1, config.NUM_ENVS, *env.obs_shape[1:])),
            jnp.zeros((1, config.NUM_ENVS)),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config.NUM_ENVS,
            config.GRU_HIDDEN_DIM,
        )
        network_params = network.init(rng, init_hstate, init_x)
        return network_params

    def create_learning_rate_fn():
        base_learning_rate = config.LR

        lr_warmup = config.LR_WARMUP
        total_update_steps = config.NUM_TRAINING_STEPS
        warmup_steps = int(lr_warmup * total_update_steps)

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps,
        )

        cosine_steps = max(total_update_steps - warmup_steps, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_steps
        )

        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps],
        )
        return schedule_fn

    def schedule():
        # 学習率スケジューリング
        if config.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.LR, eps=1e-5),
            )
        return tx

    # GAE: generalized advantage estimate
    # TODO: gamma, lambdaを引数に追加すれば純粋関数として共通化できる
    # gamma(float): discount factor
    # lambda(float): GAE mixing parameter
    def _calculate_gae(rollout_buffer, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + config.GAMMA * next_value * (1 - done) - value
            gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
            return (gae, value), gae

        # unrollはXLAの最適化に関するパラメータで計算結果には関係しない
        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            rollout_buffer,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + rollout_buffer.value

    def save_checkpoint(train_state, hstate, metric, step: int):
        # CheckpointManager.saveのstep数はintでなければならないのでcallbackで実装(update_stepはjitのint32[]でNG)
        checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(train_state.params),
                obs_shape=ocp.args.StandardSave(env.obs_shape),
            ),
        )
        checkpoint_manager.wait_until_finished()
        # 進捗を更新
        if config.progress:
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    k: f"{int(metric[k] * 1000) / 1000:.3f}"
                    for k in ["combined_reward", "shaped_reward", "original_reward"]
                }
            )

    def save_metrics(metrics, seed_idx):
        save_dir = Path(config.MODEL_DIR) / f"metrics_{seed_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.csv", "w") as f:
            for key in [
                "original_reward",
                "shaped_reward",
                "anneal_factor",
                "combined_reward",
            ]:
                out = f"{key}," + ",".join([str(x) for x in metrics[key]])
                print(out, file=f)
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    def visualize_state(states):
        if config.visualize:
            viz.render_multi(
                states,
                config.viz_rows,
                config.viz_cols,
                title=f"{states.time[0]} / {env.max_steps} step",
                caption="caption",
            )

    def train(rng: jax.Array, seed_idx: int):
        # NUM_SEEDS並列に実行
        network = ActorCriticRNN(env.num_actions, config=config)

        # COLLECT TRAJECTORIES
        def _env_step(last_runner_state, _):
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
            obs_batch = last_obs["agents"].reshape(
                config.NUM_ACTORS, *env.obs_shape[1:]
            )
            # ac_in.shape:
            # (1, NUM_ACTORS, height, width, info_layer),
            # (1, NUM_ACTORS)
            ac_in = (
                obs_batch[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
            )
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
            env_act = action.reshape((config.NUM_ENVS, env.num_agents))
            # env_act:         agent0, agent1, ...
            #          env0  [ [act00, act01, ...],
            #          env1  [ [act10, act11, ...],
            #            :

            # 乱数の更新
            rng, _rng = jax.random.split(rng)
            # 並行環境にそれぞれ違う乱数を適用し、様々な状態が現れるようにする
            rng_step = jax.random.split(_rng, config.NUM_ENVS)

            # STEP ENV
            new_obsv, new_env_state, original_reward, shaped_rewards, done = jax.vmap(
                env.step_env, in_axes=(0, 0, 0)
            )(last_env_state, env_act, rng_step)
            # 終了時間に達したら状態を初期化する
            # すべての環境が同時にterminateするのでall(done)で判定してよい。
            new_obsv, new_env_state = jax.lax.cond(
                jnp.all(done),
                jax.vmap(env.reset, in_axes=(0,)),
                lambda _: (new_obsv, new_env_state),
                rng_step,
            )
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
            info = jax.tree_util.tree_map(lambda x: x.reshape(config.NUM_ACTORS), info)
            # doneはenvにつき1つなのでagent数分複製して(NUM_ACTORS,)の形にする
            done_batch = jnp.tile(done, env.num_agents)

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
                reward=combined_reward.reshape(-1, combined_reward.size).squeeze(),
                done=done_batch,
                info=info,
            )
            jax.debug.callback(visualize_state, new_env_state)
            # jax.debug.print("transition: {}", transition)
            return new_runner_state, transition

        def _loss_fn(params, init_hstate, rollout_buffer, gae, targets):
            # 方策勾配法では方策の更新によって得られるデータが変わっていくため、損失の値そのものには意味がない
            # https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

            # RERUN NETWORK
            _, pi, value = network.apply(
                params,
                init_hstate,
                (rollout_buffer.obs, rollout_buffer.done),
            )

            log_prob = pi.log_prob(rollout_buffer.action)

            # CALCULATE VALUE LOSS
            value_pred_clipped = rollout_buffer.value + (
                value - rollout_buffer.value
            ).clip(-config.CLIP_EPS, config.CLIP_EPS)
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - rollout_buffer.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config.CLIP_EPS,
                    1.0 + config.CLIP_EPS,
                )
                * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                loss_actor + config.VF_COEF * value_loss - config.ENT_COEF * entropy
            )
            return total_loss, (value_loss, loss_actor, entropy)

        # UPDATE NETWORK
        def _update_epoch(epoch_update_state, _):
            def _update_minibatch(train_state, batch_info):
                init_hstate, rollout_buffer, advantages, targets = batch_info

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, rollout_buffer, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, init_hstate, rollout_buffer, advantages, targets, rng = (
                epoch_update_state
            )

            # エージェント単位の履歴をシャッフルしてminibatchを作成する
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config.NUM_ACTORS)

            init_hstate = jnp.reshape(init_hstate, (1, config.NUM_ACTORS, -1))
            batch = (
                init_hstate,
                rollout_buffer,
                advantages.squeeze(),
                targets.squeeze(),
            )

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config.NUM_MINIBATCHES, -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )
            # jax.debug.print("batch: {}", batch)
            # jax.debug.print("shuffled_batch: {}", shuffled_batch)
            # jax.debug.print("minibatches: {}", minibatches)

            train_state, total_loss = jax.lax.scan(
                _update_minibatch, train_state, minibatches
            )
            epoch_update_state = (
                train_state,
                init_hstate.squeeze(),
                rollout_buffer,
                advantages,
                targets,
                rng,
            )
            return epoch_update_state, total_loss

        def _update_step(runner_state, _):  # jax.lax.scanに渡すため未使用の引数が必要
            stepwise_initial_hstate = runner_state[-2]  # (NUM_ACTORS, GRU_HIDDEN_DIM)
            #################################################
            # 現在の方策に従って行動し、学習データを収集する
            #################################################
            # NUM_ENVS個の環境をそれぞれTIMESTEPS分更新する
            # runner_state: TIMESTEPS更新後の状態
            # rollout_buffer: TIMESTEPS分の状態遷移のリスト(rollout buffer) (TIMESTEPS, NUM_ACTORS)
            runner_state, rollout_buffer = jax.lax.scan(
                _env_step, runner_state, None, config.TIMESTEPS
            )
            # jax.debug.print("rollout: {}", rollout_buffer)
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )
            # last_obs_batch: (NUM_ACTORS, height, width, info_layers)
            last_obs_batch = last_obs["agents"].reshape(
                config.NUM_ACTORS, *env.obs_shape[1:]
            )

            #################################################
            # CALCULATE ADVANTAGE
            #################################################
            # ac_in.shape:
            # (1, NUM_ACTORS, height, width, info_layers),
            # (1, NUM_ACTORS)
            ac_in = (
                last_obs_batch[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
            )
            # last_val は critic の評価値
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()  # (NUM_ACTORS, )

            advantages, targets = _calculate_gae(rollout_buffer, last_val)
            # advantages: 行動価値 - 状態価値
            # targets: 行動価値
            # advantages, targets: (TIMESTEPS, NUM_ACTORS)
            # jax.debug.print("advantages: {}", advantages)
            # jax.debug.print("targets: {}", targets)

            #################################################
            # パラメータの更新
            #################################################
            update_state = (
                train_state,
                stepwise_initial_hstate,  # env_stepに従って隠し状態が更新されるが、学習はそれとは別に行うので回す前の状態を保存していた？？？
                rollout_buffer,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.NUM_UPDATE_EPOCHS
            )
            train_state = update_state[0]
            metric = rollout_buffer.info
            rng = update_state[-1]
            #################################################

            # 結果の記録
            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config.TIMESTEPS * config.NUM_ENVS

            # (参考) https://github.com/luchris429/purejaxrl/issues/13#issuecomment-1823925382
            jax.debug.callback(
                save_checkpoint, train_state, hstate, metric, update_step
            )

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
            init_network_params = initialize_network_params(network, initialize_rng)
            tx = schedule()
            init_train_state = TrainState.create(
                apply_fn=network.apply,
                params=init_network_params,
                tx=tx,
            )

            # 環境の初期化
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config.NUM_ENVS)
            init_obsv, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            init_done = jnp.zeros((config.NUM_ACTORS), dtype=bool)
            init_step = 0
            # ActorCritic学習の初期隠れ状態  (NUM_ACTORS = NUM_ENVS * num_agents, hidden_dim)
            init_hstate = ScannedRNN.initialize_carry(
                config.NUM_ACTORS, config.GRU_HIDDEN_DIM
            )

            rng, init_rng = jax.random.split(rng)
            init_runner_state = (
                init_train_state,  # ネットワークパラメータ初期値
                init_env_state,  # 初期化した環境 (NUM_ENVS,)
                init_obsv,  # 初期状態の観測値 (NUM_ENVS,)
                init_done,  # is_terminal初期値 (NUM_ACTORS,)
                init_step,  # 学習ステップ数初期値(=0)
                init_hstate,  # RNN隠れ状態 (NUM_ACTORS, GRU_HIDDEN_DIM)
                init_rng,
            )
            return init_runner_state

        ###################################################
        # train の処理本体
        ###################################################
        init_runner_state = _initialize_runner_state(rng)
        # TRAIN LOOP
        final_runner_state, metric = jax.lax.scan(
            _update_step, init_runner_state, None, config.NUM_TRAINING_STEPS
        )
        jax.debug.callback(save_metrics, metric, seed_idx)
        return {"runner_state": final_runner_state, "metrics": metric}

    return train


def load_config(config: DictConfig):
    layout = config.layout.get(str(config.get("stage", None)), None)
    if layout is None:
        print("select one of stages by stage=(stage_name)")
        print(list(config.layout.keys()))
        exit()
    with open_dict(config):
        config.train["progress"] = config.progress
        config.train["visualize"] = config.visualize
        config.train["aspect_row"] = config.aspect_row
        config.train["aspect_col"] = config.aspect_col
        config.env["layout"] = layout
        del config.layout
    return config


@hydra.main(version_base=None, config_path="../config", config_name="ippo_rnn")
def main(config):
    config = load_config(config)

    num_seeds = config.train.NUM_SEEDS
    with jax.disable_jit(False):
        rng = jax.random.key(config.train.SEED)
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config))
        # train_jit = make_train(config)
        out = jax.vmap(train_jit)(rngs, jnp.arange(num_seeds))


if __name__ == "__main__":
    main()
