from itertools import product
from pathlib import Path

import absl.logging
import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from config import EvalConfig, evaluate_config_from_omegaconf
from environment.overcooked import OvercookedCustom
from environment.reward import RewardType
from operation.controller import Controller, _OperationConfig
from visualize.visualizer import OvercookedCustomVisualizer

# モデル読み込み時にログが出力されるのを抑止
absl.logging.set_verbosity(absl.logging.WARNING)

class Evaluator:
    # ui_config: {controller_type: [option1, option2, ...]}（listは選択肢リスト）
    def __init__(self, config: EvalConfig, ui_config: dict[str, _OperationConfig | list[_OperationConfig]]):
        self.visualize = config.visualize
        self.loop = config.loop
        self.outdir = Path(HydraConfig.get().runtime.output_dir).parent
        self.save_gif = config.save_gif
        self.gif_filename = Path(config.gif_filename)
        self.frame_seq = []
        self.iter_num = 0
        self.key = jax.random.wrap_key_data(jnp.array(config.seed, dtype=jnp.uint32))
        self.env = OvercookedCustom(config.env, config.random_agent_position)
        if self.visualize:
            self.viz = OvercookedCustomVisualizer()
            self.viz.show()
        # {player: 設定} のリストを作成（listは選択肢リスト → 展開して control_configs にする）
        self.control_configs: list[dict[str, _OperationConfig]] = []
        for k, vs in ui_config.items():
            if isinstance(vs, list):
                self.control_configs += [{k: v} for v in vs]
            else:
                self.control_configs.append({k: vs})

        combinations = product(self.control_configs, repeat=self.env.num_agents)
        self.num_models = len(self.control_configs)
        # 可視化時に num_models**num_agents 通りの組み合わせを縦横(num_models**rows, num_models**cols)に並べる
        self.num_rows = self.num_models ** (self.env.num_agents // 2 + self.env.num_agents % 2)
        self.num_cols = self.num_models ** (self.env.num_agents // 2)
        self.num_parallel = self.num_models**self.env.num_agents
        self.total_rewards = jnp.zeros((self.loop, self.num_parallel, len(RewardType) + 1, self.env.num_agents))

        self.controllers = []
        for condition in combinations:
            player = sum([[p for p in c.keys()] for c in condition], [])
            ui: dict[str, list[_OperationConfig]] = {}
            for c in condition:
                for k, vs in c.items():
                    if k in ui:
                        ui[k] = ui[k] + [vs]
                    else:
                        ui[k] = [vs]
            controller = Controller(env=self.env, ui=ui, player=player, verbose=False, confirm=False)
            if not controller.is_auto():
                exit("interactive control is prohibited.")
            self.controllers.append(controller)

    def run(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        for cnt in tqdm(range(self.loop)):
            self.eval_once()
            self.iter_num += 1
        # 繰り返しごとの評価結果をそのまま保存
        self.save_raw_results()
        # 繰り返し回数とエージェントについての平均、標準偏差
        # self.total_rewards : (loop, モデル組み合わせ, 報酬種別, エージェント)
        mean_ = jnp.mean(self.total_rewards, axis=(0, 3)).transpose()
        std_ = jnp.std(self.total_rewards, axis=(0, 3)).transpose()
        max_ = jnp.max(self.total_rewards, axis=(0, 3)).transpose()
        agent_means = jnp.mean(self.total_rewards, axis=0).transpose(1, 0, 2)
        agent_stds = jnp.std(self.total_rewards, axis=0).transpose(1, 0, 2)
        agent_maxs = jnp.max(self.total_rewards, axis=0).transpose(1, 0, 2)
        self.save_stats(mean_, agent_means, self.outdir / "means.csv", "means")
        self.save_stats(std_, agent_stds, self.outdir / "stds.csv", "stds")
        self.save_stats(max_, agent_maxs, self.outdir / "maxs.csv", "maxs")

    def _write_model_controlls(self, fout):
        # 使用したモデルを記録
        for idx, config in enumerate(self.control_configs):
            type_ = list(config.keys()).pop()
            print(f"{idx},{type_},{str(config[type_]).replace(',', '')}", file=fout)
        print("=" * 64, file=fout)

    def save_raw_results(self):
        iter_header = ",".join([str(n) for n in jnp.repeat(jnp.arange(self.loop), self.env.num_agents)])
        agent_header = ",".join([str(n) for n in jnp.tile(jnp.arange(self.env.num_agents), self.loop)])
        combinations = jnp.array([x for x in product(range(self.num_models), repeat=self.env.num_agents)])
        labels = [t.name for t in RewardType] + ["original"]
        # self.total_rewards : (loop, モデル組み合わせ, 報酬種別, エージェント)
        results = self.total_rewards.transpose(1, 2, 0, 3).reshape(self.num_parallel * (len(RewardType) + 1), -1)
        with open(self.outdir / "raw_result.csv", "w") as f:
            self._write_model_controlls(f)
            print("," * self.env.num_agents + f"iter,{iter_header}", file=f)
            print("," * self.env.num_agents + f"agent,{agent_header}", file=f)
            for i, row in enumerate(results):
                combination = ",".join([str(n) for n in combinations[i // (len(RewardType) + 1)]])
                label = labels[i % (len(RewardType) + 1)]
                values = ",".join([str(v) for v in row])
                print(f"{combination},{label},{values}", file=f)

    def save_stats(self, data: jax.Array, agents_data: jax.Array, output_file: Path, stat_type: str):
        labels = [t.name for t in RewardType] + ["original"]
        with open(output_file, "w") as f:
            self._write_model_controlls(f)
            combinations = jnp.array([x for x in product(range(self.num_models), repeat=self.env.num_agents)])
            for idx, row in enumerate(combinations.transpose()):
                print(f"agent_{idx}," + ",".join([str(value) for value in row]), file=f)
            for i, label in enumerate(labels):
                print(f"{label}_{stat_type}," + ",".join([str(value) for value in data[i]]), file=f)
            print("=" * 64, file=f)
            for i, label in enumerate(labels):
                print(f"{label}_{stat_type}", file=f)
                for agent_idx in range(agents_data.shape[2]):
                    print(
                        f"agent_{agent_idx}," + ",".join([str(value) for value in agents_data[i, :, agent_idx]]), file=f
                    )

    def _reset(self):
        # 並列環境を同じ乱数で初期化する
        self.key, subkey = jax.random.split(self.key)
        subkeys = jnp.vstack([subkey] * self.num_parallel)
        init_obs, init_state_tree = jax.vmap(self.env.reset)(subkeys)

        init_done = jnp.zeros((self.num_parallel), dtype=bool)
        task_rewards = jnp.zeros((self.num_parallel, len(RewardType) + 1, self.env.num_agents))
        self.frame_seq.clear()

        runner = (init_state_tree, init_done, task_rewards)

        for i, controller in enumerate(self.controllers):
            controller.reset()
            controller.input_observation(init_obs[i])
        return runner

    def eval_once(self):

        def _accum_reward(reward_array, types, shaped_rewards):
            def _accum(current_rewards, x):
                reward_type, reward, agent_idx = x
                cur_value = current_rewards[reward_type, agent_idx]
                return current_rewards.at[reward_type, agent_idx].set(cur_value + reward), None

            reward_array, _ = jax.lax.scan(
                _accum, reward_array, (types, shaped_rewards, jnp.arange(self.env.num_agents))
            )
            return reward_array

        def _step_env(runner_state):
            last_env_state, _, task_rewards = runner_state
            self.key, subkey = jax.random.split(self.key)
            actions = jnp.array([controller.operate() for controller in self.controllers])
            obs, env_states, reward, shaped_rewards, reward_types, done = jax.vmap(
                self.env.step_env, in_axes=(0, 0, None)
            )(last_env_state, actions, subkey)
            if self.visualize:
                self.viz.render_multi(
                    env_states, self.num_rows, self.num_cols, title=f"{env_states.time[0]} / {self.env.max_steps} step"
                )
            if self.save_gif:
                self.frame_seq.append(env_states)
            task_rewards = jax.vmap(_accum_reward)(task_rewards, reward_types, shaped_rewards)
            # task_rewards: (コントロール組み合わせ, 報酬種別, エージェント)
            task_rewards = task_rewards.at[:, -1, :].set(task_rewards[:, -1, :] + reward)
            for i, controller in enumerate(self.controllers):
                controller.input_observation(obs[i])
            return (env_states, done, task_rewards)

        runner = self._reset()
        done = False
        while not done:
            runner = _step_env(runner)
            done = runner[1][0]

        rewards = runner[2]
        self.total_rewards = self.total_rewards.at[self.iter_num].set(rewards)
        if self.save_gif:
            gif_name = str(self.outdir / self.gif_filename.with_stem(self.gif_filename.stem + f"_{self.iter_num}"))
            self.viz.grid_animate(self.frame_seq, self.num_rows, self.num_cols, filename=gif_name)


def load_config(config: DictConfig) -> DictConfig:
    layout = config.layout.get(str(config.get("stage", None)), None)
    if layout is None:
        print("select one of stages by stage=(stage_name)")
        print(list(config.layout.keys()))
        exit()
    with open_dict(config):
        config.env["layout"] = layout
        del config.layout
    return config


@hydra.main(config_path="../config", config_name="evaluate_models", version_base=None)
def main(config: DictConfig):
    config = load_config(config)
    interactive_config = evaluate_config_from_omegaconf(config)
    ui_config: dict[str, _OperationConfig | list[_OperationConfig]] = OmegaConf.to_container(config.ui, resolve=True)  # type: ignore[assignment]
    evaluator = Evaluator(interactive_config, ui_config)
    evaluator.run()


if __name__ == "__main__":
    main()
