import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, open_dict

from environment.overcooked import OvercookedCustom
from operation.controller import Controller
from utils.schema import PlayConfig
from visualize.visualizer import OvercookedCustomVisualizer

# print時の表示設定はNumpyに依存する
np.set_printoptions(threshold=100000, linewidth=30000)


class InteractiveOvercookedCustom:
    def __init__(self, config: PlayConfig):
        option = config.option
        self.verbose = option.verbose
        self.visualize = option.visualize
        self.loop = option.loop
        self.profile = option.profile
        self.obs = option.observation
        self.log = option.log
        self.log_dir = Path(option.log_dir)
        if self.loop:
            self.iter_num = 0

        self.key = jax.random.wrap_key_data(jnp.array(option.seed, dtype=jnp.uint32))
        self.env = OvercookedCustom(config.env, option.random_agent_position)
        self.viz = OvercookedCustomVisualizer()
        self.controller = Controller(self.env, config.interface, self.verbose, option.confirm)

        if self.verbose:
            print(f"取りうるアクションの種類: {self.env.num_actions}")
            print(f"観測結果のサイズ： {self.env.obs_shape}")

    def run(self):
        self._run()

    def _run(self):
        self._reset()

        if self.visualize:
            if self.controller.is_auto():
                self._handle_keyboard_input()
            self.viz.window.reg_key_handler(self._handle_input)
            self.viz.show(block=True)
        else:
            self._handle_keyboard_input()

    def _handle_input(self, event):
        match event.key:
            case "escape":
                self.viz.close()
            case "backspace":
                self._reset()
            case key if key in ["enter", "return"]:
                self._step()
            case key:
                done = self.controller.keyboard_input(key)
                if done:
                    self._step()

    def _handle_keyboard_input(self):
        convert = {"w": "up", "d": "right", "s": "down", "a": "left"}
        while True:
            if self.controller.is_auto():
                self._step()
            else:
                user_input = input("input>")
                match user_input:
                    case "q":
                        exit()
                    case "r":
                        self._reset()
                    case "":
                        self._step()
                    case k:
                        key = convert.get(k, k)
                        done = self.controller.keyboard_input(key)
                        if done:
                            self._step()

    def _redraw(self):
        if self.visualize:
            self.viz.render(self.state, title=f"{self.state.time} / {self.env.max_steps} step")

    def _reset(self):
        if self.log:
            self.action_log = []
            self.init_key = self.key
        self.key, subkey = jax.random.split(self.key)
        init_obs, self.state = jax.jit(self.env.reset)(subkey)
        if self.verbose:
            print("==== layout ====")
            print(self.state.grid[:, :, 0])
        self.controller.input_observation(init_obs)
        self._redraw()

    def _step(self):
        print("=" * 80)
        print(f"==== {self.state.time} / {self.env.max_steps} step")
        print("=" * 80)

        actions = self.controller.operate()
        if self.log:
            self.action_log.append(actions.tolist())
        if self.profile:
            start = time.perf_counter()
        self.key, subkey = jax.random.split(self.key)
        obs, state, reward, shaped_reward, done = self.env.step_env(self.state, actions, subkey)
        if self.profile:
            end = time.perf_counter()
            print(f"elapsed time: {(end - start) * 1000: .3f} [msec]")
        self.controller.input_observation(obs)
        self.state = state
        if self.verbose:

            @jax.jit
            def _info():
                jax.debug.print("{}", state, ordered=True)
                jax.debug.print("-" * 60, ordered=True)
                jax.debug.print("■■ 報酬 ■■", ordered=True)
                jax.debug.print("reward: {},  shaped_reward: {}", reward, shaped_reward, ordered=True)
                jax.debug.print("-" * 60, ordered=True)
                jax.debug.print("■■ 観測 ■■", ordered=True)
                jax.debug.print("schedule\n{}", obs["schedule"], ordered=True)
                jax.debug.print("customer\n{}", obs["customer"], ordered=True)
                jax.debug.print("line\n{}", obs["line"], ordered=True)

            _info()
            if self.obs:
                # transposed_obs = jnp.transpose(obs["agent_0"], (2, 0, 1))
                transposed_obs = jnp.transpose(obs["all_agents"], (2, 0, 1))
                for i in range(transposed_obs.shape[0]):
                    print(f"------- channel {i} ----------------------------------")
                    print(transposed_obs[i])

        if done and self.loop:
            self.save_log(self.iter_num)
            self.iter_num += 1
            self._reset()
        elif done and not self.loop:
            self.save_log(None)
            exit()
        else:
            self._redraw()

    def save_log(self, iter_num: int | None):
        if self.log:
            actions = jnp.array(self.action_log, dtype=jnp.int8)
            init_key = jax.random.key_data(self.init_key)
            self.log_dir.mkdir(exist_ok=True, parents=True)
            filename = f"action_log_iter{iter_num}.npz" if iter_num is not None else "action_log.npz"
            log_file = self.log_dir / filename
            jnp.savez(log_file, actions=actions, init_key=init_key)


def load_config(config: DictConfig):
    layout = config.layout.get(str(config.get("stage", None)), None)
    if layout is None:
        print("select one of stages by stage=(stage_name)")
        print(list(config.layout.keys()))
        exit()
    with open_dict(config):
        config.env["layout"] = layout
        del config.layout
        # uiの設定から実行時の各エージェント操作方式に対応するものを抽出
        config.interface = [{kind: config.ui[kind]} for kind in config.player]
    return config


@hydra.main(config_path="../config", config_name="interactive", version_base=None)
def main(config: DictConfig):
    config: PlayConfig = load_config(config)
    interactive = InteractiveOvercookedCustom(config)
    interactive.run()


if __name__ == "__main__":
    main()
