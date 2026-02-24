import jax.numpy as jnp
from omegaconf import DictConfig, ListConfig

from operation.agent_controller import AgentController


class Controller:
    def __init__(self, env, config: DictConfig):
        self.num_agents = env.num_agents
        controller_config = self.apply_config(config.ui, config.player)
        self.controllers = []
        for i in range(self.num_agents):
            self.controllers.append(
                AgentController.create_controller(
                    controller_config[i], i, env.num_actions, config.verbose, config.confirm
                )
            )
        self.controller_idx = 0

    def apply_config(self, ui: DictConfig, player: str | ListConfig) -> ListConfig:
        player_list = [player] if isinstance(player, str) else player
        # レイアウトに含まれるエージェント数に対して操作方法指定が不足する場合は、最後のものを繰り返し適用
        operation_types = player_list + [player_list[-1]] * (self.num_agents - len(player_list))
        operation_types = operation_types[: self.num_agents]
        print(f"エージェント数: {self.num_agents}, 操作種別: {operation_types}")
        # 全エージェント分になるよう操作設定を補う
        filled_settings = {}
        for k, v in ui.items():
            if isinstance(v, ListConfig):
                filled_settings[k] = v + [v[-1]] * (self.num_agents - len(v))
            else:
                filled_settings[k] = [v] * self.num_agents
        interfaces = []
        for operation_type in operation_types:
            interfaces.append({operation_type: filled_settings[operation_type].pop(0)})
        return ListConfig(interfaces)

    def input_observation(self, obs):
        # stepごとの初期化
        self.controller_idx = 0
        for controller in self.controllers:
            controller.input_observation(obs)

    def is_auto(self) -> bool:
        auto = True
        for controller in self.controllers:
            auto &= controller.is_auto
        return auto

    def is_done(self) -> bool:
        return all(c.is_done for c in self.controllers)

    def keyboard_input(self, key: str) -> bool:
        for _ in range(self.num_agents):
            accept = self.controllers[self.controller_idx].input_key(key)
            self.controller_idx = (self.controller_idx + 1) % self.num_agents
            if accept:
                return self.is_done()
        return self.is_done()

    def operate(self):
        return jnp.array([controller.get_action() for controller in self.controllers])
