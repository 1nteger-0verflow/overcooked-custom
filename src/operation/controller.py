import jax.numpy as jnp
from omegaconf import ListConfig

from operation.agent_controller import AgentController


class Controller:
    def __init__(self, env, config: ListConfig, verbose: bool, confirm: bool):
        # エージェントが何体いるか、
        # 操作用の設定が不足していたら最後のものを繰り返して使う
        self.num_agents = env.num_agents
        num_actions = env.num_actions
        if len(config) < self.num_agents:
            config += [config[-1]] * (self.num_agents - len(config))
        self.controllers = []
        for i in range(self.num_agents):
            self.controllers.append(
                AgentController.create_controller(
                    config[i], i, num_actions, verbose, confirm
                )
            )
        self.controller_idx = 0

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
        done = True
        for controller in self.controllers:
            done &= controller.is_done
        return done

    def keyboard_input(self, key: str) -> bool:
        for _ in range(self.num_agents):
            accept = self.controllers[self.controller_idx].input_key(key)
            self.controller_idx = (self.controller_idx + 1) % self.num_agents
            if accept:
                return self.is_done()
        return self.is_done()

    def operate(self):
        return jnp.array([controller.get_action() for controller in self.controllers])
