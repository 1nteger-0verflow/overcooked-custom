from collections import abc
from pathlib import Path

import jax.numpy as jnp

from operation.agent_controller import AgentController


class ReplayLog(AgentController):
    def __init__(self, agent_id: int, config: abc.Mapping[str, str]):
        self.agent_id = agent_id
        log = Path(config.get("action_log"))
        log_data = jnp.load(log)
        self.step = 0
        init_key = log_data["init_key"]
        self.actions = log_data["actions"].astype(jnp.int32)
        print(f"agent_{agent_id} is replaying log with initial key {init_key}")

    def get_action(self):
        action = self.actions[self.step, self.agent_id]
        self.step += 1
        return action
