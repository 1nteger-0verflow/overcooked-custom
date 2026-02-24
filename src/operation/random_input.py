import random

from operation.agent_controller import AgentController


class RandomInput(AgentController):
    def __init__(self, agent_id: int, num_actions: int):
        self.agent_id = agent_id
        self.num_actions = num_actions

    def get_action(self):
        return random.choice(range(self.num_actions))
