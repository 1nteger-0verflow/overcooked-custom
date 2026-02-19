from omegaconf import DictConfig

from operation.ippo_model_controller import IPPOModelInput
from operation.keyboard_input import KeyboardInput
from operation.random_input import RandomInput
from operation.replay_log import ReplayLog


def create_controller(
    operation: DictConfig, agent_id: int, num_actions: int, verbose: bool, confirm: bool
):
    optype = list(operation.keys())[0]

    if optype == "keyboard":
        return KeyboardInput(agent_id, verbose, confirm)
    elif optype == "random":
        return RandomInput(agent_id, num_actions)
    elif optype == "replay":
        return ReplayLog(agent_id, operation[optype])
    elif optype == "ippo":
        return IPPOModelInput(agent_id, operation[optype], num_actions)
