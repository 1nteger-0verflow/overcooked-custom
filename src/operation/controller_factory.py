from operation.ippo_model_controller import IPPOModelInput
from operation.keyboard_input import KeyboardInput
from operation.random_input import RandomInput
from operation.replay_log import ReplayLog


def create_controller(operation: dict[str, Any], agent_id: int, num_actions: int, verbose: bool, confirm: bool):
    optype = next(iter(operation.keys()))

    if optype == "keyboard":
        return KeyboardInput(agent_id, verbose, confirm)
    if optype == "random":
        return RandomInput(agent_id, num_actions)
    if optype == "replay":
        return ReplayLog(agent_id, operation[optype])
    if optype == "ippo":
        return IPPOModelInput(agent_id, operation[optype], num_actions)
    raise
