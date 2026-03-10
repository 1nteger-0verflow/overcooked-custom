from collections import abc
from typing import Any

from config import IPPOModelConfig, ReplayConfig
from operation.ippo_model_controller import IPPOModelInput
from operation.keyboard_input import KeyboardInput
from operation.random_input import RandomInput
from operation.replay_log import ReplayLog


def create_controller(
    operation: abc.Mapping[str, Any], agent_id: int, num_actions: int, *, verbose: bool, confirm: bool
):
    optype = next(iter(operation.keys()))

    if optype == "keyboard":
        return KeyboardInput(agent_id, verbose=verbose, confirm=confirm)
    if optype == "random":
        return RandomInput(agent_id, num_actions)
    if optype == "replay":
        sub = dict(operation[optype] or {})
        return ReplayLog(agent_id, ReplayConfig(**sub))
    if optype == "ippo":
        sub = dict(operation[optype] or {})
        return IPPOModelInput(agent_id, IPPOModelConfig(**sub), num_actions, verbose=verbose)
    msg = f"Unknown operation type: {optype!r}"
    raise ValueError(msg)
