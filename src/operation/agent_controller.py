from collections import abc

import jax.numpy as jnp

_OperationConfig = None | abc.Mapping[str, str | int]


class AgentController:
    @classmethod
    def create_controller(
        cls,
        operation: abc.Mapping[str, _OperationConfig],
        agent_id: int,
        num_actions: int,
        *,
        verbose: bool,
        confirm: bool,
    ):
        from operation.controller_factory import create_controller

        return create_controller(operation, agent_id, num_actions, verbose=verbose, confirm=confirm)

    def input_key(self, key: str) -> bool:
        print(f"AgentController: {key}")  # noqa: T201
        # key入力を受け付けたかどうかを返す。
        return False

    @property
    def is_auto(self) -> bool:
        return True

    @property
    def is_done(self) -> bool:
        # 必要な入力が終わったかどうか(keyboard/no_confirm -> 1回入力するとTrue, keyboard -> False)
        return True

    def input_observation(self, obs: jnp.ndarray):
        # モデルの場合観測をもとに次の行動を決定する
        pass
