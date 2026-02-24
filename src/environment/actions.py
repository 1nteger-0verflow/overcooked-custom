from enum import IntEnum
from typing import List

import jax
import jax.numpy as jnp


class ActionType(IntEnum):
    MOVE = 0
    NOP = 1
    INTERACTION = 2
    PICK_PLACE = 3


class Actions(IntEnum):
    # Turn left, turn right, move forward
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    STAY = 4
    # InteractionするアクションはINTERACT以降にする（大小で判定するため）
    INTERACT = 5
    PICK_PLACE_BASE = INTERACT + 1
    # REACT = 6  # 現状未使用だが、エージェントの状態がbusyのとき自動的にこれを選ぶようにし、複数stepにわたって拘束されるようにする
    # PICK_PLACE_BASE = REACT + 1

    @staticmethod
    def declare_action_set(storage_sizes: List[int]):
        return jnp.array(jnp.arange(max(storage_sizes) + Actions.PICK_PLACE_BASE))

    @staticmethod
    def action_type(action: int):
        type_branch = jnp.array(
            [
                action < 0,  # 持てる数以上のpick_placeを指定したとき
                action < Actions.STAY,
                action == Actions.STAY,
                action == Actions.INTERACT,
                action > Actions.INTERACT,
            ]
        )
        return jax.lax.switch(
            jnp.argmax(type_branch),
            [
                lambda: (ActionType.NOP, -1),
                lambda: (ActionType.MOVE, -1),
                lambda: (ActionType.NOP, -1),
                lambda: (ActionType.INTERACTION, -1),
                lambda: (
                    ActionType.PICK_PLACE,
                    jnp.clip(action - Actions.PICK_PLACE_BASE, min=0),
                ),
            ],
        )

    @classmethod
    def action_to_string(cls, value):
        for member in cls:
            if member.value == value and value < cls.PICK_PLACE_BASE:
                return member.name
        else:
            return f"PICK_PLACE_{value - cls.PICK_PLACE_BASE}"

    @staticmethod
    def action_to_direction(action):
        ACT_TO_DIR = jnp.array(
            [
                (0, +1),
                (+1, 0),
                (0, -1),
                (-1, 0),
                (0, 0),
            ]
        )
        return ACT_TO_DIR[action]
