from enum import IntEnum

import jax
import jax.numpy as jnp


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
    def declare_action_set(storage_sizes: list[int]):
        return jnp.array(jnp.arange(max(storage_sizes) + Actions.PICK_PLACE_BASE))

    @staticmethod
    def interpret_interact(action: int):
        return jax.lax.cond(
            action >= Actions.INTERACT,
            lambda: (True, jnp.clip(action - Actions.PICK_PLACE_BASE, min=0)),
            lambda: (False, -1),
        )

    @staticmethod
    def action_to_direction(action):
        ACT_TO_DIR = jnp.array([(0, +1), (+1, 0), (0, -1), (-1, 0), (0, 0)])
        return ACT_TO_DIR[action]
