from enum import IntEnum

import jax.numpy as jnp
from flax.struct import PyTreeNode, dataclass

from environment.agent import Agent
from environment.customer import (
    Customer,
    CustomerLine,
    RegisterLine,
)
from environment.dynamic_object import DynamicObject


class Channel(IntEnum):
    # gridの何チャンネル目が何を意味しているか
    env = 0  # Static items(変化しないレイアウト)
    obj = 1  # Dynamic Object (plates, ingredients, dirt)
    extra = 2  # 残調理時間  TODO: 多分別で管理する方がよい


@dataclass
class State(PyTreeNode):
    agents: Agent
    customer: Customer  # 客席
    line: CustomerLine  # 待ち行列
    register: RegisterLine  # 会計待ち

    # width x height x 3
    # 1 channel: static items
    # 2 channel: dynamic items (plates, ingredients, dirt)
    # 3 channel: extra info  # TODO: 調理の進行の管理をCustomerのような別クラスにする
    grid: jnp.ndarray

    time: jnp.ndarray

    def __str__(self):
        expr = str(self.grid[:, :, Channel.obj])
        expr += "\n"
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x, Channel.obj] != DynamicObject.EMPTY:
                    expr += DynamicObject.decode(self.grid[y, x, Channel.obj])
                    expr += ", "
            if jnp.sum(self.grid[y, :, Channel.obj]) > 0:
                expr += "\n"
        expr += (
            self.agents.__str__()
            + self.customer.__str__()
            + self.line.__str__()
            + self.register.__str__()
        )
        return expr
