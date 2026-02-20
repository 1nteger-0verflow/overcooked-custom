import warnings

import jax
import jax.numpy as jnp

from environment.agent import Agent
from environment.customer import Customer, CustomerLine, CustomerStatus, RegisterLine
from environment.dynamic_object import DynamicObject
from environment.layouts import Layout
from environment.menus import MenuList
from environment.state import Channel, State
from environment.static_object import StaticObject
from utils.schema import EnvConfig


def _compute_enclosed_spaces(empty_mask: jnp.ndarray) -> jnp.ndarray:
    """JaxMARL/jaxmarl/environments/overcooked_v2/utils.py
    エージェントが移動して到達できる範囲内に同じIDを振ったarrayを返す。
    (カウンターなどで区切られたエリア内でエージェントの初期位置をランダムにするために使用する)
    empty_mask には grid==StaticObject.EMPTY のマップを入力する。
    """
    height, width = empty_mask.shape
    # 各マスに左上からナンバリング
    id_grid = jnp.arange(empty_mask.size, dtype=jnp.int32).reshape(empty_mask.shape)
    # 移動不可能な範囲は-1で埋める
    id_grid = jnp.where(empty_mask, id_grid, -1)
    # 上下左右の移動方向
    directions = jnp.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])

    def _body_fun(val):
        _, current_grid = val

        def _next_val(pos):
            @jax.vmap
            def _move_in_bounds(dir: jax.Array):
                return jnp.clip(pos + dir, min=0, max=jnp.array([height - 1, width - 1]))

            neighbors = _move_in_bounds(directions)
            neighbour_values = current_grid[*neighbors.T]
            self_value = current_grid[*pos]
            values = jnp.concatenate([neighbour_values, self_value[jnp.newaxis]], axis=0)
            new_val = jnp.max(values)
            return jax.lax.select(self_value == -1, self_value, new_val)

        # 全てのマスで周囲4マスの値との最大値に置き換える(移動可能でないマスは除く)
        pos = jnp.stack(jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij"), axis=-1)
        new_grid = jax.vmap(jax.vmap(_next_val))(pos)
        # 変化がなくなったら終了(-1で区切られた各連結成分内の最大番号になる)
        stop = jnp.all(current_grid == new_grid)
        return stop, new_grid

    def _cond_fun(val):
        return ~val[0]

    initial_val = (False, id_grid)
    _, res = jax.lax.while_loop(_cond_fun, _body_fun, initial_val)
    return res


class Initializer:
    def __init__(self, config: EnvConfig, layout: Layout, menu: MenuList, random_agent_position: bool):
        self.layout = layout
        self.menu = menu
        self.num_agents = len(layout.agent_positions)
        self.num_customers = layout.num_customers
        #################################
        # 視野範囲のパラメータ読み込み
        #################################
        # 前方の視野範囲
        self.forward_view_size = config.parameter.forward_view_size
        if isinstance(self.forward_view_size, int):
            self.forward_view_size = [self.forward_view_size] * self.num_agents
        if len(self.forward_view_size) < self.num_agents:
            warnings.warn(
                "insufficient parameters specified.\n"
                f"forward_view_size({len(self.forward_view_size)}) is not sufficient for num_agents({self.num_agents})"
            )
            # 不足分は最後の設定を繰り返し用いる
            self.forward_view_size += [self.forward_view_size[-1]] * (self.num_agents - len(self.forward_view_size))
        # 多い場合はエージェント数までの設定を使用する
        self.forward_view_size = self.forward_view_size[: self.num_agents]

        # 左右の視野範囲
        self.side_view_size = config.parameter.side_view_size
        if isinstance(self.side_view_size, int):
            self.side_view_size = [self.side_view_size] * self.num_agents
        if len(self.side_view_size) < self.num_agents:
            warnings.warn(
                "insufficient parameters specified.\n"
                f"side_view_size({len(self.side_view_size)}) is not sufficient for num_agents({self.num_agents})"
            )
            # 不足分は最後の設定を繰り返し用いる
            self.side_view_size += [self.side_view_size[-1]] * (self.num_agents - len(self.side_view_size))
        # 多い場合はエージェント数までの設定を使用する
        self.side_view_size = self.side_view_size[: self.num_agents]

        # [前方, 左右]のリストにする
        self.agent_view_size = list(zip(self.forward_view_size, self.side_view_size))

        #################################
        # 把持可能数の読み込み
        #################################
        self.capacity = config.parameter.capacity
        if isinstance(self.capacity, int):
            self.capacity = [self.capacity] * self.num_agents
        if len(self.capacity) < self.num_agents:
            warnings.warn(
                "insufficient parameters specified.\n"
                f"capacity({len(self.capacity)}) is not sufficient for num_agents({self.num_agents})"
            )
            # 不足分は最後の設定を繰り返し用いる
            self.capacity += [self.capacity[-1]] * (self.num_agents - len(self.capacity))
        # 多い場合はエージェント数までの設定を使用する
        self.capacity = self.capacity[: self.num_agents]
        config.parameter.capacity = self.capacity  # TODO: Check mutability

        self.order_max = config.parameter.order_max
        self.plate_count = config.parameter.plate_count
        self.wait_line_max = config.parameter.wait_line_max
        self.reservation = config.schedule.reservation
        self.random_agent_position = random_agent_position

    def initialize(self, key: jax.Array):
        static_objects = self.layout.static_objects
        grid = jnp.stack(
            [
                static_objects,
                jnp.zeros_like(static_objects),  # dynamic object channel
                jnp.zeros_like(static_objects),  # extra info channel
            ],
            axis=-1,
            dtype=jnp.int32,
        )

        # エージェントを配置
        num_agents = self.num_agents
        max_storage_size = max(self.capacity)
        view_sizes = jnp.array(self.agent_view_size)
        view_sizes = jnp.where(view_sizes == 0, self.layout.size_limit, view_sizes)
        grid_observed = jnp.full((num_agents, self.layout.height, self.layout.width), -1, dtype=jnp.int32)
        agents = Agent(
            pos=jnp.array(self.layout.agent_positions, dtype=jnp.int32),
            dir=jnp.stack([jnp.array([-1, 0], dtype=jnp.int32)] * num_agents),  # UP
            capacity=jnp.array(self.capacity),
            inventory=jnp.zeros((num_agents, max_storage_size), dtype=jnp.int32),
            view_sizes=view_sizes,
            grid_observed_step=grid_observed,
        )
        if self.random_agent_position:
            agents = self._randomize_agent_positions(agents, key)
        # 皿を配置
        if len(self.layout.plate_positions) > 0:
            plate_pile = DynamicObject.get_clean_plates(self.plate_count)
            for y, x in self.layout.plate_positions:
                grid = grid.at[y, x, Channel.obj].set(plate_pile)
        # 入口を配置
        waiting_line = CustomerLine(
            entrance_pos=jnp.array(self.layout.entrance_positions),
            line_length=jnp.array(0, dtype=jnp.int32),
            queued_time=jnp.zeros((self.wait_line_max,), dtype=jnp.int32),
            reserved_line_length=jnp.array(0, dtype=jnp.int32),
            reserved_queued_time=jnp.zeros((len(self.reservation)), dtype=jnp.int32),
            reserve_time=jnp.array(self.reservation, dtype=jnp.int32),
        )

        # 客席を配置
        customers = Customer(
            table_pos=jnp.array(self.layout.table_positions),
            chair_pos=jnp.array(self.layout.chair_positions),
            menu=self.menu,
            used=jnp.zeros((self.num_customers,), dtype=int),
            status=jnp.full((self.num_customers,), CustomerStatus.empty, dtype=jnp.int32),
            time=jnp.zeros((self.num_customers,), dtype=int),
            ordered_menu=jnp.full((self.num_customers, self.order_max), -1, dtype=jnp.int32),
            food=jnp.full((self.num_customers, self.order_max), DynamicObject.EMPTY, dtype=jnp.int32),
        )

        # 会計待ちの列
        register = RegisterLine(
            register_pos=jnp.array(self.layout.register_positions),
            queued_time=jnp.array(0, dtype=jnp.int32),
            service_time=jnp.array(0, dtype=jnp.int32),
        )

        return State(
            agents=agents,
            customer=customers,
            line=waiting_line,
            register=register,
            grid=grid,
            time=jnp.array(0, dtype=jnp.int32),
        )

    def _randomize_agent_positions(self, agents: Agent, key: jax.Array):
        # 空の場所からエージェントの初期位置を選択
        # （元々の位置から移動可能な範囲内でランダムにする）
        enclosed_spaces = _compute_enclosed_spaces(self.layout.static_objects == StaticObject.EMPTY)

        def _select_agent_position(taken_mask: jnp.ndarray, x):
            pos, key = x

            allowed_positions = (enclosed_spaces == enclosed_spaces[*pos]) & ~taken_mask
            allowed_positions = allowed_positions.flatten()
            # 配置可能なマスから等確率で選択
            unif_p = allowed_positions / jnp.sum(allowed_positions)
            agent_pos_idx = jax.random.choice(key, allowed_positions.size, (), p=unif_p)
            agent_position = jnp.array([agent_pos_idx // self.layout.width, agent_pos_idx % self.layout.width])
            new_taken_mask = taken_mask.at[*agent_position].set(True)
            return new_taken_mask, agent_position

        # 先に決めたエージェントと同じ位置にならないようにするためのマスク
        taken_mask = jnp.zeros_like(enclosed_spaces, dtype=jnp.bool_)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, self.num_agents)
        _, agent_positions = jax.lax.scan(_select_agent_position, taken_mask, (agents.pos, keys))
        # 向きをランダムにする
        key, subkey = jax.random.split(key)
        direction_idxs = jax.random.randint(subkey, (self.num_agents,), 0, 4)
        dir_vectors = jnp.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
        directions = dir_vectors.take(direction_idxs, axis=0)
        return agents.replace(pos=agent_positions, dir=directions)
