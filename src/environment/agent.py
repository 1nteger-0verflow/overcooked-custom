import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode, dataclass

from environment.actions import Actions
from environment.dynamic_object import DynamicObject


@dataclass
class Agent(PyTreeNode):
    # Remark: ndarrayの最初の次元の長さをそろえる(=num_agents)。そうでないとvmapなどの並列化が失敗する。
    pos: jnp.ndarray  # エージェントごとの位置 (y,x)の順
    dir: jnp.ndarray  # (y, x)
    capacity: jnp.ndarray  # 各エージェントが持ち運べるものの個数(inventoryの使える列数)
    inventory: jnp.ndarray  # 縦：AgentID, 横：持てる個数のエージェント間最大値
    view_sizes: jnp.ndarray  # 視野範囲　[前方, 左右]
    # 各マスを視野範囲内に最後に見たstep数をエージェントごとに格納
    grid_observed_step: jnp.ndarray

    # interactionするためにエージェントの前の位置を取得するためのメソッド
    def get_fwd_pos(self):
        return self.pos + self.dir

    # エージェント1体ずつ処理されるためposは1-dimになる
    def move_in_bounds(self, dir: jnp.ndarray, height: int, width: int):
        new_pos = self.pos + dir
        return jnp.array([jnp.clip(new_pos[0], 0, height - 1), jnp.clip(new_pos[1], 0, width - 1)])

    def modify_action(self, action: int):
        return jax.lax.cond(action - Actions.PICK_PLACE_BASE < self.capacity, lambda: action, lambda: -1)

    @jax.jit
    def compute_view_box(self, height: int, width: int) -> jax.Array:
        # レイアウト全体のサイズを受け取って、レイアウト内でエージェントの観測できる範囲をy_min, y_max, x_min, x_maxで求める
        def _compute(yx: jnp.ndarray, dir: jnp.ndarray, view_size: jnp.ndarray) -> jax.Array:
            fwd_view, side_view = view_size
            # 向きによる視野範囲を計算（レイアウトは考慮しない）
            # x_min = pos.x + coeff1*side_view + coeff2*fwd_view などで計算する
            branch = jnp.prod(
                jnp.array(
                    [
                        dir == jnp.array([-1, 0]),
                        dir == jnp.array([+1, 0]),
                        dir == jnp.array([0, +1]),
                        dir == jnp.array([0, -1]),
                    ]
                ),
                axis=1,
            )
            branch_idx = jnp.argmax(branch)
            x_min_coef, x_max_coef, y_min_coef, y_max_coef = jax.lax.switch(
                branch_idx,
                [
                    lambda: ((-1, 0), (+1, 0), (0, -1), (0, 0)),
                    lambda: ((-1, 0), (+1, 0), (0, 0), (0, +1)),
                    lambda: ((0, 0), (0, +1), (-1, 0), (+1, 0)),
                    lambda: ((0, -1), (0, 0), (-1, 0), (+1, 0)),
                ],
            )
            x_min = yx[1] + x_min_coef[0] * side_view + x_min_coef[1] * fwd_view
            x_max = yx[1] + x_max_coef[0] * side_view + x_max_coef[1] * fwd_view
            y_min = yx[0] + y_min_coef[0] * side_view + y_min_coef[1] * fwd_view
            y_max = yx[0] + y_max_coef[0] * side_view + y_max_coef[1] * fwd_view
            # レイアウト全体のサイズを考慮
            bounded_x_min = jnp.clip(x_min, min=0)
            bounded_x_max = jnp.clip(x_max + 1, max=width)
            bounded_y_min = jnp.clip(y_min, min=0)
            bounded_y_max = jnp.clip(y_max + 1, max=height)
            return jnp.array([bounded_x_min, bounded_x_max, bounded_y_min, bounded_y_max])

        return jax.vmap(_compute)(self.pos, self.dir, self.view_sizes)

    def update_observed_grid(self, time: jnp.ndarray, height: int, width: int):
        observable_area = self.compute_view_box(height, width)

        def _update_observed_area(observed, x):
            area, idx = x
            xmin, xmax, ymin, ymax = area
            observed = jax.lax.fori_loop(
                ymin,
                ymax,
                lambda y, observed: jax.lax.fori_loop(
                    xmin, xmax, lambda x, observed: observed.at[idx, y, x].set(time), observed
                ),
                observed,
            )
            return observed, None

        num_agents = self.dir.shape[0]
        new_observed, _ = jax.lax.scan(
            _update_observed_area, self.grid_observed_step, (observable_area, jnp.arange(num_agents))
        )
        return self.replace(grid_observed_step=new_observed)

    def __str__(self):
        def _discribe_agent(idx):
            disc = f"agent{idx}  pos:({self.pos[idx]}), dir: ({self.dir[idx]}), 持てる個数:{self.capacity[idx]}\n"
            disc += "持ってるもの："
            for i in range(self.capacity[idx]):
                disc += f"{DynamicObject.decode(self.inventory[idx, i])}, "
            disc += "\n"
            return disc

        expr = "-" * 60 + "\n"
        expr += "■■ エージェント ■■\n"
        for i in range(self.inventory.shape[0]):
            expr += _discribe_agent(i)
        return expr
