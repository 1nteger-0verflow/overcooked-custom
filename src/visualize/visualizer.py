
import imageio
import jax
import jax.numpy as jnp

from environment.actions import Actions, ActionType
from environment.agent import Agent
from environment.customer import CustomerLine
from environment.dynamic_object import DynamicObject
from environment.state import Channel
from environment.state import State as EnvState
from environment.static_object import StaticObject
from visualize.renderer import render_grid
from visualize.window import Window


class OvercookedCustomVisualizer:
    """Manages a window and renders contents of EnvState instances to it."""

    def __init__(self):
        self.window = Window("Overcooked V2-customized")

    def show(self, *, block: bool = False):
        self.window.show(block=block)

    def render(self, state: EnvState, title: str = "", caption: str = ""):
        """Method for rendering the state in a window. Esp. useful for interactive mode."""
        img = self._render_state(state)

        self.window.set_title(title)
        self.window.set_caption(caption)
        self.window.show_img(img)

    def _render_multi_state(self, states: EnvState, rows: int, cols: int):
        imgs = jax.vmap(self._render_state)(states)  # (NUM_ENVS, height, width, color)
        # imgsのaxis=0のlength(=並列環境数)がrows*colsに満たない場合は0で埋める
        padding = jnp.zeros((rows * cols - imgs.shape[0], *imgs.shape[1:]), dtype=jnp.uint8)
        # 各環境の可視化画像を(rows, cols)に並べる
        padded_imgs = jnp.concat([imgs, padding])
        hstack_img = jnp.hstack(padded_imgs)
        row_split_img = jnp.split(hstack_img, rows, axis=1)
        return jnp.vstack(row_split_img)

    def render_multi(self, states: EnvState, rows: int, cols: int, title: str = "", caption: str = ""):
        grid_img = self._render_multi_state(states, rows, cols)
        self.window.set_title(title)
        self.window.set_caption(caption)
        self.window.show_img(grid_img)

    def close(self):
        self.window.close()

    def animate(self, state_seq: list[EnvState], filename: str = "animation.gif"):
        """Animate a gif give a state sequence and save if to file."""
        # frame_seq = jax.vmap(self._render_state)(state_seq)
        frame_seq = [self._render_state(state) for state in state_seq]
        # print("frame_seq", frame_seq)
        # print("frame_seq.shape", frame_seq.shape)
        # print("frame_seq.dtype", frame_seq.dtype)

        imageio.mimsave(filename, frame_seq, "GIF", duration=0.5)

    def grid_animate(self, states_seq, rows: int, cols: int, filename="animation.gif"):
        frame_seq = [self._render_multi_state(states, rows, cols) for states in states_seq]
        imageio.mimsave(filename, frame_seq, "GIF", duration=0.5)

    @classmethod
    def _encode_agent_extras(cls, direction: jax.Array, idx: int):
        dir_order = jnp.array([[-1, 0], [+1, 0], [0, +1], [0, -1]])
        dir_idx = jnp.argmax(jnp.all(direction == dir_order, axis=1))
        dir_num = jax.lax.switch(
            dir_idx,
            [
                # 右向きから時計回りに90度回転させる回数
                lambda: 3,
                lambda: 1,
                lambda: 0,
                lambda: 2,
            ],
        )
        return dir_num | (idx << 4)

    def aux_info(self, state: EnvState, grid, highlight_mask):
        # gridはレイアウトのサイズに描画用情報を付与したもの
        height, width = grid.shape[:2]
        agents = state.agents
        customer = state.customer
        # エージェントもStaticObject.EMPTYに描画用情報(idx<<4|direction)があるかどうかで表示されるので、
        # extraにそれと被らない値を入れて区別する
        is_agent_aux = 1 << 2
        is_customer_aux = 1 << 3
        aux_height = agents.num_agents
        aux_width = customer.ordered_menu.shape[1]

        # エージェントの行動、持ち物
        agent_aux = jnp.zeros((aux_height, width, 3), dtype=jnp.int32)
        agent_aux = agent_aux.at[:, :, Channel.extra].set(is_agent_aux)

        def _include_agent_aux(row, agent, action, idx):
            # エージェント情報表示の各グリッドに行動内容を含める

            def _encode_direction(agent, idx):
                return OvercookedCustomVisualizer._encode_agent_extras(agent.dir, idx)

            def _encode(i, row):
                action_type = Actions.action_type(i)[0]
                column = jax.lax.switch(action_type, [lambda: 0, lambda: 1, lambda: 2, lambda: i - 3])
                select = i == action
                # 移動のときは移動方向をエンコードする
                select_move = select & (action_type == ActionType.MOVE)
                # pick_placeは実行したかどうかによらず常に持っているものを表示する
                is_pick_place = action_type == ActionType.PICK_PLACE
                # 移動は同じマスを上書きするので注意
                obj = row[column, Channel.obj]
                extra = row[column, Channel.extra]
                obj = jax.lax.cond(select_move, _encode_direction, lambda a, i: obj, agent, idx)
                obj = jax.lax.cond(is_pick_place, lambda: agent.inventory[i - Actions.PICK_PLACE_BASE], lambda: obj)
                extra = jax.lax.cond(
                    select,
                    lambda: is_agent_aux | action_type << 8 | select << 16,
                    lambda: extra | is_agent_aux | action_type << 8,
                )
                row = row.at[column, Channel.obj].set(obj).at[column, Channel.extra].set(extra)
                return row

            row = jax.lax.fori_loop(0, agents.num_actions[idx], _encode, row)
            return row

        agent_aux = jax.vmap(_include_agent_aux)(agent_aux, agents, state.prev_actions, jnp.arange(agents.num_agents))
        # 注文内容
        # カウンターに置いた料理と同様に出来上がりの料理を描画できるようにする
        customer_aux = jnp.zeros((height, aux_width, 3), dtype=jnp.int32)
        customer_aux = customer_aux.at[:, :, Channel.extra].set(is_customer_aux)

        def _include_order(aux, x):
            order, table_id = x
            order = jnp.where(order > 0, order, 0)
            aux = aux.at[table_id, :, Channel.obj].set(order)
            return aux, None

        customer_aux, _ = jax.lax.scan(
            _include_order, customer_aux, (customer.ordered_menu, jnp.arange(customer.num_customers))
        )
        # jax.debug.print("{}", customer_aux)
        # 右下部分を埋める
        # 最終形が長方形になるよう右下の埋める部分を作成
        padding = jnp.zeros((aux_height, aux_width, 3), dtype=jnp.uint8)
        padding = padding.at[:, :, 0].set(StaticObject.EMPTY).at[:, :, 2].set(is_agent_aux)
        aux_grid = jnp.concat(
            [jnp.concat([grid, customer_aux], axis=1), jnp.concat([agent_aux, padding], axis=1)], axis=0
        )
        extend_highlight_mask = jnp.block(
            [
                [highlight_mask, jnp.zeros((height, aux_width), dtype=bool)],
                [jnp.zeros((aux_height, width), dtype=bool), jnp.zeros((aux_height, aux_width), dtype=bool)],
            ]
        )
        return aux_grid, extend_highlight_mask

    @jax.jit(static_argnums=(0,))
    def _render_state(self, state: EnvState):
        """Render the state."""
        grid = state.grid
        agents = state.agents
        customer = state.customer
        register = state.register
        line = state.line

        ###########################################
        # 表示用の情報をextra_infoに格納しておく
        ###########################################
        # agentの向きを格納
        def _include_agents(grid: jax.Array, x: tuple[Agent, jax.Array]):
            agent, idx = x
            pos = agent.pos
            inventory = agent.inventory[0]
            direction = agent.dir
            # we have to do the encoding because we don't really have a way to also pass the agent's id
            extra_info = OvercookedCustomVisualizer._encode_agent_extras(direction, idx)

            # gridのshapeは変わらないがchの中身は表示用で変更あり
            new_grid = grid.at[*pos].set([StaticObject.AGENT, inventory, extra_info])
            return new_grid, None

        grid, _ = jax.lax.scan(_include_agents, grid, (agents, jnp.arange(agents.num_agents)))

        # 客の待ち人数を格納
        def _include_line(grid: jax.Array, line: CustomerLine):
            # 入口は1か所
            entrance_pos = line.entrance_pos[0]
            extra_info = (
                line.reserved_line_length
                | len(line.reserved_queued_time) << 8
                | line.line_length << 16
                | len(line.queued_time) << 24
            )
            return grid.at[*entrance_pos, Channel.extra].set(extra_info)

        grid = _include_line(grid, line)

        # 客席に出されている料理、食べ終わり、着席状況を格納
        def _include_customer(grid: jax.Array, x: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
            table_pos, chair_pos, used, status, food = x
            table_extra_info = (
                jnp.sum(DynamicObject.get_count(food) > 0)
                | (jnp.sum(DynamicObject.is_plate(food) > 0) - jnp.sum(DynamicObject.get_count(food) > 0)) << 8
                | len(food) << 16
            )
            chair_extra_info = jnp.sum(used << 8 | status)
            new_grid = (
                grid.at[*table_pos, Channel.extra]
                .set(table_extra_info)
                .at[*chair_pos, Channel.extra]
                .set(chair_extra_info)
            )
            return new_grid, None

        grid, _ = jax.lax.scan(
            _include_customer,
            grid,
            (customer.table_pos, customer.chair_pos, customer.used, customer.status, customer.food),
        )

        # レジの待ち有無を格納
        def _include_register(grid: jax.Array):
            # レジは1か所の想定
            register_pos = register.register_pos[0]
            extra_info = register.service_time
            return grid.at[*register_pos, Channel.extra].set(extra_info)

        grid = _include_register(grid)

        highlight_mask = jnp.zeros(grid.shape[:2], dtype=bool)
        view_area_tips = state.agents.compute_view_box(grid.shape[0], grid.shape[1])

        def _view_area(area: jax.Array, tips: jax.Array):
            xmin, xmax, ymin, ymax = tips
            area_mask = jax.lax.fori_loop(
                ymin,
                ymax,
                lambda y, area: jax.lax.fori_loop(xmin, xmax, lambda x, area: area.at[y, x].set(True), area),
                area,
            )
            return area_mask, None

        highlight_mask, _ = jax.lax.scan(_view_area, highlight_mask, view_area_tips)

        grid, highlight_mask = self.aux_info(state, grid, highlight_mask)
        # Render the whole grid
        return render_grid(grid, highlight_mask)
