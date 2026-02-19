import math

import imageio
import jax
import jax.numpy as jnp

import visualize.grid_rendering_v2 as rendering
from environment.customer import CustomerStatus
from environment.dynamic_object import DynamicObject
from environment.state import Channel
from environment.static_object import StaticObject
from visualize.window import Window

POT_COOK_TIME = 10  # TODO: メニューにより調理時間が異なる
TILE_PIXELS = 32

COLORS = {
    "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
    "green": jnp.array([0, 255, 0], dtype=jnp.uint8),
    "blue": jnp.array([0, 0, 255], dtype=jnp.uint8),
    "purple": jnp.array([160, 32, 240], dtype=jnp.uint8),
    "yellow": jnp.array([255, 255, 0], dtype=jnp.uint8),
    "grey": jnp.array([100, 100, 100], dtype=jnp.uint8),
    "white": jnp.array([255, 255, 255], dtype=jnp.uint8),
    "black": jnp.array([25, 25, 25], dtype=jnp.uint8),
    "orange": jnp.array([230, 180, 0], dtype=jnp.uint8),
    "pink": jnp.array([255, 105, 180], dtype=jnp.uint8),
    "brown": jnp.array([139, 69, 19], dtype=jnp.uint8),
    "cyan": jnp.array([0, 255, 255], dtype=jnp.uint8),
    "light_blue": jnp.array([173, 216, 230], dtype=jnp.uint8),
    "dark_green": jnp.array([0, 150, 0], dtype=jnp.uint8),
}

INGREDIENT_COLORS = jnp.array(
    [
        COLORS["yellow"],
        COLORS["dark_green"],
        COLORS["purple"],
        COLORS["cyan"],
        COLORS["red"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["blue"],
        COLORS["pink"],
        COLORS["brown"],
    ]
)


AGENT_COLORS = jnp.array(
    [COLORS["red"], COLORS["blue"], COLORS["green"], COLORS["purple"], COLORS["yellow"], COLORS["orange"]]
)


class OvercookedCustomVisualizer:
    """Manages a window and renders contents of EnvState instances to it."""

    tile_cache = {}

    def __init__(self, tile_size: int = TILE_PIXELS, subdivs: int = 3):
        self.window = Window("Overcooked V2-customized")

        self.tile_size = tile_size
        self.subdivs = subdivs

    def show(self, block=False):
        self.window.show(block=block)

    def render(self, state, title: str = "", caption: str = ""):
        """Method for rendering the state in a window. Esp. useful for interactive mode."""
        img = self._render_state(state)

        self.window.set_title(title)
        self.window.set_caption(caption)
        self.window.show_img(img)

    def render_multi(self, states, rows: int, cols: int, title: str = "", caption: str = ""):
        imgs = jax.vmap(self._render_state)(states)  # (NUM_ENVS, height, width, color)
        # imgsのaxis=0のlength(=並列環境数)がrows*colsに満たない場合は0で埋める
        padding = jnp.zeros((rows * cols - imgs.shape[0], *imgs.shape[1:]), dtype=jnp.uint8)
        # 各環境の可視化画像を(rows, cols)に並べる
        padded_imgs = jnp.concat([imgs, padding])
        hstack_img = jnp.hstack(padded_imgs)
        row_split_img = jnp.split(hstack_img, rows, axis=1)
        grid_img = jnp.vstack(row_split_img)
        self.window.set_title(title)
        self.window.set_caption(caption)
        self.window.show_img(grid_img)

    def close(self):
        self.window.close()

    def animate(self, state_seq, filename="animation.gif", agent_view_size=None):
        """Animate a gif give a state sequence and save if to file."""
        frame_seq = jax.vmap(self._render_state, in_axes=(0, None))(state_seq, agent_view_size)
        # print("frame_seq", frame_seq)
        # print("frame_seq.shape", frame_seq.shape)
        # print("frame_seq.dtype", frame_seq.dtype)

        imageio.mimsave(filename, frame_seq, "GIF", duration=0.5)

    def render_sequence(self, state_seq, agent_view_size=None):
        return jax.vmap(self._render_state, in_axes=(0, None))(state_seq, agent_view_size)

    @classmethod
    def _encode_agent_extras(cls, direction, idx):
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

    @classmethod
    def _decode_agent_extras(cls, extras):
        direction = extras & 0x3
        idx = extras >> 4
        return direction, idx

    @jax.jit(static_argnums=(0,))
    def _render_state(self, state):
        """Render the state."""
        grid = state.grid
        agents = state.agents
        customer = state.customer
        register = state.register
        line = state.line

        num_agents = agents.dir.shape[0]

        ###########################################
        # 表示用の情報をextra_infoに格納しておく
        ###########################################
        # agentの向きを格納
        def _include_agents(grid, x):
            agent, idx = x
            pos = agent.pos
            inventory = agent.inventory[0]
            direction = agent.dir
            # we have to do the encoding because we don't really have a way to also pass the agent's id
            extra_info = OvercookedCustomVisualizer._encode_agent_extras(direction, idx)

            # gridのshapeは変わらないがchの中身は表示用で変更あり
            new_grid = grid.at[*pos].set([StaticObject.AGENT, inventory, extra_info])
            return new_grid, None

        grid, _ = jax.lax.scan(_include_agents, grid, (agents, jnp.arange(num_agents)))

        # 客の待ち人数を格納
        def _include_line(grid, line):
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
        def _include_customer(grid, x):
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
        def _include_register(grid):
            # レジは1か所の想定
            register_pos = register.register_pos[0]
            extra_info = register.service_time
            return grid.at[*register_pos, Channel.extra].set(extra_info)

        grid = _include_register(grid)

        highlight_mask = jnp.zeros(grid.shape[:2], dtype=bool)
        view_area_tips = state.agents.compute_view_box(grid.shape[0], grid.shape[1])

        def _view_area(area, tips):
            xmin, xmax, ymin, ymax = tips
            area_mask = jax.lax.fori_loop(
                ymin,
                ymax,
                lambda y, area: jax.lax.fori_loop(xmin, xmax, lambda x, area: area.at[y, x].set(True), area),
                area,
            )
            return area_mask, None

        highlight_mask, _ = jax.lax.scan(_view_area, highlight_mask, view_area_tips)

        # Render the whole grid
        return self._render_grid(grid, highlight_mask)

    @staticmethod
    def _render_dynamic_item(
        object,
        img,
        plate_fn=rendering.point_in_circle(0.5, 0.5, 0.3),
        ingredient_fn=rendering.point_in_circle(0.5, 0.5, 0.15),
        dish_positions=jnp.array([(0.5, 0.4), (0.4, 0.6), (0.6, 0.6)]),
    ):
        def _no_op(img, object):
            return img

        def _render_plate(img, object):
            return rendering.fill_coords(img, plate_fn, COLORS["white"])

        def _render_ingredient(img, object):
            idx = DynamicObject.get_ingredient_idx(object)
            return rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])

        def _render_dish(img, object):
            img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(object)

            for idx, ingredient_idx in enumerate(ingredient_indices):
                color = INGREDIENT_COLORS[ingredient_idx]
                pos = dish_positions[idx]
                ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
                img_ing = rendering.fill_coords(img, ingredient_fn, color)

                img = jax.lax.select(ingredient_idx != -1, img_ing, img)

            return img

        def _render_used_plate(img, object):
            img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            drop_fn1 = rendering.point_in_circle(0.6, 0.6, 0.15)
            drop_fn2 = rendering.point_in_circle(0.5, 0.3, 0.05)
            drop_fn3 = rendering.point_in_circle(0.3, 0.5, 0.05)
            img = rendering.fill_coords(img, drop_fn1, COLORS["brown"])
            img = rendering.fill_coords(img, drop_fn2, COLORS["brown"])
            return rendering.fill_coords(img, drop_fn3, COLORS["brown"])

        def _render_dirt(img, object):
            return rendering.fill_coords(img, plate_fn, COLORS["dark_green"])

        branches = jnp.array(
            [
                object == 0,
                (object & DynamicObject.PLATE > 0) & (object & DynamicObject.COOKED == 0),
                DynamicObject.is_ingredient(object),
                (object & DynamicObject.COOKED > 0) & (object & DynamicObject.USED == 0),
                (object & DynamicObject.PLATE > 0) & (object & DynamicObject.COOKED > 0),
                object & DynamicObject.DIRT,
            ]
        )
        branch_idx = jnp.argmax(branches)

        return jax.lax.switch(
            branch_idx,
            [_no_op, _render_plate, _render_ingredient, _render_dish, _render_used_plate, _render_dirt],
            img,
            object,
        )

    @staticmethod
    def _render_line(encoded, img):
        max_line_length = encoded >> 24
        line_length = (encoded >> 16) & (2**8 - 1)
        max_reserved_length = (encoded >> 8) & (2**8 - 1)
        reserved_length = encoded & (2**8 - 1)

        def _render_reserved_line(i, img):
            pos = ((i + 1) / (max_reserved_length + 1), 0.25)
            r = 1 / (2 * max_reserved_length + 2)
            reserved_fn = rendering.point_in_circle(pos[0], pos[1], r)
            return rendering.fill_coords(img, reserved_fn, COLORS["red"])

        img = jax.lax.fori_loop(0, reserved_length, _render_reserved_line, img)

        def _render_line(i, img):
            pos = ((i + 1) / (max_line_length + 1), 0.75)
            r = 1 / (2 * max_line_length + 2)
            line_fn = rendering.point_in_circle(pos[0], pos[1], r)
            return rendering.fill_coords(img, line_fn, COLORS["yellow"])

        return jax.lax.fori_loop(0, line_length, _render_line, img)

    @staticmethod
    def _render_cell(cell, img):
        static_object = cell[0]

        def _render_empty(cell, img):
            return OvercookedCustomVisualizer._render_dynamic_item(cell[1], img)

        def _render_wall(cell, img):
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["brown"])
            return OvercookedCustomVisualizer._render_dynamic_item(cell[1], img)

        def _render_agent(cell, img):
            tri_fn = rendering.point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))

            direction, idx = OvercookedCustomVisualizer._decode_agent_extras(cell[Channel.extra])

            # A bit hacky, but needed so that actions order matches the one of Overcooked-AI
            # direction_reordering = jnp.array([3, 1, 0, 2])
            # direction = direction_reordering[direction]

            agent_color = AGENT_COLORS[idx]

            tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction)
            img = rendering.fill_coords(img, tri_fn, agent_color)

            return OvercookedCustomVisualizer._render_dynamic_item(
                cell[1],
                img,
                plate_fn=rendering.point_in_circle(0.75, 0.75, 0.2),
                ingredient_fn=rendering.point_in_circle(0.75, 0.75, 0.15),
                dish_positions=jnp.array([(0.65, 0.65), (0.85, 0.65), (0.75, 0.85)]),
            )

        def _render_agent_self(cell, img):
            # Note: This should not ever be called
            return img

        def _render_pot(cell, img):
            return OvercookedCustomVisualizer._render_pot(cell, img)

        def _render_table(cell, img):
            return OvercookedCustomVisualizer._render_table(cell, img)

        def _render_chair(cell, img):
            return OvercookedCustomVisualizer._render_chair(cell, img)

        def _render_counter(cell, img):
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            return OvercookedCustomVisualizer._render_dynamic_item(cell[1], img)

        def _render_sink(cell, img):
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["cyan"])
            plate_count = DynamicObject.get_count(cell[1])

            def _render_plate_in_sink(i, img):
                pos = ((i + 1) / (plate_count + 1), 0.5)
                r = 1 / (plate_count + 2)
                plate_fn = rendering.point_in_circle(pos[0], pos[1], r)
                return rendering.fill_coords(img, plate_fn, COLORS["white"])

            return jax.lax.fori_loop(0, plate_count, _render_plate_in_sink, img)

        def _render_register(cell, img):
            img = rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["yellow"])
            check_fn = rendering.point_in_circle(0.5, 0.5, 0.3)
            img = jax.lax.cond(cell[2] > 0, lambda: rendering.fill_coords(img, check_fn, COLORS["red"]), lambda: img)
            return img

        def _render_entrance(cell, img):
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["light_blue"])
            return OvercookedCustomVisualizer._render_line(cell[2], img)

        def _render_plate_pile(cell, img):
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            plate_fns = [rendering.point_in_circle(*coord, 0.2) for coord in [(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)]]
            for plate_fn in plate_fns:
                img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            return img

        def _render_ingredient_pile(cell, img):
            ingredient_idx = cell[0] - StaticObject.INGREDIENT_PILE_BASE

            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            ingredient_fns = [
                rendering.point_in_circle(*coord, 0.15)
                for coord in [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35), (0.4, 0.8), (0.75, 0.75)]
            ]

            for ingredient_fn in ingredient_fns:
                img = rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[ingredient_idx])

            return img

        def _render_garbage_can(cell, img):
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["white"])
            img = rendering.fill_coords(img, rendering.point_in_rect(0.4, 0.6, 0.1, 0.2), COLORS["blue"])
            img = rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.2, 0.25), COLORS["blue"])
            img = rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.3, 0.95), COLORS["blue"])
            img = rendering.fill_coords(img, rendering.point_in_rect(0.25, 0.35, 0.4, 0.85), COLORS["white"])
            img = rendering.fill_coords(img, rendering.point_in_rect(0.45, 0.55, 0.4, 0.85), COLORS["white"])
            return rendering.fill_coords(img, rendering.point_in_rect(0.65, 0.75, 0.4, 0.85), COLORS["white"])

        render_fns_dict = {
            StaticObject.EMPTY: _render_empty,
            StaticObject.WALL: _render_wall,
            StaticObject.AGENT: _render_agent,
            StaticObject.SELF_AGENT: _render_agent_self,
            StaticObject.POT: _render_pot,
            StaticObject.PLATE_PILE: _render_plate_pile,
            StaticObject.COUNTER: _render_counter,
            StaticObject.SINK: _render_sink,
            StaticObject.REGISTER: _render_register,
            StaticObject.ENTRANCE: _render_entrance,
            StaticObject.TABLE: _render_table,
            StaticObject.CHAIR: _render_chair,
            StaticObject.GARBAGE_CAN: _render_garbage_can,
        }
        # for i in range(MAX_CUSTOMERS):
        #    render_fns_dict[StaticObject.TABLE_BASE + i] = _render_table
        #    render_fns_dict[StaticObject.CHAIR_BASE + i] = _render_chair

        render_fns = [_render_empty] * (max(render_fns_dict.keys()) + 2)
        for key, value in render_fns_dict.items():
            render_fns[key] = value
        render_fns[-1] = _render_ingredient_pile

        branch_idx = jnp.clip(static_object, 0, len(render_fns) - 1)

        return jax.lax.switch(branch_idx, render_fns, cell, img)

    @staticmethod
    def _render_pot(cell, img):
        ingredients = cell[1]
        time_left = cell[2]

        is_cooking = time_left > 0
        is_cooked = (ingredients & DynamicObject.COOKED) != 0
        is_idle = ~is_cooking & ~is_cooked
        ingredients = DynamicObject.get_ingredient_idx_list_jit(ingredients)
        has_ingredients = ingredients[0] != -1

        img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])

        ingredient_fns = [
            rendering.point_in_circle(*coord, 0.13) for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]
        ]

        for i, ingredient_idx in enumerate(ingredients):
            img_ing = rendering.fill_coords(img, ingredient_fns[i], INGREDIENT_COLORS[ingredient_idx])
            img = jax.lax.select(ingredient_idx != -1, img_ing, img)

        pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
        lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
        handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

        lid_fn_open = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
        handle_fn_open = rendering.rotate_fn(handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
        pot_open = is_idle & has_ingredients

        img = rendering.fill_coords(img, pot_fn, COLORS["black"])

        img_closed = rendering.fill_coords(img, lid_fn, COLORS["black"])
        img_closed = rendering.fill_coords(img_closed, handle_fn, COLORS["black"])

        img_open = rendering.fill_coords(img, lid_fn_open, COLORS["black"])
        img_open = rendering.fill_coords(img_open, handle_fn_open, COLORS["black"])

        img = jax.lax.select(pot_open, img_open, img_closed)

        # Render progress bar
        progress_fn = rendering.point_in_rect(0.1, 0.9 - (0.9 - 0.1) / POT_COOK_TIME * time_left, 0.83, 0.88)
        img_timer = rendering.fill_coords(img, progress_fn, COLORS["green"])
        return jax.lax.select(is_cooking, img_timer, img)

    @staticmethod
    def _render_table(cell, img):
        img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
        img = rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["blue"])
        eating = cell[2] & (2**8 - 1)  # 食事中の皿の枚数
        finished_plates = (cell[2] >> 8) & (2**8 - 1)  # 食べ終わった皿の枚数
        capacity = cell[2] >> 16  # テーブルに乗せられる食事の数
        food_color = COLORS["brown"]
        plate_color = COLORS["white"]

        def _render_food_on_table(i, img):
            pos = ((i + 1) / (capacity + 1), 0.25)
            r = 1 / (2 * capacity + 2)
            plate_fn = rendering.point_in_circle(pos[0], pos[1], r)
            food_fn = rendering.point_in_rect(pos[0] - r / 2, pos[0] + r / 2, pos[1] - r / 2, pos[1] + r / 2)
            return rendering.fill_coords(rendering.fill_coords(img, plate_fn, plate_color), food_fn, food_color)

        img = jax.lax.fori_loop(0, eating, _render_food_on_table, img)

        def _render_finished_plate(i, img):
            pos = ((i + 1) / (capacity + 1), 0.75)
            r = 1 / (2 * capacity + 2)
            plate_fn = rendering.point_in_circle(pos[0], pos[1], r)
            return rendering.fill_coords(img, plate_fn, plate_color)

        return jax.lax.fori_loop(0, finished_plates, _render_finished_plate, img)

    @staticmethod
    def _render_chair(cell, img):
        img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
        img = rendering.fill_coords(img, rendering.point_in_circle(0.5, 0.5, 0.4), COLORS["black"])
        used = cell[2] >> 8
        status = cell[2] & (2**8 - 1)
        img = jax.lax.cond(
            used > 0,
            lambda: rendering.fill_coords(img, rendering.point_in_circle(0.5, 0.5, 0.3), COLORS["red"]),
            lambda: img,
        )
        img = jax.lax.cond(
            status == CustomerStatus.ordering,
            lambda: rendering.fill_coords(
                img, rendering.point_in_triangle((0.0, 0.3), (0.3, 0.0), (0.5, 0.5)), COLORS["yellow"]
            ),
            lambda: img,
        )
        return img

    def _render_tile(self, obj, highlight=False):
        """Render a tile and cache the result."""
        # key = (*obj.tolist(), highlight, tile_size)

        # if key in OvercookedCustomVisualizer.tile_cache:
        #     return OvercookedCustomVisualizer.tile_cache[key]

        img = jnp.zeros(shape=(self.tile_size * self.subdivs, self.tile_size * self.subdivs, 3), dtype=jnp.uint8)

        # Draw the grid lines (top and left edges)
        # グリッドの縦線
        img = rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["grey"])
        # グリッドの横線
        img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["grey"])

        img = OvercookedCustomVisualizer._render_cell(obj, img)

        img_highlight = rendering.highlight_img(img)
        img = jax.lax.select(highlight, img_highlight, img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = rendering.downsample(img, self.subdivs)

        # Cache the rendered tile
        # OvercookedCustomVisualizer.tile_cache[key] = img

        return img

    def _render_grid(self, grid, highlight_mask):
        img_grid = jax.vmap(jax.vmap(self._render_tile))(grid, highlight_mask)

        grid_rows, grid_cols, tile_height, tile_width, channels = img_grid.shape

        return img_grid.transpose(0, 2, 1, 3, 4).reshape(grid_rows * tile_height, grid_cols * tile_width, channels)
