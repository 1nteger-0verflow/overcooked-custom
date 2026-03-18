import math

import jax
import jax.numpy as jnp

import visualize.grid_rendering_v2 as rendering
from environment.actions import ActionType
from environment.customer import CustomerStatus
from environment.dynamic_object import DynamicObject
from environment.state import Channel
from environment.static_object import StaticObject

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
    "light_gray": jnp.array([210, 210, 210], dtype=jnp.uint8),
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

TILE_PIXELS = 96
SUB_DIVS = 3

POT_COOK_TIME = 10  # TODO: メニューにより調理時間が異なる


def decode_agent_extras(extras):
    direction = extras & 0x3
    idx = extras >> 4
    return direction, idx


def render_empty(cell, img):
    img = render_dynamic_item(cell, img)
    return img


def render_wall(cell, img):
    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["brown"])
    img = render_dynamic_item(cell, img)

    return img


def render_agent(cell, img):
    tri_fn = rendering.point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))

    direction, idx = decode_agent_extras(cell[Channel.extra])

    # A bit hacky, but needed so that actions order matches the one of Overcooked-AI
    # direction_reordering = jnp.array([3, 1, 0, 2])
    # direction = direction_reordering[direction]

    tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction)
    img = rendering.fill_coords(img, tri_fn, AGENT_COLORS[idx])

    #    img = _render_dynamic_item(
    #        cell,
    #        img,
    #        plate_fn=rendering.point_in_circle(0.75, 0.75, 0.2),
    #        ingredient_fn=rendering.point_in_circle(0.75, 0.75, 0.15),
    #        dish_positions=jnp.array([(0.65, 0.65), (0.85, 0.65), (0.75, 0.85)]),
    #    )

    return img


def render_agent_self(cell, img):
    # Note: This should not ever be called
    return img


def render_counter(cell, img):
    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
    img = render_dynamic_item(cell, img)
    return img


def render_sink(cell, img):
    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["cyan"])
    plate_count = DynamicObject.get_count(cell[1])

    def _render_plate_in_sink(i, img):
        pos = ((i + 1) / (plate_count + 1), 0.5)
        r = 1 / (plate_count + 2)
        plate_fn = rendering.point_in_circle(pos[0], pos[1], r)
        return rendering.fill_coords(img, plate_fn, COLORS["white"])

    return jax.lax.fori_loop(0, plate_count, _render_plate_in_sink, img)


def render_register(cell, img):
    img = rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["yellow"])
    check_fn = rendering.point_in_circle(0.5, 0.5, 0.3)
    img = jax.lax.cond(cell[2] > 0, lambda: rendering.fill_coords(img, check_fn, COLORS["red"]), lambda: img)
    return img


def render_entrance(cell, img):
    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["light_blue"])
    img = render_line(cell[2], img)
    return img


def render_plate_pile(cell, img):
    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
    plate_fns = [rendering.point_in_circle(*coord, 0.2) for coord in [(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)]]
    for plate_fn in plate_fns:
        img = rendering.fill_coords(img, plate_fn, COLORS["white"])
    return img


def render_ingredient_pile(cell, img):
    ingredient_idx = cell[0] - StaticObject.INGREDIENT_PILE_BASE

    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
    ingredient_fns = [
        rendering.point_in_circle(*coord, 0.15)
        for coord in [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35), (0.4, 0.8), (0.75, 0.75)]
    ]

    for ingredient_fn in ingredient_fns:
        img = rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[ingredient_idx])

    return img


def render_garbage_can(cell, img):
    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["white"])
    img = rendering.fill_coords(img, rendering.point_in_rect(0.4, 0.6, 0.1, 0.2), COLORS["blue"])
    img = rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.2, 0.25), COLORS["blue"])
    img = rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.3, 0.95), COLORS["blue"])
    img = rendering.fill_coords(img, rendering.point_in_rect(0.25, 0.35, 0.4, 0.85), COLORS["white"])
    img = rendering.fill_coords(img, rendering.point_in_rect(0.45, 0.55, 0.4, 0.85), COLORS["white"])
    img = rendering.fill_coords(img, rendering.point_in_rect(0.65, 0.75, 0.4, 0.85), COLORS["white"])
    return img


def render_line(encoded, img):
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

    img = jax.lax.fori_loop(0, line_length, _render_line, img)
    return img


def render_pot(cell, img):
    ingredients = cell[1]
    time_left = cell[2]

    is_cooking = time_left > 0
    is_cooked = (ingredients & DynamicObject.COOKED) != 0
    is_idle = ~is_cooking & ~is_cooked
    ingredients = DynamicObject.get_ingredient_idx_list_jit(ingredients)
    has_ingredients = ingredients[0] != -1

    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])

    ingredient_fns = [rendering.point_in_circle(*coord, 0.13) for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]]

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
    img = jax.lax.select(is_cooking, img_timer, img)

    return img


def render_table(cell, img):
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

    img = jax.lax.fori_loop(0, finished_plates, _render_finished_plate, img)
    return img


def render_chair(cell, img):
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


def render_dynamic_item(cell, img):
    plate_fn = rendering.point_in_circle(0.5, 0.5, 0.3)
    ingredient_fn = rendering.point_in_circle(0.5, 0.5, 0.15)
    dish_positions = jnp.array([(0.5, 0.4), (0.4, 0.6), (0.6, 0.6)])

    def _no_op(cell, img):
        return img

    def _render_plate(cell, img):
        return rendering.fill_coords(img, plate_fn, COLORS["white"])

    def _render_ingredient(cell, img):
        idx = DynamicObject.get_ingredient_idx(cell[Channel.obj])
        return rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])

    def _render_dish(cell, img):
        img = rendering.fill_coords(img, plate_fn, COLORS["white"])
        ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(cell[Channel.obj])

        for idx, ingredient_idx in enumerate(ingredient_indices):
            color = INGREDIENT_COLORS[ingredient_idx]
            pos = dish_positions[idx]
            ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
            img_ing = rendering.fill_coords(img, ingredient_fn, color)

            img = jax.lax.select(ingredient_idx != -1, img_ing, img)

        return img

    def _render_used_plate(cell, img):
        img = rendering.fill_coords(img, plate_fn, COLORS["white"])
        drop_fn1 = rendering.point_in_circle(0.6, 0.6, 0.15)
        drop_fn2 = rendering.point_in_circle(0.5, 0.3, 0.05)
        drop_fn3 = rendering.point_in_circle(0.3, 0.5, 0.05)
        img = rendering.fill_coords(img, drop_fn1, COLORS["brown"])
        img = rendering.fill_coords(img, drop_fn2, COLORS["brown"])
        img = rendering.fill_coords(img, drop_fn3, COLORS["brown"])
        return img

    def _render_dirt(cell, img):
        n = DynamicObject.get_count(cell[Channel.obj])
        cx, cy = jax.lax.cond(
            n > 1, lambda: (0.5 + 0.3 * ((n + 1) % 2) / n, 0.2), lambda: (0.5 + 0.3 * ((n + 1) % 2) / n, 0.5)
        )
        rotate = 2 * math.pi / n
        dirt_fn = rendering.point_in_circle(cx, cy, 0.2)

        def _render_dirty_multi(i, img):
            single_dirt_fn = rendering.rotate_fn(dirt_fn, cx=0.5, cy=0.5, theta=rotate * (i + 1))
            return rendering.fill_coords(img, single_dirt_fn, COLORS["dark_green"])

        return jax.lax.fori_loop(0, n, _render_dirty_multi, img)

    def _render_agent_aux(cell, img):
        def _render_emphasis(img):
            # TODO: jax.lax.scan
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 0.05, 0, 1), COLORS["red"])
            img = rendering.fill_coords(img, rendering.point_in_rect(0.95, 1, 0, 1), COLORS["red"])
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.05), COLORS["red"])
            img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0.95, 1), COLORS["red"])
            return img

        def _render_move_action(img, direction, idx):
            tri_fn = rendering.point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
            tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction)
            img = rendering.fill_coords(img, tri_fn, AGENT_COLORS[idx])
            return img

        def _render_nop(img):
            return rendering.fill_coords(img, rendering.point_in_rect(0.2, 0.8, 0.45, 0.55), COLORS["black"])

        def _render_interaction(img):
            return rendering.fill_coords(img, rendering.point_in_circle(0.5, 0.5, 0.3), COLORS["red"])

        def _render_inventory(cell, img):
            object = cell[Channel.obj]
            branches = jnp.array(
                [
                    (object & DynamicObject.PLATE > 0)
                    & (object & DynamicObject.COOKED == 0)
                    & (object & DynamicObject.USED == 0),
                    DynamicObject.is_ingredient(object),
                    (object & DynamicObject.COOKED > 0) & (object & DynamicObject.USED == 0),
                    (object & DynamicObject.PLATE > 0) & (object & DynamicObject.USED > 0),
                    object == 0,
                ]
            )
            branch_idx = jnp.argmax(branches)
            img = jax.lax.switch(
                branch_idx, [_render_plate, _render_ingredient, _render_dish, _render_used_plate, _no_op], cell, img
            )
            return img

        img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["light_gray"])
        # グリッドの縦線
        img = rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["black"])
        # グリッドの横線
        img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["black"])

        obj = cell[Channel.obj]
        extra = cell[Channel.extra]
        direction, idx = decode_agent_extras(obj)
        action_type = (extra >> 8) & (2**3 - 1)
        # TODO: jax.lax.switch
        is_move = (action_type == ActionType.MOVE) & (extra >> 16)
        is_nop = (action_type == ActionType.NOP) & (extra >> 16)
        is_interaction = (action_type == ActionType.INTERACTION) & (extra >> 16)
        img = jax.lax.cond(is_move, _render_move_action, lambda img, dir, idx: img, img, direction, idx)
        img = jax.lax.cond(is_nop, _render_nop, lambda img: img, img)
        img = jax.lax.cond(is_interaction, _render_interaction, lambda img: img, img)
        # 持っているものを常に表示
        is_pick_place = action_type == ActionType.PICK_PLACE
        img = jax.lax.cond(is_pick_place, _render_inventory, lambda cell, img: img, cell, img)
        # 選択された行動を赤枠で囲む
        is_selected_action = cell[Channel.extra] & (1 << 16)
        img = jax.lax.cond(is_selected_action, _render_emphasis, lambda img: img, img)
        return img

    def _render_customer_aux(cell, img):
        def _render_order(cell, img):
            object = cell[Channel.obj]
            is_valid_order = (object & DynamicObject.COOKED > 0) & (object & DynamicObject.USED == 0)
            img = jax.lax.cond(is_valid_order, _render_dish, _no_op, cell, img)
            return img

        img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["light_gray"])
        # グリッドの縦線
        img = rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["black"])
        # グリッドの横線
        img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["black"])
        valid_order = cell[Channel.obj] > 0
        img = jax.lax.cond(valid_order, _render_order, lambda cell, img: img, cell, img)
        return img

    object = cell[1]
    aux = cell[2]
    branches = jnp.array(
        [
            aux & (1 << 2) > 0,
            aux & (1 << 3) > 0,
            (object & DynamicObject.PLATE > 0)
            & (object & DynamicObject.COOKED == 0)
            & (object & DynamicObject.USED == 0),
            DynamicObject.is_ingredient(object),
            (object & DynamicObject.COOKED > 0) & (object & DynamicObject.USED == 0),
            (object & DynamicObject.PLATE > 0) & (object & DynamicObject.USED > 0),
            object & DynamicObject.DIRT,
            object == 0,
        ]
    )
    branch_idx = jnp.argmax(branches)

    img = jax.lax.switch(
        branch_idx,
        [
            _render_agent_aux,
            _render_customer_aux,
            _render_plate,
            _render_ingredient,
            _render_dish,
            _render_used_plate,
            _render_dirt,
            _no_op,
        ],
        cell,
        img,
    )

    return img


def render_cell(cell, img):

    render_fns_dict = {
        StaticObject.EMPTY: render_empty,
        StaticObject.WALL: render_wall,
        StaticObject.AGENT: render_agent,
        StaticObject.SELF_AGENT: render_agent_self,
        StaticObject.POT: render_pot,
        StaticObject.PLATE_PILE: render_plate_pile,
        StaticObject.COUNTER: render_counter,
        StaticObject.SINK: render_sink,
        StaticObject.REGISTER: render_register,
        StaticObject.ENTRANCE: render_entrance,
        StaticObject.TABLE: render_table,
        StaticObject.CHAIR: render_chair,
        StaticObject.GARBAGE_CAN: render_garbage_can,
    }

    render_fns = [render_empty] * (max(render_fns_dict.keys()) + 2)
    for key, value in render_fns_dict.items():
        render_fns[key] = value
    render_fns[-1] = render_ingredient_pile

    static_object = cell[0]
    branch_idx = jnp.clip(static_object, 0, len(render_fns) - 1)

    return jax.lax.switch(branch_idx, render_fns, cell, img)


def render_tile(obj, highlight=False):

    img = jnp.zeros(shape=(TILE_PIXELS, TILE_PIXELS, 3), dtype=jnp.uint8)

    # Draw the grid lines (top and left edges)
    # グリッドの縦線
    img = rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["grey"])
    # グリッドの横線
    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["grey"])

    img = render_cell(obj, img)

    img_highlight = rendering.highlight_img(img)
    img = jax.lax.select(highlight, img_highlight, img)

    # Downsample the image to perform supersampling/anti-aliasing
    img = rendering.downsample(img, SUB_DIVS)

    return img


def render_grid(grid, highlight_mask):
    img_grid = jax.vmap(jax.vmap(render_tile))(grid, highlight_mask)

    grid_rows, grid_cols, tile_height, tile_width, channels = img_grid.shape

    big_image = img_grid.transpose(0, 2, 1, 3, 4).reshape(grid_rows * tile_height, grid_cols * tile_width, channels)

    return big_image
