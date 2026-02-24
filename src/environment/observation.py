import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from environment.dynamic_object import Digits, DynamicObject
from environment.layouts import Layout
from environment.state import Channel, State
from environment.static_object import StaticObject

static_encoding = jnp.array(
    [
        StaticObject.WALL,
        StaticObject.COUNTER,
        StaticObject.ENTRANCE,
        StaticObject.REGISTER,
        StaticObject.POT,
        StaticObject.PLATE_PILE,
        StaticObject.SINK,
        StaticObject.TABLE,
        StaticObject.CHAIR,
        StaticObject.GARBAGE_CAN,
    ]
)

static_channels = [
    "WALL",
    "COUNTER",
    "ENTRANCE",
    "REGISTER",
    "POT",
    "PLATE_PILE",
    "SINK",
    "TABLE",
    "CHAIR",
    "GARBAGE_CAN",
]


class Observer:
    def __init__(self, config: DictConfig, layout: Layout):
        self.config = config
        self.layout = layout
        self.width = layout.width
        self.height = layout.height
        # 初期状態ができる前にエージェント数が必要なのでレイアウトから取得
        self.num_agents = layout.num_agents
        self.capacity = self.config.parameter.capacity
        self.obs_shape = self._get_obs_shape()
        self.layer_infos = self._get_obs_layers()

    def _get_obs_shape(self) -> tuple[int, int, int, int]:
        # チャンネル数の合計を求める
        num_ingredients = self.layout.num_ingredients
        # 皿、調理済み, 使用済み,汚れ 4ch + 素材数ch + カウント 1 ch
        num_obj_layers = 4 + num_ingredients + 1
        # 自エージェントに関する情報のレイヤー数
        num_inventory_layers = 3 + num_ingredients
        num_self_agent_layers = 3 + num_inventory_layers * max(self.capacity) + 2
        # 他エージェントは位置と向き(y,x)と持っているもの(観測時刻の記憶なし)
        num_other_agents_layers = 3 + num_inventory_layers * max(self.capacity)
        # WALL～GARBAGE_CANまで各種 + 食材
        num_static_layers = static_encoding.size + num_ingredients
        num_context_layers = 4  # 現在時刻, 開店・閉店時刻、次の予約時刻
        num_line_layers = 2  # 予約客の待ち列長、一般客の待ち列長
        order_max = self.config.parameter.order_max
        # 客席状態(1ch), レシピ(食材3つずつ*order数), 空き皿(order数)
        num_customer_layers = 1 + 4 * order_max
        num_extra_layers = 1
        # エージェントは自身のみのレイヤーと、自分以外のエージェントのレイヤーがある
        num_layers = int(
            num_self_agent_layers
            + num_other_agents_layers
            + num_static_layers
            + num_context_layers
            + num_line_layers
            + num_customer_layers
            + num_obj_layers
            + num_extra_layers
        )

        # 依存パラメータ： エージェントの持てる個数, 素材の数
        obs_shapes = (self.num_agents, self.height, self.width, num_layers)
        # 客席のステータスと客の待ち開始時刻を客席のセルに格納する
        # 客席の情報(客席数×[注文, 料理])
        # obs_shapes["customer"] = 2 * self.layout.num_customers
        return obs_shapes

    def _get_obs_layers(self) -> dict[str, dict[int, str]]:

        def _register_layer(info_dict: dict[int, str], contents: list[str], start: int = 0):
            for ch, content in enumerate(contents, start=start):
                info_dict[ch] = content

        layer_infos = {}
        agent_channels = ["position", "direction_y", "direction_x"]
        observed_channels = ["observed_step", "fresheness"]
        ingredient_channels = [f"ingredient{n}" for n in range(self.layout.num_ingredients)]
        inventory_channels = ["plate", "cooked", "used"] + ingredient_channels

        self_obs_layers = {}
        _register_layer(self_obs_layers, agent_channels)
        for i in range(max(self.capacity)):
            _register_layer(self_obs_layers, [f"({i}) {inv}" for inv in inventory_channels], len(self_obs_layers))
        _register_layer(self_obs_layers, observed_channels, len(self_obs_layers))
        layer_infos["self"] = self_obs_layers

        base_count = sum([len(d) for d in layer_infos.values()])
        other_obs_layers = {}
        _register_layer(other_obs_layers, agent_channels, base_count)
        for i in range(max(self.capacity)):
            _register_layer(
                other_obs_layers, [f"({i}) {inv}" for inv in inventory_channels], base_count + len(other_obs_layers)
            )
        layer_infos["other"] = other_obs_layers

        object_channels = ["plate", "cooked", "used", "dirt"] + ingredient_channels + ["count"]
        order_max = self.config.parameter.order_max
        order_channels = sum([[f"order{n}(0)", f"order{n}(1)", f"order{n}(2)"] for n in range(order_max)], [])
        used_plate_channels = [f"used_plate{n}" for n in range(order_max)]
        channel_lists = {
            "static": static_channels + ingredient_channels,
            "context": ["current_step", "open", "close", "next_reservation"],
            "line": ["reserved_line_length", "customer_line_length"],
            "customer": ["status"] + order_channels + used_plate_channels,
            "object": object_channels,
            "extra": ["pot_timer"],
        }
        for obs_type, channels in channel_lists.items():
            base_count = sum([len(d) for d in layer_infos.values()])
            layers = {}
            _register_layer(layers, channels, base_count)
            layer_infos[obs_type] = layers

        return layer_infos

    def print_layer_info(self):
        for type_, info_dict in self.layer_infos.items():
            print(f"--- {type_} ---")
            for cnt, (ch, content) in enumerate(info_dict.items(), start=1):
                print(f"{ch:>3}: {content:<10},  ", end="")
                if cnt % 8 == 0:
                    print()
            print()

    def observe_static_objects(self, state: State):
        static_objects = state.grid[:, :, Channel.env]
        static_layers = static_objects[..., None] == static_encoding
        # 食材置き場
        num_ingredients = self.layout.num_ingredients
        ingredient_pile_encoding = jnp.array([StaticObject.INGREDIENT_PILE_BASE + i for i in range(num_ingredients)])

        # 食材pileのレイヤー(num_ingredientsチャンネル)
        ingredient_pile_layers = static_objects[..., None] == ingredient_pile_encoding
        return jnp.concatenate([static_layers, ingredient_pile_layers], axis=-1)

    def observe_context(self, state: State):
        # 現在時刻、開店・閉店時刻、次の予約時刻までの時間（予約客が全員来店後は無効値として-1を設定）
        entrance_pos = jnp.array(self.layout.entrance_positions).squeeze()
        schedule = self.config.schedule
        reservations = jnp.array(schedule.reservation)
        reservation_remaining = state.time <= jnp.max(reservations)

        def next_reservation_time():
            time_to_arrival = reservations - state.time
            return jnp.min(time_to_arrival, initial=1000, where=time_to_arrival >= 0).astype(int)

        next_reservation_time = jax.lax.cond(reservation_remaining, next_reservation_time, lambda: -1)
        context = jnp.array([state.time, schedule.opening_time, schedule.closing_time, next_reservation_time])
        context_layers = jnp.zeros((self.height, self.width, context.size))
        context_layers = context_layers.at[*entrance_pos].set(context)
        return context_layers

    def observe_line(self, state: State):
        # 客の待機列の観測(入口のgridに列の長さを設定する)
        entrance_pos = jnp.array(self.layout.entrance_positions).squeeze()
        length_info = jnp.array([state.line.reserved_line_length, state.line.line_length])
        line_layers = jnp.zeros((self.height, self.width, length_info.size))
        line_layers = line_layers.at[*entrance_pos].set(length_info)
        return line_layers

    def observe_customer(self, state: State):
        def _recipe_to_ingredients(recipe):
            # 完成品(cookedフラグONの料理)を入力すると素材に分解
            num_ingredients = self.layout.num_ingredients
            # DynamicObjectのビットに合わせる
            shift = jnp.array([Digits.INGREDIENTS + 2 * i for i in range(num_ingredients)])
            mask = jnp.array([0x3] * num_ingredients)

            layers = recipe[..., None] >> shift
            layers = layers & mask
            return layers

        def _observe_recipe(obj):
            # 注文1品ずつを食材の組み合わせに分解、注文なしのとき無効(-1)
            return jax.lax.cond(obj > 0, DynamicObject.get_ingredient_idx_list_jit, lambda _: jnp.full((3,), -1), obj)

        def _food_finished(food):
            # 料理を提供した後は空いた皿の有無だけ見る
            return jax.lax.cond(food == DynamicObject.USED | DynamicObject.PLATE, lambda: 1, lambda: 0)

        customers = state.customer
        table_pos = jnp.array(customers.table_pos)
        status = customers.status[:, jnp.newaxis]
        orders = customers.ordered_menu
        foods = customers.food
        ordered_recipes = jax.vmap(jax.vmap(_observe_recipe))(orders)
        finished_plates = jax.vmap(jax.vmap(_food_finished))(foods)
        flatten_recipes = ordered_recipes.reshape((ordered_recipes.shape[0], -1))
        customer_features = jnp.concat([status, flatten_recipes, finished_plates], axis=1)
        customer_layers = jnp.zeros((self.height, self.width, customer_features.shape[1]))

        def _set_feature(carry, x):
            pos, val = x
            return carry.at[*pos].set(val), None

        customer_layers, _ = jax.lax.scan(_set_feature, customer_layers, (table_pos, customer_features))
        return customer_layers

    def observe_pot(self, state: State):
        static_objects = state.grid[:, :, Channel.env]
        extra_info = state.grid[:, :, Channel.extra]
        pot_timer_layer = jnp.where(static_objects == StaticObject.POT, extra_info, 0)

        extra_layers = [pot_timer_layer]

        return jnp.stack(extra_layers, axis=-1)

    def observe_obj(self, ingredients):
        num_ingredients = self.layout.num_ingredients
        # DynamicObjectのビットに合わせる
        shift = jnp.array(
            [Digits.PLATE, Digits.COOKED, Digits.USED, Digits.DIRT]
            + [Digits.INGREDIENTS + 2 * i for i in range(num_ingredients)]
        )
        mask = jnp.array([0x1, 0x1, 0x1, 0x1] + [0x3] * num_ingredients)
        counts = DynamicObject.get_count(ingredients)

        layers = ingredients[..., None] >> shift
        layers = layers & mask

        return jnp.concatenate([layers, jnp.expand_dims(counts, -1)], axis=-1)

    def observe_dynamic_objects(self, state: State):
        ingredients = state.grid[:, :, Channel.obj]
        return self.observe_obj(ingredients)

    def get_obs(self, state: State) -> tuple[jax.Array, jax.Array]:
        width = self.layout.width
        height = self.layout.height

        ingredients = state.grid[:, :, Channel.obj]

        static_layers = self.observe_static_objects(state)
        context_layers = self.observe_context(state)
        line_layers = self.observe_line(state)
        customer_layers = self.observe_customer(state)
        # 置かれているもののレイヤー(4+num_ingredient+1 チャンネル)
        ingredients_layers = self.observe_dynamic_objects(state)
        extra_layers = self.observe_pot(state)

        def _agent_layers(agent, storage):
            pos = agent.pos
            direction = agent.dir
            inv = agent.inventory

            pos_layers = jnp.zeros((height, width, 1), dtype=jnp.int32).at[*pos, 0].set(1)
            dir_layers = jnp.zeros((height, width, 2), dtype=jnp.int32).at[*pos].set(direction)
            # 最後に見たステップ数
            observe_step_layer = jnp.expand_dims(agent.grid_observed_step, -1)
            # 最後に見てからのステップ数
            freshness_layer = jnp.full_like(observe_step_layer, state.time) - observe_step_layer
            # エージェントの持っているものを表すため必要なch数(皿、調理済み, 使用済み 3ch + 素材数ch)
            obj_layers = 3 + self.layout.num_ingredients

            def _observe_inventory(ingredients):
                num_ingredients = self.layout.num_ingredients
                shift = jnp.array(
                    [Digits.PLATE, Digits.COOKED, Digits.USED]
                    + [Digits.INGREDIENTS + 2 * i for i in range(num_ingredients)]
                )
                mask = jnp.array([0x1, 0x1, 0x1] + [0x3] * num_ingredients)

                layers = ingredients[..., None] >> shift
                layers = layers & mask

                return layers

            def _obs_inventory(i, val):
                inv_grid = jnp.zeros_like(ingredients).at[*pos].set(inv[i])
                return val.at[:, :, i].set(_observe_inventory(inv_grid))

            init_inv_grid = jnp.zeros((*ingredients.shape, obj_layers * max(self.capacity))).reshape(
                *ingredients.shape, -1, obj_layers
            )
            inv_layers = jax.lax.fori_loop(0, storage, _obs_inventory, init_inv_grid)
            inv_layers = inv_layers.reshape(*ingredients.shape, -1)

            return jnp.concatenate(
                [
                    pos_layers,  # 1
                    dir_layers,  # 2
                    inv_layers,  # (3 + num_ingredients) * max(capacity)
                    observe_step_layer,  # 1
                    freshness_layer,  # 1
                ],
                axis=-1,
            )

        def _agent_obs(agent_id: jnp.ndarray):
            agent_layers = jax.vmap(_agent_layers)(state.agents, jnp.array(self.capacity))
            agent_layer = agent_layers[agent_id]
            all_agent_layers = jnp.sum(agent_layers, axis=0)

            other_agent_layers = all_agent_layers - agent_layer
            other_agent_layers = other_agent_layers[:, :, :-2]  # いつ見たかの情報は不要

            # エージェントごとに観測結果を見るため、共通の情報もそれぞれに含める
            raw_obs = jnp.concatenate(
                [
                    agent_layer,  # 3 + (3+num_ingredients)*max(capacity) + 2
                    other_agent_layers,  # 3 + (3+num_ingredients)*max(capacity)
                    static_layers,  # 10 + num_ingredients
                    context_layers,  # 4
                    line_layers,  # 2
                    customer_layers,  # 1 + 4*order_max
                    ingredients_layers,  # 4 + num_ingredients + 1
                    extra_layers,  # 1
                ],
                axis=-1,
            )
            view_area = state.agents.grid_observed_step[agent_id] == state.time - 1
            mask = jnp.stack([view_area] * raw_obs.shape[-1], axis=-1)
            # 視野範囲内に観測情報を制限
            return jax.lax.cond(
                self.config.parameter.restrict_observation, lambda: jnp.where(mask, raw_obs, -1), lambda: raw_obs
            )

        def _obs_all():
            agent_layers = jax.vmap(_agent_layers)(state.agents, jnp.array(self.capacity))
            all_agent_layers = jnp.sum(agent_layers, axis=0)
            all_agent_layers = all_agent_layers[:, :, :-2]
            return jnp.concatenate(
                [
                    all_agent_layers,  # 3 + (3+num_ingredients)*max(capacity)
                    static_layers,  # 10 + num_ingredients
                    context_layers,  # 4
                    line_layers,  # 2
                    customer_layers,  # 1 + 4*order_max
                    ingredients_layers,  # 4 + num_ingredients + 1
                    extra_layers,  # 1
                ],
                axis=-1,
            )

        return jax.vmap(_agent_obs)(jnp.arange(state.agents.num_agents)), _obs_all()
