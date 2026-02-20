import jax
import jax.numpy as jnp

from environment.dynamic_object import DynamicObject
from environment.layouts import Layout
from environment.state import Channel, State
from environment.static_object import StaticObject
from utils.schema import EnvConfig


class Observer:
    def __init__(self, config: EnvConfig, layout: Layout):
        self.config = config
        self.layout = layout
        self.width = layout.width
        self.height = layout.height
        self.num_agents = layout.num_agents
        self.capacity = self.config.parameter.capacity
        self.obs_shape = self._get_obs_shape()

    def _get_obs_shape(self):  # TODO: remove (called once)
        # チャンネル数の設定
        num_ingredients = self.layout.num_ingredients
        obj_layers = 4 + num_ingredients + 1  # 皿、調理済み, 使用済み,汚れ 4ch + 素材数ch + カウント 1 ch
        agent_layers = 5 + obj_layers * max(self.capacity)
        static_layers = 10 + num_ingredients  # WALL～GARBAGE_CANまで10種類 + 食材
        extra_layers = 1
        # エージェントは自身のみのレイヤーと、自分以外のエージェントのレイヤーがある
        num_layers = agent_layers * 2 + static_layers + obj_layers + extra_layers

        # 依存パラメータ： エージェントの持てる個数, 素材の数
        # 客席のステータスと客の待ち開始時刻を客席のセルに格納する
        return {
            "agents": (self.num_agents, self.height, self.width, num_layers),
            "all_agents": (
                self.height,
                self.width,
                agent_layers + static_layers + obj_layers + extra_layers,
            ),  # コンテキスト情報
            "schedule": 3 + len(self.config.schedule.reservation),  # 時刻に関する情報  TODO: ピーク時間
            "customer": 2 * self.layout.num_customers,  # 客席の情報(客席数×[注文, 料理])
            "line": len(self.config.schedule.reservation) + self.config.parameter.wait_line_max,  # 待ち行列に関する情報
        }

    def observe_static_objects(self, state: State):  # TODO: remove (called once)
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
        static_objects = state.grid[:, :, Channel.env]
        static_layers = static_objects[..., None] == static_encoding
        # 食材置き場
        num_ingredients = self.layout.num_ingredients
        ingredient_pile_encoding = jnp.array([StaticObject.INGREDIENT_PILE_BASE + i for i in range(num_ingredients)])

        # 食材pileのレイヤー(num_ingredientsチャンネル)
        ingredient_pile_layers = static_objects[..., None] == ingredient_pile_encoding
        return jnp.concatenate([static_layers, ingredient_pile_layers], axis=-1)

    def observe_pot(self, state: State):  # TODO: remove (called once)
        static_objects = state.grid[:, :, Channel.env]
        extra_info = state.grid[:, :, Channel.extra]
        pot_timer_layer = jnp.where(static_objects == StaticObject.POT, extra_info, 0)

        extra_layers = [pot_timer_layer]

        return jnp.stack(extra_layers, axis=-1)

    def observe_obj(self, ingredients):
        num_ingredients = self.layout.num_ingredients
        # DynamicObjectのビットに合わせる
        shift = jnp.array([6, 7, 8, 9] + [10 + 2 * i for i in range(num_ingredients)])
        mask = jnp.array([0x1, 0x1, 0x1, 0x1] + [0x3] * num_ingredients)
        counts = DynamicObject.get_count(ingredients)

        layers = ingredients[..., None] >> shift
        layers = layers & mask

        return jnp.concatenate([layers, jnp.expand_dims(counts, -1)], axis=-1)

    def observe_dynamic_objects(self, state: State):
        ingredients = state.grid[:, :, Channel.obj]
        return self.observe_obj(ingredients)

    def get_obs(self, state: State) -> dict[str, jax.Array]:
        obs = {}
        agent_obs, all_obs = self.get_agentwise_observations(state)
        context_obs = self.get_context_observations(state)
        obs.update(agent_obs)
        obs.update(context_obs)
        obs.update(all_obs)
        return obs

    def get_agentwise_observations(self, state: State) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        agent_observations, all_observation = self.observe_agents(state)

        return {"agents": agent_observations}, {"all_agents": all_observation}

    def observe_agents(self, state: State) -> tuple[jax.Array, jax.Array]:
        width = self.layout.width
        height = self.layout.height

        ingredients = state.grid[:, :, Channel.obj]

        static_layers = self.observe_static_objects(state)
        extra_layers = self.observe_pot(state)
        # 置かれているもののレイヤー(4+num_ingredient+1 チャンネル)
        ingredients_layers = self.observe_dynamic_objects(state)

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
            # エージェントの持っているもの
            obj_layers = (
                4 + self.layout.num_ingredients + 1
            )  # 皿、調理済み, 使用済み,汚れ 4ch + 素材数ch + カウント 1 ch

            def _obs_inventory(i, val):
                inv_grid = jnp.zeros_like(ingredients).at[*pos].set(inv[i])
                return val.at[:, :, i].set(self.observe_obj(inv_grid))

            init_inv_grid = jnp.zeros((*ingredients.shape, obj_layers * max(self.capacity))).reshape(
                *ingredients.shape, -1, obj_layers
            )
            inv_layers = jax.lax.fori_loop(0, storage, _obs_inventory, init_inv_grid)
            inv_layers = inv_layers.reshape(*ingredients.shape, -1)

            return jnp.concatenate(
                [
                    pos_layers,  # 1
                    dir_layers,  # 2
                    observe_step_layer,  # 1
                    freshness_layer,  # 1
                    inv_layers,  # (4 + num_ingredients) * max(capacity)
                ],
                axis=-1,
            )

        def _agent_obs(agent_id: jnp.ndarray):
            agent_layers = jax.vmap(_agent_layers)(state.agents, jnp.array(self.capacity))
            agent_layer = agent_layers[agent_id]
            all_agent_layers = jnp.sum(agent_layers, axis=0)

            other_agent_layers = all_agent_layers - agent_layer

            # エージェントごとに観測結果を見るため、共通の情報もそれぞれに含める
            return jnp.concatenate(
                [
                    agent_layer,  # 5 + (4+num_ingredients+1)*max(capacity)
                    other_agent_layers,  # 5 + (4+num_ingredients+1)*max(capacity)
                    static_layers,  # 10 + num_ingredients
                    ingredients_layers,  # 4 + num_ingredients + 1
                    extra_layers,  # 1
                ],
                axis=-1,
            )

        def _obs_all():
            agent_layers = jax.vmap(_agent_layers)(state.agents, jnp.array(self.capacity))
            all_agent_layers = jnp.sum(agent_layers, axis=0)
            return jnp.concatenate(
                [
                    all_agent_layers,  # 5 + (4+num_ingredients+1)*max(capacity)
                    static_layers,  # 10 + num_ingredients
                    ingredients_layers,  # 4 + num_ingredients + 1
                    extra_layers,  # 1
                ],
                axis=-1,
            )

        return jax.vmap(_agent_obs)(jnp.arange(self.num_agents)), _obs_all()

    def get_context_observations(self, state: State) -> dict[str, jax.Array]:
        schedule = self.config.schedule
        # 時刻に関する公開情報
        obs_schedule = jnp.concat(
            [jnp.array([state.time]), jnp.array([schedule.opening_time, schedule.closing_time] + schedule.reservation)]
        )
        # 客席に関する公開情報
        obs_customer = jnp.vstack([state.customer.ordered_menu, state.customer.food])
        # 待ち行列に関する公開情報
        obs_line = state.line.get_obs()
        return {"schedule": obs_schedule, "customer": obs_customer, "line": obs_line}
