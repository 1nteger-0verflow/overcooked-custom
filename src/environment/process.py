import jax
import jax.numpy as jnp

from environment.actions import Actions
from environment.agent import Agent
from environment.customer import Customer, CustomerStatus, RegisterLine
from environment.dynamic_object import DynamicObject
from environment.layouts import Layout
from environment.menus import MenuList
from environment.state import Channel, State
from environment.static_object import StaticObject
from utils.schema import EnvConfig


def _tree_select(predicate, a, b):
    return jax.tree_util.tree_map(lambda x, y: jax.lax.select(predicate, x, y), a, b)


class Processor:
    def __init__(self, config: EnvConfig, layout: Layout, menu: MenuList):
        self.config = config
        self.layout = layout
        self.menu = menu
        self.width = layout.width
        self.height = layout.height
        self.num_agents = layout.num_agents
        self.sink_capacity = config.parameter.sink_capacity
        self.dirt_appear_rate = config.parameter.dirt_appear_rate

    @jax.jit(static_argnums=(0,))
    def step(self, key: jax.Array, state: State, actions: jax.Array) -> tuple[State, float, dict[str, float]]:
        key, interact_key = jax.random.split(key)
        # Move action:
        state = self.update_positions(state, actions)
        # interaction between agent and environment
        (state, reward, shaped_rewards) = self.execute_interaction(state, actions, interact_key)
        # 時間経過による変化(客の状態遷移、汚れの発生、調理・食事の進行)
        state = self.update_step(state, key)

        return (state, reward, shaped_rewards)

    def update_positions(self, state: State, actions: jax.Array) -> State:
        # 1. move agent to new position (if possible on the grid)
        new_agents = self.move_agents(state, actions)
        # 2. resolve collisions
        new_agents = self.resolve_collisions(state, new_agents)
        # 3. prevent swapping
        new_agents = self.prevent_swapping(state, new_agents)
        # 4. update view area
        new_agents = new_agents.update_observed_grid(state.time, self.layout.height, self.layout.width)
        # ここまでの処理で更新後のエージェント位置は確定するので状態に反映させる
        return state.replace(agents=new_agents)

    def move_agents(self, state: State, actions: jax.Array) -> Agent:
        # 各エージェントを環境内で移動させる(エージェント間の衝突は考えない)
        grid = state.grid

        def _move_wrapper(agent: Agent, action: jax.Array):
            direction = Actions.action_to_direction(action)

            def _move(agent: Agent, dir: jnp.ndarray):
                new_pos = agent.move_in_bounds(dir, self.height, self.width)

                new_pos = _tree_select(
                    (grid[*new_pos, Channel.env] == StaticObject.EMPTY)
                    & (grid[*new_pos, Channel.obj] & DynamicObject.DIRT == 0),
                    new_pos,
                    agent.pos,
                )

                return agent.replace(pos=new_pos, dir=dir)

            return jax.lax.cond(jnp.all(direction == jnp.array([0, 0])), lambda a, _: a, _move, agent, direction)

        return jax.vmap(_move_wrapper)(state.agents, actions)

    def resolve_collisions(self, state: State, new_agents: Agent) -> Agent:
        # エージェント同士が衝突しないようにする
        def _masked_positions(agent_mask: jax.Array):
            return jax.vmap(lambda is_collide, cur_pos, new_pos: jax.lax.select(is_collide, cur_pos, new_pos))(
                agent_mask, state.agents.pos, new_agents.pos
            )

        def _get_collisions(agent_mask: jax.Array):
            positions = _masked_positions(agent_mask)

            # 移動後の各マス上のエージェント数>1なら衝突あり
            collision_grid = jnp.zeros((self.height, self.width))
            collision_grid, _ = jax.lax.scan(lambda grid, pos: (grid.at[*pos].add(1), None), collision_grid, positions)

            collision_mask = collision_grid > 1
            return jax.vmap(lambda p: collision_mask[*p])(positions)

        initial_agent_mask = jnp.zeros((self.num_agents,), dtype=bool)
        agent_mask = jax.lax.while_loop(
            # 衝突なしになったら終了
            lambda agent_mask: jnp.any(_get_collisions(agent_mask)),
            # エージェント若い順に移動キャンセルするかを決定
            lambda agent_mask: agent_mask | _get_collisions(agent_mask),
            initial_agent_mask,
        )
        new_agents = new_agents.replace(pos=_masked_positions(agent_mask))
        return new_agents

    def prevent_swapping(self, state: State, new_agents: Agent) -> Agent:
        # エージェントがすり抜けないようにする
        def _masked_positions(agent_mask: jax.Array):
            return jax.vmap(lambda is_collide, cur_pos, new_pos: jax.lax.select(is_collide, cur_pos, new_pos))(
                agent_mask, state.agents.pos, new_agents.pos
            )

        def _compute_swapped_agents(original_positions, new_positions):
            original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
            new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

            # 各エージェントの現在位置と移動後位置が重なるか比較
            swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
            # 自身の移動前後が同じなのは不問(対角線をFalseにする)
            swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

            swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)

            return jnp.any(swap_pairs, axis=0)

        swap_mask = _compute_swapped_agents(state.agents.pos, new_agents.pos)
        new_agents = new_agents.replace(pos=_masked_positions(swap_mask))

        return new_agents

    def execute_interaction(self, state: State, actions: jax.Array, interact_key: jax.Array):
        # Interact action:
        def _interact_wrapper(carry, x):
            agent, action = x
            is_interact, storage_idx = Actions.interpret_interact(agent.modify_action(action))

            def _interact(carry: tuple[State, float], agent: Agent):
                state, reward = carry

                (new_state, new_agent, interact_reward, shaped_reward) = self.process_interact(
                    state, agent, interact_key, storage_idx
                )

                carry = (new_state, reward + interact_reward)
                return carry, (new_agent, shaped_reward)

            return jax.lax.cond(is_interact, _interact, lambda c, a: (c, (a, 0.0)), carry, agent)

        # エージェント1体ずつinteractionの処理
        carry = (state, 0.0)
        xs = (state.agents, actions)
        # returnの1つ目はcarryが全ループを経たもの、2つ目はループごとの出力のstack
        ((new_state, reward), (new_agents, shaped_rewards)) = jax.lax.scan(_interact_wrapper, carry, xs)

        return (new_state.replace(agents=new_agents), reward, shaped_rewards)

    def process_interact(self, state: State, agent: Agent, key: jax.Array, storage_idx: jax.Array):
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""
        # 1体のエージェントの前方のセル1か所に対する処理
        inventory = agent.inventory[storage_idx]
        fwd_pos = agent.get_fwd_pos()

        interact_cell = state.grid[*fwd_pos]
        interact_item = interact_cell[Channel.env]
        interact_object = interact_cell[Channel.obj]
        interact_extra = interact_cell[Channel.extra]

        def _no_op(state, agent):
            return (state, agent, 0.0, self.config.reward.penalty.ineffective_interaction)

        def _deal_customer(state, agent):
            def _invite(customer, line):
                new_customer, new_line, invite_reward = jax.lax.cond(
                    jnp.greater(line.reserved_line_length, 0) | jnp.greater(line.line_length, 0),
                    lambda: (
                        customer.append(state.time, line.reserved_line_length > 0),
                        line.dequeue(),
                        self.config.reward.shaped_reward.invite_customer,
                    ),
                    lambda: (customer, line, 0.0),
                )
                return new_customer, new_line, invite_reward

            def _refuse(customer, line):
                refuse_reward = jax.lax.cond(
                    line.line_length > 0, lambda: self.config.reward.shaped_reward.refuse_customer, lambda: 0.0
                )
                new_line = line.replace(
                    line_length=jnp.clip(line.line_length - 1, min=0),
                    queued_time=jnp.roll(line.queued_time, -1).at[-1].set(0),
                )
                return customer, new_line, refuse_reward

            new_customer, new_line, shaped_reward = jax.lax.cond(
                state.customer.empty_count > 0, _invite, _refuse, state.customer, state.line
            )
            return (state.replace(customer=new_customer, line=new_line), agent, 0.0, shaped_reward)

        def _take_order(state, agent):
            customer = state.customer
            # 対象のテーブルIDを取得する
            tableID = customer.get_tableID(fwd_pos)
            new_status, new_time, new_order, order_reward = jax.lax.cond(
                customer.status[tableID] == CustomerStatus.ordering,
                lambda: (
                    customer.status.at[tableID].set(CustomerStatus.waiting_food),
                    customer.time.at[tableID].set(state.time),
                    customer.order(tableID, self.menu, key),
                    self.config.reward.shaped_reward.take_order,
                ),
                lambda: (customer.status, customer.time, customer.ordered_menu, 0.0),
            )
            new_customer = customer.replace(status=new_status, time=new_time, ordered_menu=new_order)

            return state.replace(customer=new_customer), agent, 0.0, order_reward

        def _clean_table(state, agent):
            customer = state.customer
            tableID = customer.get_tableID(fwd_pos)
            picked_up, new_customer = customer.cleanup(tableID)
            new_inventory = agent.inventory.at[storage_idx].set(picked_up)
            new_agent = agent.replace(inventory=new_inventory)
            return (state.replace(customer=new_customer), new_agent, 0.0, self.config.reward.shaped_reward.clean_table)

        def _add_ingredient(state, agent):
            def _start_cooking(inventory):
                new_obj = DynamicObject.add_ingredient(interact_object, inventory[storage_idx])
                is_correct_recipe, cooking_duration = self.menu.get_duration(new_obj)
                # cooking_durationを指定範囲内の倍率でばらつかせる
                range_min, range_max = self.config.parameter.cooking_duration_range
                duration_coeff = jax.random.uniform(key, (), minval=range_min, maxval=range_max)
                cooking_duration = jnp.floor(cooking_duration * duration_coeff).astype(int)
                new_cell = interact_cell.at[Channel.obj].set(new_obj).at[Channel.extra].set(cooking_duration)
                return new_cell, inventory.at[storage_idx].set(DynamicObject.EMPTY)

            def _add(inventory):
                new_obj = DynamicObject.add_ingredient(interact_object, inventory[storage_idx])
                new_cell = interact_cell.at[Channel.obj].set(new_obj)
                return new_cell, inventory.at[storage_idx].set(DynamicObject.EMPTY)

            pot_is_cooking = interact_extra > 0
            pot_is_cooked = interact_object & DynamicObject.COOKED != 0
            pot_is_full = pot_is_cooking | pot_is_cooked
            pot_is_full_after_drop = DynamicObject.ingredient_count(interact_object) == 2
            pot_is_idle = ~pot_is_cooking * ~pot_is_cooked * ~pot_is_full_after_drop
            new_cell, new_inventory = jax.lax.switch(
                jnp.argmax(jnp.array([pot_is_full, pot_is_full_after_drop, pot_is_idle])),
                [
                    lambda _: (state.grid[*fwd_pos], agent.inventory),  # 調理中、調理済みのとき何もしない
                    _start_cooking,  # 3つ目の食材を追加し、調理を開始する
                    _add,  # 食材を追加する
                ],
                agent.inventory,
            )
            new_grid = state.grid.at[*fwd_pos].set(new_cell)
            new_agent = agent.replace(inventory=new_inventory)
            # successful_pot_placement = pot_is_idle * inventory_is_ingredient * ~pot_full
            # shaped_rewardに加算
            return state.replace(grid=new_grid), new_agent, 0.0, 0.0

        def _put_food_on_plate(state, agent):
            def _do_plating(inventory):
                plated_food = inventory.at[storage_idx].set(interact_object | DynamicObject.PLATE)
                return (DynamicObject.EMPTY, plated_food, self.config.reward.shaped_reward.dish_pickup)

            pot_is_cooked = interact_object & DynamicObject.COOKED != 0
            new_object, new_inventory, dish_reward = jax.lax.cond(
                pot_is_cooked, _do_plating, lambda _: (interact_object, agent.inventory, 0.0), agent.inventory
            )
            new_grid = state.grid.at[*fwd_pos, Channel.obj].set(new_object)
            new_agent = agent.replace(inventory=new_inventory)
            return state.replace(grid=new_grid), new_agent, 0.0, dish_reward

        def _deliver_dish(state, agent):
            customer = state.customer
            tableID = customer.get_tableID(fwd_pos)
            is_correct_dish, correct_order_idx = self.menu.correct(
                agent.inventory[storage_idx], customer.ordered_menu[tableID]
            )
            new_inventory, new_customer = jax.lax.cond(
                is_correct_dish,
                lambda: (
                    agent.inventory.at[storage_idx].set(DynamicObject.EMPTY),
                    customer.put_dish_on_table(tableID, agent.inventory[storage_idx], correct_order_idx),
                ),
                lambda: (agent.inventory, customer),
            )
            new_agent = agent.replace(inventory=new_inventory)
            # 経過時間により報酬を割り引く
            reward = (
                is_correct_dish
                * self.config.reward.delivery_reward
                * jnp.clip((1.0 - (state.time - customer.time[tableID]) / 100.0), min=0.0)
            )
            return state.replace(customer=new_customer), new_agent, reward, 0.0

        def _retrieve_plate(state, agent):
            customer = state.customer
            tableID = customer.get_tableID(fwd_pos)

            def _retrieve(customer):
                idx = jnp.argmax(customer.food[tableID] == DynamicObject.USED | DynamicObject.PLATE)
                new_food = customer.food.at[tableID, idx].set(DynamicObject.EMPTY)
                new_inventory = agent.inventory.at[storage_idx].set(DynamicObject.PLATE | DynamicObject.USED | 1)
                return new_inventory, new_food

            exists_empty_plate = jnp.sum(customer.food[tableID] == DynamicObject.USED | DynamicObject.PLATE) > 0
            new_inventory, new_food = jax.lax.cond(
                exists_empty_plate, _retrieve, lambda _: (agent.inventory, customer.food), customer
            )
            plate_retrieve_reward = jax.lax.cond(
                exists_empty_plate, lambda: self.config.reward.shaped_reward.retrieve_plate, lambda: 0.0
            )
            new_agent = agent.replace(inventory=new_inventory)
            new_customer = customer.replace(food=new_food)
            return (state.replace(customer=new_customer), new_agent, 0.0, plate_retrieve_reward)

        def _pickup(state, agent):
            picked_obj, remainings = DynamicObject.pick(interact_object)
            new_grid = state.grid.at[fwd_pos[0], fwd_pos[1], Channel.obj].set(remainings)
            new_inventory = agent.inventory.at[storage_idx].set(picked_obj)
            new_agent = agent.replace(inventory=new_inventory)
            return state.replace(grid=new_grid), new_agent, 0.0, 0.0

        def _pickup_ingredient(state, agent):
            ingredient = StaticObject.get_ingredient(interact_item)
            new_inventory = agent.inventory.at[storage_idx].set(ingredient)
            new_agent = agent.replace(inventory=new_inventory)
            return state, new_agent, 0.0, 0.0

        def _place(state, agent):
            placed_obj, stackings = DynamicObject.place(interact_object, agent.inventory[storage_idx])
            new_grid = state.grid.at[*fwd_pos, Channel.obj].set(stackings)
            new_inventory = agent.inventory.at[storage_idx].set(placed_obj)
            new_agent = agent.replace(inventory=new_inventory)
            return state.replace(grid=new_grid), new_agent, 0.0, 0.0

        def _soak_plate(state, agent):
            soaked_plate_count = DynamicObject.get_count(interact_object)
            new_obj, stackings = jax.lax.cond(
                soaked_plate_count < self.sink_capacity,
                DynamicObject.place,
                lambda obj, inv: (agent.inventory[storage_idx], interact_object),
                interact_object,
                agent.inventory[storage_idx],
            )
            soak_reward = jax.lax.cond(
                soaked_plate_count < self.sink_capacity,
                lambda: self.config.reward.shaped_reward.soak_plate,
                lambda: 0.0,
            )
            new_grid = state.grid.at[*fwd_pos, Channel.obj].set(stackings)
            new_inventory = agent.inventory.at[storage_idx].set(new_obj)
            new_agent = agent.replace(inventory=new_inventory)
            return state.replace(grid=new_grid), new_agent, 0.0, soak_reward

        def _wash_plate(state, agent):
            soaked_plate_count = DynamicObject.get_count(interact_object)
            plate_pile_pos = self.layout.plate_positions[0]
            plate_pile_obj = state.grid[*plate_pile_pos, Channel.obj]
            clean_plate_count = DynamicObject.get_count(state.grid[*plate_pile_pos, Channel.obj])
            new_grid, wash_reward = jax.lax.cond(
                soaked_plate_count > 0,
                lambda: (
                    state.grid.at[*fwd_pos, Channel.obj]
                    .set(DynamicObject.set_count(interact_object, soaked_plate_count - 1))
                    .at[plate_pile_pos[0], plate_pile_pos[1], Channel.obj]
                    .set(DynamicObject.set_count(plate_pile_obj, clean_plate_count + 1)),
                    self.config.reward.shaped_reward.wash_plate,
                ),
                lambda: (state.grid, 0.0),
            )
            return state.replace(grid=new_grid), agent, 0.0, wash_reward

        def _process_payment(state, agent):
            def _leave(customer):
                # 会計中だった客を退店させる
                idx = jnp.argmax(customer.status == CustomerStatus.checking)
                new_customer = customer.leave(idx)
                return new_customer, self.config.reward.shaped_reward.finish_payment

            new_register = state.register.replace(service_time=jnp.clip(state.register.service_time - 1, min=0))
            new_customer, payment_reward = jax.lax.cond(
                new_register.service_time == 0, _leave, lambda _: (state.customer, 0.0), state.customer
            )
            return (state.replace(customer=new_customer, register=new_register), agent, 0.0, payment_reward)

        def _clean_dirt(state, agent):
            new_grid = state.grid.at[*fwd_pos].set(
                jnp.array([interact_item, DynamicObject.clean_dirt(interact_object), 0])
            )
            return (state.replace(grid=new_grid), agent, 0.0, self.config.reward.shaped_reward.clean_dirt)

        def _dispose_garbage(state, agent):
            inventory = agent.inventory[storage_idx]
            new_inventory = jax.lax.cond(
                inventory & DynamicObject.PLATE,
                lambda: DynamicObject.set_count(DynamicObject.PLATE | DynamicObject.USED, 1),
                lambda: DynamicObject.EMPTY,
            )
            new_inventory = agent.inventory.at[storage_idx].set(new_inventory)
            new_agent = agent.replace(inventory=new_inventory)
            return state, new_agent, 0.0, 0.0

        # Booleans depending on what the agent is in front of
        in_front_of_counter = interact_item == StaticObject.COUNTER
        in_front_of_entrance = interact_item == StaticObject.ENTRANCE
        in_front_of_register = interact_item == StaticObject.REGISTER
        in_front_of_pot = interact_item == StaticObject.POT
        in_front_of_plate_pile = interact_item == StaticObject.PLATE_PILE
        in_front_of_sink = interact_item == StaticObject.SINK
        in_front_of_table = interact_item == StaticObject.TABLE
        in_front_of_ingredient_pile = StaticObject.is_ingredient_pile(interact_item)
        in_front_of_garbage_can = interact_item == StaticObject.GARBAGE_CAN

        # Booleans depending on what the agent interact
        object_is_plate = interact_object & DynamicObject.PLATE > 0
        object_is_dish = interact_object & DynamicObject.COOKED > 0
        object_is_used_plate = interact_object & DynamicObject.USED > 0
        object_is_ingredient = DynamicObject.is_ingredient(interact_object)
        object_is_pickable = object_is_plate | object_is_ingredient  # plateにdish, used_plateも含まれる
        object_is_dirt = interact_object & DynamicObject.DIRT > 0

        # Booleans depending on what agent have
        inventory_is_empty = inventory == DynamicObject.EMPTY
        inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
        inventory_is_cooked = (inventory & DynamicObject.COOKED) > 0
        inventory_is_plated = (inventory & DynamicObject.PLATE) > 0
        inventory_is_dish = inventory_is_cooked * inventory_is_plated
        inventory_is_used_plate = (inventory & DynamicObject.USED) > 0
        inventory_is_new_plate = (inventory & DynamicObject.PLATE > 0) & ~inventory_is_dish & ~inventory_is_used_plate

        # Booleans depending on customer status
        customer_status = jax.lax.cond(
            state.customer.is_table(fwd_pos),
            lambda: state.customer.status[state.customer.get_tableID(fwd_pos)],
            lambda: CustomerStatus.empty,
        )
        is_customer_ordering = customer_status == CustomerStatus.ordering
        is_customer_waiting_food = customer_status == CustomerStatus.waiting_food
        is_customer_eating = customer_status == CustomerStatus.eating_food
        is_customer_waiting_delivery = is_customer_waiting_food | is_customer_eating
        is_customer_waiting_check = customer_status == CustomerStatus.waiting_check
        is_customer_checking = customer_status == CustomerStatus.checking
        is_plate_is_retrievable = is_customer_eating | is_customer_waiting_check | is_customer_checking
        is_table_need_cleaning = customer_status == CustomerStatus.cleaning

        # Booleans depending on payment
        is_register_executing = jnp.any(state.customer.status == CustomerStatus.checking) | jnp.any(
            state.customer.status == CustomerStatus.waiting_check
        )

        # interact対象とエージェントの状態によって分岐
        # TODO: 分岐した時点でinteractionが成功するか決まっているかどうかが処理によって異なる
        #     : 関数が呼ばれたときは必ず状態が変化するように判定する
        branches = jnp.array(
            [
                # ものを持つ
                in_front_of_counter & object_is_pickable & inventory_is_empty,
                in_front_of_plate_pile & inventory_is_empty,
                in_front_of_ingredient_pile & inventory_is_empty,
                # カウンターにものを置く
                in_front_of_counter & ~inventory_is_empty,
                # 並んでいる客への対応
                in_front_of_entrance,
                # 注文を取る
                in_front_of_table & is_customer_ordering,
                # テーブルを片付ける
                in_front_of_table & inventory_is_empty & is_table_need_cleaning,
                # 調理する
                in_front_of_pot & inventory_is_ingredient,
                in_front_of_pot & inventory_is_new_plate,
                # 料理を提供する
                in_front_of_table & inventory_is_dish & is_customer_waiting_delivery,
                # 空いた皿を回収する
                in_front_of_table & inventory_is_empty & is_plate_is_retrievable,
                # 皿を洗う
                in_front_of_sink & inventory_is_used_plate,
                in_front_of_sink & inventory_is_empty,
                # 会計する
                in_front_of_register & is_register_executing,
                # 床の掃除
                object_is_dirt,
                # ゴミを捨てる
                in_front_of_garbage_can,
                # default
                1,
            ]
        )
        branch_idx = jnp.argmax(branches)
        interact_functions = [
            # ものを持つ
            _pickup,
            _pickup,
            _pickup_ingredient,
            # カウンターにものを置く
            _place,
            # 並んでいる客への対応
            _deal_customer,
            # 注文を取る
            _take_order,
            # テーブルを片付ける
            _clean_table,
            # 調理する
            _add_ingredient,
            _put_food_on_plate,
            # 料理を提供する
            _deliver_dish,
            # 空いた皿を回収する
            _retrieve_plate,
            # 皿を洗う
            _soak_plate,
            _wash_plate,
            # 会計する
            _process_payment,
            # 床の掃除
            _clean_dirt,
            # ゴミを捨てる
            _dispose_garbage,
            # default
            _no_op,
        ]

        (new_state, new_agent, reward, shaped_reward) = jax.lax.switch(branch_idx, interact_functions, state, agent)
        # jax.debug.print("interact branch: {}, target_idx: {}", branches, branch_idx)

        return (new_state, new_agent, reward, shaped_reward)

    def update_step(self, state: State, key: jax.Array) -> State:
        grid_key, line_key, customer_key, cook_key, check_key = jax.random.split(key, 5)
        state = self.disturb_env(state, grid_key)
        state = self.arrive_customer(state, line_key)
        state = self.call_order(state, customer_key)
        state = self.progress_cooking(state, cook_key)
        state = self.progress_eating(state)
        return self.get_the_check(state, check_key)

    def disturb_env(self, state: State, key: jax.Array) -> State:
        grid = state.grid
        # 床に汚れがランダムに出現
        # 対象のマスを１つ選択し、次に汚れを発生させるかどうかを判定する
        cell_select_key, dirt_key = jax.random.split(key, 2)
        cell_idx = jax.random.choice(cell_select_key, grid.shape[0] * grid.shape[1])
        cell = jnp.unravel_index(cell_idx, grid.shape[:2])
        target_cell = grid[cell]
        # TODO: state.agents.agent_posとtarget_cellが同じ場合は汚れなし
        # on_agent = jnp.any(jnp.all(cell==agent_pos))  # Positionを１つのjnp.ndarrayにする

        def _appear_dirt(cell):
            # TODO: エージェントの現在いるマスにも出現してしまう
            rn = jax.random.uniform(dirt_key, (), minval=0.0, maxval=1.0)
            dirt_level = jax.random.randint(dirt_key, (), minval=1, maxval=5)
            dirt_appear = rn < self.dirt_appear_rate
            dirt = DynamicObject.create_dirt(dirt_level)
            new_obj = jax.lax.select(dirt_appear, dirt, cell[Channel.obj])

            return cell.at[Channel.obj].set(new_obj)

        is_empty = (target_cell[Channel.env] == StaticObject.EMPTY) * (target_cell[Channel.obj] == DynamicObject.EMPTY)
        new_cell = jax.lax.cond(is_empty, _appear_dirt, lambda x: x, target_cell)
        new_grid = grid.at[cell].set(new_cell)

        return state.replace(grid=new_grid)

    def generate_congestion_rate(self, time: jax.Array):
        # [step数、 出現頻度(%)] を参照し、現在の時刻での出現頻度(0~1)を求める
        temporal_rates = jnp.array(self.config.schedule.congestion_rates)
        current_timezone_idx = jnp.argmax(temporal_rates[:, 0] > time) - 1
        return temporal_rates[current_timezone_idx, 1] / 100.0

    def arrive_customer(self, state: State, key: jax.Array) -> State:
        line = state.line
        current_step = state.time
        # JAXの配列はサイズが変わらない、array.at[n].set(value)として、nがサイズ以上の場合は変更されない
        # これを利用してサイズの判定をなくすことができる
        rn = jax.random.uniform(key, (), minval=0.0, maxval=1.0)
        thres = self.generate_congestion_rate(current_step)
        is_arrived = (rn < thres) * (line.line_length < line.queued_time.shape[0])
        line_appended = line.queued_time.at[line.line_length].set(current_step)
        new_queued_time, new_line_length = jax.lax.cond(
            is_arrived, lambda: (line_appended, line.line_length + 1), lambda: (line.queued_time, line.line_length)
        )
        # 予約時間になったとき、予約客を追加
        new_reserved_queued_time, new_reserved_line_length = jax.lax.cond(
            jnp.any(line.reserve_time == current_step),
            lambda: (
                line.reserved_queued_time.at[line.reserved_line_length].set(current_step),
                line.reserved_line_length + 1,
            ),
            lambda: (line.reserved_queued_time, line.reserved_line_length),
        )
        new_line = line.replace(
            line_length=new_line_length,
            queued_time=new_queued_time,
            reserved_line_length=new_reserved_line_length,
            reserved_queued_time=new_reserved_queued_time,
        )
        return state.replace(line=new_line)

    def call_order(self, state: State, key: jax.Array) -> State:
        def _order(i, val):
            customer, to_order, time = val
            # 着席してからのstep数で10%ずつ注文待ちに遷移する確率を増やす(10step以内に必ず注文する)
            new_status, new_time = jax.lax.cond(
                (customer.status[i] == CustomerStatus.sitting) & (to_order[i] < 0.1 * (time - customer.time[i])),
                lambda: (customer.status.at[i].set(CustomerStatus.ordering), customer.time.at[i].set(time)),
                lambda: (customer.status, customer.time),
            )
            customer = customer.replace(status=new_status, time=new_time)
            return customer, to_order, time

        # 座っている客がorderを要求する
        rn = jax.random.uniform(key, (self.layout.num_customers,), minval=0.0, maxval=1.0)
        # statusがsitting -> call orderに変わる
        new_customer, _, _ = jax.lax.fori_loop(0, state.customer.seat_count, _order, (state.customer, rn, state.time))
        return state.replace(customer=new_customer)

    def progress_cooking(self, state: State, key: jax.Array) -> State:
        # Update extra info:
        def _timestep_wrapper(cell):
            def _cook(cell):
                is_cooking = cell[Channel.extra] > 0
                new_extra = jax.lax.select(is_cooking, cell[Channel.extra] - 1, cell[Channel.extra])
                finished_cooking = is_cooking * (new_extra == 0)
                correct, volume = self.menu.get_volume(cell[Channel.obj])
                # volumeを指定範囲内の倍率でばらつかせる
                range_min, range_max = self.config.parameter.volume_range
                volume_coeff = jax.random.uniform(key, (), minval=range_min, maxval=range_max)
                volume = jnp.floor(volume * volume_coeff).astype(int)
                new_ingredients = jax.lax.cond(
                    finished_cooking,
                    lambda: DynamicObject.set_count(cell[Channel.obj] | DynamicObject.COOKED, volume),
                    lambda: cell[Channel.obj],
                )

                return jnp.array([cell[Channel.env], new_ingredients, new_extra])

            branches = jnp.array([cell[Channel.env] == StaticObject.POT, 1])
            branch_idx = jnp.argmax(branches)

            return jax.lax.switch(
                branch_idx,  # select index
                [_cook, lambda x: x],
                cell,  # operand
            )

        new_grid = jax.vmap(jax.vmap(_timestep_wrapper))(state.grid)
        return state.replace(grid=new_grid)

    def progress_eating(self, state: State) -> State:
        def _eat(status: jax.Array, food: jax.Array, prev_phase_time: jax.Array):
            is_eating = status == CustomerStatus.eating_food
            volumes = DynamicObject.get_count(food)
            decreased_volumes, new_time = jax.lax.switch(
                jnp.argmax(
                    jnp.array(
                        [
                            is_eating * jnp.all(jnp.max(volumes) == 1),
                            is_eating * jnp.any(jnp.max(volumes) > 1),
                            ~is_eating,
                        ]
                    )
                ),
                [
                    lambda: (jnp.zeros_like(food), state.time),
                    lambda: (jnp.clip(volumes - 1, min=0), prev_phase_time),
                    lambda: (volumes, prev_phase_time),
                ],
            )
            new_food = jax.vmap(DynamicObject.set_count)(food, decreased_volumes)
            return new_food, new_time

        customer = state.customer
        # TODO: 全部の料理が来る前に食べ終わると、提供待ちの開始時間が更新され報酬が高くなるが良いか
        new_food, new_time = jax.vmap(_eat)(customer.status, customer.food, customer.time)

        def _finish(obj):
            return jax.lax.cond(
                DynamicObject.is_cooked(obj) & (DynamicObject.get_count(obj) == 0),
                lambda: DynamicObject.PLATE | DynamicObject.USED,
                lambda: obj,
            )

        new_food = jax.vmap(jax.vmap(_finish))(new_food)
        food_remains = jnp.sum(DynamicObject.get_count(new_food), axis=1) > 0
        order_undelivered = jnp.sum(customer.ordered_menu >= 0, axis=1) > 0
        meal_done = ~food_remains & ~order_undelivered & (customer.status == CustomerStatus.eating_food)
        new_status = customer.status * ~meal_done + CustomerStatus.waiting_check * meal_done
        new_customer = customer.replace(status=new_status, food=new_food, time=new_time)
        return state.replace(customer=new_customer)

    def get_the_check(self, state: State, key: jax.Array) -> State:
        def _start_checking(customer: Customer, register: RegisterLine):
            # 会計待ちになってからの時間が最も長い客席のindex
            dequeue_idx = jnp.argmax(
                (customer.status == CustomerStatus.waiting_check) * (state.time + 1 - customer.time)
            )
            new_status = customer.status.at[dequeue_idx].set(CustomerStatus.checking)
            new_time = customer.time.at[dequeue_idx].set(state.time)
            # 会計にかかるステップ数をランダムにする
            duration_max = self.config.parameter.check_time_max
            duration = jax.random.randint(key, (), minval=1, maxval=duration_max)
            new_customer = customer.replace(status=new_status, time=new_time)
            new_register = register.replace(queued_time=state.time, service_time=duration)
            return new_customer, new_register

        customer = state.customer
        register = state.register
        customers_waiting_check = jnp.sum(customer.status == CustomerStatus.waiting_check) > 0
        register_is_free = register.service_time == 0
        new_customer, new_register = jax.lax.cond(
            customers_waiting_check & register_is_free,
            _start_checking,
            lambda c, r: (customer, register),
            customer,
            register,
        )
        return state.replace(customer=new_customer, register=new_register)
