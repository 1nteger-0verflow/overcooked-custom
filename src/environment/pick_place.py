from functools import partial

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Int, Key

from config import EnvParameterConfig, RewardConfig
from environment.agent import Agent
from environment.customer import Customer, CustomerStatus
from environment.dynamic_object import DynamicObject
from environment.reward import RewardType
from environment.state import Channel, State
from environment.static_object import StaticObject


def pick_and_place(
    state: State,
    agent: Agent,
    key: Key[Array, ""],
    storage_idx: Int[Array, ""],
    reward: RewardConfig,
    parameter: EnvParameterConfig,
):
    """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""
    # 1体のエージェントの前方のセル1か所に対する処理
    inventory = agent.inventory[storage_idx]
    inventory_is_empty = inventory == DynamicObject.EMPTY
    fwd_pos = agent.get_fwd_pos()

    interact_cell = state.grid[*fwd_pos]
    interact_item = interact_cell[Channel.env]
    interact_object = interact_cell[Channel.obj]
    interact_extra = interact_cell[Channel.extra]

    # Booleans depending on what agent have
    inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
    inventory_is_cooked = (inventory & DynamicObject.COOKED) > 0
    inventory_is_plated = (inventory & DynamicObject.PLATE) > 0
    inventory_is_dish = inventory_is_cooked * inventory_is_plated
    inventory_is_used_plate = (inventory & DynamicObject.USED) > 0
    inventory_is_new_plate = (inventory & DynamicObject.PLATE > 0) & ~inventory_is_dish & ~inventory_is_used_plate

    # Booleans depending on customer status
    customer_status = jax.lax.cond(
        state.customer.is_table(fwd_pos),
        lambda: state.customer.status[state.customer.get_table_id(fwd_pos)],
        lambda: CustomerStatus.empty,
    )
    is_customer_waiting_food = customer_status == CustomerStatus.waiting_food
    is_customer_eating = customer_status == CustomerStatus.eating_food
    is_customer_waiting_delivery = is_customer_waiting_food | is_customer_eating
    is_customer_waiting_check = customer_status == CustomerStatus.waiting_check
    is_customer_checking = customer_status == CustomerStatus.checking
    is_plate_is_retrievable = is_customer_eating | is_customer_waiting_check | is_customer_checking
    is_table_need_cleaning = customer_status == CustomerStatus.cleaning

    shaped = reward.shaped_reward
    penalty = reward.penalty
    sink_capacity = parameter.sink_capacity

    def _no_op(state: State, agent: Agent):
        return (state, agent, 0.0, -penalty.ineffective_interaction, RewardType.FAIL_PICK_PLACE)

    def _add_ingredient(state: State, agent: Agent):
        def _start_cooking(inventory: Int[Array, "max_storage"]):
            new_obj = DynamicObject.add_ingredient(interact_object, inventory[storage_idx])
            _is_correct_recipe, cooking_duration = state.menu.get_duration(new_obj)
            # cooking_durationを指定範囲内の倍率でばらつかせる
            range_min, range_max = parameter.cooking_duration_range
            duration_coeff = jax.random.uniform(key, (), minval=range_min, maxval=range_max)
            cooking_duration = jnp.clip(jnp.floor(cooking_duration * duration_coeff), min=1).astype(int)
            new_cell = interact_cell.at[Channel.obj].set(new_obj).at[Channel.extra].set(cooking_duration)
            # メニューにあるかどうかではなく注文されているかどうかで正しい調理かを判定
            ordered_menus = state.customer.ordered_menu.flatten()
            dish_to_cook = DynamicObject.set_count(new_obj | DynamicObject.COOKED | DynamicObject.PLATE, 0)
            is_ordered = jnp.any(dish_to_cook == ordered_menus)
            cook_reward = jax.lax.cond(is_ordered, lambda: shaped.pot_start_cooking, lambda: -penalty.erroneous_cooking)
            return (new_cell, inventory.at[storage_idx].set(DynamicObject.EMPTY), cook_reward)

        def _add(inventory: Int[Array, "max_storage"]):
            new_obj = DynamicObject.add_ingredient(interact_object, inventory[storage_idx])
            new_cell = interact_cell.at[Channel.obj].set(new_obj)
            return (new_cell, inventory.at[storage_idx].set(DynamicObject.EMPTY), shaped.placement_in_pot)

        pot_is_cooking = interact_extra > 0
        pot_is_cooked = interact_object & DynamicObject.COOKED != 0
        pot_is_full = pot_is_cooking | pot_is_cooked
        pot_is_full_after_drop = DynamicObject.ingredient_count(interact_object) == 2
        pot_is_idle = ~pot_is_cooking * ~pot_is_cooked * ~pot_is_full_after_drop
        new_cell, new_inventory, shaped_reward = jax.lax.switch(
            jnp.argmax(jnp.array([pot_is_full, pot_is_full_after_drop, pot_is_idle])),
            [
                lambda _: (
                    state.grid[*fwd_pos],
                    agent.inventory,
                    -penalty.ineffective_placement,
                ),  # 調理中、調理済みのとき何もしない
                _start_cooking,  # 3つ目の食材を追加し、調理を開始する
                _add,  # 食材を追加する
            ],
            agent.inventory,
        )
        new_grid = state.grid.at[*fwd_pos].set(new_cell)
        new_agent = agent.replace(inventory=new_inventory)
        return (state.replace(grid=new_grid), new_agent, 0.0, shaped_reward, RewardType.ADD_INGREDIENT)

    def _put_food_on_plate(state: State, agent: Agent):
        def _plate():
            def _do_plating(inventory: Int[Array, "max_storage"]):
                plated_food = inventory.at[storage_idx].set(interact_object | DynamicObject.PLATE)
                return (DynamicObject.EMPTY, plated_food, shaped.plate_dish, RewardType.PLATING)

            pot_is_cooked = interact_object & DynamicObject.COOKED != 0
            new_object, new_inventory, dish_reward, reward_type = jax.lax.cond(
                pot_is_cooked,
                _do_plating,
                lambda _: (
                    interact_object,
                    agent.inventory,
                    -penalty.ineffective_interaction,
                    RewardType.FAIL_PICK_PLACE,
                ),
                agent.inventory,
            )
            new_grid = state.grid.at[*fwd_pos, Channel.obj].set(new_object)
            new_agent = agent.replace(inventory=new_inventory)
            return (state.replace(grid=new_grid), new_agent, 0.0, dish_reward, reward_type)

        return jax.lax.cond(
            inventory_is_new_plate,
            _plate,
            lambda: (state, agent, 0.0, -penalty.ineffective_pickup, RewardType.FAIL_PICK_PLACE),
        )

    def _deliver_dish(state: State, agent: Agent):
        def _success_delivery():
            customer = state.customer
            table_id = customer.get_table_id(fwd_pos)
            is_correct_dish, correct_order_idx = state.menu.correct(
                agent.inventory[storage_idx], customer.ordered_menu[table_id]
            )
            new_inventory, new_customer = jax.lax.cond(
                is_correct_dish,
                lambda: (
                    agent.inventory.at[storage_idx].set(DynamicObject.EMPTY),
                    customer.put_dish_on_table(table_id, agent.inventory[storage_idx], correct_order_idx),
                ),
                lambda: (agent.inventory, customer),
            )
            new_agent = agent.replace(inventory=new_inventory)
            # 経過時間により報酬を割り引く
            discount = reward.discount
            enable_discount = discount.decline_deliver_reward
            transition_steps = discount.deliver_limit_step - discount.deliver_ramp_step
            discount_rate = jax.lax.cond(
                enable_discount,
                optax.schedules.linear_schedule(
                    init_value=1,
                    end_value=discount.deliver_discount_rate,
                    transition_steps=transition_steps,
                    transition_begin=discount.deliver_ramp_step,
                ),
                lambda _: 1.0,
                state.time - customer.time[table_id],
            )
            delivery_reward = jax.lax.cond(
                is_correct_dish,
                lambda: discount_rate * reward.shaped_reward.deliver_food,
                lambda: -penalty.erroneous_delivery,
            )
            return (state.replace(customer=new_customer), new_agent, 0.0, delivery_reward, RewardType.DELIVERY)

        return jax.lax.cond(
            inventory_is_dish & is_customer_waiting_delivery,
            _success_delivery,
            lambda: (state, agent, 0.0, -penalty.ineffective_placement, RewardType.FAIL_PICK_PLACE),
        )

    def _retrieve_plate(state: State, agent: Agent):
        def _success_plate_retrieval():
            customer = state.customer
            table_id = customer.get_table_id(fwd_pos)

            def _retrieve(customer: Customer):
                idx = jnp.argmax(customer.food[table_id] == DynamicObject.USED | DynamicObject.PLATE)
                new_food = customer.food.at[table_id, idx].set(DynamicObject.EMPTY)
                new_inventory = agent.inventory.at[storage_idx].set(DynamicObject.PLATE | DynamicObject.USED | 1)
                return new_inventory, new_food

            exists_empty_plate = jnp.sum(customer.food[table_id] == DynamicObject.USED | DynamicObject.PLATE) > 0
            new_inventory, new_food = jax.lax.cond(
                exists_empty_plate, _retrieve, lambda _: (agent.inventory, customer.food), customer
            )
            plate_retrieve_reward, reward_type = jax.lax.cond(
                exists_empty_plate,
                lambda: (shaped.pickup_used_plate_from_table, RewardType.RETRIEVE_PLATE),
                lambda: (-penalty.ineffective_pickup, RewardType.FAIL_PICK_PLACE),
            )
            new_agent = agent.replace(inventory=new_inventory)
            new_customer = customer.replace(food=new_food)
            return (state.replace(customer=new_customer), new_agent, 0.0, plate_retrieve_reward, reward_type)

        def _clean_table():
            customer = state.customer
            table_id = customer.get_table_id(fwd_pos)
            picked_up, new_customer = customer.cleanup(table_id)
            new_inventory = agent.inventory.at[storage_idx].set(picked_up)
            new_agent = agent.replace(inventory=new_inventory)
            return (state.replace(customer=new_customer), new_agent, 0.0, shaped.clean_table, RewardType.CLEAN_TABLE)

        return jax.lax.switch(
            jnp.argmax(jnp.array([is_plate_is_retrievable, is_table_need_cleaning, 1])),
            [
                _success_plate_retrieval,
                _clean_table,
                lambda: (state, agent, 0.0, -penalty.ineffective_pickup, RewardType.FAIL_PICK_PLACE),
            ],
        )

    def _pickup_from_counter(state: State, agent: Agent):
        # Booleans depending on what the agent interact
        object_is_ingredient = DynamicObject.is_ingredient(interact_object)
        object_is_new_plate = (
            (interact_object & DynamicObject.PLATE > 0)
            & (interact_object & DynamicObject.COOKED == 0)
            & (interact_object & DynamicObject.USED == 0)
        )
        object_is_dish = (interact_object & DynamicObject.PLATE > 0) & (interact_object & DynamicObject.COOKED > 0)
        object_is_used_plate = (interact_object & DynamicObject.PLATE > 0) & (interact_object & DynamicObject.USED > 0)
        branch = jnp.array([object_is_ingredient, object_is_new_plate, object_is_dish, object_is_used_plate, 1])
        branch_idx = jnp.argmax(branch)

        def _pick(state: State, agent: Agent, rew: float):
            picked_obj, remainings = DynamicObject.pick(interact_object)
            new_grid = state.grid.at[fwd_pos[0], fwd_pos[1], Channel.obj].set(remainings)
            new_inventory = agent.inventory.at[storage_idx].set(picked_obj)
            new_agent = agent.replace(inventory=new_inventory)
            return state.replace(grid=new_grid), new_agent, 0.0, rew, RewardType.PICKUP

        return jax.lax.switch(
            branch_idx,
            [
                partial(_pick, rew=shaped.pickup_ingredient_from_counter),
                partial(_pick, rew=shaped.pickup_new_plate_from_counter),
                partial(_pick, rew=shaped.pickup_dish_from_counter),
                partial(_pick, rew=shaped.pickup_used_plate_from_counter),
                lambda state, agent: (state, agent, 0.0, -penalty.ineffective_pickup, RewardType.FAIL_PICK_PLACE),
            ],
            state,
            agent,
        )

    def _pickup_plate(state: State, agent: Agent):
        picked_obj, remainings = DynamicObject.pick(interact_object)
        new_grid = state.grid.at[fwd_pos[0], fwd_pos[1], Channel.obj].set(remainings)
        new_inventory = agent.inventory.at[storage_idx].set(picked_obj)
        new_agent = agent.replace(inventory=new_inventory)
        return jax.lax.cond(
            inventory_is_empty,
            lambda: (
                state.replace(grid=new_grid),
                new_agent,
                0.0,
                shaped.pickup_new_plate_from_pile,
                RewardType.PICKUP_PLATE,
            ),
            lambda: (state, agent, 0.0, -penalty.ineffective_pickup, RewardType.FAIL_PICK_PLACE),
        )

    def _pickup_ingredient(state: State, agent: Agent):
        ingredient = StaticObject.get_ingredient(interact_item)
        new_inventory = agent.inventory.at[storage_idx].set(ingredient)
        new_agent = agent.replace(inventory=new_inventory)
        return jax.lax.cond(
            inventory_is_empty,
            lambda: (state, new_agent, 0.0, shaped.pickup_ingredient_from_pile, RewardType.PICKUP_INGREDIENT),
            lambda: (state, agent, 0.0, -penalty.ineffective_pickup, RewardType.FAIL_PICK_PLACE),
        )

    def _place_on_counter(state: State, agent: Agent):
        placed_obj, stackings = DynamicObject.place(interact_object, agent.inventory[storage_idx])
        object_is_placed = placed_obj == DynamicObject.EMPTY

        def _place_object():
            rew = jax.lax.switch(
                jnp.argmax(
                    jnp.array(
                        [inventory_is_ingredient, inventory_is_dish, inventory_is_used_plate, inventory_is_new_plate, 1]
                    )
                ),
                [
                    lambda: shaped.place_ingredient_on_counter,
                    lambda: shaped.place_food_on_counter,
                    lambda: shaped.place_used_plate_on_counter,
                    lambda: shaped.place_new_plate_on_counter,
                    lambda: -penalty.ineffective_placement,
                ],
            )
            new_grid = state.grid.at[*fwd_pos, Channel.obj].set(stackings)
            new_inventory = agent.inventory.at[storage_idx].set(placed_obj)
            new_agent = agent.replace(inventory=new_inventory)
            return (state.replace(grid=new_grid), new_agent, 0.0, rew, RewardType.PLACE)

        return jax.lax.cond(
            object_is_placed,
            _place_object,
            lambda: (state, agent, 0.0, -penalty.ineffective_placement, RewardType.FAIL_PICK_PLACE),
        )

    def _soak_plate(state: State, agent: Agent):
        def _soak():
            soaked_plate_count = DynamicObject.get_count(interact_object)
            new_obj, stackings = jax.lax.cond(
                soaked_plate_count < sink_capacity,
                DynamicObject.place,
                lambda obj, inv: (agent.inventory[storage_idx], interact_object),
                interact_object,
                agent.inventory[storage_idx],
            )
            soak_reward, reward_type = jax.lax.cond(
                soaked_plate_count < sink_capacity,
                lambda: (shaped.soak_plate, RewardType.SOAK_PLATE),
                lambda: (-penalty.ineffective_interaction, RewardType.FAIL_PICK_PLACE),
            )
            new_grid = state.grid.at[*fwd_pos, Channel.obj].set(stackings)
            new_inventory = agent.inventory.at[storage_idx].set(new_obj)
            new_agent = agent.replace(inventory=new_inventory)
            return (state.replace(grid=new_grid), new_agent, 0.0, soak_reward, reward_type)

        return jax.lax.cond(
            inventory_is_used_plate,
            _soak,
            lambda: (state, agent, 0.0, -penalty.ineffective_placement, RewardType.FAIL_PICK_PLACE),
        )

    def _dispose_garbage(state: State, agent: Agent):
        def _dispose():
            inventory = agent.inventory[storage_idx]
            new_inventory = jax.lax.cond(
                inventory & DynamicObject.PLATE,
                lambda: DynamicObject.set_count(DynamicObject.PLATE | DynamicObject.USED, 1),
                lambda: DynamicObject.EMPTY,
            )
            new_inventory = agent.inventory.at[storage_idx].set(new_inventory)
            new_agent = agent.replace(inventory=new_inventory)
            # 食材をそのまま捨てたり、正しく調理したものを捨てて報酬を稼ぐ懸念があるのでペナルティを与える
            dispose_cost = jax.lax.cond(
                inventory_is_ingredient, lambda: shaped.dispose_ingredient, lambda: shaped.dispose_food
            )
            return state, new_agent, 0.0, dispose_cost, RewardType.DISPOSE

        return jax.lax.cond(
            inventory_is_ingredient | inventory_is_dish,
            _dispose,
            lambda: (state, agent, 0.0, -penalty.ineffective_placement, RewardType.FAIL_PICK_PLACE),
        )

    # Booleans depending on what the agent is in front of
    in_front_of_counter = interact_item == StaticObject.COUNTER
    in_front_of_pot = interact_item == StaticObject.POT
    in_front_of_plate_pile = interact_item == StaticObject.PLATE_PILE
    in_front_of_sink = interact_item == StaticObject.SINK
    in_front_of_table = interact_item == StaticObject.TABLE
    in_front_of_ingredient_pile = StaticObject.is_ingredient_pile(interact_item)
    in_front_of_garbage_can = interact_item == StaticObject.GARBAGE_CAN

    # interact対象とエージェントの状態によって分岐
    # TODO: 分岐した時点でinteractionが成功するか決まっているかどうかが処理によって異なる
    #     : 有効なinteractionには報酬、無効なinteractionにはペナルティを与えられるよう関数内で判
    branches = jnp.array(
        [
            # ものを持つ
            in_front_of_counter & inventory_is_empty,
            in_front_of_plate_pile,
            in_front_of_ingredient_pile,
            # カウンターにものを置く
            in_front_of_counter & ~inventory_is_empty,
            # 調理する
            in_front_of_pot & inventory_is_ingredient,
            in_front_of_pot,
            # 料理を提供する
            in_front_of_table & ~inventory_is_empty,
            # 空いた皿を回収する
            in_front_of_table,
            # 皿を洗う
            in_front_of_sink,
            # ゴミを捨てる
            in_front_of_garbage_can,
            # default
            1,
        ]
    )
    branch_idx = jnp.argmax(branches)
    interact_functions = [
        # ものを持つ
        _pickup_from_counter,
        _pickup_plate,
        _pickup_ingredient,
        # カウンターにものを置く
        _place_on_counter,
        # 調理する
        _add_ingredient,
        _put_food_on_plate,
        # 料理を提供する
        _deliver_dish,
        # 空いた皿を回収する
        _retrieve_plate,
        # 皿を洗う
        _soak_plate,
        # ゴミを捨てる
        _dispose_garbage,
        # default
        _no_op,
    ]

    (new_state, new_agent, reward, shaped_reward, reward_type) = jax.lax.switch(
        branch_idx, interact_functions, state, agent
    )
    # jax.debug.print("interact branch: {}, target_idx: {}", branches, branch_idx)

    return (new_state, new_agent, reward, shaped_reward, reward_type)
