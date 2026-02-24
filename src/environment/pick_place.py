import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from environment.agent import Agent
from environment.customer import CustomerStatus
from environment.dynamic_object import DynamicObject
from environment.reward import RewardType
from environment.state import Channel, State
from environment.static_object import StaticObject


def pick_and_place(
    state: State,
    agent: Agent,
    key: jax.Array,
    storage_idx: jax.Array,
    config: DictConfig,
):
    """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""
    # 1体のエージェントの前方のセル1か所に対する処理
    inventory = agent.inventory[storage_idx]
    fwd_pos = agent.get_fwd_pos()

    interact_cell = state.grid[*fwd_pos]
    interact_item = interact_cell[Channel.env]
    interact_object = interact_cell[Channel.obj]
    interact_extra = interact_cell[Channel.extra]

    shaped = config.reward.shaped_reward
    penalty = config.reward.penalty
    sink_capacity = config.parameter.sink_capacity

    def _no_op(state, agent):
        return (
            state,
            agent,
            0.0,
            -penalty.ineffective_interaction,
            RewardType.FAIL_PICK_PLACE,
        )

    def _add_ingredient(state, agent):
        def _start_cooking(inventory):
            new_obj = DynamicObject.add_ingredient(
                interact_object, inventory[storage_idx]
            )
            is_correct_recipe, cooking_duration = state.menu.get_duration(new_obj)
            # cooking_durationを指定範囲内の倍率でばらつかせる
            range_min, range_max = config.parameter.cooking_duration_range
            duration_coeff = jax.random.uniform(
                key, (), minval=range_min, maxval=range_max
            )
            cooking_duration = jnp.floor(cooking_duration * duration_coeff).astype(int)
            new_cell = (
                interact_cell.at[Channel.obj]
                .set(new_obj)
                .at[Channel.extra]
                .set(cooking_duration)
            )
            return (
                new_cell,
                inventory.at[storage_idx].set(DynamicObject.EMPTY),
                shaped.pot_start_cooking,
            )

        def _add(inventory):
            new_obj = DynamicObject.add_ingredient(
                interact_object, inventory[storage_idx]
            )
            new_cell = interact_cell.at[Channel.obj].set(new_obj)
            return (
                new_cell,
                inventory.at[storage_idx].set(DynamicObject.EMPTY),
                shaped.placement_in_pot,
            )

        pot_is_cooking = interact_extra > 0
        pot_is_cooked = interact_object & DynamicObject.COOKED != 0
        pot_is_full = pot_is_cooking | pot_is_cooked
        pot_is_full_after_drop = DynamicObject.ingredient_count(interact_object) == 2
        pot_is_idle = ~pot_is_cooking * ~pot_is_cooked * ~pot_is_full_after_drop
        new_cell, new_inventory, shaped_reward = jax.lax.switch(
            jnp.argmax(
                jnp.array(
                    [
                        pot_is_full,
                        pot_is_full_after_drop,
                        pot_is_idle,
                    ]
                )
            ),
            [
                lambda _: (
                    state.grid[*fwd_pos],
                    agent.inventory,
                    0.0,
                ),  # 調理中、調理済みのとき何もしない
                _start_cooking,  # 3つ目の食材を追加し、調理を開始する
                _add,  # 食材を追加する
            ],
            agent.inventory,
        )
        new_grid = state.grid.at[*fwd_pos].set(new_cell)
        new_agent = agent.replace(inventory=new_inventory)
        return (
            state.replace(grid=new_grid),
            new_agent,
            0.0,
            shaped_reward,
            RewardType.ADD_INGREDIENT,
        )

    def _put_food_on_plate(state, agent):
        def _do_plating(inventory):
            plated_food = inventory.at[storage_idx].set(
                interact_object | DynamicObject.PLATE
            )
            return (
                DynamicObject.EMPTY,
                plated_food,
                shaped.dish_pickup,
                RewardType.PLATING,
            )

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
        return state.replace(grid=new_grid), new_agent, 0.0, dish_reward, reward_type

    def _deliver_dish(state, agent):
        customer = state.customer
        tableID = customer.get_tableID(fwd_pos)
        is_correct_dish, correct_order_idx = state.menu.correct(
            agent.inventory[storage_idx], customer.ordered_menu[tableID]
        )
        new_inventory, new_customer = jax.lax.cond(
            is_correct_dish,
            lambda: (
                agent.inventory.at[storage_idx].set(DynamicObject.EMPTY),
                customer.put_dish_on_table(
                    tableID, agent.inventory[storage_idx], correct_order_idx
                ),
            ),
            lambda: (
                agent.inventory,
                customer,
            ),
        )
        new_agent = agent.replace(inventory=new_inventory)
        # 経過時間により報酬を割り引く
        delivery_reward = (
            is_correct_dish
            * config.reward.shaped_reward.deliver_food
            * jnp.clip((1.0 - (state.time - customer.time[tableID]) / 100.0), min=0.0)
        ) - (1 - is_correct_dish) * penalty.erroneous_delivery
        # TODO: 誤提供はshaped_reward
        return (
            state.replace(customer=new_customer),
            new_agent,
            0.0,
            delivery_reward,
            RewardType.DELIVERY,
        )

    def _retrieve_plate(state, agent):
        customer = state.customer
        tableID = customer.get_tableID(fwd_pos)

        def _retrieve(customer):
            idx = jnp.argmax(
                customer.food[tableID] == DynamicObject.USED | DynamicObject.PLATE
            )
            new_food = customer.food.at[tableID, idx].set(DynamicObject.EMPTY)
            new_inventory = agent.inventory.at[storage_idx].set(
                DynamicObject.PLATE | DynamicObject.USED | 1
            )
            return new_inventory, new_food

        exists_empty_plate = (
            jnp.sum(customer.food[tableID] == DynamicObject.USED | DynamicObject.PLATE)
            > 0
        )
        new_inventory, new_food = jax.lax.cond(
            exists_empty_plate,
            _retrieve,
            lambda _: (agent.inventory, customer.food),
            customer,
        )
        plate_retrieve_reward, reward_type = jax.lax.cond(
            exists_empty_plate,
            lambda: (shaped.retrieve_plate, RewardType.RETRIEVE_PLATE),
            lambda: (-penalty.ineffective_interaction, RewardType.FAIL_PICK_PLACE),
        )
        new_agent = agent.replace(inventory=new_inventory)
        new_customer = customer.replace(food=new_food)
        return (
            state.replace(customer=new_customer),
            new_agent,
            0.0,
            plate_retrieve_reward,
            reward_type,
        )

    def _clean_table(state, agent):
        customer = state.customer
        tableID = customer.get_tableID(fwd_pos)
        picked_up, new_customer = customer.cleanup(tableID)
        new_inventory = agent.inventory.at[storage_idx].set(picked_up)
        new_agent = agent.replace(inventory=new_inventory)
        return (
            state.replace(customer=new_customer),
            new_agent,
            0.0,
            shaped.clean_table,
            RewardType.CLEAN_TABLE,
        )

    def _pickup(state, agent):
        picked_obj, remainings = DynamicObject.pick(interact_object)
        new_grid = state.grid.at[fwd_pos[0], fwd_pos[1], Channel.obj].set(remainings)
        new_inventory = agent.inventory.at[storage_idx].set(picked_obj)
        new_agent = agent.replace(inventory=new_inventory)
        return state.replace(grid=new_grid), new_agent, 0.0, 0.0, RewardType.PICKUP

    def _pickup_ingredient(state, agent):
        ingredient = StaticObject.get_ingredient(interact_item)
        new_inventory = agent.inventory.at[storage_idx].set(ingredient)
        new_agent = agent.replace(inventory=new_inventory)
        return state, new_agent, 0.0, 0.0, RewardType.PICKUP

    def _place(state, agent):
        placed_obj, stackings = DynamicObject.place(
            interact_object, agent.inventory[storage_idx]
        )
        new_grid = state.grid.at[*fwd_pos, Channel.obj].set(stackings)
        new_inventory = agent.inventory.at[storage_idx].set(placed_obj)
        new_agent = agent.replace(inventory=new_inventory)
        return state.replace(grid=new_grid), new_agent, 0.0, 0.0, RewardType.PLACE

    def _soak_plate(state, agent):
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
        return state.replace(grid=new_grid), new_agent, 0.0, soak_reward, reward_type

    def _dispose_garbage(state, agent):
        inventory = agent.inventory[storage_idx]
        new_inventory = jax.lax.cond(
            inventory & DynamicObject.PLATE,
            lambda: DynamicObject.set_count(
                DynamicObject.PLATE | DynamicObject.USED, 1
            ),
            lambda: DynamicObject.EMPTY,
        )
        new_inventory = agent.inventory.at[storage_idx].set(new_inventory)
        new_agent = agent.replace(inventory=new_inventory)
        # 食材をそのまま捨てたり、正しく調理したものを捨てて報酬を稼ぐ懸念があるので報酬を与えない
        return state, new_agent, 0.0, 0.0, RewardType.DISPOSE

    # Booleans depending on what the agent is in front of
    in_front_of_counter = interact_item == StaticObject.COUNTER
    in_front_of_pot = interact_item == StaticObject.POT
    in_front_of_plate_pile = interact_item == StaticObject.PLATE_PILE
    in_front_of_sink = interact_item == StaticObject.SINK
    in_front_of_table = interact_item == StaticObject.TABLE
    in_front_of_ingredient_pile = StaticObject.is_ingredient_pile(interact_item)
    in_front_of_garbage_can = interact_item == StaticObject.GARBAGE_CAN

    # Booleans depending on what the agent interact
    object_is_plate = interact_object & DynamicObject.PLATE > 0
    object_is_ingredient = DynamicObject.is_ingredient(interact_object)
    # plateにdish, used_plateも含まれる
    object_is_pickable = object_is_plate | object_is_ingredient

    # Booleans depending on what agent have
    inventory_is_empty = inventory == DynamicObject.EMPTY
    inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
    inventory_is_cooked = (inventory & DynamicObject.COOKED) > 0
    inventory_is_plated = (inventory & DynamicObject.PLATE) > 0
    inventory_is_dish = inventory_is_cooked * inventory_is_plated
    inventory_is_used_plate = (inventory & DynamicObject.USED) > 0
    inventory_is_new_plate = (
        (inventory & DynamicObject.PLATE > 0)
        & ~inventory_is_dish
        & ~inventory_is_used_plate
    )

    # Booleans depending on customer status
    customer_status = jax.lax.cond(
        state.customer.is_table(fwd_pos),
        lambda: state.customer.status[state.customer.get_tableID(fwd_pos)],
        lambda: CustomerStatus.empty,
    )
    is_customer_waiting_food = customer_status == CustomerStatus.waiting_food
    is_customer_eating = customer_status == CustomerStatus.eating_food
    is_customer_waiting_delivery = is_customer_waiting_food | is_customer_eating
    is_customer_waiting_check = customer_status == CustomerStatus.waiting_check
    is_customer_checking = customer_status == CustomerStatus.checking
    is_plate_is_retrievable = (
        is_customer_eating | is_customer_waiting_check | is_customer_checking
    )
    is_table_need_cleaning = customer_status == CustomerStatus.cleaning

    # interact対象とエージェントの状態によって分岐
    # TODO: 分岐した時点でinteractionが成功するか決まっているかどうかが処理によって異なる
    #     : 有効なinteractionには報酬、無効なinteractionにはペナルティを与えられるよう関数内で判
    branches = jnp.array(
        [
            # ものを持つ
            in_front_of_counter & object_is_pickable & inventory_is_empty,
            in_front_of_plate_pile & inventory_is_empty,
            in_front_of_ingredient_pile & inventory_is_empty,
            # カウンターにものを置く
            in_front_of_counter & ~inventory_is_empty,
            # 調理する
            in_front_of_pot & inventory_is_ingredient,
            in_front_of_pot & inventory_is_new_plate,
            # 料理を提供する
            in_front_of_table & inventory_is_dish & is_customer_waiting_delivery,
            # 空いた皿を回収する
            in_front_of_table & inventory_is_empty & is_plate_is_retrievable,
            # テーブルを片付ける
            in_front_of_table & inventory_is_empty & is_table_need_cleaning,
            # 皿を洗う
            in_front_of_sink & inventory_is_used_plate,
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
        # 調理する
        _add_ingredient,
        _put_food_on_plate,
        # 料理を提供する
        _deliver_dish,
        # 空いた皿を回収する
        _retrieve_plate,
        # テーブルを片付ける
        _clean_table,
        # 皿を洗う
        _soak_plate,
        # ゴミを捨てる
        _dispose_garbage,
        # default
        _no_op,
    ]

    (
        new_state,
        new_agent,
        reward,
        shaped_reward,
        reward_type,
    ) = jax.lax.switch(
        branch_idx,
        interact_functions,
        state,
        agent,
    )
    # jax.debug.print("interact branch: {}, target_idx: {}", branches, branch_idx)

    return (
        new_state,
        new_agent,
        reward,
        shaped_reward,
        reward_type,
    )
