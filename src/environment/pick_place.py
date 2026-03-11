import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Key

from config import EnvParameterConfig, RewardConfig
from environment.agent import Agent
from environment.customer import Customer, CustomerStatus
from environment.dynamic_object import DynamicObject
from environment.reward import RewardType
from environment.state import Channel, State
from environment.static_object import StaticObject


def _pp_no_op(state: State, agent: Agent, *, penalty) -> tuple:
    return (state, agent, 0.0, -penalty.ineffective_interaction, RewardType.FAIL_PICK_PLACE)


def _pp_pickup(state: State, agent: Agent, *, fwd_pos, interact_object, storage_idx) -> tuple:
    picked_obj, remainings = DynamicObject.pick(interact_object)
    new_grid = state.grid.at[fwd_pos[0], fwd_pos[1], Channel.obj].set(remainings)
    return state.replace(grid=new_grid), agent.replace(inventory=agent.inventory.at[storage_idx].set(picked_obj)), 0.0, 0.0, RewardType.PICKUP


def _pp_pickup_ingredient(state: State, agent: Agent, *, interact_item, storage_idx) -> tuple:
    ingredient = StaticObject.get_ingredient(interact_item)
    return state, agent.replace(inventory=agent.inventory.at[storage_idx].set(ingredient)), 0.0, 0.0, RewardType.PICKUP


def _pp_place(state: State, agent: Agent, *, fwd_pos, interact_object, storage_idx) -> tuple:
    placed_obj, stackings = DynamicObject.place(interact_object, agent.inventory[storage_idx])
    new_grid = state.grid.at[*fwd_pos, Channel.obj].set(stackings)
    return state.replace(grid=new_grid), agent.replace(inventory=agent.inventory.at[storage_idx].set(placed_obj)), 0.0, 0.0, RewardType.PLACE


def _pp_start_cooking(inventory, *, interact_object, interact_cell, storage_idx, key, parameter, state, shaped) -> tuple:
    new_obj = DynamicObject.add_ingredient(interact_object, inventory[storage_idx])
    _is_correct_recipe, cooking_duration = state.menu.get_duration(new_obj)
    range_min, range_max = parameter.cooking_duration_range
    duration_coeff = jax.random.uniform(key, (), minval=range_min, maxval=range_max)
    cooking_duration = jnp.floor(cooking_duration * duration_coeff).astype(int)
    new_cell = interact_cell.at[Channel.obj].set(new_obj).at[Channel.extra].set(cooking_duration)
    return (new_cell, inventory.at[storage_idx].set(DynamicObject.EMPTY), shaped.pot_start_cooking)


def _pp_add_to_pot(inventory, *, interact_object, interact_cell, storage_idx, shaped) -> tuple:
    new_obj = DynamicObject.add_ingredient(interact_object, inventory[storage_idx])
    new_cell = interact_cell.at[Channel.obj].set(new_obj)
    return (new_cell, inventory.at[storage_idx].set(DynamicObject.EMPTY), shaped.placement_in_pot)


def _pp_add_ingredient(
    state: State, agent: Agent,
    *, fwd_pos, interact_object, interact_extra, interact_cell, storage_idx, key, parameter, shaped,
) -> tuple:
    pot_is_cooking = interact_extra > 0
    pot_is_cooked = interact_object & DynamicObject.COOKED != 0
    pot_is_full_after_drop = DynamicObject.ingredient_count(interact_object) == 2
    pot_is_full = pot_is_cooking | pot_is_cooked
    pot_is_idle = ~pot_is_cooking * ~pot_is_cooked * ~pot_is_full_after_drop
    start_cooking = functools.partial(
        _pp_start_cooking,
        interact_object=interact_object, interact_cell=interact_cell,
        storage_idx=storage_idx, key=key, parameter=parameter, state=state, shaped=shaped,
    )
    add_to_pot = functools.partial(
        _pp_add_to_pot,
        interact_object=interact_object, interact_cell=interact_cell, storage_idx=storage_idx, shaped=shaped,
    )
    new_cell, new_inventory, shaped_reward = jax.lax.switch(
        jnp.argmax(jnp.array([pot_is_full, pot_is_full_after_drop, pot_is_idle])),
        [lambda _: (state.grid[*fwd_pos], agent.inventory, 0.0), start_cooking, add_to_pot],
        agent.inventory,
    )
    new_grid = state.grid.at[*fwd_pos].set(new_cell)
    return (state.replace(grid=new_grid), agent.replace(inventory=new_inventory), 0.0, shaped_reward, RewardType.ADD_INGREDIENT)


def _pp_do_plating(inventory, *, interact_object, storage_idx, shaped) -> tuple:
    plated_food = inventory.at[storage_idx].set(interact_object | DynamicObject.PLATE)
    return (DynamicObject.EMPTY, plated_food, shaped.dish_pickup, RewardType.PLATING)


def _pp_put_food_on_plate(
    state: State, agent: Agent,
    *, fwd_pos, interact_object, interact_cell, storage_idx, shaped, penalty,
) -> tuple:
    pot_is_cooked = interact_object & DynamicObject.COOKED != 0
    do_plating = functools.partial(_pp_do_plating, interact_object=interact_object, storage_idx=storage_idx, shaped=shaped)
    new_object, new_inventory, dish_reward, reward_type = jax.lax.cond(
        pot_is_cooked,
        do_plating,
        lambda _: (interact_object, agent.inventory, -penalty.ineffective_interaction, RewardType.FAIL_PICK_PLACE),
        agent.inventory,
    )
    new_grid = state.grid.at[*fwd_pos, Channel.obj].set(new_object)
    return state.replace(grid=new_grid), agent.replace(inventory=new_inventory), 0.0, dish_reward, reward_type


def _pp_deliver_dish(state: State, agent: Agent, *, fwd_pos, storage_idx, shaped, penalty) -> tuple:
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
    # 経過時間により報酬を割り引く
    delivery_reward = (
        is_correct_dish
        * shaped.deliver_food
        * jnp.clip((1.0 - (state.time - customer.time[table_id]) / 100.0), min=0.0)
    ) - (1 - is_correct_dish) * penalty.erroneous_delivery
    # TODO: 誤提供はshaped_reward
    return (state.replace(customer=new_customer), agent.replace(inventory=new_inventory), 0.0, delivery_reward, RewardType.DELIVERY)


def _pp_retrieve(customer: Customer, *, table_id, storage_idx, agent: Agent) -> tuple:
    idx = jnp.argmax(customer.food[table_id] == DynamicObject.USED | DynamicObject.PLATE)
    new_food = customer.food.at[table_id, idx].set(DynamicObject.EMPTY)
    new_inventory = agent.inventory.at[storage_idx].set(DynamicObject.PLATE | DynamicObject.USED | 1)
    return new_inventory, new_food


def _pp_retrieve_plate(state: State, agent: Agent, *, fwd_pos, storage_idx, shaped, penalty) -> tuple:
    customer = state.customer
    table_id = customer.get_table_id(fwd_pos)
    exists_empty_plate = jnp.sum(customer.food[table_id] == DynamicObject.USED | DynamicObject.PLATE) > 0
    retrieve = functools.partial(_pp_retrieve, table_id=table_id, storage_idx=storage_idx, agent=agent)
    new_inventory, new_food = jax.lax.cond(
        exists_empty_plate, retrieve, lambda _: (agent.inventory, customer.food), customer
    )
    plate_retrieve_reward, reward_type = jax.lax.cond(
        exists_empty_plate,
        lambda: (shaped.retrieve_plate, RewardType.RETRIEVE_PLATE),
        lambda: (-penalty.ineffective_interaction, RewardType.FAIL_PICK_PLACE),
    )
    return (
        state.replace(customer=customer.replace(food=new_food)),
        agent.replace(inventory=new_inventory),
        0.0, plate_retrieve_reward, reward_type,
    )


def _pp_clean_table(state: State, agent: Agent, *, fwd_pos, storage_idx, shaped) -> tuple:
    customer = state.customer
    table_id = customer.get_table_id(fwd_pos)
    picked_up, new_customer = customer.cleanup(table_id)
    return (
        state.replace(customer=new_customer),
        agent.replace(inventory=agent.inventory.at[storage_idx].set(picked_up)),
        0.0, shaped.clean_table, RewardType.CLEAN_TABLE,
    )


def _pp_soak_plate(
    state: State, agent: Agent,
    *, fwd_pos, interact_object, storage_idx, shaped, penalty, sink_capacity,
) -> tuple:
    soaked_plate_count = DynamicObject.get_count(interact_object)
    new_obj, stackings = jax.lax.cond(
        soaked_plate_count < sink_capacity,
        DynamicObject.place,
        lambda _obj, _inv: (agent.inventory[storage_idx], interact_object),
        interact_object,
        agent.inventory[storage_idx],
    )
    soak_reward, reward_type = jax.lax.cond(
        soaked_plate_count < sink_capacity,
        lambda: (shaped.soak_plate, RewardType.SOAK_PLATE),
        lambda: (-penalty.ineffective_interaction, RewardType.FAIL_PICK_PLACE),
    )
    new_grid = state.grid.at[*fwd_pos, Channel.obj].set(stackings)
    return state.replace(grid=new_grid), agent.replace(inventory=agent.inventory.at[storage_idx].set(new_obj)), 0.0, soak_reward, reward_type


def _pp_dispose_garbage(state: State, agent: Agent, *, storage_idx) -> tuple:
    inventory = agent.inventory[storage_idx]
    new_inv_val = jax.lax.cond(
        inventory & DynamicObject.PLATE,
        lambda: DynamicObject.set_count(DynamicObject.PLATE | DynamicObject.USED, 1),
        lambda: DynamicObject.EMPTY,
    )
    # 食材をそのまま捨てたり、正しく調理したものを捨てて報酬を稼ぐ懸念があるので報酬を与えない
    return state, agent.replace(inventory=agent.inventory.at[storage_idx].set(new_inv_val)), 0.0, 0.0, RewardType.DISPOSE


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
    fwd_pos = agent.get_fwd_pos()

    interact_cell = state.grid[*fwd_pos]
    interact_item = interact_cell[Channel.env]
    interact_object = interact_cell[Channel.obj]
    interact_extra = interact_cell[Channel.extra]

    shaped = reward.shaped_reward
    penalty = reward.penalty
    sink_capacity = parameter.sink_capacity

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

    pickup = functools.partial(_pp_pickup, fwd_pos=fwd_pos, interact_object=interact_object, storage_idx=storage_idx)
    interact_functions = [
        # ものを持つ
        pickup,
        pickup,
        functools.partial(_pp_pickup_ingredient, interact_item=interact_item, storage_idx=storage_idx),
        # カウンターにものを置く
        functools.partial(_pp_place, fwd_pos=fwd_pos, interact_object=interact_object, storage_idx=storage_idx),
        # 調理する
        functools.partial(
            _pp_add_ingredient,
            fwd_pos=fwd_pos, interact_object=interact_object, interact_extra=interact_extra,
            interact_cell=interact_cell, storage_idx=storage_idx, key=key, parameter=parameter, shaped=shaped,
        ),
        functools.partial(
            _pp_put_food_on_plate,
            fwd_pos=fwd_pos, interact_object=interact_object, interact_cell=interact_cell,
            storage_idx=storage_idx, shaped=shaped, penalty=penalty,
        ),
        # 料理を提供する
        functools.partial(_pp_deliver_dish, fwd_pos=fwd_pos, storage_idx=storage_idx, shaped=shaped, penalty=penalty),
        # 空いた皿を回収する
        functools.partial(_pp_retrieve_plate, fwd_pos=fwd_pos, storage_idx=storage_idx, shaped=shaped, penalty=penalty),
        # テーブルを片付ける
        functools.partial(_pp_clean_table, fwd_pos=fwd_pos, storage_idx=storage_idx, shaped=shaped),
        # 皿を洗う
        functools.partial(
            _pp_soak_plate,
            fwd_pos=fwd_pos, interact_object=interact_object, storage_idx=storage_idx,
            shaped=shaped, penalty=penalty, sink_capacity=sink_capacity,
        ),
        # ゴミを捨てる
        functools.partial(_pp_dispose_garbage, storage_idx=storage_idx),
        # default
        functools.partial(_pp_no_op, penalty=penalty),
    ]

    (new_state, new_agent, reward, shaped_reward, reward_type) = jax.lax.switch(
        branch_idx, interact_functions, state, agent
    )

    return (new_state, new_agent, reward, shaped_reward, reward_type)
