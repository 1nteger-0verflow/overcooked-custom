from typing import Tuple

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from environment.agent import Agent
from environment.customer import CustomerStatus
from environment.dynamic_object import DynamicObject
from environment.layouts import Layout
from environment.reward import RewardType
from environment.state import Channel, State
from environment.static_object import StaticObject


def process_interact(
    state: State,
    agent: Agent,
    key: jax.Array,
    config: DictConfig,
    layout: Layout,
) -> Tuple[State, Agent, float, float, RewardType]:
    """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""
    # 1体のエージェントの前方のセル1か所に対する処理
    fwd_pos = agent.get_fwd_pos()

    interact_cell = state.grid[*fwd_pos]
    interact_item = interact_cell[Channel.env]
    interact_object = interact_cell[Channel.obj]

    original = config.reward.original_reward
    shaped = config.reward.shaped_reward
    penalty = config.reward.penalty

    def _no_op(state, agent):
        return (
            state,
            agent,
            0.0,
            -penalty.ineffective_interaction,
            RewardType.FAIL_INTERACT,
        )

    def _deal_customer(state, agent):
        def _invite(customer, line):
            new_customer, new_line, invite_reward, reward_type = jax.lax.cond(
                jnp.greater(line.reserved_line_length, 0)
                | jnp.greater(line.line_length, 0),
                lambda: (
                    customer.append(state.time, line.reserved_line_length > 0),
                    line.dequeue(),
                    shaped.invite_customer,
                    RewardType.INVITATION,
                ),
                lambda: (
                    customer,
                    line,
                    -penalty.ineffective_interaction,
                    RewardType.FAIL_INTERACT,
                ),
            )
            return new_customer, new_line, invite_reward, reward_type

        def _refuse(customer, line):
            refuse_reward, reward_type = jax.lax.cond(
                line.line_length > 0,
                lambda: (shaped.refuse_customer, RewardType.REFUSE_CUSTOMER),
                lambda: (-penalty.ineffective_interaction, RewardType.FAIL_INTERACT),
            )
            new_line = line.replace(
                line_length=jnp.clip(line.line_length - 1, min=0),
                queued_time=jnp.roll(line.queued_time, -1).at[-1].set(0),
            )
            return customer, new_line, refuse_reward, reward_type

        # 空席ありのとき待ち行列から客を案内、空席なしのとき断る
        new_customer, new_line, shaped_reward, reward_type = jax.lax.cond(
            state.customer.empty_count > 0,
            _invite,
            _refuse,
            state.customer,
            state.line,
        )
        return (
            state.replace(customer=new_customer, line=new_line),
            agent,
            0.0,
            shaped_reward,
            reward_type,
        )

    def _take_order(state, agent):
        customer = state.customer
        # 対象のテーブルIDを取得する
        tableID = customer.get_tableID(fwd_pos)
        new_status, new_time, new_order, order_reward, reward_type = jax.lax.cond(
            customer.status[tableID] == CustomerStatus.ordering,
            lambda: (
                customer.status.at[tableID].set(CustomerStatus.waiting_food),
                customer.time.at[tableID].set(state.time),
                customer.order(tableID, state.menu, key),
                shaped.take_order,
                RewardType.TAKE_ORDER,
            ),
            lambda: (
                customer.status,
                customer.time,
                customer.ordered_menu,
                -penalty.ineffective_interaction,
                RewardType.FAIL_INTERACT,
            ),
        )
        new_customer = customer.replace(
            status=new_status, time=new_time, ordered_menu=new_order
        )

        return (
            state.replace(customer=new_customer),
            agent,
            0.0,
            order_reward,
            reward_type,
        )

    def _wash_plate(state, agent):
        soaked_plate_count = DynamicObject.get_count(interact_object)
        plate_pile_pos = layout.plate_positions[0]
        plate_pile_obj = state.grid[*plate_pile_pos, Channel.obj]
        clean_plate_count = DynamicObject.get_count(
            state.grid[*plate_pile_pos, Channel.obj]
        )
        new_grid, wash_reward, reward_type = jax.lax.cond(
            soaked_plate_count > 0,
            lambda: (
                state.grid.at[*fwd_pos, Channel.obj]
                .set(DynamicObject.set_count(interact_object, soaked_plate_count - 1))
                .at[plate_pile_pos[0], plate_pile_pos[1], Channel.obj]
                .set(DynamicObject.set_count(plate_pile_obj, clean_plate_count + 1)),
                shaped.wash_plate,
                RewardType.WASH_PLATE,
            ),
            lambda: (
                state.grid,
                -penalty.ineffective_interaction,
                RewardType.FAIL_INTERACT,
            ),
        )
        return state.replace(grid=new_grid), agent, 0.0, wash_reward, reward_type

    def _process_payment(state, agent):
        def _leave(customer):
            # 会計中だった客を退店させる
            idx = jnp.argmax(customer.status == CustomerStatus.checking)
            new_customer = customer.leave(idx)
            # 会計が終了し、客が退店させたときに大報酬を与える
            return new_customer, original.finish_payment, 0.0, RewardType.CHECKING

        new_register = state.register.replace(
            service_time=jnp.clip(state.register.service_time - 1, min=0)
        )
        new_customer, final_reward, payment_reward, reward_type = jax.lax.cond(
            new_register.service_time == 0,
            _leave,
            lambda _: (
                state.customer,
                0.0,
                shaped.process_payment,
                RewardType.CHECKING,
            ),
            state.customer,
        )
        return (
            state.replace(customer=new_customer, register=new_register),
            agent,
            final_reward,
            payment_reward,
            reward_type,
        )

    def _clean_dirt(state, agent):
        new_grid = state.grid.at[*fwd_pos].set(
            jnp.array([interact_item, DynamicObject.clean_dirt(interact_object), 0])
        )
        return (
            state.replace(grid=new_grid),
            agent,
            0.0,
            shaped.clean_dirt,
            RewardType.CLEAN_DIRT,
        )

    # Booleans depending on what the agent is in front of
    in_front_of_entrance = interact_item == StaticObject.ENTRANCE
    in_front_of_register = interact_item == StaticObject.REGISTER
    in_front_of_sink = interact_item == StaticObject.SINK
    in_front_of_table = interact_item == StaticObject.TABLE

    # Booleans depending on what the agent interact
    object_is_dirt = interact_object & DynamicObject.DIRT > 0

    # Booleans depending on customer status
    customer_status = jax.lax.cond(
        state.customer.is_table(fwd_pos),
        lambda: state.customer.status[state.customer.get_tableID(fwd_pos)],
        lambda: CustomerStatus.empty,
    )
    is_customer_ordering = customer_status == CustomerStatus.ordering

    # Booleans depending on payment
    is_register_executing = jnp.any(
        state.customer.status == CustomerStatus.checking
    ) | jnp.any(state.customer.status == CustomerStatus.waiting_check)

    # interact対象とエージェントの状態によって分岐
    # TODO: 分岐した時点でinteractionが成功するか決まっているかどうかが処理によって異なる
    #     : 有効なinteractionには報酬、無効なinteractionにはペナルティを与えられるよう関数内で判定
    branches = jnp.array(
        [
            # 並んでいる客への対応
            in_front_of_entrance,
            # 注文を取る
            in_front_of_table & is_customer_ordering,
            # 皿を洗う
            in_front_of_sink,
            # 会計する
            in_front_of_register & is_register_executing,
            # 床の掃除
            object_is_dirt,
            # default
            1,
        ]
    )
    branch_idx = jnp.argmax(branches)
    interact_functions = [
        # 並んでいる客への対応
        _deal_customer,
        # 注文を取る
        _take_order,
        # 皿を洗う
        _wash_plate,
        # 会計する
        _process_payment,
        # 床の掃除
        _clean_dirt,
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
