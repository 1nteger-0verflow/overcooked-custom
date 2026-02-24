import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from environment.customer import Customer, CustomerStatus, RegisterLine
from environment.dynamic_object import DynamicObject
from environment.state import Channel, State
from environment.static_object import StaticObject


def update_step(state: State, config: DictConfig, key: jax.Array) -> State:
    # 時間経過による環境の更新
    grid_key, line_key, customer_key, cook_key, check_key = jax.random.split(key, 5)
    state = disturb_env(state, config, grid_key)
    state = arrive_customer(state, config, line_key)
    state = call_order(state, customer_key)
    state = progress_cooking(state, config, cook_key)
    state = progress_eating(state)
    return get_the_check(state, config, check_key)


def disturb_env(state: State, config: DictConfig, key: jax.Array) -> State:
    grid = state.grid
    # 床に汚れがランダムに出現
    # 対象のマスを１つ選択し、次に汚れを発生させるかどうかを判定する
    cell_select_key, dirt_key = jax.random.split(key, 2)
    cell_idx = jax.random.choice(cell_select_key, grid.shape[0] * grid.shape[1])
    cell = jnp.unravel_index(cell_idx, grid.shape[:2])
    target_cell = grid[cell]
    dirt_appear_rate = config.parameter.dirt_appear_rate
    dirtiness_max = config.parameter.dirtiness_max
    # TODO: state.agents.agent_posとtarget_cellが同じ場合は汚れなし
    # on_agent = jnp.any(jnp.all(cell==agent_pos))  # Positionを１つのjnp.ndarrayにする

    def _appear_dirt(cell):
        # TODO: エージェントの現在いるマスにも出現してしまう
        rn = jax.random.uniform(dirt_key, (), minval=0.0, maxval=1.0)
        dirt_level = jax.random.randint(dirt_key, (), minval=1, maxval=dirtiness_max)
        dirt_appear = rn < dirt_appear_rate
        dirt = DynamicObject.create_dirt(dirt_level)
        new_obj = jax.lax.select(dirt_appear, dirt, cell[Channel.obj])

        return cell.at[Channel.obj].set(new_obj)

    is_empty = (target_cell[Channel.env] == StaticObject.EMPTY) * (target_cell[Channel.obj] == DynamicObject.EMPTY)
    new_cell = jax.lax.cond(is_empty, _appear_dirt, lambda x: x, target_cell)
    new_grid = grid.at[cell].set(new_cell)

    return state.replace(grid=new_grid)


def arrive_customer(state: State, config: DictConfig, key: jax.Array) -> State:
    line = state.line
    current_step = state.time
    congestion_rates = config.schedule.congestion_rates

    def _generate_congestion_rate(time: jax.Array):
        # [step数、 出現頻度(%)] を参照し、現在の時刻での出現頻度(0~1)を求める
        temporal_rates = jnp.array(congestion_rates)
        current_timezone_idx = jnp.argmax(temporal_rates[:, 0] > time) - 1
        return temporal_rates[current_timezone_idx, 1] / 100.0

    # JAXの配列はサイズが変わらない、array.at[n].set(value)として、nがサイズ以上の場合は変更されない
    # これを利用してサイズの判定をなくすことができる
    rn = jax.random.uniform(key, (), minval=0.0, maxval=1.0)
    thres = _generate_congestion_rate(current_step)
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


def call_order(state: State, key: jax.Array) -> State:
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
    rn = jax.random.uniform(key, (state.customer.num_customers,), minval=0.0, maxval=1.0)
    # statusがsitting -> call orderに変わる
    new_customer, _, _ = jax.lax.fori_loop(0, state.customer.seat_count, _order, (state.customer, rn, state.time))
    return state.replace(customer=new_customer)


def progress_cooking(state: State, config: DictConfig, key: jax.Array) -> State:
    # Update extra info:
    def _timestep_wrapper(cell):
        def _cook(cell):
            is_cooking = cell[Channel.extra] > 0
            new_extra = jax.lax.select(is_cooking, cell[Channel.extra] - 1, cell[Channel.extra])
            finished_cooking = is_cooking * (new_extra == 0)
            correct, volume = state.menu.get_volume(cell[Channel.obj])
            # volumeを指定範囲内の倍率でばらつかせる
            range_min, range_max = config.parameter.volume_range
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


def progress_eating(state: State) -> State:
    def _eat(status: jax.Array, food: jax.Array, prev_phase_time: jax.Array):
        is_eating = status == CustomerStatus.eating_food
        volumes = DynamicObject.get_count(food)
        decreased_volumes, new_time = jax.lax.switch(
            jnp.argmax(
                jnp.array(
                    [is_eating * jnp.all(jnp.max(volumes) == 1), is_eating * jnp.any(jnp.max(volumes) > 1), ~is_eating]
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


def get_the_check(state: State, config: DictConfig, key: jax.Array) -> State:
    def _start_checking(customer: Customer, register: RegisterLine):
        # 会計待ちになってからの時間が最も長い客席のindex
        dequeue_idx = jnp.argmax((customer.status == CustomerStatus.waiting_check) * (state.time + 1 - customer.time))
        new_status = customer.status.at[dequeue_idx].set(CustomerStatus.checking)
        new_time = customer.time.at[dequeue_idx].set(state.time)
        # 会計にかかるステップ数をランダムにする
        duration_max = config.parameter.check_time_max
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
