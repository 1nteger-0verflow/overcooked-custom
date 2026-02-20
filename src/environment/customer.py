from enum import IntEnum

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode, dataclass

from environment.dynamic_object import DynamicObject
from environment.menus import MenuList


class CustomerStatus(IntEnum):
    empty = 0  # 空席
    sitting = 1  # 案内されて着席した状態
    ordering = 2  # 注文待ち
    waiting_food = 3  # 料理提供待ち
    eating_food = 4  # 食事中
    waiting_check = 5  # 会計待ち
    checking = 6  # 会計中
    cleaning = 7  # 片付け中


@dataclass
class Customer(PyTreeNode):
    table_pos: jnp.ndarray
    chair_pos: jnp.ndarray
    menu: MenuList
    used: jnp.ndarray  # 0,1   TODO: status!=emptyでわかるので不要では？
    status: jnp.ndarray  # CustomerStatus
    time: jnp.ndarray  # 客席数 x 最後に何かした時刻(statusが変わった時刻)
    ordered_menu: jnp.ndarray  # 注文内容  客席数 x 注文上限数
    food: jnp.ndarray  # 客席数 x テーブルに置ける上限数

    def __str__(self):
        status_string = {
            CustomerStatus.empty: "空席",
            CustomerStatus.sitting: "着席",
            CustomerStatus.ordering: "注文待ち",
            CustomerStatus.waiting_food: "料理提供待ち",
            CustomerStatus.eating_food: "食事中",
            CustomerStatus.waiting_check: "会計待ち",
            CustomerStatus.checking: "会計中",
            CustomerStatus.cleaning: "片付け中",
        }

        def _discribe_order(idx):
            if self.status[idx] in [CustomerStatus.waiting_food, CustomerStatus.eating_food]:
                return ", 注文: " + ",".join(
                    [f"(#{order!s}){self.menu.menu[order]}" for order in self.ordered_menu[idx] if order >= 0]
                )
            return ""

        def _discribe_food(idx):
            if (
                self.status[idx] == CustomerStatus.eating_food
                or self.status[idx] == CustomerStatus.waiting_check
                or self.status[idx] == CustomerStatus.checking
                or self.status[idx] == CustomerStatus.cleaning
            ):
                return ", 配膳: " + ", ".join([DynamicObject.decode(food) for food in self.food[idx] if food != 0])
            return ""

        def _discribe_table(idx, table_pos, status, time):
            discription = f"[Table {idx}]({table_pos}):"
            discription += f" status: {status_string.get(CustomerStatus(status), '不明')}, from: {time}"
            discription += _discribe_order(idx)
            discription += _discribe_food(idx)
            discription += "\n"
            return discription

        expr = "-" * 60 + "\n"
        expr += "■■ 客席状況 ■■\n"
        for i, (table_pos, status, time) in enumerate(zip(self.table_pos, self.status, self.time)):
            expr += _discribe_table(i, table_pos, status, time)
        return expr

    @property
    def empty_count(self):
        return self.used.shape[0] - jnp.count_nonzero(self.used)

    def empty_seat(self):
        # 新しい客を通す客席のIDを取得する
        # jax.numpy.argmaxは最小のインデックスを見つける
        return jnp.argmax(self.used == 0)

    @property
    def seat_count(self):
        return self.used.shape[0]

    def is_table(self, pos: jnp.ndarray):
        # テーブルに向いているかを判定
        return jnp.any(jnp.all(self.table_pos == pos, axis=1))

    def get_tableID(self, pos: jnp.ndarray):
        # テーブルの位置からIDを取得
        return jnp.argmax(jnp.all(self.table_pos == pos, axis=1))

    def append(self, time: jax.Array, is_reserved: bool):
        return self.replace(
            used=self.used.at[self.empty_seat()].set(1),
            status=self.status.at[self.empty_seat()].set(CustomerStatus.sitting),
            time=self.time.at[self.empty_seat()].set(time),
        )

    def order(self, idx: int, menu: MenuList, key: jax.Array):
        # key, *_ = jax.random.split(key, idx)   # 同じstepに複数エージェントが同時に注文を取るケースに対応
        order_max = min(menu.num_menus, self.ordered_menu.shape[1])
        ordernum = jax.random.randint(key, (), minval=1, maxval=order_max + 1)  # 注文件数を決める
        order_menus = jax.random.choice(key, jnp.arange(menu.num_menus), shape=(order_max,), replace=False)
        order_menus = jax.lax.fori_loop(ordernum, self.ordered_menu.shape[1], lambda i, o: o.at[i].set(-1), order_menus)
        order_menus = jnp.sort(order_menus)
        return self.ordered_menu.at[idx].set(order_menus)

    def put_dish_on_table(self, table_idx: int, dish: DynamicObject, delivered_idx: int):
        place_idx = jnp.argmax(self.food[table_idx] == DynamicObject.EMPTY)  # 空いている最も若い場所のidx
        new_food = self.food.at[table_idx, place_idx].set(dish)
        new_order = self.ordered_menu.at[table_idx, delivered_idx].set(-1)
        return self.replace(
            status=self.status.at[table_idx].set(CustomerStatus.eating_food), food=new_food, ordered_menu=new_order
        )

    def leave(self, idx):
        # 会計したが皿を下げていないとき片付け中の状態にする
        new_status, new_time, new_used = jax.lax.cond(
            jnp.any(DynamicObject.is_plate(self.food) > 0),
            lambda: (CustomerStatus.cleaning, self.time[idx], 1),
            lambda: (CustomerStatus.empty, 0, 0),
        )
        return self.replace(
            used=self.used.at[idx].set(new_used),
            status=self.status.at[idx].set(new_status),
            time=self.time.at[idx].set(new_time),
            ordered_menu=self.ordered_menu.at[idx].set(0),
        )

    def cleanup(self, idx):
        # 残った皿を片付ける
        # jax.debug.print("clean table {}, before cleaning: {}", idx, self.food)
        need_cleaning_idx = jnp.argmax(self.food[idx] != DynamicObject.EMPTY)
        # jax.debug.print("remove at {}", need_cleaning_idx)
        pickup = DynamicObject.set_count(
            self.food[idx][need_cleaning_idx], 1
        )  # 食べ終わりでカウントが0になっているので、皿1枚に修正
        new_food = self.food.at[idx, need_cleaning_idx].set(DynamicObject.EMPTY)
        finished = jnp.all(new_food[idx] == DynamicObject.EMPTY)
        # jax.debug.print("cleanup finished?: {}, foods: {}", finished, new_food[idx])
        return jax.lax.cond(
            finished,
            lambda: (
                pickup,
                self.replace(
                    used=self.used.at[idx].set(0), status=self.status.at[idx].set(CustomerStatus.empty), food=new_food
                ),
            ),
            lambda: (pickup, self.replace(food=new_food)),
        )


@dataclass
class CustomerLine(PyTreeNode):
    entrance_pos: jnp.ndarray
    # arr = jnp.roll(arr, -1).at[-1].set(0)  1個ずらして終端を0にすることで案内に代える
    line_length: jnp.ndarray  # 一般客の並んでいる人数
    queued_time: jnp.ndarray  # 一般客の並び始めた時間
    reserved_line_length: jnp.ndarray  # 予約客の並んでいる人数
    reserved_queued_time: jnp.ndarray  # 予約客の並び始めた時間
    reserve_time: jnp.ndarray  # 予約時間

    def __str__(self):
        expr = "-" * 60 + "\n"
        expr += "■■ 案内待ち ■■\n"
        expr += f"一般客: {self.line_length}人, 並んだ時間 = "
        expr += ",".join([str(self.queued_time[i]) for i in range(self.line_length)])
        expr += "\n"
        expr += f"予約客: {self.reserved_line_length}人, 予約客来店時間 = "
        expr += ",".join([str(self.reserved_queued_time[i]) for i in range(self.reserved_line_length)])
        expr += ", 予約時間 = "
        expr += ",".join([str(t) for t in self.reserve_time])
        expr += "\n"
        return expr

    def dequeue(self):
        (new_line_length, new_queued_time, new_reserved_line_length, new_reserved_queued_time) = jax.lax.switch(
            jnp.argmax(jnp.array([self.reserved_line_length > 0, self.line_length > 0, 1])),
            [
                lambda: (
                    self.line_length,
                    self.queued_time,
                    jnp.clip(self.reserved_line_length - 1, min=0),
                    jnp.roll(self.reserved_queued_time, -1).at[-1].set(0),
                ),
                lambda: (
                    jnp.clip(self.line_length - 1, min=0),
                    jnp.roll(self.queued_time, -1).at[-1].set(0),
                    self.reserved_line_length,
                    self.reserved_queued_time,
                ),
                lambda: (self.line_length, self.queued_time, self.reserved_line_length, self.reserved_queued_time),
            ],
        )
        return self.replace(
            line_length=new_line_length,
            queued_time=new_queued_time,
            reserved_line_length=new_reserved_line_length,
            reserved_queued_time=new_reserved_queued_time,
        )

    def get_obs(self):
        return jnp.append(self.reserved_queued_time, self.queued_time)


@dataclass
class RegisterLine(PyTreeNode):
    # 会計待ちの客の管理用
    register_pos: jnp.ndarray
    queued_time: jnp.ndarray  # 会計待ちになった時間
    service_time: jnp.ndarray  # 会計にかかる残り時間(step数)

    def __str__(self):
        expr = ""
        if self.service_time > 0:
            expr = "-" * 60 + "\n"
            expr = "■■ 会計 ■■\n"
            expr += f"レジに並んだ時間： {self.queued_time}, 会計にかかる時間残: {self.service_time}"
        return expr
