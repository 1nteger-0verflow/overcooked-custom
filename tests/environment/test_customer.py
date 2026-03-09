"""Tests for environment.customer: Customer・CustomerLine・RegisterLine."""

import jax.numpy as jnp

from environment.customer import Customer, CustomerLine, CustomerStatus, RegisterLine
from environment.dynamic_object import DynamicObject
from environment.menus import MenuList


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _make_customer(num_seats: int = 2, order_max: int = 2) -> Customer:
    all_table_pos = [[2, 3], [4, 5], [6, 7]]
    all_chair_pos = [[2, 2], [4, 4], [6, 6]]
    return Customer(
        table_pos=jnp.array(all_table_pos[:num_seats], dtype=jnp.int32),
        chair_pos=jnp.array(all_chair_pos[:num_seats], dtype=jnp.int32),
        used=jnp.zeros((num_seats,), dtype=jnp.int32),
        status=jnp.full((num_seats,), CustomerStatus.empty, dtype=jnp.int32),
        time=jnp.zeros((num_seats,), dtype=jnp.int32),
        ordered_menu=jnp.full((num_seats, order_max), -1, dtype=jnp.int32),
        food=jnp.full((num_seats, order_max), DynamicObject.EMPTY, dtype=jnp.int32),
    )


def _make_customer_line(reserved: int = 0, general: int = 0) -> CustomerLine:
    return CustomerLine(
        entrance_pos=jnp.array([[0, 3]], dtype=jnp.int32),
        line_length=jnp.array(general, dtype=jnp.int32),
        queued_time=jnp.array([5, 10, 0], dtype=jnp.int32),
        reserved_line_length=jnp.array(reserved, dtype=jnp.int32),
        reserved_queued_time=jnp.array([3, 0], dtype=jnp.int32),
        reserve_time=jnp.array([30, 50], dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Customer プロパティ
# ---------------------------------------------------------------------------
class TestCustomerProperties:
    def test_num_customers(self):
        assert _make_customer(2).num_customers == 2

    def test_empty_count_all_empty(self):
        assert int(_make_customer(2).empty_count) == 2

    def test_seat_count(self):
        assert _make_customer(3).seat_count == 3


# ---------------------------------------------------------------------------
# empty_seat / is_table / get_tableID
# ---------------------------------------------------------------------------
class TestCustomerEmptySeat:
    def test_empty_seat_returns_first_when_all_empty(self):
        assert int(_make_customer(2).empty_seat()) == 0

    def test_empty_seat_skips_used(self):
        c = _make_customer(2)
        c = c.replace(used=c.used.at[0].set(1))
        assert int(c.empty_seat()) == 1


class TestCustomerIsTable:
    def test_is_table_true(self):
        assert bool(_make_customer(2).is_table(jnp.array([2, 3])))

    def test_is_table_false(self):
        assert not bool(_make_customer(2).is_table(jnp.array([0, 0])))

    def test_get_table_id_second_seat(self):
        assert int(_make_customer(2).get_tableID(jnp.array([4, 5]))) == 1


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------
class TestCustomerAppend:
    def test_append_sets_sitting_status(self):
        c2 = _make_customer(2).append(time=jnp.array(5), is_reserved=False)
        assert int(c2.status[0]) == int(CustomerStatus.sitting)

    def test_append_marks_seat_as_used(self):
        c2 = _make_customer(2).append(time=jnp.array(5), is_reserved=False)
        assert int(c2.used[0]) == 1

    def test_append_records_time(self):
        c2 = _make_customer(2).append(time=jnp.array(10), is_reserved=False)
        assert int(c2.time[0]) == 10


# ---------------------------------------------------------------------------
# leave
# ---------------------------------------------------------------------------
class TestCustomerLeave:
    def test_leave_no_food_clears_seat(self):
        c = _make_customer(2)
        c = c.replace(
            used=c.used.at[0].set(1),
            status=c.status.at[0].set(CustomerStatus.waiting_check),
        )
        c2 = c.leave(0)
        assert int(c2.used[0]) == 0
        assert int(c2.status[0]) == int(CustomerStatus.empty)

    def test_leave_with_food_sets_cleaning(self):
        plate = DynamicObject.get_clean_plates(1)
        c = _make_customer(2)
        c = c.replace(
            used=c.used.at[0].set(1),
            status=c.status.at[0].set(CustomerStatus.waiting_check),
            food=c.food.at[0, 0].set(plate),
        )
        c2 = c.leave(0)
        assert int(c2.status[0]) == int(CustomerStatus.cleaning)
        assert int(c2.used[0]) == 1


# ---------------------------------------------------------------------------
# put_dish_on_table
# ---------------------------------------------------------------------------
class TestCustomerPutDishOnTable:
    def test_put_dish_sets_eating_status(self):
        c = _make_customer(2)
        c = c.replace(
            used=c.used.at[0].set(1),
            status=c.status.at[0].set(CustomerStatus.waiting_food),
        )
        plate = DynamicObject.get_clean_plates(1)
        c2 = c.put_dish_on_table(0, plate, 0)
        assert int(c2.status[0]) == int(CustomerStatus.eating_food)

    def test_put_dish_stores_food(self):
        plate = DynamicObject.get_clean_plates(1)
        c2 = _make_customer(2).put_dish_on_table(0, plate, 0)
        assert int(c2.food[0, 0]) == int(plate)

    def test_put_dish_clears_ordered_menu_slot(self):
        plate = DynamicObject.get_clean_plates(1)
        c = _make_customer(2)
        c = c.replace(ordered_menu=c.ordered_menu.at[0, 0].set(plate))
        c2 = c.put_dish_on_table(0, plate, 0)
        assert int(c2.ordered_menu[0, 0]) == -1


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------
class TestCustomerCleanup:
    def _seated_with_plate(self) -> Customer:
        plate = DynamicObject.get_clean_plates(1)
        c = _make_customer(2, order_max=2)
        return c.replace(
            used=c.used.at[0].set(1),
            status=c.status.at[0].set(CustomerStatus.cleaning),
            food=c.food.at[0, 0].set(plate),
        )

    def test_cleanup_removes_plate_from_food(self):
        _pickup, c2 = self._seated_with_plate().cleanup(0)
        assert int(c2.food[0, 0]) == int(DynamicObject.EMPTY)

    def test_cleanup_final_plate_clears_seat(self):
        _pickup, c2 = self._seated_with_plate().cleanup(0)
        assert int(c2.status[0]) == int(CustomerStatus.empty)
        assert int(c2.used[0]) == 0

    def test_cleanup_pickup_is_plate(self):
        pickup, _ = self._seated_with_plate().cleanup(0)
        assert bool(DynamicObject.is_plate(pickup))


class TestCustomerDiscribeCustomer:
    def test_returns_string(self, menu_cfg):
        menu = MenuList.load(menu_cfg)
        result = _make_customer(2).discribe_customer(menu)
        assert isinstance(result, str)

    def test_contains_empty_status_label(self, menu_cfg):
        menu = MenuList.load(menu_cfg)
        result = _make_customer(2).discribe_customer(menu)
        assert "空席" in result

    def test_contains_order_info_when_waiting_food(self, menu_cfg):
        menu = MenuList.load(menu_cfg)
        food = menu.order_to_complete_food(0)
        c = _make_customer(2)
        c = c.replace(
            status=c.status.at[0].set(CustomerStatus.waiting_food),
            ordered_menu=c.ordered_menu.at[0, 0].set(food),
        )
        assert "注文" in c.discribe_customer(menu)

    def test_contains_food_info_when_eating(self, menu_cfg):
        menu = MenuList.load(menu_cfg)
        food = menu.order_to_complete_food(0)
        c = _make_customer(2)
        c = c.replace(
            status=c.status.at[0].set(CustomerStatus.eating_food),
            food=c.food.at[0, 0].set(food),
        )
        assert "配膳" in c.discribe_customer(menu)

    def test_contains_food_info_when_waiting_check(self, menu_cfg):
        menu = MenuList.load(menu_cfg)
        food = menu.order_to_complete_food(0)
        c = _make_customer(2)
        c = c.replace(
            status=c.status.at[0].set(CustomerStatus.waiting_check),
            food=c.food.at[0, 0].set(food),
        )
        assert "配膳" in c.discribe_customer(menu)


class TestCustomerLineStr:
    def test_str_returns_string(self):
        assert isinstance(str(_make_customer_line()), str)

    def test_str_contains_general_count(self):
        assert "2" in str(_make_customer_line(reserved=0, general=2))

    def test_str_contains_reserved_count(self):
        assert "1" in str(_make_customer_line(reserved=1, general=0))


# ---------------------------------------------------------------------------
# CustomerLine.dequeue (予約客→一般客→空の優先順)
# ---------------------------------------------------------------------------
class TestCustomerLineDequeue:
    def test_dequeue_reserved_first(self):
        cl = _make_customer_line(reserved=1, general=1)
        cl2 = cl.dequeue()
        assert int(cl2.reserved_line_length) == 0
        assert int(cl2.line_length) == 1

    def test_dequeue_general_when_no_reserved(self):
        cl = _make_customer_line(reserved=0, general=2)
        cl2 = cl.dequeue()
        assert int(cl2.line_length) == 1

    def test_dequeue_empty_is_noop(self):
        cl = _make_customer_line(reserved=0, general=0)
        cl2 = cl.dequeue()
        assert int(cl2.line_length) == 0
        assert int(cl2.reserved_line_length) == 0

    def test_get_obs_shape(self):
        cl = _make_customer_line(reserved=1, general=2)
        obs = cl.get_obs()
        expected_len = cl.reserved_queued_time.shape[0] + cl.queued_time.shape[0]
        assert obs.shape == (expected_len,)


class TestRegisterLineStr:
    def test_str_empty_when_no_service(self):
        rl = RegisterLine(
            register_pos=jnp.array([[3, 6]], dtype=jnp.int32),
            queued_time=jnp.array(0, dtype=jnp.int32),
            service_time=jnp.array(0, dtype=jnp.int32),
        )
        assert str(rl) == ""

    def test_str_shows_info_when_service_active(self):
        rl = RegisterLine(
            register_pos=jnp.array([[3, 6]], dtype=jnp.int32),
            queued_time=jnp.array(5, dtype=jnp.int32),
            service_time=jnp.array(3, dtype=jnp.int32),
        )
        assert "会計" in str(rl)

    def test_str_contains_queued_time(self):
        rl = RegisterLine(
            register_pos=jnp.array([[3, 6]], dtype=jnp.int32),
            queued_time=jnp.array(7, dtype=jnp.int32),
            service_time=jnp.array(2, dtype=jnp.int32),
        )
        assert "7" in str(rl)
