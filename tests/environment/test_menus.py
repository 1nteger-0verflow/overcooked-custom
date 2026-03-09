"""Tests for environment.menus."""

import chex
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from environment.dynamic_object import DynamicObject
from environment.menus import MenuList


def _make_menu(recipes, durations, volumes) -> MenuList:
    cfg = OmegaConf.create([{"recipe": r, "duration": d, "volume": v} for r, d, v in zip(recipes, durations, volumes)])
    return MenuList.load(cfg)


@pytest.fixture
def simple_menu():
    return _make_menu(recipes=[[0, 0, 1], [0, 1, 1], [1, 1, 2]], durations=[2, 5, 8], volumes=[4, 6, 10])


class TestMenuListLoad:
    def test_num_menus(self, simple_menu):
        assert simple_menu.num_menus == 3

    def test_menu_shape(self, simple_menu):
        chex.assert_shape(simple_menu.menu, (3, 3))

    def test_duration_shape(self, simple_menu):
        chex.assert_shape(simple_menu.duration, (3,))

    def test_volume_shape(self, simple_menu):
        chex.assert_shape(simple_menu.volume, (3,))

    def test_menu_values(self, simple_menu):
        row0 = list(map(int, simple_menu.menu[0]))
        assert row0 == [0, 0, 1]

    def test_empty_menu_raises(self):
        with pytest.raises(ValueError, match="At least one recipe"):
            _make_menu([], [], [])


class TestMenuListOrder:
    def test_order_returns_correct_row(self, simple_menu):
        result = simple_menu.order(0)
        chex.assert_shape(result, (3,))
        assert list(map(int, result)) == [0, 0, 1]

    def test_order_invalid_index_raises(self, simple_menu):
        with pytest.raises(ValueError, match="invalid order"):
            simple_menu.order(99)

    def test_order_to_ingredients(self, simple_menu):
        result = simple_menu.order_to_ingredients(1)
        assert list(map(int, result)) == [0, 1, 1]


class TestOrderToCompleteFood:
    def test_complete_food_has_plate_flag(self, simple_menu):
        food = simple_menu.order_to_complete_food(0)
        assert bool(DynamicObject.is_plate(food))

    def test_complete_food_has_cooked_flag(self, simple_menu):
        food = simple_menu.order_to_complete_food(0)
        assert bool(DynamicObject.is_cooked(food))

    def test_complete_food_count_is_zero(self, simple_menu):
        food = simple_menu.order_to_complete_food(0)
        assert int(DynamicObject.get_count(food)) == 0

    def test_different_recipes_different_food(self, simple_menu):
        food0 = simple_menu.order_to_complete_food(0)
        food1 = simple_menu.order_to_complete_food(1)
        assert int(food0) != int(food1)


class TestGetDuration:
    def test_valid_recipe_returns_in_menu_true(self, simple_menu):
        # recipe [0,0,1] → build the pot contents
        pot = DynamicObject.EMPTY
        pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(0))
        pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(0))
        pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(1))
        in_menu, duration = simple_menu.get_duration(pot)
        assert bool(in_menu)
        assert int(duration) == 2

    def test_unknown_recipe_returns_in_menu_false(self, simple_menu):
        # recipe [2,2,2] はメニューにない
        pot = DynamicObject.EMPTY
        for _ in range(3):
            pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(2))
        in_menu, _ = simple_menu.get_duration(pot)
        assert not bool(in_menu)


class TestGetVolume:
    def test_valid_recipe_returns_correct_volume(self, simple_menu):
        pot = DynamicObject.EMPTY
        pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(1))
        pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(1))
        pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(2))
        in_menu, volume = simple_menu.get_volume(pot)
        assert bool(in_menu)
        assert int(volume) == 10  # recipe [1,1,2] → volume=10

    def test_unknown_recipe_returns_false(self, simple_menu):
        pot = DynamicObject.ingredient(0)  # 1個だけ
        in_menu, _ = simple_menu.get_volume(pot)
        assert not bool(in_menu)


class TestCorrect:
    def test_correct_dish_is_accepted(self, simple_menu):
        # メニュー0番の完成品を注文リストと照合
        food = simple_menu.order_to_complete_food(0)
        ordered = jnp.array([food, -1])
        is_correct, idx = simple_menu.correct(food, ordered)
        assert bool(is_correct)
        assert int(idx) == 0

    def test_wrong_dish_is_rejected(self, simple_menu):
        food0 = simple_menu.order_to_complete_food(0)
        food1 = simple_menu.order_to_complete_food(1)
        ordered = jnp.array([food1, -1])
        is_correct, _ = simple_menu.correct(food0, ordered)
        assert not bool(is_correct)

    def test_volume_variation_is_ignored(self, simple_menu):
        food = simple_menu.order_to_complete_food(0)
        # 分量を変えた料理を正しい注文として検証
        food_with_volume = DynamicObject.set_count(food, 3)
        ordered = jnp.array([food, -1])
        is_correct, _ = simple_menu.correct(food_with_volume, ordered)
        assert bool(is_correct)
