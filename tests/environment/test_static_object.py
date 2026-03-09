"""Tests for environment.static_object."""

import pytest

from environment.dynamic_object import DynamicObject
from environment.static_object import StaticObject


class TestStaticObjectValues:
    """Enum値の正常系テスト."""

    def test_empty_is_zero(self):
        assert StaticObject.EMPTY == 0

    def test_wall_value(self):
        assert StaticObject.WALL == 1

    def test_ingredient_pile_base(self):
        assert StaticObject.INGREDIENT_PILE_BASE == 13

    def test_ingredient_0_equals_base(self):
        assert StaticObject.INGREDIENT_0 == StaticObject.INGREDIENT_PILE_BASE

    def test_ingredient_9_equals_base_plus_9(self):
        assert StaticObject.INGREDIENT_9 == StaticObject.INGREDIENT_PILE_BASE + 9

    def test_all_values_unique(self):
        values = [m.value for m in StaticObject]
        assert len(values) == len(set(values))


class TestIsIngredientPile:
    """is_ingredient_pile の境界値テスト."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (StaticObject.INGREDIENT_PILE_BASE - 1, False),  # 境界直下
            (StaticObject.INGREDIENT_PILE_BASE, True),  # 境界 (0番食材)
            (StaticObject.INGREDIENT_PILE_BASE + 5, True),  # 中間
            (StaticObject.INGREDIENT_9, True),  # 最大食材
            (StaticObject.EMPTY, False),
            (StaticObject.WALL, False),
            (StaticObject.COUNTER, False),
        ],
    )
    def test_is_ingredient_pile(self, obj, expected):
        assert StaticObject.is_ingredient_pile(obj) == expected


class TestGetIngredient:
    """get_ingredient: 食材IDを返す整合性テスト."""

    @pytest.mark.parametrize("idx", range(10))
    def test_get_ingredient_idx(self, idx):
        pile = StaticObject.ingredient_pile(idx)
        result = StaticObject.get_ingredient(pile)
        # get_ingredient はDynamicObject.ingredient(idx) を返す
        expected = DynamicObject.ingredient(idx)
        assert int(result) == int(expected)


class TestIngredientPile:
    """ingredient_pile: 食材置き場IDの計算テスト."""

    @pytest.mark.parametrize("idx", range(10))
    def test_ingredient_pile_roundtrip(self, idx):
        pile = StaticObject.ingredient_pile(idx)
        assert pile == StaticObject.INGREDIENT_PILE_BASE + idx

    def test_ingredient_pile_zero(self):
        assert StaticObject.ingredient_pile(0) == StaticObject.INGREDIENT_0

    def test_ingredient_pile_nine(self):
        assert StaticObject.ingredient_pile(9) == StaticObject.INGREDIENT_9
