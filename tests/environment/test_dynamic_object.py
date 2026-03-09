"""Tests for environment.dynamic_object.

チェック対象:
- DynamicObject のビット演算ユーティリティ全体
- ingredient_count / pick / place / add_ingredient の境界値
- chex によるArray形状・型検証
- jax.jit 互換性
"""
import chex
import jax
import jax.numpy as jnp
import pytest

from environment.dynamic_object import MAX_INGREDIENTS, DynamicObject


class TestIngredientEncoding:
    """ingredient() と is_ingredient() の整合性."""

    @pytest.mark.parametrize("idx", range(5))
    def test_ingredient_is_nonzero(self, idx):
        ing = DynamicObject.ingredient(idx)
        assert ing != 0

    @pytest.mark.parametrize("idx", range(5))
    def test_is_ingredient_true(self, idx):
        ing = DynamicObject.ingredient(idx)
        assert bool(DynamicObject.is_ingredient(ing))

    def test_is_ingredient_false_for_empty(self):
        assert not bool(DynamicObject.is_ingredient(DynamicObject.EMPTY))

    def test_is_ingredient_false_for_plate(self):
        plate = DynamicObject.get_clean_plates(3)
        assert not bool(DynamicObject.is_ingredient(plate))

    def test_different_ingredients_have_different_encodings(self):
        encs = [DynamicObject.ingredient(i) for i in range(5)]
        assert len(set(encs)) == 5


class TestIngredientCount:
    """ingredient_count の境界値テスト."""

    def test_empty_has_zero_ingredients(self):
        assert int(DynamicObject.ingredient_count(DynamicObject.EMPTY)) == 0

    def test_single_ingredient(self):
        ing = DynamicObject.ingredient(0)
        assert int(DynamicObject.ingredient_count(ing)) == 1

    def test_add_two_same_type(self):
        pot = DynamicObject.EMPTY
        pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(0))
        pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(0))
        assert int(DynamicObject.ingredient_count(pot)) == 2

    def test_add_max_ingredients(self):
        pot = DynamicObject.EMPTY
        for _ in range(MAX_INGREDIENTS):
            pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(0))
        assert int(DynamicObject.ingredient_count(pot)) == MAX_INGREDIENTS

    def test_ingredient_count_jit_compatible(self):
        fn = jax.jit(DynamicObject.ingredient_count)
        result = fn(DynamicObject.ingredient(1))
        assert int(result) == 1


class TestGetIngredientIdx:
    """get_ingredient_idx の正確性テスト."""

    @pytest.mark.parametrize("idx", range(5))
    def test_single_ingredient_idx(self, idx):
        ing = DynamicObject.ingredient(idx)
        assert int(DynamicObject.get_ingredient_idx(ing)) == idx

    def test_empty_returns_minus_one(self):
        assert int(DynamicObject.get_ingredient_idx(DynamicObject.EMPTY)) == -1


class TestGetIngredientIdxListJit:
    """get_ingredient_idx_list_jit の形状・値テスト."""

    def test_empty_returns_all_minus_one(self):
        result = DynamicObject.get_ingredient_idx_list_jit(DynamicObject.EMPTY)
        chex.assert_shape(result, (MAX_INGREDIENTS,))
        assert all(int(v) == -1 for v in result)

    def test_single_ingredient_first_slot(self):
        ing = DynamicObject.ingredient(0)
        result = DynamicObject.get_ingredient_idx_list_jit(ing)
        chex.assert_shape(result, (MAX_INGREDIENTS,))
        assert int(result[0]) == 0

    def test_three_same_ingredients(self):
        pot = DynamicObject.EMPTY
        for _ in range(MAX_INGREDIENTS):
            pot = DynamicObject.add_ingredient(pot, DynamicObject.ingredient(1))
        result = DynamicObject.get_ingredient_idx_list_jit(pot)
        chex.assert_shape(result, (MAX_INGREDIENTS,))
        assert all(int(v) == 1 for v in result)

    def test_returns_jax_array(self):
        result = DynamicObject.get_ingredient_idx_list_jit(DynamicObject.ingredient(0))
        assert isinstance(result, jax.Array)


class TestCount:
    """get_count と set_count の整合性テスト."""

    def test_get_count_zero(self):
        assert int(DynamicObject.get_count(DynamicObject.EMPTY)) == 0

    def test_get_clean_plates_count(self):
        plates = DynamicObject.get_clean_plates(5)
        assert int(DynamicObject.get_count(plates)) == 5

    def test_set_count_preserves_flags(self):
        plate = DynamicObject.get_clean_plates(3)
        new_plate = DynamicObject.set_count(plate, 7)
        assert int(DynamicObject.get_count(new_plate)) == 7
        assert bool(DynamicObject.is_plate(new_plate))

    def test_set_count_zero(self):
        plate = DynamicObject.get_clean_plates(3)
        new_plate = DynamicObject.set_count(plate, 0)
        assert int(DynamicObject.get_count(new_plate)) == 0


class TestPick:
    """pick の返り値テスト."""

    def test_pick_from_empty_returns_empty(self):
        picked, remaining = DynamicObject.pick(DynamicObject.EMPTY)
        assert int(picked) == int(DynamicObject.EMPTY)
        assert int(remaining) == int(DynamicObject.EMPTY)

    def test_pick_single_plate(self):
        plates = DynamicObject.get_clean_plates(1)
        picked, remaining = DynamicObject.pick(plates)
        assert bool(DynamicObject.is_plate(picked))
        assert int(remaining) == int(DynamicObject.EMPTY)

    def test_pick_from_multiple_decrements_count(self):
        plates = DynamicObject.get_clean_plates(3)
        _, remaining = DynamicObject.pick(plates)
        assert int(DynamicObject.get_count(remaining)) == 2

    def test_pick_returns_one_item(self):
        plates = DynamicObject.get_clean_plates(3)
        picked, _ = DynamicObject.pick(plates)
        assert int(DynamicObject.get_count(picked)) == 1


class TestPlace:
    """place の同種スタック・異種拒否テスト."""

    def test_place_on_empty_succeeds(self):
        # stack=EMPTY(count=0) に obj(1枚) を置く → count は stack(0)+1 = 1
        plate = DynamicObject.get_clean_plates(1)
        leftover, result = DynamicObject.place(DynamicObject.EMPTY, plate)
        assert int(leftover) == int(DynamicObject.EMPTY)
        assert int(DynamicObject.get_count(result)) == 1

    def test_place_same_type_stacks(self):
        plates_on_counter = DynamicObject.get_clean_plates(2)
        plate_in_hand = DynamicObject.get_clean_plates(1)
        leftover, result = DynamicObject.place(plates_on_counter, plate_in_hand)
        assert int(leftover) == int(DynamicObject.EMPTY)
        assert int(DynamicObject.get_count(result)) == 3

    def test_place_different_type_stays(self):
        counter_obj = DynamicObject.ingredient(0)
        in_hand = DynamicObject.get_clean_plates(1)
        leftover, result = DynamicObject.place(counter_obj, in_hand)
        # 異種は置けないので手持ちが残る
        assert int(leftover) == int(in_hand)
        assert int(result) == int(counter_obj)


class TestDirtOperations:
    """汚れ関連メソッドのテスト."""

    def test_create_dirt_positive(self):
        dirt = DynamicObject.create_dirt(jnp.array(3))
        assert bool(DynamicObject.is_dirt(dirt))
        assert int(DynamicObject.get_count(dirt)) == 3

    def test_create_dirt_zero_returns_empty(self):
        result = DynamicObject.create_dirt(jnp.array(0))
        assert int(result) == 0

    def test_is_dirt_false_for_plate(self):
        plate = DynamicObject.get_clean_plates(1)
        assert not bool(DynamicObject.is_dirt(plate))

    def test_clean_dirt_decrements(self):
        dirt = DynamicObject.create_dirt(jnp.array(3))
        cleaned = DynamicObject.clean_dirt(dirt)
        assert int(DynamicObject.get_count(cleaned)) == 2
        assert bool(DynamicObject.is_dirt(cleaned))

    def test_clean_dirt_to_zero_removes_dirt(self):
        dirt = DynamicObject.create_dirt(jnp.array(1))
        cleaned = DynamicObject.clean_dirt(dirt)
        assert not bool(DynamicObject.is_dirt(cleaned))

    def test_clean_dirt_does_not_affect_non_dirt(self):
        plate = DynamicObject.get_clean_plates(2)
        result = DynamicObject.clean_dirt(plate)
        assert int(result) == int(plate)


class TestPlateFlags:
    """is_plate / is_cooked / get_clean_plates テスト."""

    def test_is_plate_true(self):
        plate = DynamicObject.get_clean_plates(1)
        assert bool(DynamicObject.is_plate(plate))

    def test_is_plate_false_for_ingredient(self):
        assert not bool(DynamicObject.is_plate(DynamicObject.ingredient(0)))

    def test_is_cooked_false_for_clean_plate(self):
        plate = DynamicObject.get_clean_plates(1)
        assert not bool(DynamicObject.is_cooked(plate))

    def test_is_cooked_flag(self):
        cooked = int(DynamicObject.PLATE) | int(DynamicObject.COOKED) | 1
        assert bool(DynamicObject.is_cooked(cooked))


class TestGetRecipeEncoding:
    """get_recipe_encoding の冪等性テスト."""

    def test_encoding_is_scalar(self):
        recipe = jnp.array([0, 0, 1])
        result = DynamicObject.get_recipe_encoding(recipe)
        assert result.ndim == 0

    def test_encoding_depends_on_recipe(self):
        recipe_a = jnp.array([0, 0, 1])
        recipe_b = jnp.array([0, 1, 1])
        assert int(DynamicObject.get_recipe_encoding(recipe_a)) != int(DynamicObject.get_recipe_encoding(recipe_b))
