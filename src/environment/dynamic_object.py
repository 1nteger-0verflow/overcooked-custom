from enum import IntEnum

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

MAX_INGREDIENTS = 3  # = 2^2 -1 　素材1種類につき2bitを割り当てている理由
COUNTS_BIT_WIDTH = 6


class Digits(IntEnum):
    # 何ビット目が何の管理箇所かを示す
    PLATE = COUNTS_BIT_WIDTH  # 6: 皿の有無
    COOKED = PLATE + 1  # 7: 料理が盛り付けられている
    USED = COOKED + 1  # 8: 食べ終わった皿のとき1
    DIRT = USED + 1  # 9: 床の汚れ
    INGREDIENTS = DIRT + 1  # 12 ~ 32:  最大10種類 * 2bit


class DynamicObject(IntEnum):
    EMPTY = 0
    PLATE = 1 << Digits.PLATE
    COOKED = 1 << Digits.COOKED
    USED = 1 << Digits.USED
    DIRT = 1 << Digits.DIRT

    # every ingredient has two unique bit
    BASE_INGREDIENT = 1 << Digits.INGREDIENTS

    @staticmethod
    def is_cooked(obj) -> bool:
        return (obj & DynamicObject.COOKED) != 0

    @staticmethod
    def ingredient(idx):
        # 素材1個を取り出す
        return (DynamicObject.BASE_INGREDIENT << 2 * idx) | 1

    @staticmethod
    def is_ingredient(obj) -> bool:
        return ((obj >> Digits.INGREDIENTS) != 0) & ((obj & DynamicObject.PLATE) == 0)

    @staticmethod
    def ingredient_count(obj):
        initial_val = (obj >> Digits.INGREDIENTS, jnp.array(0))

        # １つの食材に2bit充てることにより個数を0～3まで管理しているので、
        # 2bitずつずらして下位2bitをカウントしていく
        def _count_ingredients(x):
            obj, count = x
            return (obj >> 2, count + (obj & 0x3))

        _, count = jax.lax.while_loop(
            lambda x: x[0] > 0, _count_ingredients, initial_val
        )
        return count

    @staticmethod
    def add_ingredient(obj, add):  # obj: potの中身、add: エージェントが入れた素材(1つ)
        idx = DynamicObject.get_ingredient_idx(add)
        return obj + (DynamicObject.BASE_INGREDIENT << 2 * idx)

    @staticmethod
    def get_count(obj) -> int:
        return obj & (2**COUNTS_BIT_WIDTH - 1)

    @staticmethod
    def set_count(obj, new_count):
        # 個数のみ指定の値に変える
        return ((obj >> COUNTS_BIT_WIDTH) << COUNTS_BIT_WIDTH) + (
            new_count & (2**COUNTS_BIT_WIDTH - 1)
        )

    @staticmethod
    def pick(obj):
        # 皿、料理、素材を１つ取り出したときの取り出したものと残りのものを返す
        current_count = DynamicObject.get_count(obj)
        return jax.lax.switch(
            jnp.argmax(
                jnp.array(
                    [
                        current_count < 1,
                        current_count == 1,
                        current_count > 1,
                    ]
                )
            ),
            [
                lambda: (DynamicObject.EMPTY, obj),
                lambda: (obj, DynamicObject.EMPTY),
                lambda: (
                    DynamicObject.set_count(obj, 1),
                    DynamicObject.set_count(obj, current_count - 1),
                ),
            ],
        )

    @staticmethod
    def place(
        stack, obj
    ):  # stack: 既に置いてあるもの、 obj: エージェントが置こうとしているもの
        return jax.lax.cond(
            ((stack >> COUNTS_BIT_WIDTH) - (obj >> COUNTS_BIT_WIDTH) == 0)
            | (stack == DynamicObject.EMPTY),
            # 個数だけが違う場合(同種のオブジェクト)は、複数個置くことが可能
            lambda: (
                DynamicObject.EMPTY,
                DynamicObject.set_count(obj, DynamicObject.get_count(stack) + 1),
            ),
            # 違う種類のものだった場合は置けないのでそのまま
            lambda: (obj, stack),
        )

    @staticmethod
    def get_ingredient_idx_list_jit(obj):
        def _loop_body(carry):
            obj, pos, idx, res = carry
            count = obj & 0x3

            cond = jnp.arange(MAX_INGREDIENTS)
            cond = (cond >= pos) & (res == -1) & (cond < pos + count)

            res = jnp.where(
                cond,
                idx,
                res,
            )

            return (obj >> 2, pos + count, idx + 1, res)

        def _loop_cond(carry):
            obj, pos, _, _ = carry
            return (obj > 0) & (pos < MAX_INGREDIENTS)

        initial_res = jnp.full((MAX_INGREDIENTS,), -1, dtype=jnp.int32)
        carry = (obj >> Digits.INGREDIENTS, 0, 0, initial_res)

        val = jax.lax.while_loop(_loop_cond, _loop_body, carry)
        return val[-1]

    @staticmethod
    def get_ingredient_idx(obj):
        def _body_fun(val):
            obj, idx, res = val
            new_res = jax.lax.select(obj & 0x3 != 0, idx, res)
            return (obj >> 2, idx + 1, new_res)

        def _cond_fun(val):
            obj, _, res = val
            return (obj > 0) & (res == -1)

        initial_val = (obj >> Digits.INGREDIENTS, 0, -1)
        val = jax.lax.while_loop(_cond_fun, _body_fun, initial_val)
        return val[-1]

    @staticmethod
    def get_recipe_encoding(recipe: ArrayLike):
        ingredients = jax.vmap(DynamicObject.ingredient)(recipe)
        return jnp.sum(ingredients)

    @staticmethod
    def is_plate(obj):
        return (obj & DynamicObject.PLATE) != 0

    @staticmethod
    def get_clean_plates(count: int):
        plate_count = count & (2**COUNTS_BIT_WIDTH - 1)
        return plate_count | DynamicObject.PLATE

    @staticmethod
    def create_dirt(dirtiness: jnp.ndarray):
        return jax.lax.cond(
            dirtiness > 0,
            lambda: DynamicObject.DIRT | dirtiness,
            lambda: 0,
        )

    @staticmethod
    def is_dirt(obj):
        return obj & DynamicObject.DIRT > 0

    @staticmethod
    def clean_dirt(obj, efficiency: int = 1):
        def _clean(val):
            new_dirtiness = jnp.clip(DynamicObject.get_count(val) - efficiency, min=0)
            return DynamicObject.create_dirt(new_dirtiness)

        return jax.lax.cond(
            obj & DynamicObject.DIRT,
            _clean,
            lambda x: x,
            obj,
        )

    @staticmethod
    def decode(obj):
        if obj == DynamicObject.EMPTY:
            return ""
        expr = f"[{obj}]=>"
        count = DynamicObject.get_count(obj)
        ingredients = DynamicObject.get_ingredient_idx_list_jit(obj)
        if obj & DynamicObject.DIRT > 0:
            expr += f"汚れ レベル{count}"
        elif (
            len(ingredients) == 3
            and (obj & DynamicObject.COOKED > 0)
            and (obj & DynamicObject.PLATE > 0)
        ):
            expr += "料理/皿: " + ",".join(
                [str(ingredient) for ingredient in ingredients]
            )
            expr += f"（残り：{DynamicObject.get_count(obj)}）"
        elif (
            len(ingredients) == 3
            and (obj & DynamicObject.COOKED > 0)
            and (obj & DynamicObject.PLATE == 0)
        ):
            expr += "調理済: " + ",".join(
                [str(ingredient) for ingredient in ingredients]
            )
        elif len(ingredients) == 3 and (DynamicObject.COOKED == 0):
            expr += "調理中: " + ",".join(
                [str(ingredient) for ingredient in ingredients]
            )
        elif len(ingredients) == 2 or len(ingredients) == 1:
            expr += "食材: " + ",".join([str(ingredient) for ingredient in ingredients])
        elif obj & DynamicObject.PLATE > 0:
            if obj & DynamicObject.COOKED > 0:
                expr += "調理済み"
            if obj & DynamicObject.USED > 0:
                expr += "使用済み"
            expr += f"皿 {count}枚"
        return expr
