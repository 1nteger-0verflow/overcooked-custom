import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode, dataclass

from environment.dynamic_object import DynamicObject


@dataclass
class MenuList(PyTreeNode):
    menu: jnp.ndarray
    duration: jnp.ndarray
    volume: jnp.ndarray

    def __post_init__(self):
        if self.menu.shape[0] < 1:
            raise ValueError("At least one recipe must be provided")

    @property
    def num_menus(self) -> int:
        return self.menu.shape[0]

    def order(self, menu_index: int) -> jax.Array:
        if menu_index >= self.num_menus:
            raise ValueError("invalid order")
        return self.menu[menu_index]

    @staticmethod
    def load(menus):
        menu = jnp.array([config.recipe for config in menus], dtype=int)
        duration = jnp.array([config.duration for config in menus], dtype=int)
        volume = jnp.array([config.volume for config in menus], dtype=int)
        return MenuList(menu=menu, duration=duration, volume=volume)

    def __str__(self):
        expr = "---- Menu ----\n"
        for i in range(self.num_menus):
            expr += f"({i}) 素材:{self.menu[i]}, 調理時間:{self.duration[i]}, 量:{self.volume[i]}\n"
        return expr

    def order_to_ingredients(self, order_idx: int):
        return self.menu[order_idx]

    def get_duration(self, obj):
        ingredient_idxs = DynamicObject.get_ingredient_idx_list_jit(obj)
        # 調理中の食材の組み合わせがメニュー中にあるか
        is_menu_table = jnp.all(jnp.sort(self.menu) == ingredient_idxs, axis=1)
        menu_idx = jnp.argmax(is_menu_table)
        in_menu = jnp.any(is_menu_table)
        return jax.lax.cond(
            in_menu, lambda: (True, self.duration[menu_idx]), lambda: (False, 1)
        )

    def get_volume(self, obj):
        ingredient_idxs = DynamicObject.get_ingredient_idx_list_jit(obj)
        # 調理中の食材の組み合わせがメニュー中にあるか
        is_menu_table = jnp.all(jnp.sort(self.menu) == ingredient_idxs, axis=1)
        menu_idx = jnp.argmax(is_menu_table)
        in_menu = jnp.any(is_menu_table)
        return jax.lax.cond(
            in_menu, lambda: (True, self.volume[menu_idx]), lambda: (False, 1)
        )

    def correct(self, dish, ordered_menus: jnp.ndarray):
        def _get_orders(i, ordered_menu_recipes):
            return jax.lax.cond(
                ordered_menus[i] >= 0,
                lambda: ordered_menu_recipes.at[i].set(
                    jnp.sort(self.menu[ordered_menus[i]])
                ),
                lambda: ordered_menu_recipes.at[i].set(-1),
            )

        order_nums = ordered_menus.shape[0]
        correct_recipes = jax.lax.fori_loop(
            0, order_nums, _get_orders, jnp.zeros((order_nums, 3), dtype=int)
        )
        # idx順になるので正解レシピもソートして格納する必要がある(jnp.sort)
        ingredient_idxs = DynamicObject.get_ingredient_idx_list_jit(dish)
        # jax.debug.print("ingredient in dish:{}", ingredient_idxs)
        # jax.debug.print("correct recipes: {}", correct_recipes)
        is_correct_order = jnp.all(correct_recipes == ingredient_idxs, axis=1)
        correct_order_idx = jnp.argmax(
            is_correct_order
        )  # 正しいレシピを提供したときのみ使用する
        is_correct = jnp.any(is_correct_order, axis=None)
        return is_correct, correct_order_idx
