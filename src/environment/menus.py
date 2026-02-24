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

    def order_to_complete_food(self, menu_index: int):
        # メニュー番号から完成品の料理を構成(分量は不問)
        ingredients = self.menu[menu_index]
        food = DynamicObject.get_recipe_encoding(ingredients)
        return DynamicObject.PLATE | DynamicObject.COOKED | DynamicObject.set_count(food, 0)

    def get_duration(self, obj):
        ingredient_idxs = DynamicObject.get_ingredient_idx_list_jit(obj)
        # 調理中の食材の組み合わせがメニュー中にあるか
        is_menu_table = jnp.all(jnp.sort(self.menu) == ingredient_idxs, axis=1)
        menu_idx = jnp.argmax(is_menu_table)
        in_menu = jnp.any(is_menu_table)
        return jax.lax.cond(in_menu, lambda: (True, self.duration[menu_idx]), lambda: (False, 1))

    def get_volume(self, obj):
        ingredient_idxs = DynamicObject.get_ingredient_idx_list_jit(obj)
        # 調理中の食材の組み合わせがメニュー中にあるか
        is_menu_table = jnp.all(jnp.sort(self.menu) == ingredient_idxs, axis=1)
        menu_idx = jnp.argmax(is_menu_table)
        in_menu = jnp.any(is_menu_table)
        return jax.lax.cond(in_menu, lambda: (True, self.volume[menu_idx]), lambda: (False, 1))

    def correct(self, dish, ordered_menus: jnp.ndarray):
        # ordered_menusのなかにdishと分量の違いを除いて一致するものがあれば正しい
        # 分量はランダムに変動するので除外する
        # 料理が-1になることはないのでそのまま比較する
        delivered_dish = DynamicObject.set_count(dish, 0)
        is_correct_order = jnp.any(delivered_dish == ordered_menus)
        # 正しいレシピを提供したときのみ使用する
        correct_order_idx = jnp.argmax(delivered_dish == ordered_menus)
        is_correct = jnp.any(is_correct_order, axis=None)
        return is_correct, correct_order_idx
