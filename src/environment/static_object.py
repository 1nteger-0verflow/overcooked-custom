from enum import IntEnum

from environment.dynamic_object import DynamicObject


class StaticObject(IntEnum):
    EMPTY = 0
    WALL = 1
    COUNTER = 2
    ENTRANCE = 3
    REGISTER = 4

    # Agents are only included in the observation grid
    AGENT = 5
    SELF_AGENT = 6

    POT = 7
    PLATE_PILE = 8
    SINK = 9

    TABLE = 10
    CHAIR = 11
    GARBAGE_CAN = 12
    INGREDIENT_PILE_BASE = 13
    INGREDIENT_0 = 13
    INGREDIENT_1 = 14
    INGREDIENT_2 = 15
    INGREDIENT_3 = 16
    INGREDIENT_4 = 17
    INGREDIENT_5 = 18
    INGREDIENT_6 = 19
    INGREDIENT_7 = 20
    INGREDIENT_8 = 21
    INGREDIENT_9 = 22

    @staticmethod
    def is_ingredient_pile(obj):
        return obj >= StaticObject.INGREDIENT_PILE_BASE

    @staticmethod
    def get_ingredient(obj):
        idx = obj - StaticObject.INGREDIENT_PILE_BASE
        return DynamicObject.ingredient(idx)

    @staticmethod
    def ingredient_pile(idx):
        return StaticObject.INGREDIENT_PILE_BASE + idx
