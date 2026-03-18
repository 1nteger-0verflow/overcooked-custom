from enum import IntEnum, auto


# 報酬を獲得したときの行動ごとに集計できるようにする
class RewardType(IntEnum):
    STAY = 0
    MOVE = auto()
    PICKUP = auto()
    PICKUP_INGREDIENT = auto()
    PICKUP_PLATE = auto()
    PLACE = auto()
    INVITATION = auto()
    REFUSE_CUSTOMER = auto()
    TAKE_ORDER = auto()
    ADD_INGREDIENT = auto()
    PLATING = auto()
    DELIVERY = auto()
    RETRIEVE_PLATE = auto()
    CLEAN_TABLE = auto()
    SOAK_PLATE = auto()
    WASH_PLATE = auto()
    CHECKING = auto()
    CLEAN_DIRT = auto()
    FAIL_INTERACT = auto()
    FAIL_PICK_PLACE = auto()
    DISPOSE = auto()
