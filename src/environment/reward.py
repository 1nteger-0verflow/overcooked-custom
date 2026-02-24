from enum import IntEnum, auto


# 報酬を獲得したときの行動ごとに集計できるようにする
class RewardType(IntEnum):
    MOVE = 0
    STAY = auto()
    FAIL_INTERACT = auto()
    INVITATION = auto()
    REFUSE_CUSTOMER = auto()
    TAKE_ORDER = auto()
    WASH_PLATE = auto()
    CHECKING = auto()
    CLEAN_DIRT = auto()
    FAIL_PICK_PLACE = auto()
    ADD_INGREDIENT = auto()
    PLATING = auto()
    DELIVERY = auto()
    RETRIEVE_PLATE = auto()
    CLEAN_TABLE = auto()
    PICKUP = auto()
    PLACE = auto()
    SOAK_PLATE = auto()
    DISPOSE = auto()
