from dataclasses import dataclass

import numpy as np

from environment.static_object import StaticObject

_CHAR_TO_STATIC_ITEM: dict[str, StaticObject] = {
    " ": StaticObject.EMPTY,
    "W": StaticObject.WALL,
    "C": StaticObject.COUNTER,
    "E": StaticObject.ENTRANCE,
    "R": StaticObject.REGISTER,
    "B": StaticObject.PLATE_PILE,
    "P": StaticObject.POT,
    "S": StaticObject.SINK,
    "T": StaticObject.TABLE,
    "c": StaticObject.CHAIR,
    "G": StaticObject.GARBAGE_CAN,
    "0": StaticObject.INGREDIENT_0,
    "1": StaticObject.INGREDIENT_1,
    "2": StaticObject.INGREDIENT_2,
    "3": StaticObject.INGREDIENT_3,
    "4": StaticObject.INGREDIENT_4,
    "5": StaticObject.INGREDIENT_5,
    "6": StaticObject.INGREDIENT_6,
    "7": StaticObject.INGREDIENT_7,
    "8": StaticObject.INGREDIENT_8,
    "9": StaticObject.INGREDIENT_9,
}


def _split_rows(grid: str) -> list[str]:
    if not isinstance(grid, str):
        msg = "Invalid layout, must be a string layout"
        raise TypeError(msg)
    rows = grid.split("\n")
    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]
    return rows


def _validate_grid(
    num_tables: int,
    num_chairs: int,
    entrance_positions: list[tuple[int, int]],
    register_positions: list[tuple[int, int]],
) -> None:
    if num_tables != num_chairs:
        msg = f"Table and Chair must match. ({num_tables}Tables, {num_chairs}Chairs.)"
        raise ValueError(msg)
    if len(entrance_positions) > 1:
        msg = "Multiple Entrance is not allowed."
        raise ValueError(msg)
    if len(register_positions) < 1:
        msg = "Register is not included in layout."
        raise ValueError(msg)


@dataclass
class Layout:
    # agent positions list of positions, tuples (y, x)
    agent_positions: list[tuple[int, int]]

    # width x height grid with static items
    static_objects: np.ndarray

    num_ingredients: int
    num_customers: int
    entrance_positions: list[tuple[int, int]]
    plate_positions: list[tuple[int, int]]
    table_positions: list[tuple[int, int]]
    chair_positions: list[tuple[int, int]]
    register_positions: list[tuple[int, int]]

    def __post_init__(self):
        if self.num_agents == 0:
            msg = "At least one agent position must be provided"
            raise ValueError(msg)
        if self.num_ingredients < 1:
            msg = "At least one ingredient must be available"
            raise ValueError(msg)

    @property
    def height(self) -> int:
        return self.static_objects.shape[0]

    @property
    def width(self) -> int:
        return self.static_objects.shape[1]

    @property
    def size_limit(self) -> int:
        return max(self.height, self.width)

    @property
    def num_agents(self) -> int:
        return len(self.agent_positions)

    @staticmethod
    def from_string(grid: str):
        """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols.

        A: agent
        W: wall
        C: counter
        E: entrance
        R: register
        B: plate (bowl) pile
        P: pot location
        S: sink
        T: table
        c: chair
        G: Garbage can
        0-9: Ingredient x pile
        ' ' (space) : empty cell
        """
        rows = _split_rows(grid)
        row_lens = [len(row) for row in rows]
        static_objects = np.zeros((len(rows), max(row_lens)), dtype=int)

        agent_positions: list[tuple[int, int]] = []
        entrance_positions: list[tuple[int, int]] = []
        plate_positions: list[tuple[int, int]] = []
        table_positions: list[tuple[int, int]] = []
        chair_positions: list[tuple[int, int]] = []
        register_positions: list[tuple[int, int]] = []
        num_ingredients = 0
        num_tables = 0
        num_chairs = 0

        for r, row in enumerate(rows):
            for c, char in enumerate(row):
                pos = [r, c]
                obj = _CHAR_TO_STATIC_ITEM.get(char, StaticObject.EMPTY)
                static_objects[r, c] = obj
                if char == "A":
                    agent_positions.append(pos)
                if obj == StaticObject.PLATE_PILE:
                    plate_positions.append(pos)
                elif obj == StaticObject.ENTRANCE:
                    entrance_positions.append(pos)
                elif obj == StaticObject.TABLE:
                    num_tables += 1
                    table_positions.append(pos)
                elif obj == StaticObject.CHAIR:
                    num_chairs += 1
                    chair_positions.append(pos)
                elif obj == StaticObject.REGISTER:
                    register_positions.append(pos)
                if StaticObject.is_ingredient_pile(obj):
                    ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                    num_ingredients = max(num_ingredients, ingredient_idx + 1)

        _validate_grid(num_tables, num_chairs, entrance_positions, register_positions)
        # TODO: add some sanity checks - e.g. agent must exist, surrounded by walls, etc.

        return Layout(
            agent_positions=agent_positions,
            static_objects=static_objects,
            num_ingredients=num_ingredients,
            num_customers=num_tables,
            entrance_positions=entrance_positions,
            plate_positions=plate_positions,
            table_positions=table_positions,
            chair_positions=chair_positions,
            register_positions=register_positions,
        )
