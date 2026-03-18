"""Tests for environment.layouts."""

import jax.numpy as jnp
import pytest

from environment.layouts import Layout
from environment.static_object import StaticObject

# ---------------------------------------------------------------------------
# テスト用レイアウト文字列
# ---------------------------------------------------------------------------
MINIMAL = """\
WBTcREW
W     W
W  A  W
0     W
WWWWWWW"""

# ポット・シンクありの中型レイアウト
FULL = """\
WWWWWWWWWW
W  BGRE  W
WCP   ATcW
W0A2   TcW
W1     TcW
W        W
WCCCCSCCCW
WWWWWWWWWW"""

MULTI_ENTRANCE = """\
WBTcRW
W    E
W  A E
0    W
WWWWWW"""

NO_REGISTER = """\
WBTcEW
W    W
W  A W
0    W
WWWWWW"""

TABLE_CHAIR_MISMATCH = """\
WBTTcREW
W      W
W  A   W
0      W
WWWWWWWW"""


class TestFromStringMinimal:
    """最小レイアウトのパース結果テスト."""

    @pytest.fixture(scope="class")
    def layout(self):
        return Layout.from_string(MINIMAL)

    def test_height(self, layout):
        assert layout.height == 5

    def test_width(self, layout):
        assert layout.width == 7

    def test_num_agents(self, layout):
        assert layout.num_agents == 1

    def test_agent_position(self, layout):
        assert layout.agent_positions == [[2, 3]]

    def test_num_ingredients(self, layout):
        assert layout.num_ingredients == 1

    def test_num_customers(self, layout):
        assert layout.num_customers == 1

    def test_entrance_position(self, layout):
        assert len(layout.entrance_positions) == 1
        assert layout.entrance_positions[0].tolist() == [0, 5]

    def test_plate_pile_position(self, layout):
        assert layout.plate_positions[0].tolist() == [0, 1]

    def test_table_chair_match(self, layout):
        assert len(layout.table_positions) == len(layout.chair_positions) == 1

    def test_register_present(self, layout):
        assert len(layout.register_positions) == 1

    def test_static_objects_dtype(self, layout):
        assert jnp.issubdtype(layout.static_objects.dtype, jnp.integer)

    def test_wall_in_static_objects(self, layout):
        assert layout.static_objects[0, 0] == StaticObject.WALL

    def test_agent_cell_is_empty(self, layout):
        # エージェント位置はEMPTY として static_objects に入る
        assert layout.static_objects[2, 3] == StaticObject.EMPTY

    def test_ingredient_pile_in_static_objects(self, layout):
        assert layout.static_objects[3, 0] == StaticObject.INGREDIENT_0


class TestFromStringFull:
    """フル機能レイアウトのテスト."""

    @pytest.fixture(scope="class")
    def layout(self):
        return Layout.from_string(FULL)

    def test_num_agents(self, layout):
        assert layout.num_agents == 2

    def test_num_ingredients(self, layout):
        assert layout.num_ingredients == 3  # 0, 1, 2

    def test_num_customers(self, layout):
        assert layout.num_customers == 3

    def test_pot_in_static_objects(self, layout):
        # compact レイアウトの (2,2) がポット
        assert layout.static_objects[2, 2] == StaticObject.POT

    def test_sink_in_static_objects(self, layout):
        assert layout.static_objects[6, 5] == StaticObject.SINK

    def test_size_limit(self, layout):
        assert layout.size_limit == max(layout.height, layout.width)


class TestFromStringValidation:
    """不正レイアウトのエラー処理テスト."""

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            Layout.from_string(123)

    def test_multiple_entrances_raise(self):
        with pytest.raises(ValueError, match="Multiple Entrance"):
            Layout.from_string(MULTI_ENTRANCE)

    def test_no_register_raises(self):
        with pytest.raises(ValueError, match="Register"):
            Layout.from_string(NO_REGISTER)

    def test_table_chair_mismatch_raises(self):
        with pytest.raises(ValueError, match="Table and Chair"):
            Layout.from_string(TABLE_CHAIR_MISMATCH)

    def test_no_agent_raises(self):
        no_agent = """\
WBREW
W   W
W   W
0   W
WWWWW"""
        with pytest.raises(ValueError, match="agent"):
            Layout.from_string(no_agent)

    def test_no_ingredient_raises(self):
        no_ing = """\
WBREW
W   W
W A W
W   W
WWWWW"""
        with pytest.raises(ValueError, match="ingredient"):
            Layout.from_string(no_ing)


class TestLayoutProperties:
    def test_size_limit_equals_max_dimension(self):
        layout = Layout.from_string(MINIMAL)
        assert layout.size_limit == max(layout.height, layout.width)

    def test_static_objects_shape(self):
        layout = Layout.from_string(MINIMAL)
        assert layout.static_objects.shape == (layout.height, layout.width)
