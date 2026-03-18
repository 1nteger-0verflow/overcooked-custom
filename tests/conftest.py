"""Shared fixtures for all test modules."""

import jax
import pytest
from omegaconf import OmegaConf

from environment.overcooked import OvercookedCustom

# ---------------------------------------------------------------------------
# Layout strings
# ---------------------------------------------------------------------------
# 1エージェント・最小構成レイアウト (7x5)
MINIMAL_LAYOUT = """\
WBTcREW
W     W
W  A  W
0     W
WWWWWWW"""

# 2エージェント・フル機能レイアウト (10x8)
COMPACT_LAYOUT = """\
WWWWWWWWWW
W  BGRE  W
WCP   ATcW
W0A2   TcW
W1     TcW
W        W
WCCCCSCCCW
WWWWWWWWWW"""


# ---------------------------------------------------------------------------
# Menu configs (OmegaConf)
# ---------------------------------------------------------------------------
_MENU_CFG = OmegaConf.create(
    [{"recipe": [0, 0, 0], "duration": 2, "volume": 4}, {"recipe": [0, 0, 1], "duration": 5, "volume": 6}]
)

_MENU_CFG_3TYPES = OmegaConf.create(
    [
        {"recipe": [0, 0, 1], "duration": 2, "volume": 4},
        {"recipe": [0, 1, 2], "duration": 8, "volume": 22},
        {"recipe": [1, 1, 2], "duration": 5, "volume": 6},
    ]
)


def _make_config(layout: str, menu_cfg: OmegaConf, num_agents: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "layout": layout,
            "menu": OmegaConf.to_container(menu_cfg),
            "parameter": {
                "forward_view_size": [0] * num_agents,
                "side_view_size": [0] * num_agents,
                "restrict_observation": False,
                "capacity": [3] * num_agents,
                "plate_count": 5,
                "sink_capacity": 5,
                "order_max": 2,
                "wait_line_max": 3,
                "dirt_appear_rate": 0.0,
                "dirtiness_max": 3,
                "cooking_duration_range": [1.0, 1.0],
                "volume_range": [1.0, 1.0],
                "check_time_max": 5,
            },
            "reward": {
                "original_reward": {"finish_payment": 20.0},
                "shaped_reward": {
                    "invite_customer": 3.0,
                    "refuse_customer": 0.1,
                    "take_order": 3.0,
                    "wash_plate": 3.0,
                    "process_payment": 3.0,
                    "clean_dirt": 3.0,
                    "pickup_ingredient_from_pile": 0.5,
                    "pickup_ingredient_from_counter": 0.0,
                    "pickup_new_plate_from_pile": 0.5,
                    "pickup_new_plate_from_counter": 0.0,
                    "plate_dish": 6.0,
                    "pickup_dish_from_counter": 0.0,
                    "pickup_used_plate_from_table": 3.0,
                    "clean_table": 3.0,
                    "pickup_used_plate_from_counter": 0.0,
                    "placement_in_pot": 3.0,
                    "pot_start_cooking": 3.0,
                    "place_ingredient_on_counter": -0.2,
                    "dispose_ingredient": -2.0,
                    "deliver_food": 8.0,
                    "place_food_on_counter": -0.2,
                    "dispose_food": -10.0,
                    "place_new_plate_on_counter": -0.2,
                    "place_used_plate_on_counter": -0.2,
                    "soak_plate": 3.0,
                },
                "penalty": {
                    "ineffective_interaction": 0.0,
                    "ineffective_pickup": 0.001,
                    "ineffective_placement": 0.001,
                    "erroneous_cooking": 1.0,
                    "erroneous_delivery": 1.0,
                    "step_cost": 0.0,
                    "block_cost": 0.0,
                },
                "discount": {
                    "decline_deliver_reward": False,
                    "deliver_ramp_step": 10,
                    "deliver_limit_step": 100,
                    "deliver_discount_rate": 0.8,
                },
            },
            "schedule": {
                "opening_time": 5,
                "closing_time": 50,
                "terminal_time": 60,
                "reservation": [30],
                "congestion_rates": [[0, 0], [5, 50]],
            },
            "customer": {"patience_mean": 10, "patience_std": 4, "digestion_speed": 1},
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def prng_key():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="session")
def minimal_layout_str():
    return MINIMAL_LAYOUT


@pytest.fixture(scope="session")
def compact_layout_str():
    return COMPACT_LAYOUT


@pytest.fixture(scope="session")
def menu_cfg():
    return _MENU_CFG


@pytest.fixture(scope="session")
def menu_cfg_3types():
    return _MENU_CFG_3TYPES


@pytest.fixture(scope="session")
def minimal_config():
    """1エージェント・最小構成の設定."""
    return _make_config(MINIMAL_LAYOUT, _MENU_CFG, num_agents=1)


@pytest.fixture(scope="session")
def compact_config():
    """2エージェント・フル機能の設定."""
    return _make_config(COMPACT_LAYOUT, _MENU_CFG_3TYPES, num_agents=2)


@pytest.fixture(scope="session")
def minimal_env(minimal_config):
    return OvercookedCustom(minimal_config)


@pytest.fixture(scope="session")
def compact_env(compact_config):
    return OvercookedCustom(compact_config)


@pytest.fixture(scope="session")
def minimal_state(minimal_env, prng_key):
    _, state = minimal_env.reset(prng_key)
    return state


@pytest.fixture(scope="session")
def compact_state(compact_env, prng_key):
    _, state = compact_env.reset(prng_key)
    return state
