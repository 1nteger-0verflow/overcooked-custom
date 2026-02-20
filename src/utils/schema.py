from typing import Any

from flax.struct import dataclass


@dataclass
class TrainConfig:
    NUM_SEEDS: int
    SEED: int
    LR: float
    ANNEAL_LR: bool
    LR_WARMUP: float
    NUM_ENVS: int
    MINIBATCH_SIZE: int
    NUM_UPDATE_EPOCHS: int
    NUM_TRAINING_STEPS: int
    REW_SHAPING_HORIZON: int
    TIMESTEPS: int
    FC_DIM_SIZE: int
    GRU_HIDDEN_DIM: int
    ACTIVATION: str
    CLIP_EPS: float
    GAMMA: float
    GAE_LAMBDA: float
    MAX_GRAD_NORM: float
    VF_COEF: float
    ENT_COEF: float
    RANDOM_AGENT_POS: bool
    CHECKPOINT_SAVE_DIR: str  # TODO: Check unuse
    CHECKPOINT_INTERVAL_STEP: int
    progress: bool
    visualize: bool
    aspect_row: int
    aspect_col: int


@dataclass
class CustomerConfig:
    patience_mean: int
    patience_std: int
    digestion_speed: int


@dataclass
class MenuConfig:
    recipe: list[int]
    duration: int
    volume: int


@dataclass
class ParameterConfig:
    forward_view_size: list[int]
    side_view_size: list[int]
    capacity: list[int]
    plate_count: int
    sink_capacity: int
    order_max: int
    wait_line_max: int
    dirt_appear_rate: float
    cooking_duration_range: list[float]
    volume_range: list[float]
    check_time_max: int


@dataclass
class PenaltyConfig:
    ineffective_interaction: float


@dataclass
class ShapedRewardConfig:
    invite_customer: float
    refuse_customer: float
    take_order: float
    retrieve_plate: float
    clean_table: float
    soak_plate: float
    wash_plate: float
    finish_payment: float
    clean_dirt: float
    placement_in_pot: float  # TODO: unuse ?
    pot_start_cooking: float  # TODO: unuse ?
    dish_pickup: float


@dataclass
class RewardConfig:
    delivery_reward: float
    shaped_reward: ShapedRewardConfig
    penalty: PenaltyConfig


@dataclass
class ScheduleConfig:
    opening_time: int
    closing_time: int
    terminal_time: int
    reservation: list[int]
    congestion_rates: list[tuple[int, int]]


@dataclass
class EnvConfig:
    parameter: ParameterConfig
    reward: RewardConfig
    schedule: ScheduleConfig
    menu: list[MenuConfig]
    customer: CustomerConfig  # TODO: check unuse
    layout: str


@dataclass
class FileConfig:
    env: EnvConfig
    train: TrainConfig
    progress: bool
    visualize: bool
    aspect_row: int
    aspect_col: int


@dataclass
class IppoConfig(TrainConfig):
    NUM_ACTORS: int
    NUM_MINIBATCHES: int
    MODEL_DIR: str


@dataclass
class PlayOption:
    verbose: bool
    observation: bool
    seed: list[int]
    confirm: bool
    visualize: bool
    loop: bool
    profile: bool
    log: bool
    log_dir: str
    random_agent_position: bool


@dataclass
class PlayConfig:
    env: EnvConfig
    interface: list[dict[str, Any]]
    option: PlayOption
    player: list[str]
    ui: dict[str, Any]
