import dataclasses
from dataclasses import dataclass, field

from flax import struct
from omegaconf import DictConfig


@dataclass(frozen=True)
class CustomerConfig:
    patience_mean: float = 10.0
    patience_std: float = 4.0
    digestion_speed: int = 1


@dataclass(frozen=True)
class EnvParameterConfig:
    forward_view_size: tuple[int, ...] = field(default_factory=lambda: (0, 3))
    side_view_size: tuple[int, ...] = field(default_factory=lambda: (2, 1))
    restrict_observation: bool = True
    capacity: tuple[int, ...] = field(default_factory=lambda: (3, 3))
    plate_count: int = 30
    sink_capacity: int = 5
    order_max: int = 3
    wait_line_max: int = 5
    dirt_appear_rate: float = 0.05
    dirtiness_max: int = 5
    cooking_duration_range: tuple[float, float] = field(default_factory=lambda: (0.5, 3.0))
    volume_range: tuple[float, float] = field(default_factory=lambda: (0.5, 2.0))
    check_time_max: int = 10


@dataclass(frozen=True)
class OriginalRewardConfig:
    finish_payment: float = 20.0


@dataclass(frozen=True)
class ShapedRewardConfig:
    invite_customer: float = 3.0
    refuse_customer: float = 0.1
    take_order: float = 3.0
    deliver_food: float = 3.0
    retrieve_plate: float = 3.0
    clean_table: float = 3.0
    soak_plate: float = 3.0
    wash_plate: float = 3.0
    process_payment: float = 3.0
    clean_dirt: float = 3.0
    placement_in_pot: float = 3.0
    pot_start_cooking: float = 3.0
    dish_pickup: float = 3.0


@dataclass(frozen=True)
class PenaltyConfig:
    ineffective_interaction: float = 0.0
    erroneous_delivery: float = 1.0
    step_cost: float = 0.0
    block_cost: float = 0.0


@dataclass(frozen=True)
class RewardConfig:
    original_reward: OriginalRewardConfig = field(default_factory=OriginalRewardConfig)
    shaped_reward: ShapedRewardConfig = field(default_factory=ShapedRewardConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)


@dataclass(frozen=True)
class ScheduleConfig:
    opening_time: int = 10
    closing_time: int = 720
    terminal_time: int = 780
    reservation: tuple[int, ...] = field(default_factory=lambda: (1, 10, 30))
    congestion_rates: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: ((0, 0), (10, 10), (20, 100), (25, 20), (30, 50), (35, 0), (50, 20))
    )


@dataclass(frozen=True)
class MenuItemConfig:
    recipe: tuple[int, ...] = field(default_factory=tuple)
    duration: int = 0
    volume: int = 0


@dataclass(frozen=True)
class EnvConfig:
    customer: CustomerConfig = field(default_factory=CustomerConfig)
    parameter: EnvParameterConfig = field(default_factory=EnvParameterConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    menu: tuple[MenuItemConfig, ...] = field(default_factory=tuple)
    layout: str = ""


@struct.dataclass
class NetworkConfig:
    FC_DIM_SIZE: int = 128
    GRU_HIDDEN_DIM: int = 128
    ACTIVATION: str = "relu"


@dataclass(frozen=True)
class TrainConfig:
    NUM_SEEDS: int = 1
    SEED: int = 0
    LR: float = 0.00025
    ANNEAL_LR: bool = True
    LR_WARMUP: float = 0.0
    NUM_ENVS: int = 32
    MINIBATCH_SIZE: int = 16
    NUM_UPDATE_EPOCHS: int = 4
    NUM_TRAINING_STEPS: int = 10000
    REW_SHAPING_HORIZON: int = 60000
    TIMESTEPS: int = 32
    FC_DIM_SIZE: int = 128
    GRU_HIDDEN_DIM: int = 128
    ACTIVATION: str = "relu"
    CLIP_EPS: float = 0.2
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    MAX_GRAD_NORM: float = 0.25
    VF_COEF: float = 0.5
    ANNEAL_ENT: bool = True
    ENT_COEF: float = 0.01
    ENT_END: float = 0.0001
    ENT_COOLDOWN: float = 0.9
    RANDOM_AGENT_POS: bool = True
    CHECKPOINT_INTERVAL_STEP: int = 500
    CHECKPOINT_SAVE_DIR: str = ""


@dataclass(frozen=True)
class AppConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    stage: str = "compact"
    progress: bool = True
    visualize: bool = False
    aspect_row: int = 1
    aspect_col: int = 2


@dataclass(frozen=True)
class IPPOModelConfig:
    checkpoint: str = ""
    step: int = 0


@dataclass(frozen=True)
class ReplayConfig:
    action_log: str = "action_logs/action_log.npz"


@dataclass(frozen=True)
class InteractiveConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    verbose: bool = True
    seed: tuple[int, ...] = field(default_factory=lambda: (0, 0))
    random_agent_position: bool = False
    confirm: bool = False
    visualize: bool = True
    loop: bool = False
    save_gif: bool = False
    gif_filename: str = ""
    log: bool = False
    log_dir: str = "action_logs"
    profile: bool = False
    player: tuple[str, ...] = field(default_factory=lambda: ("keyboard",))


def env_config_from_omegaconf(cfg: DictConfig) -> EnvConfig:
    p = cfg.parameter
    s = cfg.schedule
    r = cfg.reward
    return EnvConfig(
        customer=CustomerConfig(
            patience_mean=float(cfg.customer.patience_mean),
            patience_std=float(cfg.customer.patience_std),
            digestion_speed=int(cfg.customer.digestion_speed),
        ),
        parameter=EnvParameterConfig(
            forward_view_size=tuple(int(x) for x in p.forward_view_size),
            side_view_size=tuple(int(x) for x in p.side_view_size),
            restrict_observation=bool(p.restrict_observation),
            capacity=tuple(int(x) for x in p.capacity),
            plate_count=int(p.plate_count),
            sink_capacity=int(p.sink_capacity),
            order_max=int(p.order_max),
            wait_line_max=int(p.wait_line_max),
            dirt_appear_rate=float(p.dirt_appear_rate),
            dirtiness_max=int(p.dirtiness_max),
            cooking_duration_range=(float(p.cooking_duration_range[0]), float(p.cooking_duration_range[1])),
            volume_range=(float(p.volume_range[0]), float(p.volume_range[1])),
            check_time_max=int(p.check_time_max),
        ),
        reward=RewardConfig(
            original_reward=OriginalRewardConfig(finish_payment=float(r.original_reward.finish_payment)),
            shaped_reward=ShapedRewardConfig(
                invite_customer=float(r.shaped_reward.invite_customer),
                refuse_customer=float(r.shaped_reward.refuse_customer),
                take_order=float(r.shaped_reward.take_order),
                deliver_food=float(r.shaped_reward.deliver_food),
                retrieve_plate=float(r.shaped_reward.retrieve_plate),
                clean_table=float(r.shaped_reward.clean_table),
                soak_plate=float(r.shaped_reward.soak_plate),
                wash_plate=float(r.shaped_reward.wash_plate),
                process_payment=float(r.shaped_reward.process_payment),
                clean_dirt=float(r.shaped_reward.clean_dirt),
                placement_in_pot=float(r.shaped_reward.placement_in_pot),
                pot_start_cooking=float(r.shaped_reward.pot_start_cooking),
                dish_pickup=float(r.shaped_reward.dish_pickup),
            ),
            penalty=PenaltyConfig(
                ineffective_interaction=float(r.penalty.ineffective_interaction),
                erroneous_delivery=float(r.penalty.erroneous_delivery),
                step_cost=float(r.penalty.step_cost),
                block_cost=float(r.penalty.block_cost),
            ),
        ),
        schedule=ScheduleConfig(
            opening_time=int(s.opening_time),
            closing_time=int(s.closing_time),
            terminal_time=int(s.terminal_time),
            reservation=tuple(int(x) for x in s.reservation),
            congestion_rates=tuple((int(rate[0]), int(rate[1])) for rate in s.congestion_rates),
        ),
        menu=tuple(
            MenuItemConfig(
                recipe=tuple(int(x) for x in item.recipe), duration=int(item.duration), volume=int(item.volume)
            )
            for item in cfg.menu
        ),
        layout=str(cfg.layout),
    )


def network_config_from_train(train: TrainConfig) -> NetworkConfig:
    return NetworkConfig(
        FC_DIM_SIZE=train.FC_DIM_SIZE, GRU_HIDDEN_DIM=train.GRU_HIDDEN_DIM, ACTIVATION=train.ACTIVATION
    )


def train_config_from_omegaconf(cfg: DictConfig) -> TrainConfig:
    valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in cfg.items() if k in valid_fields})


def app_config_from_omegaconf(cfg: DictConfig) -> AppConfig:
    return AppConfig(
        train=train_config_from_omegaconf(cfg.train),
        env=env_config_from_omegaconf(cfg.env),
        stage=str(cfg.stage),
        progress=bool(cfg.progress),
        visualize=bool(cfg.visualize),
        aspect_row=int(cfg.aspect_row),
        aspect_col=int(cfg.aspect_col),
    )


def interactive_config_from_omegaconf(cfg: DictConfig) -> InteractiveConfig:
    return InteractiveConfig(
        env=env_config_from_omegaconf(cfg.env),
        verbose=bool(cfg.verbose),
        seed=tuple(int(x) for x in cfg.seed),
        random_agent_position=bool(cfg.random_agent_position),
        confirm=bool(cfg.confirm),
        visualize=bool(cfg.visualize),
        loop=bool(cfg.loop),
        save_gif=bool(cfg.save_gif),
        gif_filename=str(cfg.gif_filename),
        log=bool(cfg.log),
        log_dir=str(cfg.log_dir),
        profile=bool(cfg.profile),
        player=tuple(str(p) for p in cfg.player),
    )
