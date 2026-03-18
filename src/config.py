import dataclasses
from dataclasses import dataclass, field
from typing import Any

from flax import struct
from omegaconf import DictConfig


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
    finish_payment: float = 20.0  # 会計を完了し、客が退店したら報酬


# 細かいステップごとに得られる報酬
@dataclass(frozen=True)
class ShapedRewardConfig:
    # interaction できる場合は常に推奨される
    # 実行すると状態が遷移し、繰り返し報酬を得ることはできない
    invite_customer: float = 3.0  # 客を案内
    refuse_customer: float = 0.001  # 満席時、客を帰す
    take_order: float = 3.0  # 注文を取る
    wash_plate: float = 3.0  # 皿を洗う
    process_payment: float = 3.0  # 会計処理をする(未完)
    clean_dirt: float = 3.0  # 汚れを掃除する
    # pick_place 注文内容に応じて推奨・非推奨
    ## pick
    pickup_ingredient_from_pile: float = 0.5  # 食材をpileから取り出す
    pickup_ingredient_from_counter: float = 0.0  # 食材をカウンターから取り上げる
    pickup_new_plate_from_pile: float = 0.5  # 皿をpileから取り出す
    pickup_new_plate_from_counter: float = 0.0  # 皿をカウンターから取り上げる
    plate_dish: float = 6.0  # 皿に料理を盛り付ける
    pickup_dish_from_counter: float = 0.0  # 料理をカウンターから取り上げる
    pickup_used_plate_from_table: float = 3.0  # 使用済み皿をテーブルから下げる
    clean_table: float = 3.0  # テーブルを片付ける(客が退店済み)
    pickup_used_plate_from_counter: float = 0.0  # 使用済み皿をカウンターから取り上げる
    ## place
    ### place ingredient
    placement_in_pot: float = 2.0  # 鍋に食材を入れる
    pot_start_cooking: float = 3.0  # 注文されたメニューの調理を開始する
    place_ingredient_on_counter: float = -0.2  # 食材をカウンターに置く
    dispose_ingredient: float = -2.0  # 食材をゴミ箱に捨てる
    ### place food
    deliver_food: float = 8.0  # 料理を提供する
    place_food_on_counter: float = -0.2  # 料理をカウンターに置く
    dispose_food: float = -10.0  # 料理をゴミ箱に捨てる
    ### place plate
    place_new_plate_on_counter: float = -0.2  # 新しい皿をカウンターに置く
    place_used_plate_on_counter: float = -0.2  # 使用済み皿をカウンターに置く
    soak_plate: float = 3.0  # 使用済み皿をシンクに置く


# 抑止したい行動をとったときのペナルティ、Overcookedの処理でマイナス符号にする
@dataclass(frozen=True)
class PenaltyConfig:
    ineffective_interaction: float = 0.001  # 無効なinteractionを行った
    ineffective_pickup: float = 0.001  # 無効なpickupを行った
    ineffective_placement: float = 0.001  # 無効なplacementを行った
    erroneous_cooking: float = 1.0  # 注文されていない料理を調理した(事前に調理しておく場合もペナルティになる)
    erroneous_delivery: float = 1.0  # 注文と違う料理を提供しようとした
    step_cost: float = 0.0  # 移動にかかるコスト(無駄な動きを抑制)
    block_cost: float = 0.01  # 位置・向きの変わらない移動をした(壁押し)
    # place_cost: float
    # dispose_cost: float


# 時間経過による報酬割引の設定
@dataclass
class DiscountConfig:
    decline_deliver_reward: bool = True  # 時間経過により料理提供の報酬を割り引くか
    deliver_ramp_step: int = 10  # 料理提供時の報酬が割引なしで与えられる注文からのステップ数
    deliver_limit_step: int = 100  # 料理提供の報酬が下がりきるステップ数(>deliver_ramp_step)
    deliver_discount_rate: float = 0.8  # 報酬割引率の下限

    def __post_init__(self):
        assert self.deliver_ramp_step < self.deliver_limit_step


@dataclass(frozen=True)
class RewardConfig:
    original_reward: OriginalRewardConfig = field(default_factory=OriginalRewardConfig)
    shaped_reward: ShapedRewardConfig = field(default_factory=ShapedRewardConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    discount: DiscountConfig = field(default_factory=DiscountConfig)


@dataclass(frozen=True)
class ScheduleConfig:
    opening_time: int = 10  # 開店時間
    closing_time: int = 720  # 閉店時間
    terminal_time: int = 780  # 終了時間
    reservation: tuple[int, ...] = field(default_factory=lambda: (1, 10, 30))  # 予約客の来店時間(ステップ数で指定)
    congestion_rates: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: ((0, 0), (10, 10), (20, 100), (25, 20), (30, 50), (35, 0), (50, 20))
    )  # 一般客の来店率の時間変化 [変更ステップ数, 来店率(%)]のリスト


@dataclass(frozen=True)
class MenuItemConfig:
    recipe: tuple[int, ...] = field(default_factory=tuple)  # 必要な食材番号のリスト
    duration: int = 0  # 調理にかかる時間
    volume: int = 0  # 提供から食べ終わるまでのステップ数


@dataclass(frozen=True)
class EnvConfig:
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
    # progress: bool
    # visualize: bool
    # aspect_row: int
    # aspect_col: int
    # viz_rows: int
    # viz_cols: int
    # NUM_ACTORS: int
    # NUM_MINIBATCHES: int
    MODEL_DIR: str
    ## ABOVE: ?
    NUM_SEEDS: int = 1
    SEED: int = 0
    LR: float = 0.00025
    ANNEAL_LR: bool = True
    LR_WARMUP: float = 0.0  # NUM_LEARNING_STEPSに対するウォームアップの割合
    NUM_ENVS: int = 32  # 並列実行する環境の個数(16GB: 32*2agents)
    MINIBATCH_SIZE: int = 16  # NUM_ENVS*num_agentsを割り切る数
    NUM_UPDATE_EPOCHS: int = 4  # minibatch単位の学習１周を繰り返す回数
    NUM_TRAINING_STEPS: int = 10000  # 行動->学習(UPDATE_EPOCHS回) を1単位として何回繰り返すか
    REW_SHAPING_HORIZON: int = (
        60000  # LEARNING_STEPS に応じてshaped_rewardの重みを減らすしていき、HORIZON以降は提供時のrewardのみになる
    )
    TIMESTEPS: int = 32  # 学習1ステップのために環境の更新を行いデータを収集するステップ数(unroll=16の倍数にするとよい)
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
    RANDOM_AGENT_POS: bool = True  # エージェントの初期位置を元の位置から移動可能な範囲でランダムにする
    EVAL_SEED: int = 42  # 評価に使う環境用のシード
    CHECKPOINT_INTERVAL_STEP: int = 500  # チェックポイントの保存間隔
    CHECKPOINT_KEEP: int = 5  # チェックポイントの最大保存数


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


@dataclass
class PlayConfig:
    env: EnvConfig
    ui: dict[str, Any]
    player: str | list[str]
    verbose: bool
    seed: list[int]
    random_agent_position: bool
    confirm: bool
    visualize: bool
    loop: int
    save_gif: bool
    gif_filename: str
    log: bool
    log_dir: str
    profile: bool


@dataclass(frozen=True)
class EvalConfig:
    env: EnvConfig
    # ui: dict[str, Any]
    seed: list[int]
    random_agent_position: bool
    visualize: bool
    loop: int
    save_gif: bool
    gif_filename: str


def env_config_from_omegaconf(cfg: DictConfig) -> EnvConfig:
    p = cfg.parameter
    s = cfg.schedule
    r = cfg.reward
    return EnvConfig(
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
                wash_plate=float(r.shaped_reward.wash_plate),
                process_payment=float(r.shaped_reward.process_payment),
                clean_dirt=float(r.shaped_reward.clean_dirt),
                pickup_ingredient_from_pile=float(r.shaped_reward.pickup_ingredient_from_pile),
                pickup_ingredient_from_counter=float(r.shaped_reward.pickup_ingredient_from_counter),
                pickup_new_plate_from_pile=float(r.shaped_reward.pickup_new_plate_from_pile),
                pickup_new_plate_from_counter=float(r.shaped_reward.pickup_new_plate_from_counter),
                plate_dish=float(r.shaped_reward.plate_dish),
                pickup_dish_from_counter=float(r.shaped_reward.pickup_dish_from_counter),
                pickup_used_plate_from_table=float(r.shaped_reward.pickup_used_plate_from_table),
                clean_table=float(r.shaped_reward.clean_table),
                pickup_used_plate_from_counter=float(r.shaped_reward.pickup_used_plate_from_counter),
                placement_in_pot=float(r.shaped_reward.placement_in_pot),
                pot_start_cooking=float(r.shaped_reward.pot_start_cooking),
                place_ingredient_on_counter=float(r.shaped_reward.place_ingredient_on_counter),
                dispose_ingredient=float(r.shaped_reward.dispose_ingredient),
                deliver_food=float(r.shaped_reward.deliver_food),
                place_food_on_counter=float(r.shaped_reward.place_food_on_counter),
                dispose_food=float(r.shaped_reward.dispose_food),
                place_new_plate_on_counter=float(r.shaped_reward.place_new_plate_on_counter),
                place_used_plate_on_counter=float(r.shaped_reward.place_used_plate_on_counter),
                soak_plate=float(r.shaped_reward.soak_plate),
            ),
            penalty=PenaltyConfig(
                ineffective_interaction=float(r.penalty.ineffective_interaction),
                ineffective_pickup=float(r.penalty.ineffective_pickup),
                ineffective_placement=float(r.penalty.ineffective_placement),
                erroneous_cooking=float(r.penalty.erroneous_cooking),
                erroneous_delivery=float(r.penalty.erroneous_delivery),
                step_cost=float(r.penalty.step_cost),
                block_cost=float(r.penalty.block_cost),
            ),
            discount=DiscountConfig(
                decline_deliver_reward=bool(r.discount.decline_deliver_reward),
                deliver_ramp_step=int(r.discount.deliver_ramp_step),
                deliver_limit_step=int(r.discount.deliver_limit_step),
                deliver_discount_rate=float(r.discount.deliver_discount_rate),
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


def evaluate_config_from_omegaconf(cfg: DictConfig):
    return EvalConfig(
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
