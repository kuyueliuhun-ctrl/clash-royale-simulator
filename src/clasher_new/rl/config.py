"""命名训练配置：一组参数（超参 + 奖励权重）→ 命名 → 独立输出文件夹。

用途（对应需求）：
- 「不同奖惩机制分类模型」：每个命名配置自带一套 ``reward`` 权重，即一种奖惩机制；
- 「一组参数设置 + 命名 + 分文件夹保存」：``--config <名>`` 选中预设（或用
  ``--load-config <json>`` 载入自定义），训练产物全部落在 ``out_dir/<name>/`` 下：
  断点状态、main 检查点、联赛录像、config.json 存档、Elo 状态、日志。
- ``--save-config <path>`` 把当前解析结果导出为 JSON，可编辑后 ``--load-config`` 复用。

预设：
- ``standard``   默认机制：塔血统一（打击=损失同价 0.001/0.001）+ 费差默认打开
                 （normalize_tower_dmg=True，elixir_diff_weight=0.5，1 圣水≈500 塔血 @lv11）；
- ``aggressive`` 推进：破塔/皇冠/胜利奖励更高，费差加码（1 圣水≈700 血，塔血换费差）；
- ``defensive``  防守反击：非法动作惩罚更重，费差减码（1 圣水≈300 血）；
- ``lockdown``   自闭：费差压到≈0（1 圣水≈50 血，鼓励费差换塔血，浪费仍小惩罚）；
- ``elixir``     鼓励圣水效率：每步按我方剩余圣水给正向 shaping（叠加默认费差机制）；
- ``economy``    费差经济（默认机制别名）：塔损按塔血%归一化 + 显式 Δ费差 shaping；
- ``fast``       小步快跑：步数/评估频率/单局上限都调小，用于冒烟/设备验证。

按流派区分奖惩（flow 联赛 6 模型）：``MODEL_REWARD_OVERRIDES`` 在所选预设之上按
模型 id 覆盖，main/all_decks/random_deck 用基线，推进加码费差、防反减码、自闭压到≈0。

任何超参都可用命令行覆盖（如 ``--config aggressive --lr 1e-3``）。
"""

import os
import json
from dataclasses import dataclass, field, asdict

#: 默认奖励权重（与 RLEnv 的 _DEFAULT_REWARD 保持一致；勿单独改一处）。
#: reward v2（2026 重构，替代旧"手牌圣水差逐帧费差"）：
#: - 资源账：费差项 = Δ(手牌圣水 + 场上部署份额)差（部署帧 E−c/V+c 抵消 → 下牌不罚；
#:   死亡注销、法术击杀返还 → 送死/空砸有代价、解牌赚费差）；
#: - 价格两段离散（120s 切双倍圣水，不线性）：tower_dmg_late=0.002 塔血贵 /
#:   elixir_diff_late=0.1 费贱 → 双倍期亏费换塔血、法术砸塔自动变正 EV；
#: - unit_dmg_k：单位受伤 shaping（客观伤害事件，非估值：敌方单位每掉 1 血 → 我方 +k）。
DEFAULT_REWARD = {
    "crown_weight": 8.0,        # 皇冠差系数（每差 1 皇冠 ±8：破塔里程碑，胜利太稀疏需中间大奖励）
    "tower_dmg_opp": 0.001,     # 敌方塔损 → 正奖励（前段 t<120，与 self 统一）
    "tower_dmg_self": 0.001,    # 我方塔损 → 负奖励（与 opp 统一）
    "tower_dmg_late": 0.002,    # v2 双倍期塔血系数（t≥120：斩杀/法术砸塔自动变正 EV）
    "win_bonus": 10.0,          # 获胜加成
    "lose_penalty": 10.0,       # 失败惩罚
    "draw_penalty": 10.0,       # 平局惩罚（= 失败：平局归类为败，逼策略主动求胜；0=旧行为免费平局）
    "invalid_penalty": 0.05,    # 每次非法动作惩罚
    "elixir_bonus": 0.0,        # 每步按我方剩余圣水的正向 shaping（圣水效率机制）
    "normalize_tower_dmg": True,   # 塔损按塔血%归一化到 lv11 锚（默认打开，跨等级一致）
    "elixir_diff_weight": 0.5,     # v2 资源账 edw 前段（t<120：费贵，教珍惜圣水；
                                   # lv11 下 1 圣水 ≈ 0.5/0.001 = 500 塔血的前段锚）
    "elixir_diff_late": 0.1,       # v2 双倍期 edw（t≥120：费贱，亏费换塔血可接受）
    "unit_dmg_k": 0.0005,          # v2 单位受伤 shaping（敌方单位每掉 1 血 → 我方 +k）
}

#: 按流派模型的奖惩覆盖（在所选预设之上按模型 id 覆盖；flow 联赛 6 模型用）。
#: main / all_decks / random_deck = 基线（同一参数）；推进加码费差（塔血换费差）、
#: 防守反击减码、自闭压到≈0（鼓励费差换塔血，浪费仍小惩罚——见 config 模块注释）。
MODEL_REWARD_OVERRIDES = {
    "main": {},                 # 基线
    "all_decks": {},            # 基线（与 main/random_deck 同一参数）
    "random_deck": {},          # 基线
    "push_flow": {"elixir_diff_weight": 0.7},     # 推进：增加"塔血换费差"奖励（1圣水≈700血）
    "counter_flow": {"elixir_diff_weight": 0.3},  # 防守反击：降低该奖励（1圣水≈300血）
    "lockdown_flow": {"elixir_diff_weight": 0.05},# 自闭：压到≈0，鼓励"费差换塔血"（1圣水≈50血）
}


@dataclass
class TrainConfig:
    name: str = "standard"
    description: str = ""
    # —— 训练/评估超参 ——
    total_steps: int = 20000
    steps_per_eval: int = 4000   # 评估频率：每 N 步训满再评（2000→4000：每 ckpt 训练量翻倍、评估频率减半）
    batch_size: int = 128
    update_interval: int = 128
    lr: float = 3e-4
    hidden_dim: int = 128
    seed: int = 0
    n_eval_games: int = 16    # 每对评估局数。统计契约：轮内聚合估计 SE≈347.5/√(5N)（main 5 对）
                              # N=16→SE≈39（±80 移动≈2σ，可区分学习信号）；评估开销与 N 成正比，
                              # 40→16 是降噪地板与训练速度的折中（评估量 ÷2.5）
    max_ep_steps: int = 360       # 常规时间 180s（360 步）的截断上限；180s 皇冠相同 → 自动进入
                                  # 加时窗口（overtime_open 延长到最多 300s=600 步，先破塔者胜；
                                  # 到 300s 仍无人破塔 → 按平局=失败结算，不再按塔血提前判胜）
    n_envs: int = 1               # 并行多环境（>1 用批量推理；默认 1 与旧行为一致）
    parallel: str = "mp"          # n_envs>1 时：mp=跨进程 worker（多核真并行）/ proc=单进程批量化
    card_level: int = 11          # 本局全部卡牌等级（11-16；配合 economy 的塔血归一化跨等级一致）
    eval_at_start: bool = True    # 训练开始先跑一次评估/快照，WebUI 立即有真实数据
    gamma: float = 0.997          # 终端现值修复：0.997^360≈0.34（原 0.99^600≈0.0024，胜负/平局惩罚几乎不可学）
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip: float = 0.2
    max_grad_norm: float = 0.5
    # —— 纯 RL 冷启动修复（2026）——
    # advantage 归一化：batch=整批中心化（旧默认）/ scale=只除以批 std /
    # none=原始。躺平局批内大量零优势帧被中心化抬成伪正优势（推高 STOP），
    # scale 保留原始符号只缩放幅度。
    adv_norm: str = "batch"
    # solo 训练环僵局早停判平（与 eval 同语义：连续 100 步双方塔血零变化 → 判平）。
    # 否则躺平要拖满 max_ep_steps 才在 360 帧末罚一次 −10，(γλ)^k 视野内完全不可见。
    train_stall_stop: bool = True
    # —— 数据 / 运行时 ——
    decks_path: str = None      # 三分类卡组 JSON（缺省自动探测）
    main_init: str = None       # BC 预训练 / 旧检查点
    device: str = "auto"        # cpu / cuda / auto（=cuda 可用则 cuda）
    only_vs_main: bool = False
    keep_snapshot: bool = False
    out_dir: str = "runs"
    # —— solo 自对弈（--mode solo，无联赛；原版 train.py 思路）——
    solo_copy_every: int = 2000   # 冻结副本同步间隔（步）：每 N 步把 main 权重拷给对手
    # —— 评估并行（eval_solo/play_pair 用进程池绕开 GIL；0 = 串行旧行为）——
    eval_workers: int = 0         # >1 时评估 spawn N 个进程并行打局（每 worker 独立 env+信念，
                                  # 主进程汇总；战斗模拟是纯 Python，跨进程才真正吃满多核）
    # —— 奖惩机制（每配置一套，见 DEFAULT_REWARD 注释）——
    reward: dict = field(default_factory=lambda: dict(DEFAULT_REWARD))

    # ---- 路径（全部落在 out_dir/<name>/ 下）----
    def folder(self):
        return os.path.join(self.out_dir, self.name)

    def state_path(self):
        return os.path.join(self.folder(), "league_state.json")

    def solo_state_path(self):
        return os.path.join(self.folder(), "solo_state.json")

    def solo_main_path(self):
        return os.path.join(self.folder(), "solo_main.pt")

    def solo_ckpt_path(self, step):
        """solo 历史检查点：solo_main_<step>.pt（每次评估保留一份，回溯用）。"""
        return os.path.join(self.folder(), f"solo_main_{step}.pt")

    def solo_opt_path(self):
        """solo 优化器状态（断点续练恢复 Adam 用）。"""
        return os.path.join(self.folder(), "solo_opt.pt")

    def run_state_path(self):
        return os.path.join(self.folder(), "run_state.json")

    def config_path(self):
        return os.path.join(self.folder(), "config.json")

    def replays_dir(self):
        return os.path.join(self.folder(), "replays")

    def main_final_path(self):
        return os.path.join(self.folder(), "main_final.pt")

    def ckpt_path(self, step):
        return os.path.join(self.folder(), f"main_ckpt_{step}.pt")

    def opt_path(self, step):
        return os.path.join(self.folder(), f"main_opt_{step}.pt")

    def ensure_dirs(self):
        os.makedirs(self.folder(), exist_ok=True)
        os.makedirs(self.replays_dir(), exist_ok=True)

    # ---- 序列化 ----
    def to_dict(self):
        return asdict(self)

    def save(self, path=None):
        path = path or self.config_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        rw = dict(DEFAULT_REWARD)
        rw.update(d.get("reward") or {})
        cfg.reward = rw
        return cfg

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---- 预设 ----
    @classmethod
    def presets(cls):
        return {
            "standard": cls(name="standard", description="默认奖惩机制：塔血统一 0.001/0.001、费差打开（1圣水≈500血）"),
            "aggressive": cls(
                name="aggressive",
                description="推进：破塔/皇冠/胜利奖励更高，费差加码（塔血换费差，1圣水≈700血）",
                reward={"crown_weight": 8.0, "win_bonus": 15.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.0, "elixir_diff_weight": 0.7}),
            "defensive": cls(
                name="defensive",
                description="防守反击：非法动作惩罚更重，费差减码（1圣水≈300血）",
                reward={"crown_weight": 3.0, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.1,
                        "elixir_bonus": 0.0, "elixir_diff_weight": 0.3}),
            "lockdown": cls(
                name="lockdown",
                description="自闭：费差压到≈0（1圣水≈50血，鼓励费差换塔血，浪费仍小惩罚）",
                reward={"crown_weight": 8.0, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.0, "elixir_diff_weight": 0.05}),
            "elixir": cls(
                name="elixir",
                description="鼓励圣水效率：默认机制基础上叠加每步按剩余圣水 shaping",
                reward={"crown_weight": 8.0, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.01, "elixir_diff_weight": 0.5}),
            "economy": cls(
                name="economy",
                description="费差经济（默认机制别名）：塔损按塔血%归一化 + Δ费差 shaping（1圣水≈500血）",
                reward={"crown_weight": 8.0, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.0, "normalize_tower_dmg": True,
                        "elixir_diff_weight": 0.5},
                gae_lambda=0.99,     # 纯RL冷启动：γλ=0.947→0.987，优势半衰期 13→53 帧
                                     # （终端±10 与 60-150 帧的出牌因果进入 GAE 视野）
                only_vs_main=True),   # 联赛模式评估只测 main（15 对→5 对，评估量再 ÷3）；solo 不受影响
            "fast": cls(
                name="fast", description="小步快跑（冒烟/设备验证用）",
                total_steps=2000, steps_per_eval=500,
                n_eval_games=2, max_ep_steps=300),
        }

    @classmethod
    def resolve(cls, preset, load_config=None, **overrides):
        """按名字/JSON 解析出配置，再应用命令行 overrides（只覆盖合法字段）。

        注意：预设是共享实例，先深拷贝再覆盖，避免一次运行污染下次运行。
        参数名用 preset 而非 name，避免与字段 name 的 overrides 冲突。
        """
        if load_config:
            cfg = cls.load(load_config)
            if preset:
                cfg.name = preset
        else:
            presets = cls.presets()
            if preset in presets:
                cfg = cls.from_dict(presets[preset].to_dict())  # 拷贝，不共享
            else:
                raise ValueError(
                    f"未知配置 '{preset}'，可用: {', '.join(presets)}（或 --load-config）")
        for k, v in (overrides or {}).items():
            if v is None:
                continue
            if k in cls.__dataclass_fields__:
                setattr(cfg, k, v)
            else:
                raise ValueError(f"未知配置字段 '{k}'")
        return cfg


def reward_to_env(cfg: TrainConfig) -> dict:
    """把配置的奖励权重转成 RLEnv 接受的 reward_weights 字典。"""
    return dict(DEFAULT_REWARD, **cfg.reward)


def model_reward_weights(model_id: str, cfg: TrainConfig) -> dict:
    """按流派模型的奖励权重 = 所选预设之上叠加 MODEL_REWARD_OVERRIDES。

    flow 联赛 6 模型用（推进/防反/自闭差异化，main/all/random 用基线同一参数）。
    未知 model_id 直接回退到所选预设（不改基线行为）。
    """
    rw = dict(DEFAULT_REWARD, **cfg.reward)
    rw.update(MODEL_REWARD_OVERRIDES.get(model_id) or {})
    return rw
