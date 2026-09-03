"""命名训练配置：一组参数（超参 + 奖励权重）→ 命名 → 独立输出文件夹。

用途（对应需求）：
- 「不同奖惩机制分类模型」：每个命名配置自带一套 ``reward`` 权重，即一种奖惩机制；
- 「一组参数设置 + 命名 + 分文件夹保存」：``--config <名>`` 选中预设（或用
  ``--load-config <json>`` 载入自定义），训练产物全部落在 ``out_dir/<name>/`` 下：
  断点状态、main 检查点、联赛录像、config.json 存档、Elo 状态、日志。
- ``--save-config <path>`` 把当前解析结果导出为 JSON，可编辑后 ``--load-config`` 复用。

预设：
- ``standard``   默认机制（与旧公式逐位一致，行为不变）；
- ``aggressive`` 鼓励推进：破塔/皇冠/胜利奖励更高，挨打惩罚降低；
- ``defensive``  鼓励防守：我方塔损惩罚更高、非法动作惩罚更重；
- ``elixir``     鼓励圣水效率：每步按我方剩余圣水给正向 shaping；
- ``economy``    费差经济：塔损按塔血%归一化（跨等级不变）+ 显式 Δ费差 shaping，
                 让模型学会"让塔挨打换圣水/费差"的真实游戏 trade；
- ``fast``       小步快跑：步数/评估频率/单局上限都调小，用于冒烟/设备验证。

任何超参都可用命令行覆盖（如 ``--config aggressive --lr 1e-3``）。
"""

import os
import json
from dataclasses import dataclass, field, asdict

#: 默认奖励权重（与 RLEnv 旧公式逐位一致：crown 5 / 塔损 0.001/0.0012 / 胜负 ±10 / 非法 -0.05）
DEFAULT_REWARD = {
    "crown_weight": 5.0,        # 皇冠差系数（每差 1 皇冠 ±5）
    "tower_dmg_opp": 0.001,     # 敌方塔损 → 正奖励
    "tower_dmg_self": 0.0012,   # 我方塔损 → 负奖励
    "win_bonus": 10.0,          # 获胜加成
    "lose_penalty": 10.0,       # 失败惩罚
    "invalid_penalty": 0.05,    # 每次非法动作惩罚
    "elixir_bonus": 0.0,        # 每步按我方剩余圣水的正向 shaping（圣水效率机制）
    "normalize_tower_dmg": False,  # 塔损按塔血%归一化到 lv11 锚（economy 机制；默认关=旧公式）
    "elixir_diff_weight": 0.0,     # 每步 Δ费差（我方−对方圣水）shaping 权重（economy 机制）
}


@dataclass
class TrainConfig:
    name: str = "standard"
    description: str = ""
    # —— 训练/评估超参 ——
    total_steps: int = 20000
    steps_per_eval: int = 2000
    batch_size: int = 128
    update_interval: int = 128
    lr: float = 3e-4
    hidden_dim: int = 128
    seed: int = 0
    n_eval_games: int = 4
    max_ep_steps: int = 600
    n_envs: int = 1               # 并行多环境（>1 用批量推理；默认 1 与旧行为一致）
    parallel: str = "mp"          # n_envs>1 时：mp=跨进程 worker（多核真并行）/ proc=单进程批量化
    card_level: int = 11          # 本局全部卡牌等级（11-16；配合 economy 的塔血归一化跨等级一致）
    eval_at_start: bool = True    # 训练开始先跑一次评估/快照，WebUI 立即有真实数据
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip: float = 0.2
    max_grad_norm: float = 0.5
    # —— 数据 / 运行时 ——
    decks_path: str = None      # 三分类卡组 JSON（缺省自动探测）
    main_init: str = None       # BC 预训练 / 旧检查点
    device: str = "auto"        # cpu / cuda / auto（=cuda 可用则 cuda）
    only_vs_main: bool = False
    keep_snapshot: bool = False
    out_dir: str = "runs"
    # —— 奖惩机制（每配置一套，见 DEFAULT_REWARD 注释）——
    reward: dict = field(default_factory=lambda: dict(DEFAULT_REWARD))

    # ---- 路径（全部落在 out_dir/<name>/ 下）----
    def folder(self):
        return os.path.join(self.out_dir, self.name)

    def state_path(self):
        return os.path.join(self.folder(), "league_state.json")

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
            "standard": cls(name="standard", description="默认奖惩机制（与旧公式一致）"),
            "aggressive": cls(
                name="aggressive",
                description="鼓励推进：破塔/皇冠/胜利奖励更高，挨打惩罚降低",
                reward={"crown_weight": 8.0, "tower_dmg_opp": 0.002,
                        "tower_dmg_self": 0.001, "win_bonus": 15.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.0}),
            "defensive": cls(
                name="defensive",
                description="鼓励防守：我方塔损惩罚更高、非法动作惩罚更重",
                reward={"crown_weight": 3.0, "tower_dmg_opp": 0.001,
                        "tower_dmg_self": 0.0025, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.1,
                        "elixir_bonus": 0.0}),
            "elixir": cls(
                name="elixir",
                description="鼓励圣水效率：每步按剩余圣水给正向 shaping",
                reward={"crown_weight": 5.0, "tower_dmg_opp": 0.001,
                        "tower_dmg_self": 0.0012, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.01}),
            "economy": cls(
                name="economy",
                description="费差经济：塔损按塔血%归一化（跨等级不变）+ 显式 Δ费差 shaping，"
                            "让模型学会用塔血换圣水（1 圣水≈0.9% 总塔血，lv11 锚 10928）",
                reward={"crown_weight": 5.0, "tower_dmg_opp": 0.001,
                        "tower_dmg_self": 0.0012, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.0,
                        "normalize_tower_dmg": True,
                        "elixir_diff_weight": 0.1}),
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
