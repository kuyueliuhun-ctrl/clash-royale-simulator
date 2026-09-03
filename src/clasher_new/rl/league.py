"""联赛机制（规划文档 7）：对手池 + PFSP + Elo + checkpoint + Exploiter 入口。

修复：
- P1-9：register_checkpoint 注册**新条目**并保存**权重副本**，原 main 保持 kind="main"，
  训练中的活对象不再被改名/引用；
- P1-10：提供 save_state / load_state 持久化 Elo / PFSP / 成员（策略本体另行存 checkpoint）；
- P2：add_agent 同 id 默认抛异常（replace=True 重置 rating）；add_exploiter 用递增计数器；
      补 remove_agent / retire（含清理 pfsp.winrates 键）。
"""

import os
import sys
from dataclasses import dataclass, field

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from rl.pfsp import PFSP
from rl.elo import Elo


@dataclass
class LeagueAgent:
    agent_id: str
    kind: str = "main"            # main / historical / exploiter / baseline
    policy: object = None         # callable(obs) -> ActionBundle | tuple
    path: str = None              # checkpoint 路径
    elo: float = 1500.0


class League:
    def __init__(self, pfsp_beta: float = 1.0, elo_k: float = 32.0, seed: int = 0):
        self.agents = {}          # agent_id -> LeagueAgent
        self.pfsp = PFSP(beta=pfsp_beta, seed=seed)
        self.elo = Elo(k=elo_k)
        self.history = []
        self._exploiter_counter = 0
        self._ckpt_counter = {}
        self.elo_history = {}     # agent_id -> [(step, elo), ...]（供训练网页 UI 画曲线）
        self.total_steps = 0

    def add_agent(self, agent_id, kind="main", policy=None, path=None, replace=False):
        if agent_id in self.agents:
            if not replace:
                raise ValueError(f"agent_id '{agent_id}' 已存在（用 replace=True 重置）")
            self.agents.pop(agent_id, None)
        self.agents[agent_id] = LeagueAgent(agent_id, kind=kind, policy=policy, path=path)
        self.elo.ensure(agent_id)

    def sample_opponent(self, agent_id) -> LeagueAgent:
        opponents = [aid for aid in self.agents if aid != agent_id]
        if not opponents:
            raise RuntimeError("联赛至少需要 2 个 agent")
        op_id = self.pfsp.sample(agent_id, opponents)
        return self.agents[op_id]

    def record_match(self, agent_a, agent_b, score_a: float, n_games: int = 1):
        """score_a ∈ [0,1]（0/0.5/1 逐局）。n_games>1 时按局数缩放 Elo K（P1-11）。"""
        self.elo.update(agent_a, agent_b, score_a, n_games=n_games)
        self.pfsp.update_winrate(agent_a, agent_b, score_a)
        self.pfsp.update_winrate(agent_b, agent_a, 1 - score_a)
        self.history.append((agent_a, agent_b, score_a))

    def register_checkpoint(self, agent_id, policy, path=None, metadata=None):
        """把当前 main 的**权重副本**注册为 historical 新条目（P1-9）。

        返回新条目 agent_id（f"{agent_id}_ckpt{n}"）。
        """
        from rl.follower import FollowerPolicy
        n = self._ckpt_counter.get(agent_id, 0)
        self._ckpt_counter[agent_id] = n + 1
        new_id = f"{agent_id}_ckpt{n}"
        snap = FollowerPolicy(hidden=policy.hidden_dim, plan_dim=policy.plan_dim,
                              belief_dim=policy.belief_dim)
        snap.load_state_dict(policy.state_dict())
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.add_agent(new_id, kind="historical", policy=snap, path=path)
        return new_id

    def refresh_snapshot(self, agent_id, policy, path=None):
        """把 main 的权重副本刷新到**固定槽位** f"{agent_id}_ckpt"（维持 5 模型稳定结构）。

        替换不重置 Elo：历史曲线连续。返回槽位 agent_id。
        """
        from rl.follower import FollowerPolicy
        new_id = f"{agent_id}_ckpt"
        snap = FollowerPolicy(hidden=policy.hidden_dim, plan_dim=policy.plan_dim,
                              belief_dim=policy.belief_dim)
        snap.load_state_dict(policy.state_dict())
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.add_agent(new_id, kind="historical", policy=snap, path=path, replace=True)
        return new_id

    def record_elo_history(self, step: int = None):
        """记录当前各 agent 的 (step, elo)，供训练网页 UI 画 Elo-训练次数曲线。"""
        if step is not None:
            self.total_steps = max(self.total_steps, int(step))
        for aid, rating in self.elo.ratings.items():
            self.elo_history.setdefault(aid, []).append((self.total_steps, float(rating)))

    def add_exploiter(self, policy, path=None):
        aid = f"exploiter_{self._exploiter_counter}"
        self._exploiter_counter += 1
        self.add_agent(aid, kind="exploiter", policy=policy, path=path)
        return aid

    def remove_agent(self, agent_id):
        if agent_id not in self.agents:
            return False
        self.agents.pop(agent_id, None)
        self.elo.ratings.pop(agent_id, None)
        self.pfsp.winrates = {k: v for k, v in self.pfsp.winrates.items() if agent_id not in k}
        return True

    retire = remove_agent

    def elo_table(self):
        return self.elo.table()

    # ---- 持久化（P1-10：Elo / PFSP / 成员元信息；策略本体由外部存 checkpoint）----

    def save_state(self, path):
        import json
        state = {
            "ratings": dict(self.elo.ratings),
            "winrates": {f"{a}|{b}": v for (a, b), v in self.pfsp.winrates.items()},
            "agents": [{"agent_id": a.agent_id, "kind": a.kind, "path": a.path} for a in self.agents.values()],
            "history": self.history,
            "exploiter_counter": self._exploiter_counter,
            "ckpt_counter": dict(self._ckpt_counter),
            "elo_history": {k: [list(p) for p in v] for k, v in self.elo_history.items()},
            "total_steps": int(self.total_steps),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def load_state(self, path, policies=None):
        """恢复联赛状态。policies: {agent_id: policy} 可选，用于重新挂载策略本体。"""
        import json
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.elo.ratings = {k: float(v) for k, v in state["ratings"].items()}
        self.pfsp.winrates = {
            tuple(k.split("|")): float(v) for k, v in state["winrates"].items()}
        self.history = list(state["history"])
        self._exploiter_counter = int(state.get("exploiter_counter", 0))
        self._ckpt_counter = {k: int(v) for k, v in state.get("ckpt_counter", {}).items()}
        self.elo_history = {k: [tuple(p) for p in v] for k, v in state.get("elo_history", {}).items()}
        self.total_steps = int(state.get("total_steps", 0))
        policies = policies or {}
        for a in state["agents"]:
            self.agents[a["agent_id"]] = LeagueAgent(
                agent_id=a["agent_id"], kind=a["kind"], path=a["path"],
                policy=policies.get(a["agent_id"]))
        return self
