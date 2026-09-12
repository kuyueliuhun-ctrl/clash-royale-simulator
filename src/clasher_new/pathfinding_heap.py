from pathlib import Path
import math
from core import Position
import heapq

grid_path = Path(__file__).with_name('tilemap_lane_grid.txt')
with grid_path.open('r', encoding='utf-8') as f:
    contents = [list(each) for each in f.read().splitlines()]

cell_cache = {}
neighbor_cache = {}

def position_to_cell(position: Position):
    x, y = position.x, position.y
    return math.floor(2*x), math.floor(2*y)

def cell_to_position(cell):
    if cell not in cell_cache:
        x, y = cell
        cell_cache[cell] = Position((x+0.5)/2, (y+0.5)/2)
    return cell_cache[cell]

def get_neighboring_points(x, y):
    if (x, y) in neighbor_cache: return neighbor_cache[(x,y)]
    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            new_x, new_y = x+dx, y+dy
            if dx == dy == 0: continue
            if new_x < 0 or new_y < 0 or new_x >= 36 or new_y >= 64: continue
            result.append((new_x, new_y))
    neighbor_cache[(x,y)] = result
    return result


class EntityPathfinder:
    def __init__(self, entity, target, battle_state):
        self.start_position = Position(entity.position.x, entity.position.y)
        self.target_position = Position(target.position.x, target.position.y)
        self.target = target
        self.entity = entity
        self.start_cell = position_to_cell(self.start_position)
        self.battle = battle_state
        self.goals = set()
        self.goal = None

    def heuristic(self, cell):
        x, y = cell
        gx, gy = self.goal
        return 10 * max(abs(x - gx), abs(y - gy))

    def _target_footprint_radius(self):
        """目标占用半径：塔=矩形半宽/半高较大者（_tower_rect），其余=圆形碰撞半径。

        扫描窗需要覆盖目标足迹 + 射程；塔矩形（王塔半宽 2.0）可能大于圆形碰撞
        半径（王塔 data 仍 1.0），直接取矩形半轴更稳。"""
        rect = getattr(self.target, '_tower_rect', 0)
        if rect == 0:
            self.target._bind_tower_rect()
            rect = getattr(self.target, '_tower_rect', 0)
        if rect is not None:
            return max(rect[2], rect[3])
        return self.target.data.collision_radius

    def calculate(self):
        self.goals = set()

        # —— 目标格按「到目标边缘的距离」生成（edge-based，2026-09-10 用户机制定稿）——
        # 真实 CR 寻路不是「到目标中心的最短路径」，而是每步朝目标边缘最近点走：
        #   · 左桥部署的单位过桥后靠左、右桥靠右（车道保持），不是全部吸向目标中心正对；
        #   · 目标塔被毁转向王塔时平滑偏移约 1 格，而非走到塔前才猛拐；
        # edge_distance_from 已统一口径：塔=矩形边缘、部队/建筑=圆形边缘，
        # 与 in_attack_range 的索敌射程一致（edge_dist ≤ range）。——
        edge_radius = self.entity.data.range
        target_cell = position_to_cell(self.target_position)
        scan_radius = math.ceil((self._target_footprint_radius() + edge_radius) * 2) + 2
        for x in range(target_cell[0]-scan_radius, target_cell[0]+scan_radius):
            for y in range(target_cell[1]-scan_radius, target_cell[1]+scan_radius):
                pos = cell_to_position((x, y))
                edge_dist = self.target.edge_distance_from(pos)
                # 0.375 裕量让短射程部队够得着塔（原注释语义保留：lumberjack 能打到塔
                # 而不是在侧边徘徊）
                if edge_dist < edge_radius + 0.375 and self.battle.pathfind_ground_walkable(pos, self.entity.data.collision_radius):
                    self.goals.add((x, y))
        if not self.goals:
            # 兜底：range=0 近战对建筑时，攻击半径(碰撞半径)内无可达格
            # （塔足迹 + mover 半径把目标周围全堵死）。退而求其次找全扫描区内
            # 到目标边缘最近的可达格；全无则直接以目标格为目标（由 in_attack_range 碰撞半径判定攻击时机）。
            best, best_d = None, float('inf')
            for x in range(target_cell[0]-scan_radius, target_cell[0]+scan_radius):
                for y in range(target_cell[1]-scan_radius, target_cell[1]+scan_radius):
                    pos = cell_to_position((x, y))
                    if not self.battle.pathfind_ground_walkable(pos, self.entity.data.collision_radius):
                        continue
                    d = self.target.edge_distance_from(pos)
                    if d < best_d:
                        best, best_d = (x, y), d
            if best is not None:
                self.goals.add(best)
            else:
                self.goals.add(target_cell)
        # —— 择优：目标格=「到目标边缘最近 + 到起点最近」——
        # 边缘距离对所有候选基本相等（都在射程环上），实际起主导的是「离起点最近」：
        # 单位从自己一侧接近目标边缘 → 车道保持（左桥来的靠左、右桥来的靠右），
        # 而不是全部吸向目标中心正对的全局最短路径。——
        self.goal = min(self.goals, key=lambda c: self.target.edge_distance_from(cell_to_position(c)) + cell_to_position(c).distance_to(self.start_position))

        g = {}
        f = {}
        parent = {}
        closed_set = set()
        g[self.start_cell] = 0
        f[self.start_cell] = self.heuristic(self.start_cell)
        open_heap = [(f[self.start_cell], self.start_cell)]

        while open_heap:
            current_f, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            if current_f > f[current]:
                continue
            if current == self.goal:
                break
            closed_set.add(current)
            for neighbor in get_neighboring_points(current[0], current[1]):
                if neighbor in closed_set: continue
                neighbor_position = cell_to_position(neighbor)
                if not self.battle.pathfind_ground_walkable(neighbor_position, self.entity.data.collision_radius):
                    continue
                nx, ny = neighbor
                px, py = current
                tile_char = contents[63-ny][nx]
                if tile_char == 'W':
                    tile_cost = 800 if not self.entity.data.is_air_unit else 7
                elif tile_char == '.':
                    # 桥面代价与半场地面一致（5）：真实 CR 桥 3 格宽，三车道过桥
                    # 代价相同，单位保持部署时的横向车道（2026-09-10 用户机制：
                    # 桥头左侧的单位上桥后靠左、右侧靠右，寻路按边缘最近贪心）。
                    # 旧值 8 让 A* 把单位吸向更便宜的桥中央地面格，抹平车道。
                    tile_cost = 5
                else:
                    tile_cost = 5
                if nx != px and ny != py:
                    geo_cost = 14
                else:
                    geo_cost = 10
                step_cost = tile_cost * geo_cost
                tentative_g = g[current] + step_cost
                if neighbor not in g or tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    parent[neighbor] = current
                    f[neighbor] = g[neighbor] + self.heuristic((nx, ny))
                    heapq.heappush(open_heap, (f[neighbor], neighbor))
        path = [current]
        while path[-1] != self.start_cell:
            path.append(parent[path[-1]])
        path.reverse()

        positions = [cell_to_position(each) for each in path]
        return positions

if __name__ == '__main__':
    from battle import BattleState
    from player import PlayerState

    player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
    player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']
    battle = BattleState(PlayerState(0, player_0_deck, 10), PlayerState(1, player_1_deck, 10))
    battle.deploy_card(0, 'Knight', Position(10.5, 10.5))

    pathfind = EntityPathfinder(battle.entities[7], battle.entities[2], battle)
    print(pathfind.calculate())






