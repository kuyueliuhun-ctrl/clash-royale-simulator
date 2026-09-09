from dataclasses import dataclass
from core import Position
import math

walkable_cache = {}

@dataclass
class TileGrid:
    width, height = 18, 32
    tile_size: float = 100.0
    BLUE_KING_TOWER = Position(9.0, 3.0)     # King tower centered at x=9 (middle of 18-wide arena)
    BLUE_LEFT_TOWER = Position(3.5, 6.5)     # Left princess tower
    BLUE_RIGHT_TOWER = Position(14.5, 6.5)   # Right princess tower (corrected for better symmetry)
    RED_KING_TOWER = Position(9.0, 29.0)     # King tower centered at x=9
    RED_LEFT_TOWER = Position(3.5, 25.5)     # Left princess tower  
    RED_RIGHT_TOWER = Position(14.5, 25.5)   # Right princess tower (corrected for better symmetry)
    LEFT_BRIDGE = Position(3.5, 16.0)   # Left bridge center of center tile (tiles 2,3,4 -> center at 3.5)
    RIGHT_BRIDGE = Position(14.5, 16.0) # Right bridge center of center tile (tiles 13,14,15 -> center at 14.5)
    RIVER_Y1 = 15.0
    RIVER_Y2 = 16.0
    BLOCKED_TILES = [
        # Edge tiles next to river
        (0, 15), (0, 16), (1, 15), (1, 16),
        *[(i, j) for i in range(5, 13) for j in range(15, 17)], # (5, 15) to (12, 16)
        (16, 15), (16, 16), (17, 15), (17, 16),
        
        # Top row (y=0): 6 gray fences (0-5), 6 green king area (6-11), 6 gray fences (12-17)
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),           # Left 6 gray fence tiles
        (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0),     # Right 6 gray fence tiles
        
        # Bottom row (y=31): 6 gray fences (0-5), 6 green king area (6-11), 6 gray fences (12-17)
        (0, 31), (1, 31), (2, 31), (3, 31), (4, 31), (5, 31),     # Left 6 gray fence tiles
        (12, 31), (13, 31), (14, 31), (15, 31), (16, 31), (17, 31), # Right 6 gray fence tiles
    ]

    towers = [
        # (中心, 半宽, 半高, player_id)：塔是矩形（2026-09-09 用户勘误：引擎此前全员
        # 圆形是错的）。公主塔 3×3（half 1.5/1.5）、国王塔 4×4（half 2.0/2.0）。
        # 半开区间约定：矩形 = 中心±half（边界属塔内）；部署/占位判定用脚点。
        (BLUE_LEFT_TOWER, 1.5, 1.5, 0),  # Princess tower, 3x3
        (BLUE_RIGHT_TOWER, 1.5, 1.5, 0),  # Princess tower, 3x3
        (BLUE_KING_TOWER, 2.0, 2.0, 0),  # King tower, 4x4
        (RED_LEFT_TOWER, 1.5, 1.5, 1),  # Princess tower, 3x3
        (RED_RIGHT_TOWER, 1.5, 1.5, 1),  # Princess tower, 3x3
        (RED_KING_TOWER, 2.0, 2.0, 1)  # King tower, 4x4
    ]

    @staticmethod
    def dist_to_rect(px: float, py: float, cx: float, cy: float, hw: float, hh: float) -> float:
        """点到轴对齐矩形的距离：矩形内 0，矩形外=到最近边的欧氏距离。"""
        dx = max(abs(px - cx) - hw, 0.0)
        dy = max(abs(py - cy) - hh, 0.0)
        return math.hypot(dx, dy)

    def refresh_tower_alive_cache(self, battle_state):
        """塔存活状态缓存（battle.step 每帧先刷新一次）。

        tower_rect_dist 是行军/占位热路径（is_position_occupied_by_building
        每局调用数百万次），不能每次线性扫 entities 判活。6 个布尔位的
        刷新成本 O(实体数) 每帧一次，与 building_positions 同节奏。"""
        self._tower_alive = tuple(
            self._is_tower_alive(tpos, pid, battle_state)
            for tpos, _hw, _hh, pid in self.towers)

    def tower_rect_dist(self, pos: Position, battle_state=None, player_filter=None) -> float:
        """pos 到所有存活塔矩形的最近距离（无存活塔/全部过滤 → inf）。

        battle_state=None 时视为全部存活（静态几何查询）；player_filter
        限定塔的归属方（None=双方）。battle_state 给定时用 refresh 缓存
        （战斗内每帧刷新）；缓存未建则即时建（冷启动安全）。"""
        cache = getattr(self, '_tower_alive', None)
        if battle_state is not None and cache is None:
            self.refresh_tower_alive_cache(battle_state)
            cache = self._tower_alive
        best = float('inf')
        for i, (tower_pos, hw, hh, pid) in enumerate(self.towers):
            if player_filter is not None and pid != player_filter:
                continue
            if battle_state is not None and not cache[i]:
                continue
            d = self.dist_to_rect(pos.x, pos.y, tower_pos.x, tower_pos.y, hw, hh)
            if d < best:
                best = d
        return best

    def behind_king_zone(self, player_id: int) -> tuple:
        """王塔身后 1 格带宽的禁区 (x1, y1, x2, y2)（建筑禁放带）。

        王塔 4×4 已贴住底线方向（BLUE 王 y∈[1,5]，其身后 = y<1 一行，恰 1 格宽；
        RED 王 y∈[27,31]，身后 = y>31 出界）。真实 CR 王塔后没有空间，本引擎
        蓝方王塔后残留一行可放格 → 显式禁建筑（部队仍可走位经过）。"""
        if player_id == 0:
            return (7.0, 0.0, 11.0, 1.0)
        return (7.0, 31.0, 11.0, 32.0)

    def is_behind_king(self, pos: Position, player_id: int) -> bool:
        x1, y1, x2, y2 = self.behind_king_zone(player_id)
        return x1 <= pos.x < x2 and y1 <= pos.y < y2

    def is_valid_position(self, pos):
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def is_blocked_tile(self, x: int, y: int) -> bool:
        return (x, y) in self.BLOCKED_TILES
    
    def is_walkable(self, pos: Position) -> bool:
        x, y = pos.x, pos.y
        int_pos = (int(x), int(y))
        if int_pos in walkable_cache:
            return walkable_cache[int_pos]
        if not self.is_valid_position(pos) or self.is_blocked_tile(int(pos.x), int(pos.y)):
            walkable_cache[int_pos] = False
        elif self.RIVER_Y1 <= pos.y <= self.RIVER_Y2:
            on_left_bridge = 2.0 <= pos.x < 5.0
            on_right_bridge = 13.0 <= pos.x < 16.0
            walkable_cache[int_pos] = on_left_bridge or on_right_bridge
        else:
            walkable_cache[int_pos] = True
        return walkable_cache[int_pos]

    def _is_tower_alive(self, tower_pos: Position, player_id: int, battle_state) -> bool:
        """Check if tower at given position is still alive"""
        for entity in battle_state.entities.values():
            if (hasattr(entity, 'position') and
                entity.position.x == tower_pos.x and
                entity.position.y == tower_pos.y and
                getattr(entity, 'player', getattr(entity, 'player_id', -1)) == player_id):
                return entity.is_alive
        return False  # Tower not found, assume dead

    def is_tower_tile(self, pos: Position, battle_state=None) -> bool:
        """Check if position overlaps with any living tower's occupied area"""
        for tower_pos, hw, hh, player_id in self.towers:
            # Check if tower is still alive (if battle_state provided)
            if battle_state:
                tower_alive = self._is_tower_alive(tower_pos, player_id, battle_state)
                if not tower_alive:
                    continue
            # 矩形占位（含边界）：脚点落在 中心±半宽/半高 内即塔占位
            if (abs(pos.x - tower_pos.x) <= hw) and (abs(pos.y - tower_pos.y) <= hh):
                return True

        return False
    
    def get_deploy_zones(self, player_id: int, battle_state=None):
        """Get valid deployment zones for a player (x1, y1, x2, y2)
        Expands to include bridge areas and 4 tiles back when towers are destroyed"""
        zones = []
        
        if player_id == 0:  # Player 0 (bottom half)
            # Basic deployment zone (bottom half, excluding river)
            zones.append((0, 1, self.width, self.RIVER_Y1))  # y=1 to y=14
            zones.append((6, 0, 12, 6))  # Behind blue king: x=6-11, y=0-5 (6 tiles behind own king, including edge row)
            if battle_state:
                # If red left tower is destroyed, blue player can spawn on left half of arena and 4 tiles back
                if battle_state.players[1].left_tower_hp <= 0:
                    zones.append((0, self.RIVER_Y2 + 1, 9, self.RIVER_Y2 + 5))  # Left half: x=0-8, y=17-20
                # If red right tower is destroyed, blue player can spawn on right half of arena and 4 tiles back  
                if battle_state.players[1].right_tower_hp <= 0:
                    zones.append((9, self.RIVER_Y2 + 1, self.width, self.RIVER_Y2 + 5))  # Right half: x=9-17, y=17-20
        else:  # Player 1 (top half)
            zones.append((0, self.RIVER_Y2 + 1, self.width, 31))  # y=17 to y=30
            zones.append((6, 26, 12, 32))  # Behind red king: x=6-11, y=26-31 (6 tiles behind own king, including edge row)
            if battle_state:
                # If blue left tower is destroyed, red player can spawn on left half of arena and 4 tiles back
                if battle_state.players[0].left_tower_hp <= 0:
                    zones.append((0, self.RIVER_Y1 - 4, 9, self.RIVER_Y1))  # Left half: x=0-8, y=11-14
                # If blue right tower is destroyed, red player can spawn on right half of arena and 4 tiles back
                if battle_state.players[0].right_tower_hp <= 0:
                    zones.append((9, self.RIVER_Y1 - 4, self.width, self.RIVER_Y1))  # Right half: x=9-17, y=11-14
        return zones
    
    def can_deploy_at(self, pos: Position, player_id: int, battle_state=None, is_spell=False, spell_obj=None, is_building=False) -> bool:
        """Check if position is valid for deployment"""
        # Check basic bounds
        if not self.is_valid_position(pos) or self.is_blocked_tile(int(pos.x), int(pos.y)): return False
        if not is_spell and self.is_tower_tile(pos, battle_state): return False
        # 王塔身后 1 格宽禁建筑（部队可走位经过/部署；用户口径 2026-09-09：
        # 王塔 4×4 后只有 1 格宽，不应有建筑被放出）
        if is_building and self.is_behind_king(pos, player_id): return False
        if is_spell and not self._is_rolling_projectile_spell(spell_obj): return True
        if (pos.y == 0 or pos.y == 31) and not (6 <= pos.x <= 11): return False
        zones = self.get_deploy_zones(player_id, battle_state)
        for x1, y1, x2, y2 in zones:
            if x1 <= pos.x < x2 and y1 <= pos.y < y2:
                return True
        if player_id == 0 and pos.y == 0 and 6 <= pos.x <= 11:
            return True
        elif player_id == 1 and pos.y == 31 and 6 <= pos.x <= 11:
            return True
        return False

    def get_tower_blocked_x_ranges(self, y: float, battle_state=None):
        """Get X coordinate ranges blocked by towers at a specific Y coordinate"""
        blocked_ranges = []
        for tower_pos, hw, hh, player_id in self.towers:
            # Check if tower is still alive
            if battle_state:
                tower_alive = self._is_tower_alive(tower_pos, player_id, battle_state)
                if not tower_alive:
                    continue
            # Check if Y coordinate intersects with tower area
            if abs(y - tower_pos.y) <= hh:
                # Y coordinate overlaps with tower, add X range to blocked list
                blocked_ranges.append((tower_pos.x - hw, tower_pos.x + hw))
        return blocked_ranges

if __name__ == '__main__':
    print(TileGrid().is_blocked_tile(int(6.9), int(16.001)))
