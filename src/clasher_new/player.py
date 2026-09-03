from card_utils import Card

# Hardcoded: max elixir is 10

class PlayerState:
    def __init__(self, player_id, cycle_queue, elixir, tower_hps=(4824, 3052, 3052)):
        self.player_id = player_id
        self.cycle = cycle_queue[:]
        self.elixir = elixir
        self.king_tower_hp, self.left_tower_hp, self.right_tower_hp = tower_hps
        self.last_card = None  # M1: 镜像法术需要记录上一张使用的卡
        self.evo_slots = set()  # M4: 卡组携带的觉醒位（≤2 张卡名）
        self.evo_plays = {}     # M4: 觉醒周期计数（card_name → 已打出次数）

    def set_evolution_slots(self, cards):
        """M4：声明该卡组携带的觉醒卡（最多 2 个觉醒位）"""
        self.evo_slots = set(list(cards)[:2])
    
    def regenerate_elixir(self, dt: float, base_regen_time: float = 2.8):
        elixir_per_second = 1.0 / base_regen_time
        self.elixir = min(10, self.elixir + elixir_per_second * dt)
    
    def can_play_card(self, card_name):
        return (card_name in self.cycle[:4] and
                self.elixir >= Card(card_name).elixir and
                self.king_tower_hp > 0)
    
    def play_card(self, card_name):
        """Update the player's deck when playing a card."""
        if not self.can_play_card(card_name): return False
        self.elixir -= Card(card_name).elixir
        self.cycle.remove(card_name)
        self.cycle.append(card_name)
        if card_name != 'Mirror': self.last_card = card_name
        return True

    def get_next_card(self):
        """Return the next card in cycle, if known."""
        return self.cycle[4]
    
    def get_crown_count(self) -> int:
        """Get number of crowns (destroyed towers)"""
        if self.king_tower_hp <= 0:
            return 3
        return int(self.left_tower_hp <= 0) + int(self.right_tower_hp <= 0)

if __name__ == '__main__':
    deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
    p = PlayerState(0, deck, 10)
    print(p.cycle, p.elixir)
    p.play_card('Knight')
    print(p.cycle, p.elixir)
    p.play_card('Knight')
    print(p.cycle, p.elixir)