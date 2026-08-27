from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Pos = Tuple[int, int]

# Observation channels, matching the paper's 14 x 14 x 4 description.
AGENT = 0
MONSTER = 1
SHIELD = 2
APPLE = 3
N_CHANNELS = 4

# Reward specification from Shah et al. (2022), Appendix C.2.
# Keeping every reward in one place makes the sparse reward function explicit.
APPLE_PICKUP_REWARD = 5.0
UNSHIELDED_ATTACK_REWARD = -1.0
SHIELDED_ENCOUNTER_REWARD = 0.0
SHIELD_PICKUP_REWARD = 0.0
MOVEMENT_REWARD = 0.0

STEP_EVENT_NAMES = (
    "apples_collected",
    "shields_collected",
    "monsters_destroyed",
    "unshielded_attacks",
)

# Cardinal actions. The paper appendix does not specify the exact action set.
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTION_DELTAS: Dict[int, Pos] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}


@dataclass
class MonsterGridworldConfig:
    size: int = 14
    n_monsters: int = 5
    n_shields: int = 5
    n_apples: int = 5
    max_inventory: int = 10
    episode_length: int = 25

    # Paper: every turn monsters have a 20% chance of moving twice.
    # We implement this as one global extra monster-movement phase.
    monster_double_move_prob: float = 0.20

    # The paper says apples/shields are initially present but does not fully
    # specify their respawn dynamics. Keeping counts constant is a useful
    # replica choice for long episodes and can be toggled off.
    respawn_apples: bool = True
    respawn_shields: bool = True

    # Monsters chase by greedily reducing Manhattan distance. Ties are random.
    randomize_monster_order: bool = True


@dataclass
class State:
    agent: Pos
    monsters: Set[Pos]
    shields_on_grid: Set[Pos]
    apples: Set[Pos]
    shield_inventory: int
    step_count: int = 0


class MonsterGridworld:
    """Small, dependency-light Monster Gridworld research replica.

    API intentionally resembles Gymnasium:
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(action)

    Observation:
        obs["grid"]: uint8 array [H, W, 4]
        obs["shield_inventory"]: int in [0, max_inventory]

    Step order:
        1. Agent moves.
        2. Agent picks up apple/shield on its destination cell.
        3. Monsters move once toward the agent.
        4. With probability p, monsters move a second time.

    Collision rule used here:
        - Monster attempts to enter agent cell:
            shield > 0 -> monster destroyed, one shield consumed, reward 0
            shield == 0 -> reward -1, monster remains in its old cell
        - Agent attempts to enter monster cell: same resolution; on an
          unshielded collision the agent remains in its old cell.

    These choices preserve non-overlapping entity locations, making the
    4-channel observation genuinely one-hot per occupied cell.

    Reward is additive within a step. For example, two unshielded attacks in
    the same monster-movement phase produce -2, while an attack followed by an
    apple pickup produces -1 + 1 = 0.
    """

    def __init__(self, config: Optional[MonsterGridworldConfig] = None):
        self.cfg = config or MonsterGridworldConfig()
        self.rng = np.random.default_rng()
        self.state: Optional[State] = None
        self._last_events = self._empty_step_events()

    # ---------- public API ----------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        positions = self._sample_unique_positions(
            1 + self.cfg.n_monsters + self.cfg.n_shields + self.cfg.n_apples
        )
        i = 0
        agent = positions[i]
        i += 1
        monsters = set(positions[i : i + self.cfg.n_monsters])
        i += self.cfg.n_monsters
        shields = set(positions[i : i + self.cfg.n_shields])
        i += self.cfg.n_shields
        apples = set(positions[i : i + self.cfg.n_apples])

        self.state = State(
            agent=agent,
            monsters=monsters,
            shields_on_grid=shields,
            apples=apples,
            shield_inventory=0,
            step_count=0,
        )
        self._last_events = self._empty_step_events()
        self._validate_state()
        return self.observation(), self.info()

    def step(self, action: int):
        self._require_state()
        if action not in ACTION_DELTAS:
            raise ValueError(f"action must be one of {sorted(ACTION_DELTAS)}")

        reward = MOVEMENT_REWARD
        self._last_events = self._empty_step_events()

        # The agent acts and collects before monsters respond.
        reward += self._move_agent(action)

        reward += self._move_monsters_once()
        if self.state.monsters and self.rng.random() < self.cfg.monster_double_move_prob:
            reward += self._move_monsters_once()
        self.state.step_count += 1

        truncated = self.state.step_count >= self.cfg.episode_length
        terminated = False  # no terminal condition besides time limit in this replica

        self._validate_state()
        return self.observation(), reward, terminated, truncated, self.info()

    def observation(self) -> Dict[str, object]:
        self._require_state()
        grid = np.zeros((self.cfg.size, self.cfg.size, N_CHANNELS), dtype=np.uint8)
        r, c = self.state.agent
        grid[r, c, AGENT] = 1
        for r, c in self.state.monsters:
            grid[r, c, MONSTER] = 1
        for r, c in self.state.shields_on_grid:
            grid[r, c, SHIELD] = 1
        for r, c in self.state.apples:
            grid[r, c, APPLE] = 1
        return {
            "grid": grid,
            "shield_inventory": int(self.state.shield_inventory),
        }

    def info(self) -> Dict[str, int]:
        self._require_state()
        return {
            "step": self.state.step_count,
            "n_monsters": len(self.state.monsters),
            "n_shields_on_grid": len(self.state.shields_on_grid),
            "n_apples_on_grid": len(self.state.apples),
            "shield_inventory": self.state.shield_inventory,
            **self._last_events,
        }

    def get_state(self) -> State:
        self._require_state()
        return copy.deepcopy(self.state)

    def set_state(self, state: State) -> None:
        self.state = copy.deepcopy(state)
        self._last_events = self._empty_step_events()
        self._validate_state()

    def make_state(
        self,
        *,
        agent: Pos,
        monsters: Iterable[Pos] = (),
        shields_on_grid: Iterable[Pos] = (),
        apples: Iterable[Pos] = (),
        shield_inventory: int = 0,
        step_count: int = 0,
    ) -> State:
        """Convenience constructor for controlled / counterfactual states."""
        state = State(
            agent=agent,
            monsters=set(monsters),
            shields_on_grid=set(shields_on_grid),
            apples=set(apples),
            shield_inventory=shield_inventory,
            step_count=step_count,
        )
        old = self.state
        self.state = copy.deepcopy(state)
        self._validate_state()
        self.state = old
        return state

    def remove_all_monsters(self) -> State:
        """Return a counterfactual copy of the current state with no monsters."""
        state = self.get_state()
        state.monsters = set()
        return state

    def render_ascii(self) -> str:
        self._require_state()
        chars = np.full((self.cfg.size, self.cfg.size), ".", dtype="<U1")
        for r, c in self.state.apples:
            chars[r, c] = "a"
        for r, c in self.state.shields_on_grid:
            chars[r, c] = "S"
        for r, c in self.state.monsters:
            chars[r, c] = "M"
        r, c = self.state.agent
        chars[r, c] = "P"
        body = "\n".join(" ".join(row) for row in chars)
        return (
            body
            + f"\nstep={self.state.step_count} inventory={self.state.shield_inventory} "
              f"monsters={len(self.state.monsters)}"
        )

    # ---------- dynamics ----------

    def _move_monsters_once(self) -> float:
        reward = MOVEMENT_REWARD
        monsters = list(self.state.monsters)
        if self.cfg.randomize_monster_order:
            self.rng.shuffle(monsters)

        # Update sequentially so two monsters never occupy the same square.
        for monster in monsters:
            if monster not in self.state.monsters:
                continue  # it may have been destroyed earlier in this phase

            target = self._monster_target(monster)
            if target == self.state.agent:
                reward += self._resolve_monster_encounter(monster)
                continue

            if target != monster:
                self.state.monsters.remove(monster)
                self.state.monsters.add(target)

        return reward

    def _monster_target(self, monster: Pos) -> Pos:
        candidates: List[Pos] = []
        for delta in ACTION_DELTAS.values():
            p = self._add(monster, delta)
            if not self._in_bounds(p):
                continue
            if p == self.state.agent:
                candidates.append(p)
                continue
            if p in self.state.monsters:
                continue
            if p in self.state.shields_on_grid or p in self.state.apples:
                continue
            candidates.append(p)

        if not candidates:
            return monster

        agent = self.state.agent
        distances = np.array([self._manhattan(p, agent) for p in candidates])
        best = np.flatnonzero(distances == distances.min())
        return candidates[int(self.rng.choice(best))]

    def _move_agent(self, action: int) -> float:
        reward = MOVEMENT_REWARD
        proposed = self._add(self.state.agent, ACTION_DELTAS[action])
        if not self._in_bounds(proposed):
            proposed = self.state.agent

        if proposed in self.state.monsters:
            reward += self._resolve_monster_encounter(proposed)
            if proposed in self.state.monsters:
                # The unshielded agent cannot enter an occupied cell.
                return reward
        self.state.agent = proposed

        return reward + self._collect_at_agent()

    def _resolve_monster_encounter(self, monster: Pos) -> float:
        """Resolve one attack, identically for either collision direction."""
        if self.state.shield_inventory == 0:
            self._last_events["unshielded_attacks"] += 1
            return UNSHIELDED_ATTACK_REWARD

        self.state.shield_inventory -= 1
        self.state.monsters.remove(monster)
        self._last_events["monsters_destroyed"] += 1
        return SHIELDED_ENCOUNTER_REWARD

    def _collect_at_agent(self) -> float:
        """Collect an item at the agent's position and return its reward."""
        reward = MOVEMENT_REWARD
        pos = self.state.agent
        if pos in self.state.apples:
            self.state.apples.remove(pos)
            self._last_events["apples_collected"] += 1
            reward += APPLE_PICKUP_REWARD
            if self.cfg.respawn_apples:
                self._spawn_one(self.state.apples)

        if pos in self.state.shields_on_grid:
            self.state.shields_on_grid.remove(pos)
            self._last_events["shields_collected"] += 1
            self.state.shield_inventory = min(
                self.cfg.max_inventory, self.state.shield_inventory + 1
            )
            reward += SHIELD_PICKUP_REWARD
            if self.cfg.respawn_shields:
                self._spawn_one(self.state.shields_on_grid)

        return reward

    # ---------- utilities ----------

    @staticmethod
    def _empty_step_events() -> Dict[str, int]:
        return {name: 0 for name in STEP_EVENT_NAMES}

    def _occupied(self) -> Set[Pos]:
        return (
            {self.state.agent}
            | set(self.state.monsters)
            | set(self.state.shields_on_grid)
            | set(self.state.apples)
        )

    def _spawn_one(self, collection: Set[Pos]) -> None:
        occupied = self._occupied()
        empties = [
            (r, c)
            for r in range(self.cfg.size)
            for c in range(self.cfg.size)
            if (r, c) not in occupied
        ]
        if not empties:
            return
        collection.add(empties[int(self.rng.integers(len(empties)))])

    def _sample_unique_positions(self, n: int) -> List[Pos]:
        total = self.cfg.size * self.cfg.size
        if n > total:
            raise ValueError("more entities requested than grid cells")
        inds = self.rng.choice(total, size=n, replace=False)
        return [(int(i // self.cfg.size), int(i % self.cfg.size)) for i in inds]

    def _validate_state(self) -> None:
        self._require_state()
        if not (0 <= self.state.shield_inventory <= self.cfg.max_inventory):
            raise ValueError("shield_inventory out of bounds")

        groups: Sequence[Set[Pos]] = [
            {self.state.agent},
            set(self.state.monsters),
            set(self.state.shields_on_grid),
            set(self.state.apples),
        ]
        all_positions: List[Pos] = [p for g in groups for p in g]
        if len(set(all_positions)) != len(all_positions):
            raise ValueError("entities may not overlap in this replica")
        if not all(self._in_bounds(p) for p in all_positions):
            raise ValueError("entity position out of bounds")

    def _require_state(self) -> None:
        if self.state is None:
            raise RuntimeError("call reset() or set_state() first")

    def _in_bounds(self, p: Pos) -> bool:
        return 0 <= p[0] < self.cfg.size and 0 <= p[1] < self.cfg.size

    @staticmethod
    def _add(a: Pos, b: Pos) -> Pos:
        return a[0] + b[0], a[1] + b[1]

    @staticmethod
    def _manhattan(a: Pos, b: Pos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


if __name__ == "__main__":
    env = MonsterGridworld()
    obs, info = env.reset(seed=42)
    print(env.render_ascii())
    print("grid shape:", obs["grid"].shape, "info:", info)
