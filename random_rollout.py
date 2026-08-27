import numpy as np

from monster_gridworld import MonsterGridworld, MonsterGridworldConfig


ACTION_NAMES = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}


env = MonsterGridworld(
    MonsterGridworldConfig(
        episode_length=25,
    )
)

rng = np.random.default_rng(42)

obs, info = env.reset(seed=42)

print("Initial state")
print(env.render_ascii())
print(f"Shield inventory: {obs['shield_inventory']}")
print("=" * 40)

for step in range(25):
    action = int(rng.integers(0, 4))

    obs, reward, terminated, truncated, info = env.step(action)

    grid = obs["grid"]

    n_monsters = int(grid[:, :, 1].sum())
    n_shields = int(grid[:, :, 2].sum())
    n_apples = int(grid[:, :, 3].sum())

    print(
        f"\nStep {step + 1:02d}"
        f" | {ACTION_NAMES[action]}"
        f" | reward={reward:+.0f}"
        f" | inventory={obs['shield_inventory']}"
        f" | monsters={n_monsters}"
        f" | shields_on_grid={n_shields}"
        f" | apples={n_apples}"
    )

    print(env.render_ascii())

    if terminated or truncated:
        break
