"""
Quick test for improved Phase 8A (MCTS-PIBT).
"""

import time

from pypibt import get_grid, get_scenario
from pypibt.mcts_pibt import MCTSPIBT


def test_phase8a_improved():
    """Test improved MCTS-PIBT on small instance."""
    print("Testing improved Phase 8A (MCTS-PIBT)...")
    print("=" * 80)

    grid = get_grid("assets/random-32-32-10.map")
    starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 50)

    print(f"Map: 32x32, Agents: 50")
    print(f"Testing with rollout_depth=50, num_rollouts=250, time_limit=5s")
    print()

    start_time = time.time()

    mcts = MCTSPIBT(
        grid,
        starts,
        goals,
        seed=0,
        num_rollouts=250,
        rollout_depth=50,
        time_limit_ms=5000.0,
    )

    configs = mcts.run(max_timestep=2000)

    execution_time = time.time() - start_time

    # Verify solution
    timesteps = len(configs)
    all_at_goal = configs[-1] == goals

    print()
    print("=" * 80)
    print("Results:")
    print(f"  Timesteps: {timesteps}")
    print(f"  Execution time: {execution_time:.2f}s")
    print(f"  All agents at goal: {all_at_goal}")
    print(f"  Success: {all_at_goal and timesteps < 2000}")
    print()

    if all_at_goal and timesteps < 100:
        print("✅ IMPROVED! Phase 8A now produces valid solutions under 100 steps!")
    elif all_at_goal:
        print("✅ Valid solution, but could be more optimal")
    else:
        print("❌ Failed to reach goal")

    return timesteps, execution_time, all_at_goal


if __name__ == "__main__":
    test_phase8a_improved()
