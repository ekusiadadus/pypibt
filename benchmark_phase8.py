"""
Phase 8 (8A, 8B, 8C) Benchmark

Tests the three new Phase 8 optimization variants:
- Phase 8A: MCTS-Enhanced PIBT (high performance, high complexity)
- Phase 8B: Adaptive Diverse Beam Search (balanced, recommended)
- Phase 8C: JPS-Accelerated DistTable (low complexity, sparse graph optimization)

Compares against baseline and best Phase 3/4 variants.
"""

import json
import time
from pathlib import Path

from pypibt import PIBT, get_grid, get_scenario
from pypibt.adaptive_beam_pibt import AdaptiveBeamPIBT
from pypibt.jps_dist_table import JPSDistTable
from pypibt.mcts_pibt import MCTSPIBT


def calculate_metrics(configs, starts, goals, execution_time):
    """Calculate comprehensive metrics for a solution."""
    if not configs or len(configs) == 0:
        return {
            "success": False,
            "timesteps": float("inf"),
            "makespan": float("inf"),
            "sum_of_costs": float("inf"),
            "execution_time": execution_time,
            "conflicts": float("inf"),
        }

    N = len(starts)

    # Check if all agents reached their goals
    final_config = configs[-1]
    all_reached = all(final_config[i] == goals[i] for i in range(N))

    # Calculate individual costs
    individual_costs = []
    for i in range(N):
        goal_reached_at = None
        for t, config in enumerate(configs):
            if config[i] == goals[i]:
                goal_reached_at = t
                break
        if goal_reached_at is None:
            individual_costs.append(len(configs))
        else:
            individual_costs.append(goal_reached_at)

    sum_of_costs = sum(individual_costs)
    makespan = max(individual_costs) if individual_costs else len(configs)

    # Count conflicts
    conflicts = 0
    for config in configs:
        positions = {}
        for i, pos in enumerate(config):
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(i)
        for pos, agents in positions.items():
            if len(agents) > 1:
                conflicts += len(agents) * (len(agents) - 1) // 2

    return {
        "success": all_reached,
        "timesteps": len(configs),
        "makespan": makespan,
        "sum_of_costs": sum_of_costs,
        "execution_time": execution_time,
        "conflicts": conflicts,
    }


# Test configurations
test_configs = [
    {
        "name": "Small (50 agents)",
        "map": "assets/random-32-32-10.map",
        "scen": "assets/random-32-32-10-random-1.scen",
        "num_agents": 50,
    },
    {
        "name": "Medium (100 agents)",
        "map": "assets/random-32-32-10.map",
        "scen": "assets/random-32-32-10-random-1.scen",
        "num_agents": 100,
    },
    {
        "name": "Large (200 agents)",
        "map": "assets/random-32-32-10.map",
        "scen": "assets/random-32-32-10-random-1.scen",
        "num_agents": 200,
    },
]


def run_baseline(grid, starts, goals):
    """Run baseline PIBT."""
    pibt = PIBT(
        grid,
        starts,
        goals,
        seed=0,
        enable_hindrance=False,
        enable_regret_learning=False,
        enable_anytime=False,
        enable_lns=False,
    )
    return pibt.run(max_timestep=2000)


def run_phase3d(grid, starts, goals):
    """Run Phase 3d (Optuna Best) - current best lightweight optimization."""
    pibt = PIBT(
        grid,
        starts,
        goals,
        seed=0,
        enable_hindrance=True,
        hindrance_weight=0.31,
        priority_increment=0.59,
        enable_regret_learning=False,
        enable_anytime=False,
        enable_lns=False,
    )
    return pibt.run(max_timestep=2000)


def run_phase8a_mcts(grid, starts, goals):
    """Run Phase 8A: MCTS-Enhanced PIBT."""
    mcts_pibt = MCTSPIBT(
        grid,
        starts,
        goals,
        seed=0,
        num_rollouts=250,
        rollout_depth=100,
        time_limit_ms=5000.0,  # 5 seconds
    )
    return mcts_pibt.run(max_timestep=2000)


def run_phase8b_adaptive_beam(grid, starts, goals):
    """Run Phase 8B: Adaptive Diverse Beam Search."""
    beam_pibt = AdaptiveBeamPIBT(
        grid,
        starts,
        goals,
        seed=0,
        beam_width=5,
        time_limit_ms=3000.0,  # 3 seconds
        diversity_weight=0.5,
        adaptive_beam=True,
    )
    return beam_pibt.run(max_timestep=2000)


def run_phase8c_jps(grid, starts, goals):
    """
    Run Phase 8C: PIBT with JPS-Accelerated DistTable.

    Note: Phase 8C optimizes initialization time, not runtime.
    We measure the improvement in dist_table creation.
    """
    import time as time_module

    # Measure standard DistTable initialization
    from pypibt.dist_table import DistTable

    start_std = time_module.time()
    std_tables = [DistTable(grid, goal) for goal in goals]
    std_init_time = time_module.time() - start_std

    # Measure JPS DistTable initialization
    start_jps = time_module.time()
    jps_tables = [JPSDistTable(grid, goal) for goal in goals]
    jps_init_time = time_module.time() - start_jps

    print(
        f"      DistTable init: Standard={std_init_time:.4f}s, "
        f"JPS={jps_init_time:.4f}s "
        f"({((std_init_time - jps_init_time) / std_init_time * 100):+.1f}% speedup)"
    )

    # Run PIBT with JPS dist tables (replace in PIBT class temporarily)
    pibt = PIBT(
        grid,
        starts,
        goals,
        seed=0,
        enable_hindrance=True,
        hindrance_weight=0.31,
        priority_increment=0.59,
    )

    # Replace dist_tables with JPS versions
    pibt.dist_tables = jps_tables

    return pibt.run(max_timestep=2000)


# Benchmark variants
variants = [
    {
        "phase": "Phase 1",
        "name": "Baseline PIBT",
        "runner": run_baseline,
    },
    {
        "phase": "Phase 3d",
        "name": "Optuna Best (Best Lightweight)",
        "runner": run_phase3d,
    },
    {
        "phase": "Phase 8A",
        "name": "MCTS-Enhanced PIBT",
        "runner": run_phase8a_mcts,
    },
    {
        "phase": "Phase 8B",
        "name": "Adaptive Beam Search",
        "runner": run_phase8b_adaptive_beam,
    },
    {
        "phase": "Phase 8C",
        "name": "JPS-Accelerated DistTable",
        "runner": run_phase8c_jps,
    },
]


print("=" * 100)
print("PHASE 8 OPTIMIZATION BENCHMARK")
print("=" * 100)
print()
print("Testing:")
print("  - Phase 8A: MCTS-Enhanced PIBT (high performance, 2-5s)")
print("  - Phase 8B: Adaptive Diverse Beam Search (balanced, 1-3s)")
print("  - Phase 8C: JPS-Accelerated DistTable (init speedup, sparse graphs)")
print()
print("Comparing against:")
print("  - Phase 1: Baseline PIBT")
print("  - Phase 3d: Optuna Best (current best lightweight)")
print()

all_results = {}

for config in test_configs:
    print(f"\n{'=' * 100}")
    print(f"Test Configuration: {config['name']}")
    print(f"{'=' * 100}\n")

    grid = get_grid(config["map"])
    starts, goals = get_scenario(config["scen"], config["num_agents"])

    config_results = []

    for variant in variants:
        print(f"  [{variant['phase']}] {variant['name']:35s}... ", end="", flush=True)

        try:
            start_time = time.time()
            configs = variant["runner"](grid, starts, goals)
            execution_time = time.time() - start_time

            metrics = calculate_metrics(configs, starts, goals, execution_time)

            config_results.append(
                {
                    "phase": variant["phase"],
                    "name": variant["name"],
                    **metrics,
                }
            )

            if metrics["success"]:
                print(
                    f"{metrics['timesteps']:4d} steps | "
                    f"SoC={metrics['sum_of_costs']:6d} | "
                    f"MS={metrics['makespan']:4d} | "
                    f"Conflicts={metrics['conflicts']:3d} | "
                    f"{execution_time:6.2f}s"
                )
            else:
                print(f"FAILED | {execution_time:6.2f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            config_results.append(
                {
                    "phase": variant["phase"],
                    "name": variant["name"],
                    "success": False,
                    "timesteps": float("inf"),
                    "makespan": float("inf"),
                    "sum_of_costs": float("inf"),
                    "execution_time": 0.0,
                    "conflicts": float("inf"),
                }
            )

    all_results[config["name"]] = config_results

    # Print summary
    print("\n  Summary:")
    baseline = next((r for r in config_results if r["phase"] == "Phase 1"), None)

    if baseline and baseline["success"]:
        print(f"    Baseline: {baseline['timesteps']:.0f} steps, SoC={baseline['sum_of_costs']}")

        for result in config_results:
            if result["success"] and result["phase"] != "Phase 1":
                timesteps_improvement = (
                    (baseline["timesteps"] - result["timesteps"]) / baseline["timesteps"] * 100
                )
                soc_improvement = (
                    (baseline["sum_of_costs"] - result["sum_of_costs"])
                    / baseline["sum_of_costs"]
                    * 100
                )
                print(
                    f"    {result['phase']:10s} ({result['name']:35s}): "
                    f"{result['timesteps']:4.0f} steps ({timesteps_improvement:+.1f}%), "
                    f"SoC={result['sum_of_costs']:6.0f} ({soc_improvement:+.1f}%), "
                    f"{result['execution_time']:.2f}s"
                )

# Overall summary
print("\n" + "=" * 100)
print("OVERALL SUMMARY: PHASE 8 PERFORMANCE")
print("=" * 100)
print()

for config_name, config_results in all_results.items():
    print(f"{config_name}:")

    baseline = next((r for r in config_results if r["phase"] == "Phase 1"), None)
    successful = [r for r in config_results if r["success"]]

    if len(successful) == 0:
        print("  No successful runs")
        continue

    # Best by timesteps
    best_timesteps = min(successful, key=lambda x: x["timesteps"])
    # Best by SoC
    best_soc = min(successful, key=lambda x: x["sum_of_costs"])
    # Fastest execution
    fastest = min(successful, key=lambda x: x["execution_time"])

    print(f"  Best Timesteps: {best_timesteps['name']}")
    print(f"    {best_timesteps['timesteps']:.0f} steps")
    if baseline and baseline["success"]:
        improvement = (
            (baseline["timesteps"] - best_timesteps["timesteps"]) / baseline["timesteps"] * 100
        )
        print(f"    {improvement:+.1f}% vs baseline")

    print(f"\n  Best Sum of Costs: {best_soc['name']}")
    print(f"    SoC = {best_soc['sum_of_costs']:.0f}")
    if baseline and baseline["success"]:
        improvement = (
            (baseline["sum_of_costs"] - best_soc["sum_of_costs"])
            / baseline["sum_of_costs"]
            * 100
        )
        print(f"    {improvement:+.1f}% vs baseline")

    print(f"\n  Fastest Execution: {fastest['name']}")
    print(f"    {fastest['execution_time']:.2f}s")
    print()

# Save results
output_file = "benchmark_phase8_results.json"
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("\nPhase 8 benchmark complete!")
