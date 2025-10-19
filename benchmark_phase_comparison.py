"""
Comprehensive Phase 1-7 Comparison Benchmark

Compares all PIBT optimization phases across multiple scenarios:
- Phase 1: Baseline PIBT
- Phase 2: Distance Heuristic (implicit in baseline)
- Phase 3: Hindrance Heuristic + Regret Learning
- Phase 4: Anytime PIBT
- Phase 5: Large Neighborhood Search (LNS)
- Phase 6: Priority Learning (Q-learning)
- Phase 7D: MAPF-GPT Transformer (Not yet trained)

Metrics:
- Success rate (all agents reach goals)
- Average timesteps
- Sum of Costs (SoC)
- Makespan
- Execution time
- Conflicts
"""

import time
import json
from pathlib import Path
import numpy as np
from pypibt import PIBT, get_grid, get_scenario, is_valid_mapf_solution


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

    # Calculate individual costs (timesteps until reaching goal)
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

    # Sum of Costs: Total timesteps for all agents
    sum_of_costs = sum(individual_costs)

    # Makespan: Maximum timesteps any agent needed
    makespan = max(individual_costs) if individual_costs else len(configs)

    # Count conflicts (vertex collisions)
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


# Test configurations with varying difficulty
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
    {
        "name": "Very Large (400 agents)",
        "map": "assets/random-32-32-10.map",
        "scen": "assets/random-32-32-10-random-1.scen",
        "num_agents": 400,
    },
]

# Phase variants to test
phase_variants = [
    {
        "phase": "Phase 1",
        "name": "Baseline PIBT",
        "description": "Original PIBT with distance-based priorities",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": False,
            "enable_priority_learning": False,
            "priority_increment": 1.0,
        },
    },
    {
        "phase": "Phase 3a",
        "name": "Hindrance Heuristic",
        "description": "PIBT + Hindrance term (2025 lightweight optimization)",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": False,
            "enable_priority_learning": False,
            "priority_increment": 1.0,
        },
    },
    {
        "phase": "Phase 3b",
        "name": "Regret Learning",
        "description": "PIBT + Regret Learning (iterative improvement)",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": True,
            "regret_learning_iterations": 3,
            "regret_weight": 0.2,
            "enable_anytime": False,
            "enable_lns": False,
            "enable_priority_learning": False,
            "priority_increment": 1.0,
        },
    },
    {
        "phase": "Phase 3c",
        "name": "Hindrance + Regret",
        "description": "PIBT + Hindrance + Regret Learning (combined)",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": True,
            "regret_learning_iterations": 3,
            "regret_weight": 0.2,
            "enable_anytime": False,
            "enable_lns": False,
            "enable_priority_learning": False,
            "priority_increment": 1.0,
        },
    },
    {
        "phase": "Phase 3d",
        "name": "Optuna Best",
        "description": "Hindrance with Optuna-tuned hyperparameters",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.31,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": False,
            "enable_priority_learning": False,
            "priority_increment": 0.59,
        },
    },
    {
        "phase": "Phase 4",
        "name": "Anytime PIBT",
        "description": "Continuous improvement with beam search (time-limited)",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": False,
            "enable_anytime": True,
            "anytime_time_limit_ms": 5000.0,  # 5 seconds
            "anytime_beam_width": 5,
            "enable_lns": False,
            "enable_priority_learning": False,
            "priority_increment": 1.0,
        },
    },
    {
        "phase": "Phase 5",
        "name": "LNS (Adaptive)",
        "description": "Large Neighborhood Search with adaptive destroy",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": True,
            "lns_iterations": 10,
            "lns_destroy_size": 20,
            "lns_destroy_strategy": "adaptive",
            "enable_priority_learning": False,
            "priority_increment": 1.0,
        },
    },
]

print("=" * 100)
print("PHASE 1-7 COMPREHENSIVE COMPARISON BENCHMARK")
print("=" * 100)
print()

all_results = {}

for config in test_configs:
    print(f"\n{'=' * 100}")
    print(f"Test Configuration: {config['name']}")
    print(f"{'=' * 100}\n")

    grid = get_grid(config["map"])
    starts, goals = get_scenario(config["scen"], config["num_agents"])

    config_results = []

    for variant in phase_variants:
        print(f"  [{variant['phase']}] {variant['name']:30s}...", end=" ", flush=True)

        try:
            start_time = time.time()

            pibt = PIBT(grid, starts, goals, seed=0, **variant["params"])
            configs = pibt.run(max_timestep=2000)

            execution_time = time.time() - start_time

            # Calculate comprehensive metrics
            metrics = calculate_metrics(configs, starts, goals, execution_time)

            config_results.append(
                {
                    "phase": variant["phase"],
                    "name": variant["name"],
                    "description": variant["description"],
                    **metrics,
                }
            )

            # Print results
            if metrics["success"]:
                print(
                    f"{metrics['timesteps']:4d} steps | "
                    f"SoC={metrics['sum_of_costs']:6d} | "
                    f"MS={metrics['makespan']:4d} | "
                    f"Conflicts={metrics['conflicts']:3d} | "
                    f"{execution_time:6.2f}s"
                )
            else:
                print(f"FAILED (timeout) | {execution_time:6.2f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            config_results.append(
                {
                    "phase": variant["phase"],
                    "name": variant["name"],
                    "description": variant["description"],
                    "success": False,
                    "timesteps": float("inf"),
                    "makespan": float("inf"),
                    "sum_of_costs": float("inf"),
                    "execution_time": 0.0,
                    "conflicts": float("inf"),
                }
            )

    all_results[config["name"]] = config_results

    # Print summary for this configuration
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
                    (baseline["sum_of_costs"] - result["sum_of_costs"]) / baseline["sum_of_costs"] * 100
                )
                print(
                    f"    {result['phase']:10s} ({result['name']:30s}): "
                    f"{result['timesteps']:4.0f} steps ({timesteps_improvement:+.1f}%), "
                    f"SoC={result['sum_of_costs']:6.0f} ({soc_improvement:+.1f}%), "
                    f"{result['execution_time']:.2f}s"
                )
    else:
        print("    Baseline failed - cannot compute relative improvements")

# ================================================================================
# OVERALL SUMMARY
# ================================================================================

print("\n" + "=" * 100)
print("OVERALL SUMMARY: BEST PERFORMANCE BY PHASE")
print("=" * 100)
print()

for config_name, config_results in all_results.items():
    print(f"{config_name}:")

    baseline = next((r for r in config_results if r["phase"] == "Phase 1"), None)
    successful = [r for r in config_results if r["success"]]

    if len(successful) == 0:
        print("  No successful runs")
        continue

    # Best by timesteps (makespan)
    best_timesteps = min(successful, key=lambda x: x["timesteps"])
    # Best by Sum of Costs
    best_soc = min(successful, key=lambda x: x["sum_of_costs"])
    # Best by conflicts
    best_conflicts = min(successful, key=lambda x: x["conflicts"])
    # Fastest execution
    fastest = min(successful, key=lambda x: x["execution_time"])

    print(f"  Best Timesteps (Makespan): {best_timesteps['name']}")
    print(f"    {best_timesteps['timesteps']:.0f} steps")
    if baseline and baseline["success"]:
        improvement = (baseline["timesteps"] - best_timesteps["timesteps"]) / baseline["timesteps"] * 100
        print(f"    {improvement:+.1f}% vs baseline")

    print(f"\n  Best Sum of Costs: {best_soc['name']}")
    print(f"    SoC = {best_soc['sum_of_costs']:.0f}")
    if baseline and baseline["success"]:
        improvement = (baseline["sum_of_costs"] - best_soc["sum_of_costs"]) / baseline["sum_of_costs"] * 100
        print(f"    {improvement:+.1f}% vs baseline")

    print(f"\n  Fewest Conflicts: {best_conflicts['name']}")
    print(f"    {best_conflicts['conflicts']} conflicts")

    print(f"\n  Fastest Execution: {fastest['name']}")
    print(f"    {fastest['execution_time']:.2f}s")
    print()

# ================================================================================
# SAVE RESULTS TO JSON
# ================================================================================

output_file = "benchmark_phase_comparison_results.json"
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("\nBenchmark complete!")
