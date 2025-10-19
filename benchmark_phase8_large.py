"""
Benchmark Phase 8 (JPS, Beam Search) on Large-Scale Graphs

Tests Phase 8C (JPS-Accelerated DistTable) and Phase 8B (Adaptive Beam Search)
on 64x64 and 128x128 maps with 100-500 agents to validate scalability.
"""

import argparse
import time
from pathlib import Path

from pypibt import PIBT, get_grid, get_scenario
from pypibt.adaptive_beam_pibt import AdaptiveBeamPIBT
from pypibt.dist_table import DistTable
from pypibt.jps_dist_table import JPSDistTable


def benchmark_phase8c_jps_init(
    grid, goals, num_runs=5
) -> tuple[float, float, bool]:
    """
    Benchmark JPS-accelerated DistTable initialization.

    Returns:
        (baseline_time, jps_time, jps_enabled)
    """
    # Baseline: Standard DistTable
    baseline_times = []
    for _ in range(num_runs):
        start_time = time.time()
        for goal in goals:
            _ = DistTable(grid, goal)
        baseline_times.append(time.time() - start_time)

    baseline_avg = sum(baseline_times) / len(baseline_times)

    # JPS: JPSDistTable
    jps_times = []
    jps_enabled = False
    for _ in range(num_runs):
        start_time = time.time()
        dist_tables = []
        for goal in goals:
            jps_table = JPSDistTable(grid, goal)
            dist_tables.append(jps_table)
        jps_times.append(time.time() - start_time)

    jps_avg = sum(jps_times) / len(jps_times)

    # Check if JPS was actually enabled
    if dist_tables:
        jps_enabled = dist_tables[0].use_jps

    return baseline_avg, jps_avg, jps_enabled


def benchmark_phase8b_beam_search(
    grid,
    starts,
    goals,
    max_timestep=2000,
    time_limit_ms=10000.0,
) -> tuple[int, float]:
    """
    Benchmark Phase 8B Adaptive Beam Search.

    Returns:
        (timesteps, execution_time)
    """
    start_time = time.time()

    beam_pibt = AdaptiveBeamPIBT(
        grid,
        starts,
        goals,
        seed=0,
        beam_width=5,
        time_limit_ms=time_limit_ms,
        diversity_weight=0.5,
        adaptive_beam=True,
        min_beam_width=3,
        max_beam_width=10,
    )

    configs = beam_pibt.run(max_timestep=max_timestep)

    execution_time = time.time() - start_time
    timesteps = len(configs)

    return timesteps, execution_time


def benchmark_baseline_pibt(
    grid,
    starts,
    goals,
    max_timestep=2000,
) -> tuple[int, float]:
    """
    Benchmark baseline PIBT.

    Returns:
        (timesteps, execution_time)
    """
    start_time = time.time()

    pibt = PIBT(
        grid,
        starts,
        goals,
        seed=0,
        enable_hindrance=True,
        hindrance_weight=0.3,
        enable_regret_learning=True,
        regret_learning_iterations=3,
        regret_weight=0.2,
    )

    configs = pibt.run(max_timestep=max_timestep)

    execution_time = time.time() - start_time
    timesteps = len(configs)

    return timesteps, execution_time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Phase 8 on large-scale graphs"
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=None,
        help="Map files to benchmark (default: all in assets/large/)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Scenario files to benchmark (default: all matching maps)",
    )
    parser.add_argument(
        "--max-timestep",
        type=int,
        default=2000,
        help="Maximum timesteps for PIBT",
    )
    parser.add_argument(
        "--time-limit-ms",
        type=float,
        default=10000.0,
        help="Time limit for beam search (ms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_phase8_large_results.txt",
        help="Output file for results",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline PIBT benchmark (for speed)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Phase 8 Large-Scale Benchmark")
    print("=" * 80)
    print(f"Max timesteps: {args.max_timestep}")
    print(f"Beam search time limit: {args.time_limit_ms}ms")
    print(f"Output file: {args.output}")
    print(f"Skip baseline: {args.skip_baseline}")
    print("=" * 80)
    print()

    # Find map and scenario files
    large_dir = Path("assets/large")
    if args.maps:
        map_files = [Path(m) for m in args.maps]
    else:
        map_files = sorted(large_dir.glob("*.map"))

    print(f"Found {len(map_files)} map files")
    print()

    results = []

    for map_file in map_files:
        map_name = map_file.stem
        print(f"\n{'='*60}")
        print(f"Map: {map_name}")
        print(f"{'='*60}")

        # Load grid
        grid = get_grid(str(map_file))
        print(f"Grid size: {grid.shape}")
        print(f"Free cells: {grid.sum()} / {grid.size} ({grid.sum()/grid.size:.1%})")

        # Find matching scenario files
        if args.scenarios:
            scen_files = [Path(s) for s in args.scenarios if map_name in s]
        else:
            scen_files = sorted(large_dir.glob(f"{map_name}*.scen"))

        print(f"Found {len(scen_files)} scenario files")

        for scen_file in scen_files:
            scen_name = scen_file.stem
            print(f"\n  [Scenario: {scen_name}]")

            try:
                # Extract agent count from scenario filename
                # Format: random-64-64-10-0-100agents-0.scen
                parts = scen_name.split("-")
                num_agents = None
                for part in parts:
                    if "agents" in part:
                        num_agents = int(part.replace("agents", ""))
                        break

                if num_agents is None:
                    # Fallback: try to read scenario file
                    with open(scen_file) as f:
                        lines = f.readlines()
                        num_agents = len([l for l in lines if not l.startswith("version")])

                # Load scenario (use first N agents)
                starts, goals = get_scenario(str(scen_file), num_agents)
                print(f"  Agents: {num_agents}")

                # Benchmark Phase 8C (JPS) - DistTable Initialization
                print(f"\n  [Phase 8C - JPS DistTable Initialization]")
                baseline_time, jps_time, jps_enabled = benchmark_phase8c_jps_init(
                    grid, goals, num_runs=3
                )
                speedup = baseline_time / jps_time if jps_time > 0 else 0
                print(f"    Baseline DistTable: {baseline_time:.3f}s")
                print(f"    JPS DistTable: {jps_time:.3f}s")
                print(f"    JPS Enabled: {jps_enabled}")
                print(f"    Speedup: {speedup:.2f}x")

                if speedup < 1.0:
                    print(f"    ⚠️  JPS SLOWER by {1/speedup:.2f}x")
                elif speedup > 1.0:
                    print(f"    ✅ JPS FASTER by {speedup:.2f}x")

                # Benchmark Baseline PIBT (optional, can be slow)
                baseline_timesteps = None
                baseline_exec_time = None
                if not args.skip_baseline:
                    print(f"\n  [Baseline PIBT]")
                    try:
                        baseline_timesteps, baseline_exec_time = benchmark_baseline_pibt(
                            grid, starts, goals, max_timestep=args.max_timestep
                        )
                        print(f"    Timesteps: {baseline_timesteps}")
                        print(f"    Execution time: {baseline_exec_time:.2f}s")
                    except Exception as e:
                        print(f"    ❌ Baseline PIBT failed: {e}")
                        baseline_timesteps = None
                        baseline_exec_time = None

                # Benchmark Phase 8B (Beam Search)
                print(f"\n  [Phase 8B - Adaptive Beam Search]")
                try:
                    beam_timesteps, beam_exec_time = benchmark_phase8b_beam_search(
                        grid,
                        starts,
                        goals,
                        max_timestep=args.max_timestep,
                        time_limit_ms=args.time_limit_ms,
                    )
                    print(f"    Timesteps: {beam_timesteps}")
                    print(f"    Execution time: {beam_exec_time:.2f}s")

                    if baseline_timesteps:
                        improvement = (
                            (baseline_timesteps - beam_timesteps) / baseline_timesteps * 100
                        )
                        print(f"    Improvement: {improvement:+.1f}%")
                except Exception as e:
                    print(f"    ❌ Beam search failed: {e}")
                    beam_timesteps = None
                    beam_exec_time = None

                # Store results
                results.append({
                    "map": map_name,
                    "scenario": scen_name,
                    "agents": num_agents,
                    "grid_size": grid.shape,
                    "jps_enabled": jps_enabled,
                    "jps_speedup": speedup,
                    "baseline_init_time": baseline_time,
                    "jps_init_time": jps_time,
                    "baseline_timesteps": baseline_timesteps,
                    "baseline_exec_time": baseline_exec_time,
                    "beam_timesteps": beam_timesteps,
                    "beam_exec_time": beam_exec_time,
                })

            except Exception as e:
                print(f"  ❌ Error processing scenario: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results
    print("\n" + "=" * 80)
    print("Benchmark Complete - Saving Results")
    print("=" * 80)

    with open(args.output, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 8 Large-Scale Benchmark Results\n")
        f.write("=" * 80 + "\n\n")

        # Phase 8C Summary
        f.write("Phase 8C (JPS DistTable Initialization) Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Map':<25} {'Agents':>6} {'JPS On':>7} {'Baseline':>10} {'JPS':>10} {'Speedup':>8}\n"
        )
        f.write("-" * 80 + "\n")

        for r in results:
            jps_on = "Yes" if r["jps_enabled"] else "No"
            f.write(
                f"{r['scenario']:<25} {r['agents']:>6} {jps_on:>7} "
                f"{r['baseline_init_time']:>9.3f}s {r['jps_init_time']:>9.3f}s "
                f"{r['jps_speedup']:>7.2f}x\n"
            )

        f.write("\n\n")

        # Phase 8B Summary
        f.write("Phase 8B (Adaptive Beam Search) Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Map':<25} {'Agents':>6} {'Baseline':>10} {'Beam':>10} {'Improve':>8} {'Time':>8}\n"
        )
        f.write("-" * 80 + "\n")

        for r in results:
            if r["baseline_timesteps"] and r["beam_timesteps"]:
                improvement = (
                    (r["baseline_timesteps"] - r["beam_timesteps"])
                    / r["baseline_timesteps"]
                    * 100
                )
                f.write(
                    f"{r['scenario']:<25} {r['agents']:>6} "
                    f"{r['baseline_timesteps']:>9}ts {r['beam_timesteps']:>9}ts "
                    f"{improvement:>7.1f}% {r['beam_exec_time']:>7.1f}s\n"
                )
            elif r["beam_timesteps"]:
                f.write(
                    f"{r['scenario']:<25} {r['agents']:>6} "
                    f"{'N/A':>9} {r['beam_timesteps']:>9}ts "
                    f"{'N/A':>8} {r['beam_exec_time']:>7.1f}s\n"
                )

        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"\nResults saved to: {args.output}")
    print(f"Total scenarios benchmarked: {len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
