"""
Generate Large-Scale Maps and Scenarios for Phase 8 Validation

Creates 64x64 and 128x128 maps with various obstacle densities using POGEMA.
Used for validating Phase 8C (JPS) performance at scale.
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from pogema import GridConfig
from pogema.grid import Grid as POGEMAGrid

from pypibt.mapf_utils import Coord, Config, Grid


def save_map_file(grid: Grid, filepath: str):
    """
    Save grid to .map file format (MovingAI format).

    Format:
        type octile
        height <H>
        width <W>
        map
        <map data: . = free, @ = obstacle>
    """
    height, width = grid.shape

    with open(filepath, "w") as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("map\n")

        for row in grid:
            line = "".join("." if cell == 1 else "@" for cell in row)
            f.write(line + "\n")


def save_scenario_file(
    starts: Config,
    goals: Config,
    map_filename: str,
    filepath: str,
):
    """
    Save scenario to .scen file format (MovingAI format).

    Format:
        version 1.0
        <bucket> <map_file> <map_width> <map_height> <start_x> <start_y> <goal_x> <goal_y> <optimal_length>
    """
    with open(filepath, "w") as f:
        f.write("version 1.0\n")

        for start, goal in zip(starts, goals):
            # MovingAI format uses (col, row) = (y, x) indexing
            start_y, start_x = start
            goal_y, goal_x = goal

            # We don't know optimal_length yet, use -1
            bucket = 0
            map_width = 0  # Will be filled by reader
            map_height = 0
            optimal_length = -1.0

            f.write(
                f"{bucket}\t{map_filename}\t{map_width}\t{map_height}\t"
                f"{start_x}\t{start_y}\t{goal_x}\t{goal_y}\t{optimal_length}\n"
            )


def generate_pogema_map(
    map_size: int,
    obstacle_density: float,
    seed: int,
) -> Grid:
    """
    Generate a map using POGEMA.

    Args:
        map_size: Size of the grid
        obstacle_density: Density of obstacles (0.0 to 1.0)
        seed: Random seed

    Returns:
        Grid (numpy array): 1 = free, 0 = obstacle
    """
    # Create POGEMA grid config with minimal agents (just to generate map)
    grid_config = GridConfig(
        num_agents=1,  # Minimal agents for map generation
        size=map_size,
        density=obstacle_density,
        seed=seed,
        obs_radius=5,
    )

    # Generate POGEMA grid
    pogema_grid = POGEMAGrid(grid_config=grid_config, add_artificial_border=False)

    # Extract grid (POGEMA: 1=obstacle, PIBT: 1=free)
    grid = np.logical_not(pogema_grid.obstacles).astype(int)

    return grid


def generate_pogema_scenario(
    map_size: int,
    obstacle_density: float,
    num_agents: int,
    seed: int,
) -> Tuple[Grid, Config, Config]:
    """
    Generate a scenario (map + start/goal) using POGEMA.

    Args:
        map_size: Size of the grid
        obstacle_density: Density of obstacles
        num_agents: Number of agents
        seed: Random seed

    Returns:
        Tuple of (grid, starts, goals)
    """
    # Create POGEMA grid config
    grid_config = GridConfig(
        num_agents=num_agents,
        size=map_size,
        density=obstacle_density,
        seed=seed,
        obs_radius=5,
    )

    # Generate POGEMA grid
    pogema_grid = POGEMAGrid(grid_config=grid_config, add_artificial_border=False)

    # Extract grid
    grid = np.logical_not(pogema_grid.obstacles).astype(int)

    # Extract starts and goals
    starts = tuple((x, y) for x, y in pogema_grid.positions_xy)
    goals = tuple((x, y) for x, y in pogema_grid.finishes_xy)

    return grid, starts, goals


def main():
    parser = argparse.ArgumentParser(
        description="Generate large-scale maps and scenarios for Phase 8 validation"
    )
    parser.add_argument(
        "--map-sizes",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Map sizes to generate (e.g., 64 128)",
    )
    parser.add_argument(
        "--obstacle-densities",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3],
        help="Obstacle densities (e.g., 0.1 0.2 0.3)",
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000],
        help="Agent counts for scenarios (e.g., 100 200 500 1000)",
    )
    parser.add_argument(
        "--num-maps-per-config",
        type=int,
        default=3,
        help="Number of maps per (size, density) configuration",
    )
    parser.add_argument(
        "--num-scenarios-per-map",
        type=int,
        default=3,
        help="Number of scenarios per map",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/large",
        help="Output directory for maps and scenarios",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Large-Scale Map and Scenario Generation")
    print("=" * 80)
    print(f"Map sizes: {args.map_sizes}")
    print(f"Obstacle densities: {args.obstacle_densities}")
    print(f"Agent counts: {args.agent_counts}")
    print(f"Maps per config: {args.num_maps_per_config}")
    print(f"Scenarios per map: {args.num_scenarios_per_map}")
    print(f"Output directory: {args.output_dir}")
    print(f"Base seed: {args.base_seed}")
    print("=" * 80)
    print()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_start_time = time.time()
    seed = args.base_seed

    # Generate maps and scenarios
    for map_size in args.map_sizes:
        for obstacle_density in args.obstacle_densities:
            density_pct = int(obstacle_density * 100)

            print(f"\n{'='*60}")
            print(f"Generating: {map_size}x{map_size}, {density_pct}% obstacles")
            print(f"{'='*60}")

            for map_idx in range(args.num_maps_per_config):
                # Generate map
                map_start_time = time.time()
                print(f"\n[Map {map_idx+1}/{args.num_maps_per_config}]")

                grid = generate_pogema_map(
                    map_size=map_size,
                    obstacle_density=obstacle_density,
                    seed=seed,
                )
                seed += 1

                # Save map file
                map_filename = f"random-{map_size}-{map_size}-{density_pct}-{map_idx}.map"
                map_filepath = output_path / map_filename
                save_map_file(grid, str(map_filepath))

                actual_density = 1.0 - (np.sum(grid) / grid.size)
                map_time = time.time() - map_start_time

                print(f"  Map saved: {map_filename}")
                print(f"  Actual obstacle density: {actual_density:.1%}")
                print(f"  Generation time: {map_time:.2f}s")

                # Generate scenarios for each agent count
                for agent_count in args.agent_counts:
                    print(f"\n  [Scenarios: {agent_count} agents]")

                    for scen_idx in range(args.num_scenarios_per_map):
                        scen_start_time = time.time()

                        # Generate scenario
                        try:
                            _, starts, goals = generate_pogema_scenario(
                                map_size=map_size,
                                obstacle_density=obstacle_density,
                                num_agents=agent_count,
                                seed=seed,
                            )
                            seed += 1

                            # Save scenario file
                            scen_filename = (
                                f"random-{map_size}-{map_size}-{density_pct}-"
                                f"{map_idx}-{agent_count}agents-{scen_idx}.scen"
                            )
                            scen_filepath = output_path / scen_filename
                            save_scenario_file(
                                starts=starts,
                                goals=goals,
                                map_filename=map_filename,
                                filepath=str(scen_filepath),
                            )

                            scen_time = time.time() - scen_start_time
                            print(
                                f"    Scenario {scen_idx+1}: {agent_count} agents "
                                f"({scen_time:.2f}s) -> {scen_filename}"
                            )

                        except Exception as e:
                            print(
                                f"    WARNING: Failed to generate scenario "
                                f"{scen_idx+1} with {agent_count} agents: {e}"
                            )
                            seed += 1
                            continue

    total_time = time.time() - total_start_time

    # Summary
    print("\n" + "=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Output directory: {output_path}")

    # Count generated files
    map_files = list(output_path.glob("*.map"))
    scen_files = list(output_path.glob("*.scen"))
    print(f"Generated maps: {len(map_files)}")
    print(f"Generated scenarios: {len(scen_files)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
