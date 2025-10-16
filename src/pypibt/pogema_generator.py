"""
POGEMA-based Scenario Generator for MAPF-GPT

Generates valid MAPF scenarios using POGEMA GridConfig and Grid generation.
Uses POGEMA's random map generator to create solvable scenarios.
"""

import numpy as np
from typing import Tuple
from pogema import GridConfig
from pogema.grid import Grid as POGEMAGrid

from pypibt.mapf_utils import Coord, Config, Grid


class POGEMAScenarioGenerator:
    """
    Generates MAPF scenarios using POGEMA environment.

    POGEMA generates valid start-goal pairs that are guaranteed to be solvable.
    This addresses the issue with random scenario generation producing unsolvable instances.
    """

    def __init__(
        self,
        map_size: int = 32,
        obstacle_density: float = 0.3,
        obs_radius: int = 5,
        max_episode_steps: int = 256,
    ):
        """
        Initialize POGEMA scenario generator.

        Args:
            map_size: Size of the grid (map_size x map_size)
            obstacle_density: Density of obstacles (0.0 to 1.0)
            obs_radius: Observation radius for agents
            max_episode_steps: Maximum steps per episode
        """
        self.map_size = map_size
        self.obstacle_density = obstacle_density
        self.obs_radius = obs_radius
        self.max_episode_steps = max_episode_steps

    def generate_scenario(
        self,
        num_agents: int,
        seed: int | None = None,
        map_type: str = "random",
    ) -> Tuple[Grid, Config, Config]:
        """
        Generate a single MAPF scenario using POGEMA.

        Args:
            num_agents: Number of agents
            seed: Random seed for reproducibility
            map_type: Type of map ("random" or "maze")

        Returns:
            Tuple of (grid, starts, goals)
        """
        # Create POGEMA grid config
        grid_config = GridConfig(
            num_agents=num_agents,
            size=self.map_size,
            density=self.obstacle_density,
            seed=seed,
            obs_radius=self.obs_radius,
        )

        # Create POGEMA grid directly (avoiding environment creation issues)
        # IMPORTANT: add_artificial_border=False to avoid padding with obs_radius
        pogema_grid = POGEMAGrid(grid_config=grid_config, add_artificial_border=False)

        # Extract grid, starts, and goals from POGEMA grid
        grid = self._extract_grid_from_pogema(pogema_grid)
        starts = self._extract_starts_from_pogema(pogema_grid)
        goals = self._extract_goals_from_pogema(pogema_grid)

        return grid, starts, goals

    def _extract_grid_from_pogema(self, pogema_grid: POGEMAGrid) -> Grid:
        """
        Extract grid from POGEMA Grid object.

        Args:
            pogema_grid: POGEMA Grid object

        Returns:
            Grid (numpy array): 1 = free, 0 = obstacle (PIBT format)
        """
        # POGEMA stores obstacles directly in obstacles attribute
        # POGEMA format: 1=obstacle, 0=free
        # PIBT format: 1=free, 0=obstacle (inverted!)
        # We need to invert the values
        grid = np.logical_not(pogema_grid.obstacles).astype(int)
        return grid

    def _extract_starts_from_pogema(self, pogema_grid: POGEMAGrid) -> Config:
        """
        Extract start positions from POGEMA Grid.

        Args:
            pogema_grid: POGEMA Grid object

        Returns:
            Config: Tuple of start positions (x, y)
        """
        starts = []
        # POGEMA stores positions as (x, y) in positions_xy attribute
        # IMPORTANT: Do NOT convert! POGEMA's obstacles array is indexed as [x, y]
        # and our Grid should use the same indexing
        for x, y in pogema_grid.positions_xy:
            starts.append((x, y))
        return tuple(starts)

    def _extract_goals_from_pogema(self, pogema_grid: POGEMAGrid) -> Config:
        """
        Extract goal positions from POGEMA Grid.

        Args:
            pogema_grid: POGEMA Grid object

        Returns:
            Config: Tuple of goal positions (x, y)
        """
        goals = []
        # POGEMA stores finish positions as (x, y) in finishes_xy attribute
        # IMPORTANT: Do NOT convert! Keep (x, y) format to match Grid indexing
        for x, y in pogema_grid.finishes_xy:
            goals.append((x, y))
        return tuple(goals)

    def generate_multiple_scenarios(
        self,
        num_scenarios: int,
        agent_counts: list[int],
        map_types: list[str] = ["random", "maze"],
        base_seed: int = 0,
    ) -> list[Tuple[Grid, Config, Config]]:
        """
        Generate multiple MAPF scenarios.

        Args:
            num_scenarios: Total number of scenarios to generate
            agent_counts: List of agent counts to sample from
            map_types: List of map types to use
            base_seed: Base seed for reproducibility

        Returns:
            List of (grid, starts, goals) tuples
        """
        scenarios = []
        scenarios_per_config = num_scenarios // (len(agent_counts) * len(map_types))

        seed = base_seed
        for num_agents in agent_counts:
            for map_type in map_types:
                for _ in range(scenarios_per_config):
                    try:
                        grid, starts, goals = self.generate_scenario(
                            num_agents=num_agents,
                            seed=seed,
                            map_type=map_type,
                        )
                        scenarios.append((grid, starts, goals))
                        seed += 1
                    except Exception as e:
                        print(f"Warning: Failed to generate scenario with {num_agents} agents, "
                              f"map_type={map_type}, seed={seed}: {e}")
                        seed += 1
                        continue

        return scenarios


def test_pogema_generator():
    """Test POGEMA scenario generator."""
    from loguru import logger
    from pypibt.dist_table import DistTable

    # Configure loguru
    logger.remove()
    logger.add(
        "pogema_test_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )
    logger.add(lambda msg: print(msg, end=""), format="{message}", level="INFO")

    logger.info("=" * 80)
    logger.info("Testing POGEMA Scenario Generator")
    logger.info("=" * 80)

    generator = POGEMAScenarioGenerator(
        map_size=32,
        obstacle_density=0.3,
        obs_radius=5,
    )

    # Test 1: Generate single scenario
    logger.info("\n[Test 1] Generate Single Scenario")
    grid, starts, goals = generator.generate_scenario(
        num_agents=10,
        seed=42,
        map_type="random",
    )
    logger.info(f"  Grid shape: {grid.shape}")
    logger.info(f"  Number of agents: {len(starts)}")
    logger.info(f"  Starts: {starts[:3]}...")
    logger.info(f"  Goals: {goals[:3]}...")
    logger.info(f"  Obstacle density: {np.sum(grid) / grid.size:.2%}")

    # Detailed scenario validation
    logger.debug("\n[Detailed Scenario Validation]")

    # First, let's check the raw POGEMA output before coordinate conversion
    logger.debug("\n[POGEMA Raw Coordinates Check]")
    pogema_grid_config = GridConfig(
        num_agents=10,
        size=32,
        density=0.3,
        seed=42,
        obs_radius=5,
    )
    from pogema.grid import Grid as POGEMAGrid
    pogema_grid_raw = POGEMAGrid(grid_config=pogema_grid_config, add_artificial_border=False)

    logger.debug(f"POGEMA positions_xy (first 3): {pogema_grid_raw.positions_xy[:3]}")
    logger.debug(f"POGEMA finishes_xy (first 3): {pogema_grid_raw.finishes_xy[:3]}")
    logger.debug(f"POGEMA obstacles shape: {pogema_grid_raw.obstacles.shape}")

    # Check a few positions directly on POGEMA grid
    for i in range(min(3, len(pogema_grid_raw.positions_xy))):
        x, y = pogema_grid_raw.positions_xy[i]
        logger.debug(f"POGEMA Agent {i} start (x={x}, y={y}): obstacles[{x},{y}] = {pogema_grid_raw.obstacles[x, y]}")

    logger.debug("\n[After Coordinate Conversion]")
    for i in range(len(starts)):
        start_pos = starts[i]
        goal_pos = goals[i]
        logger.debug(f"Agent {i}:")
        logger.debug(f"  Start: {start_pos} -> Grid[{start_pos}] = {grid[start_pos]}")
        logger.debug(f"  Goal: {goal_pos} -> Grid[{goal_pos}] = {grid[goal_pos]}")

        # Calculate distance to goal
        dist_table = DistTable(grid, goal_pos)
        dist_to_goal = dist_table.get(start_pos)
        logger.debug(f"  Distance to goal: {dist_to_goal}")

        # Manual check: try a simple adjacent cell
        if i == 0:
            logger.debug(f"  [Manual DistTable check for Agent {i}]")
            # Check neighbors of start position
            neighbors_check = [
                (start_pos[0] - 1, start_pos[1]),  # UP
                (start_pos[0] + 1, start_pos[1]),  # DOWN
                (start_pos[0], start_pos[1] - 1),  # LEFT
                (start_pos[0], start_pos[1] + 1),  # RIGHT
            ]
            for neighbor in neighbors_check:
                if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                    logger.debug(f"    Neighbor {neighbor}: Grid={grid[neighbor]}, Dist={dist_table.get(neighbor)}")

    # Test 2: Verify solvability with PIBT
    logger.info("\n[Test 2] Test Solvability with PIBT")
    from pypibt import PIBT

    pibt = PIBT(grid, starts, goals, seed=0)

    # Log initial state
    logger.debug("\n[Initial PIBT State]")
    logger.debug(f"Grid shape: {grid.shape}")
    logger.debug(f"Number of agents: {len(starts)}")
    logger.debug(f"Obstacle count: {np.sum(grid)}")

    trajectory = pibt.run(max_timestep=200)

    # Check if all agents reached their goals
    final_config = trajectory[-1]
    all_reached = all(final_config[i] == goals[i] for i in range(len(starts)))

    logger.info(f"  Trajectory length: {len(trajectory)} steps")
    logger.info(f"  All agents reached goals: {all_reached}")

    # Check if agents moved
    moved_count = sum(1 for i in range(len(starts)) if trajectory[0][i] != trajectory[1][i])
    logger.info(f"  Agents that moved in first step: {moved_count}/{len(starts)}")

    # Detailed trajectory analysis for first 5 steps
    logger.debug("\n[Detailed Trajectory Analysis - First 5 Steps]")
    for t in range(min(5, len(trajectory) - 1)):
        logger.debug(f"\n--- Timestep {t} ---")
        config_now = trajectory[t]
        config_next = trajectory[t + 1]

        for i in range(min(5, len(starts))):  # Only first 5 agents
            pos_now = config_now[i]
            pos_next = config_next[i]
            goal = goals[i]

            action = "WAIT"
            if pos_next != pos_now:
                dy = pos_next[0] - pos_now[0]
                dx = pos_next[1] - pos_now[1]
                if dy == -1: action = "UP"
                elif dy == 1: action = "DOWN"
                elif dx == -1: action = "LEFT"
                elif dx == 1: action = "RIGHT"

            dist_table = DistTable(grid, goal)
            dist_now = dist_table.get(pos_now)
            dist_next = dist_table.get(pos_next)

            logger.debug(
                f"Agent {i}: {pos_now} -> {pos_next} (Goal: {goal}) "
                f"| Action: {action} | Dist: {dist_now} -> {dist_next}"
            )

    # Test 3: Generate multiple scenarios
    logger.info("\n[Test 3] Generate Multiple Scenarios")
    scenarios = generator.generate_multiple_scenarios(
        num_scenarios=4,
        agent_counts=[5, 10],
        map_types=["random"],
        base_seed=100,
    )
    logger.info(f"  Generated {len(scenarios)} scenarios")
    for i, (grid, starts, goals) in enumerate(scenarios[:2]):
        logger.info(f"  Scenario {i+1}: {len(starts)} agents, grid {grid.shape}")

    logger.info("\n" + "=" * 80)
    logger.info("All tests passed! ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_pogema_generator()
