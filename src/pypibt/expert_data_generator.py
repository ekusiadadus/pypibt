"""
Expert Data Generator for MAPF-GPT

Generates expert demonstration data by running optimized PIBT and extracting
(observation, action) pairs for Transformer training.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Any

from pypibt import PIBT, get_grid, get_scenario
from pypibt.mapf_utils import Coord, Config, Configs, Grid
from pypibt.mapf_tokenizer import MAPFTokenizer
from pypibt.dist_table import DistTable


class ExpertDataGenerator:
    """
    Generates expert training data for MAPF-GPT.

    Uses optimized PIBT (Anytime + Hindrance + Regret) as expert solver
    to generate high-quality trajectories.
    """

    def __init__(
        self,
        map_file: str,
        tokenizer: MAPFTokenizer,
        expert_config: dict[str, Any] | None = None,
    ):
        """
        Initialize expert data generator.

        Args:
            map_file: Path to map file
            tokenizer: MAPF tokenizer instance
            expert_config: Configuration for expert PIBT solver
        """
        self.map_file = map_file
        self.grid = get_grid(map_file)
        self.tokenizer = tokenizer

        # Default expert configuration (optimized PIBT)
        self.expert_config = expert_config or {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": True,
            "regret_learning_iterations": 3,
            "regret_weight": 0.2,
            "enable_anytime": False,  # Disable for speed
            "enable_lns": False,
            "priority_increment": 1.0,
        }

    def generate_random_scenario(
        self, num_agents: int, seed: int | None = None
    ) -> tuple[Config, Config]:
        """
        Generate random start and goal positions for agents.

        Args:
            num_agents: Number of agents
            seed: Random seed for reproducibility

        Returns:
            Tuple of (starts, goals)
        """
        if seed is not None:
            np.random.seed(seed)

        # Find all free cells
        free_cells = []
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == 0:
                    free_cells.append((y, x))

        # Ensure we have enough free cells
        assert len(free_cells) >= num_agents * 2, (
            f"Not enough free cells for {num_agents} agents. "
            f"Available: {len(free_cells)}, Required: {num_agents * 2}"
        )

        # Randomly sample start and goal positions
        sampled_cells = np.random.choice(len(free_cells), num_agents * 2, replace=False)
        starts = tuple(free_cells[i] for i in sampled_cells[:num_agents])
        goals = tuple(free_cells[i] for i in sampled_cells[num_agents:])

        return starts, goals

    def run_expert_solver(
        self, starts: Config, goals: Config, max_timestep: int = 2000
    ) -> Configs:
        """
        Run expert PIBT solver to generate trajectory.

        Args:
            starts: Start positions
            goals: Goal positions
            max_timestep: Maximum timesteps

        Returns:
            Expert trajectory (sequence of configurations)
        """
        pibt = PIBT(
            self.grid,
            starts,
            goals,
            seed=np.random.randint(0, 1000000),
            **self.expert_config,
        )

        trajectory = pibt.run(max_timestep=max_timestep)
        return trajectory

    def extract_observation_action_pairs(
        self, trajectory: Configs, starts: Config, goals: Config
    ) -> list[tuple[list[int], int]]:
        """
        Extract (observation, action) pairs from expert trajectory.

        Args:
            trajectory: Expert trajectory
            starts: Start positions
            goals: Goal positions

        Returns:
            List of (observation_tokens, action_token) pairs
        """
        pairs = []
        num_agents = len(starts)

        # Pre-compute distance tables for each agent
        dist_tables = [DistTable(self.grid, goal) for goal in goals]

        # Extract pairs from each timestep
        for t in range(len(trajectory) - 1):
            config_now = trajectory[t]
            config_next = trajectory[t + 1]

            # For each agent
            for agent_id in range(num_agents):
                agent_pos = config_now[agent_id]
                agent_goal = goals[agent_id]
                next_pos = config_next[agent_id]

                # Skip if agent is already at goal
                if agent_pos == agent_goal:
                    continue

                # Encode observation
                observation_tokens = self.tokenizer.encode_observation(
                    grid=self.grid,
                    agent_pos=agent_pos,
                    agent_goal=agent_goal,
                    all_positions=list(config_now),
                    all_goals=list(goals),
                    agent_id=agent_id,
                    dist_table=dist_tables[agent_id],
                )

                # Encode action
                action_token = self.tokenizer.action_to_token(agent_pos, next_pos)

                pairs.append((observation_tokens, action_token))

        return pairs

    def generate_dataset_from_scenarios(
        self,
        num_scenarios: int,
        agent_counts: list[int],
        max_timestep: int = 2000,
        output_file: str | None = None,
        verbose: bool = True,
    ) -> tuple[list[list[int]], list[int]]:
        """
        Generate training dataset from multiple scenarios.

        Args:
            num_scenarios: Total number of scenarios to generate
            agent_counts: List of agent counts to sample from
            max_timestep: Maximum timesteps per scenario
            output_file: Optional file to save dataset
            verbose: Print progress

        Returns:
            Tuple of (observations, actions) lists
        """
        all_observations = []
        all_actions = []

        scenarios_per_count = num_scenarios // len(agent_counts)

        for num_agents in agent_counts:
            if verbose:
                print(f"\nGenerating {scenarios_per_count} scenarios with {num_agents} agents...")

            for scenario_id in range(scenarios_per_count):
                # Generate random scenario
                starts, goals = self.generate_random_scenario(
                    num_agents, seed=scenario_id + num_agents * 1000
                )

                try:
                    # Run expert solver
                    trajectory = self.run_expert_solver(starts, goals, max_timestep)

                    # Extract (obs, action) pairs
                    pairs = self.extract_observation_action_pairs(trajectory, starts, goals)

                    # Add to dataset
                    for obs, action in pairs:
                        all_observations.append(obs)
                        all_actions.append(action)

                    if verbose and (scenario_id + 1) % 10 == 0:
                        print(
                            f"  Progress: {scenario_id + 1}/{scenarios_per_count} scenarios, "
                            f"{len(all_observations)} total pairs"
                        )

                except Exception as e:
                    print(f"  Warning: Failed to generate scenario {scenario_id}: {e}")
                    continue

        if verbose:
            print(f"\nDataset generation complete!")
            print(f"  Total (observation, action) pairs: {len(all_observations)}")
            print(f"  Observation sequence length: {len(all_observations[0])}")
            print(f"  Vocabulary size: {self.tokenizer.vocab_size}")

        # Save dataset if output file specified
        if output_file is not None:
            self.save_dataset(all_observations, all_actions, output_file)
            if verbose:
                print(f"  Dataset saved to: {output_file}")

        return all_observations, all_actions

    def save_dataset(
        self, observations: list[list[int]], actions: list[int], output_file: str
    ) -> None:
        """
        Save dataset to file.

        Args:
            observations: List of observation token sequences
            actions: List of action tokens
            output_file: Output file path
        """
        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as pickle
        dataset = {
            "observations": observations,
            "actions": actions,
            "vocab_size": self.tokenizer.vocab_size,
            "sequence_length": self.tokenizer.get_sequence_length(),
        }

        with open(output_file, "wb") as f:
            pickle.dump(dataset, f)

    @staticmethod
    def load_dataset(input_file: str) -> tuple[list[list[int]], list[int], dict[str, int]]:
        """
        Load dataset from file.

        Args:
            input_file: Input file path

        Returns:
            Tuple of (observations, actions, metadata)
        """
        with open(input_file, "rb") as f:
            dataset = pickle.load(f)

        metadata = {
            "vocab_size": dataset["vocab_size"],
            "sequence_length": dataset["sequence_length"],
        }

        return dataset["observations"], dataset["actions"], metadata

    def filter_duplicate_observations(
        self, observations: list[list[int]], actions: list[int]
    ) -> tuple[list[list[int]], list[int]]:
        """
        Remove duplicate observations (keep one randomly).

        Based on MAPF-GPT paper: remove observations with identical context.

        Args:
            observations: List of observations
            actions: List of actions

        Returns:
            Filtered (observations, actions)
        """
        # Use hash of observation tuple as key
        unique_observations = {}

        for obs, action in zip(observations, actions):
            obs_tuple = tuple(obs)
            if obs_tuple not in unique_observations:
                unique_observations[obs_tuple] = (obs, action)

        filtered_obs = [obs for obs, _ in unique_observations.values()]
        filtered_actions = [action for _, action in unique_observations.values()]

        return filtered_obs, filtered_actions

    def balance_action_distribution(
        self, observations: list[list[int]], actions: list[int], wait_keep_ratio: float = 0.2
    ) -> tuple[list[list[int]], list[int]]:
        """
        Balance action distribution by reducing WAIT actions.

        Based on MAPF-GPT paper: discard 80% of wait-at-target actions.

        Args:
            observations: List of observations
            actions: List of actions
            wait_keep_ratio: Ratio of WAIT actions to keep (0.2 = keep 20%)

        Returns:
            Balanced (observations, actions)
        """
        balanced_obs = []
        balanced_actions = []

        for obs, action in zip(observations, actions):
            if action == MAPFTokenizer.ACTION_WAIT:
                # Randomly keep only wait_keep_ratio of WAIT actions
                if np.random.random() < wait_keep_ratio:
                    balanced_obs.append(obs)
                    balanced_actions.append(action)
            else:
                # Keep all non-WAIT actions
                balanced_obs.append(obs)
                balanced_actions.append(action)

        return balanced_obs, balanced_actions


def test_expert_data_generator():
    """Test expert data generator."""
    print("=" * 80)
    print("Testing Expert Data Generator")
    print("=" * 80)

    # Initialize
    tokenizer = MAPFTokenizer(fov_size=11, max_agents_visible=13)
    generator = ExpertDataGenerator(
        map_file="assets/random-32-32-10.map", tokenizer=tokenizer
    )

    # Test random scenario generation
    print("\n[Test 1] Random Scenario Generation")
    starts, goals = generator.generate_random_scenario(num_agents=10, seed=0)
    print(f"  Generated {len(starts)} agents")
    print(f"  Starts: {starts[:3]}...")
    print(f"  Goals: {goals[:3]}...")

    # Test expert solver
    print("\n[Test 2] Expert Solver")
    trajectory = generator.run_expert_solver(starts, goals, max_timestep=1000)
    print(f"  Trajectory length: {len(trajectory)} timesteps")

    # Test observation-action pair extraction
    print("\n[Test 3] Extract (observation, action) pairs")
    pairs = generator.extract_observation_action_pairs(trajectory, starts, goals)
    print(f"  Extracted {len(pairs)} pairs")
    if len(pairs) > 0:
        obs, action = pairs[0]
        print(f"  Observation length: {len(obs)}")
        print(f"  Action token: {action}")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_expert_data_generator()
