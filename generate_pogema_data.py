"""
Generate MAPF-GPT Training Data using POGEMA

Generates large-scale training dataset for MAPF-GPT Transformer.
Uses POGEMA for scenario generation and optimized PIBT as expert solver.
"""

import argparse
import time
import numpy as np
from pathlib import Path

from pypibt.mapf_tokenizer import MAPFTokenizer
from pypibt.pogema_generator import POGEMAScenarioGenerator
from pypibt import PIBT
from pypibt.dist_table import DistTable


def generate_training_data(
    num_scenarios: int,
    agent_counts: list[int],
    map_size: int,
    obstacle_density: float,
    max_timestep: int,
    tokenizer: MAPFTokenizer,
    base_seed: int = 0,
    verbose: bool = True,
):
    """
    Generate training data from POGEMA scenarios.

    Args:
        num_scenarios: Total number of scenarios
        agent_counts: List of agent counts to sample from
        map_size: Grid size
        obstacle_density: Obstacle density (0.0 to 1.0)
        max_timestep: Maximum timesteps per scenario
        tokenizer: MAPF tokenizer
        base_seed: Base random seed
        verbose: Print progress

    Returns:
        Tuple of (observations, actions)
    """
    all_observations = []
    all_actions = []

    # Create POGEMA generator
    generator = POGEMAScenarioGenerator(
        map_size=map_size,
        obstacle_density=obstacle_density,
        obs_radius=5,
        max_episode_steps=max_timestep,
    )

    scenarios_per_count = num_scenarios // len(agent_counts)

    for num_agents in agent_counts:
        if verbose:
            print(f"\nGenerating {scenarios_per_count} scenarios with {num_agents} agents...")

        success_count = 0
        seed = base_seed + num_agents * 10000

        while success_count < scenarios_per_count:
            try:
                # Generate POGEMA scenario
                grid, starts, goals = generator.generate_scenario(
                    num_agents=num_agents,
                    seed=seed,
                    map_type="random",
                )

                # Run expert PIBT solver
                pibt = PIBT(
                    grid, starts, goals,
                    seed=seed,
                    enable_hindrance=True,
                    hindrance_weight=0.3,
                    enable_regret_learning=True,
                    regret_learning_iterations=3,
                    regret_weight=0.2,
                )

                trajectory = pibt.run(max_timestep=max_timestep)

                # Check if solved
                final_config = trajectory[-1]
                all_reached = all(final_config[i] == goals[i] for i in range(len(starts)))

                if not all_reached:
                    if verbose and seed % 50 == 0:
                        print(f"  Warning: Scenario {seed} not fully solved, skipping...")
                    seed += 1
                    continue

                # Extract (observation, action) pairs
                dist_tables = [DistTable(grid, goal) for goal in goals]

                for t in range(len(trajectory) - 1):
                    config_now = trajectory[t]
                    config_next = trajectory[t + 1]

                    for agent_id in range(num_agents):
                        agent_pos = config_now[agent_id]
                        agent_goal = goals[agent_id]
                        next_pos = config_next[agent_id]

                        # Skip if agent is at goal
                        if agent_pos == agent_goal:
                            continue

                        # Encode observation
                        observation_tokens = tokenizer.encode_observation(
                            grid=grid,
                            agent_pos=agent_pos,
                            agent_goal=agent_goal,
                            all_positions=list(config_now),
                            all_goals=list(goals),
                            agent_id=agent_id,
                            dist_table=dist_tables[agent_id],
                        )

                        # Encode action
                        action_token = tokenizer.action_to_token(agent_pos, next_pos)

                        all_observations.append(observation_tokens)
                        all_actions.append(action_token)

                success_count += 1
                seed += 1

                if verbose and success_count % 10 == 0:
                    print(
                        f"  Progress: {success_count}/{scenarios_per_count} scenarios, "
                        f"{len(all_observations)} total pairs"
                    )

            except Exception as e:
                if verbose and seed % 50 == 0:
                    print(f"  Warning: Failed to generate scenario {seed}: {e}")
                seed += 1
                continue

    return all_observations, all_actions


def main():
    parser = argparse.ArgumentParser(description="Generate MAPF-GPT training data with POGEMA")
    parser.add_argument(
        "--num-scenarios", type=int, default=1000, help="Total number of scenarios"
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30],
        help="Agent counts to sample from",
    )
    parser.add_argument(
        "--map-size", type=int, default=32, help="Grid size (map_size x map_size)"
    )
    parser.add_argument(
        "--obstacle-density", type=float, default=0.3, help="Obstacle density (0.0 to 1.0)"
    )
    parser.add_argument(
        "--max-timestep", type=int, default=500, help="Maximum timesteps per scenario"
    )
    parser.add_argument(
        "--output", type=str, default="data/pogema_mapf_dataset.pkl", help="Output file"
    )
    parser.add_argument(
        "--fov-size", type=int, default=11, help="Field of view size"
    )
    parser.add_argument(
        "--max-agents-visible", type=int, default=13, help="Maximum visible agents"
    )
    parser.add_argument(
        "--filter-duplicates", action="store_true", help="Remove duplicate observations"
    )
    parser.add_argument(
        "--balance-actions",
        action="store_true",
        help="Balance action distribution (reduce WAIT actions)",
    )
    parser.add_argument(
        "--wait-keep-ratio", type=float, default=0.2, help="Ratio of WAIT actions to keep"
    )
    parser.add_argument(
        "--base-seed", type=int, default=0, help="Base random seed"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MAPF-GPT Data Generation with POGEMA")
    print("=" * 80)
    print(f"Number of scenarios: {args.num_scenarios}")
    print(f"Agent counts: {args.agent_counts}")
    print(f"Map size: {args.map_size}x{args.map_size}")
    print(f"Obstacle density: {args.obstacle_density}")
    print(f"Max timesteps: {args.max_timestep}")
    print(f"Output file: {args.output}")
    print(f"FOV size: {args.fov_size}")
    print(f"Max visible agents: {args.max_agents_visible}")
    print(f"Filter duplicates: {args.filter_duplicates}")
    print(f"Balance actions: {args.balance_actions}")
    print(f"Base seed: {args.base_seed}")
    print("=" * 80)

    # Initialize tokenizer
    tokenizer = MAPFTokenizer(
        fov_size=args.fov_size, max_agents_visible=args.max_agents_visible
    )

    # Generate dataset
    start_time = time.time()
    observations, actions = generate_training_data(
        num_scenarios=args.num_scenarios,
        agent_counts=args.agent_counts,
        map_size=args.map_size,
        obstacle_density=args.obstacle_density,
        max_timestep=args.max_timestep,
        tokenizer=tokenizer,
        base_seed=args.base_seed,
        verbose=True,
    )
    generation_time = time.time() - start_time

    print(f"\nData generation completed in {generation_time:.2f}s")
    print(f"Total pairs before filtering: {len(observations)}")

    # Apply filters
    if args.filter_duplicates:
        from pypibt.expert_data_generator import ExpertDataGenerator
        print("\nFiltering duplicate observations...")

        # Use static method for filtering
        unique_observations = {}
        for obs, action in zip(observations, actions):
            obs_tuple = tuple(obs)
            if obs_tuple not in unique_observations:
                unique_observations[obs_tuple] = (obs, action)

        observations = [obs for obs, _ in unique_observations.values()]
        actions = [action for _, action in unique_observations.values()]
        print(f"Total pairs after duplicate removal: {len(observations)}")

    if args.balance_actions:
        print(f"\nBalancing action distribution (keep {args.wait_keep_ratio*100}% WAIT)...")
        balanced_obs = []
        balanced_actions = []

        for obs, action in zip(observations, actions):
            if action == 1:  # WAIT
                if np.random.random() < args.wait_keep_ratio:
                    balanced_obs.append(obs)
                    balanced_actions.append(action)
            else:
                balanced_obs.append(obs)
                balanced_actions.append(action)

        observations = balanced_obs
        actions = balanced_actions
        print(f"Total pairs after balancing: {len(observations)}")

    # Action distribution statistics
    action_counts = {}
    action_names = {1: "WAIT", 2: "UP", 3: "DOWN", 4: "LEFT", 5: "RIGHT"}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1

    print("\nAction Distribution:")
    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        percentage = count / len(actions) * 100
        action_name = action_names.get(action_id, f"UNKNOWN({action_id})")
        print(f"  {action_name:6s}: {count:8d} ({percentage:5.2f}%)")

    # Save final dataset
    print(f"\nSaving dataset to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import pickle
    dataset = {
        "observations": observations,
        "actions": actions,
        "vocab_size": tokenizer.vocab_size,
        "sequence_length": tokenizer.get_sequence_length(),
    }

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)

    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print(f"  Total (observation, action) pairs: {len(observations)}")
    print(f"  Observation sequence length: {len(observations[0])}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Output file: {args.output}")
    print(f"  Total time: {generation_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
