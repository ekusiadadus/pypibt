"""
Generate MAPF-GPT Training Data

Generates large-scale training dataset for MAPF-GPT Transformer.
Uses optimized PIBT as expert to generate (observation, action) pairs.
"""

import argparse
import time
from pypibt.mapf_tokenizer import MAPFTokenizer
from pypibt.expert_data_generator import ExpertDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate MAPF-GPT training data")
    parser.add_argument(
        "--map", type=str, default="assets/random-32-32-10.map", help="Map file path"
    )
    parser.add_argument(
        "--num-scenarios", type=int, default=1000, help="Total number of scenarios"
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200],
        help="Agent counts to sample from",
    )
    parser.add_argument(
        "--max-timestep", type=int, default=2000, help="Maximum timesteps per scenario"
    )
    parser.add_argument(
        "--output", type=str, default="data/mapf_gpt_dataset.pkl", help="Output file"
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

    args = parser.parse_args()

    print("=" * 80)
    print("MAPF-GPT Data Generation")
    print("=" * 80)
    print(f"Map: {args.map}")
    print(f"Number of scenarios: {args.num_scenarios}")
    print(f"Agent counts: {args.agent_counts}")
    print(f"Max timesteps: {args.max_timestep}")
    print(f"Output file: {args.output}")
    print(f"FOV size: {args.fov_size}")
    print(f"Max visible agents: {args.max_agents_visible}")
    print(f"Filter duplicates: {args.filter_duplicates}")
    print(f"Balance actions: {args.balance_actions}")
    print("=" * 80)

    # Initialize tokenizer
    tokenizer = MAPFTokenizer(
        fov_size=args.fov_size, max_agents_visible=args.max_agents_visible
    )

    # Initialize generator
    generator = ExpertDataGenerator(map_file=args.map, tokenizer=tokenizer)

    # Generate dataset
    start_time = time.time()
    observations, actions = generator.generate_dataset_from_scenarios(
        num_scenarios=args.num_scenarios,
        agent_counts=args.agent_counts,
        max_timestep=args.max_timestep,
        output_file=None,  # Save later after filtering
        verbose=True,
    )
    generation_time = time.time() - start_time

    print(f"\nData generation completed in {generation_time:.2f}s")
    print(f"Total pairs before filtering: {len(observations)}")

    # Apply filters
    if args.filter_duplicates:
        print("\nFiltering duplicate observations...")
        observations, actions = generator.filter_duplicate_observations(observations, actions)
        print(f"Total pairs after duplicate removal: {len(observations)}")

    if args.balance_actions:
        print(f"\nBalancing action distribution (keep {args.wait_keep_ratio*100}% WAIT)...")
        observations, actions = generator.balance_action_distribution(
            observations, actions, args.wait_keep_ratio
        )
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
    generator.save_dataset(observations, actions, args.output)

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
