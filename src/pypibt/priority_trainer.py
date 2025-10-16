"""
Training system for Priority Learning.

Generates training data by running PIBT with different configurations
and learns optimal priorities from successful runs.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path
from typing import List, Tuple

from .priority_net import PriorityNet, save_model
from .pibt import PIBT
from .mapf_utils import Config, Grid, get_neighbors


class PriorityDataset(Dataset):
    """Dataset for priority learning."""

    def __init__(self, features: np.ndarray, priorities: np.ndarray):
        """
        Args:
            features: (N_samples, N_agents, feature_dim)
            priorities: (N_samples, N_agents) - target priorities
        """
        self.features = torch.from_numpy(features).float()
        self.priorities = torch.from_numpy(priorities).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.priorities[idx]


def extract_features_from_config(
    pibt: PIBT, config: Config, grid: Grid
) -> np.ndarray:
    """
    Extract features for all agents from current configuration.

    Features per agent:
    1. current_x (normalized)
    2. current_y (normalized)
    3. goal_x (normalized)
    4. goal_y (normalized)
    5. distance_to_goal (normalized)
    6. local_agent_density (周辺のエージェント密度)

    Returns:
        features: (N_agents, 6)
    """
    N = len(config)
    features = np.zeros((N, 6), dtype=np.float32)

    grid_h, grid_w = grid.shape

    for i in range(N):
        pos = config[i]
        goal = pibt.goals[i]

        # Position features (normalized to [0, 1])
        features[i, 0] = pos[0] / grid_h
        features[i, 1] = pos[1] / grid_w
        features[i, 2] = goal[0] / grid_h
        features[i, 3] = goal[1] / grid_w

        # Distance to goal (normalized)
        dist = pibt.dist_tables[i].get(pos)
        max_dist = grid_h + grid_w  # Manhattan distance upper bound
        features[i, 4] = dist / max_dist

        # Local agent density (count agents within radius 3)
        local_agents = 0
        radius = 3
        for j in range(N):
            if i == j:
                continue
            other_pos = config[j]
            manhattan_dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
            if manhattan_dist <= radius:
                local_agents += 1
        # Normalize by maximum possible agents in radius
        max_local_agents = (2 * radius + 1) ** 2 - 1
        features[i, 5] = local_agents / max_local_agents

    return features


def generate_training_data(
    grid: Grid,
    starts_list: List[Config],
    goals_list: List[Config],
    num_scenarios: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data by running PIBT with different configurations.

    Strategy:
    - Run baseline PIBT (label: lower priority)
    - Run optimized PIBT with Hindrance+Regret (label: higher priority)
    - Use solution quality as weight for priority labels

    Args:
        grid: Map grid
        starts_list: List of start configurations
        goals_list: List of goal configurations
        num_scenarios: Number of scenarios to generate

    Returns:
        features: (N_samples, N_agents, 6)
        priorities: (N_samples, N_agents)
    """
    all_features = []
    all_priorities = []

    print(f"Generating training data from {num_scenarios} scenarios...")

    for scenario_idx in range(num_scenarios):
        starts = starts_list[scenario_idx % len(starts_list)]
        goals = goals_list[scenario_idx % len(goals_list)]

        # Run baseline PIBT
        pibt_baseline = PIBT(grid, starts, goals, seed=scenario_idx)
        plan_baseline = pibt_baseline.run(max_timestep=1000)
        cost_baseline = len(plan_baseline)

        # Run optimized PIBT
        pibt_optimized = PIBT(
            grid, starts, goals, seed=scenario_idx,
            enable_hindrance=True,
            hindrance_weight=0.3,
            enable_regret_learning=True,
            regret_learning_iterations=2,
            regret_weight=0.2,
        )
        plan_optimized = pibt_optimized.run(max_timestep=1000)
        cost_optimized = len(plan_optimized)

        # Extract features from initial configuration
        features = extract_features_from_config(pibt_baseline, starts, grid)

        # Compute target priorities based on solution quality
        # Better solution = higher priorities for agents that reached goal faster
        if cost_optimized < cost_baseline:
            # Optimized is better: use distance-based priorities (closer = higher)
            N = len(starts)
            priorities = np.zeros(N, dtype=np.float32)
            for i in range(N):
                dist = pibt_baseline.dist_tables[i].get(starts[i])
                # Inverse distance: closer agents get higher priority
                max_dist = grid.shape[0] + grid.shape[1]
                priorities[i] = 1.0 - (dist / max_dist)
        else:
            # Baseline is better or equal: use uniform priorities
            priorities = np.ones(len(starts), dtype=np.float32) * 0.5

        all_features.append(features)
        all_priorities.append(priorities)

        if (scenario_idx + 1) % 10 == 0:
            print(f"  Generated {scenario_idx + 1}/{num_scenarios} scenarios")

    # Convert to arrays - handle variable agent counts by padding
    # Find max number of agents
    max_agents = max(len(f) for f in all_features)

    # Pad to max_agents
    features_padded = []
    priorities_padded = []

    for features, priorities in zip(all_features, all_priorities):
        n_agents = len(features)
        if n_agents < max_agents:
            # Pad with zeros
            features_pad = np.zeros((max_agents, features.shape[1]), dtype=np.float32)
            features_pad[:n_agents] = features
            priorities_pad = np.zeros(max_agents, dtype=np.float32)
            priorities_pad[:n_agents] = priorities
        else:
            features_pad = features
            priorities_pad = priorities

        features_padded.append(features_pad)
        priorities_padded.append(priorities_pad)

    features_array = np.array(features_padded, dtype=np.float32)
    priorities_array = np.array(priorities_padded, dtype=np.float32)

    print(f"Generated dataset: features shape={features_array.shape}, priorities shape={priorities_array.shape}")

    return features_array, priorities_array


def train_priority_network(
    features: np.ndarray,
    priorities: np.ndarray,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
) -> PriorityNet:
    """
    Train priority network.

    Args:
        features: (N_samples, N_agents, 6)
        priorities: (N_samples, N_agents)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Validation split ratio

    Returns:
        Trained PriorityNet model
    """
    print("\nTraining Priority Network...")
    print(f"  Input shape: {features.shape}")
    print(f"  Target shape: {priorities.shape}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # Split into train/val
    n_samples = len(features)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = PriorityDataset(features[train_indices], priorities[train_indices])
    val_dataset = PriorityDataset(features[val_indices], priorities[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = PriorityNet(input_dim=6, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, batch_priorities in train_loader:
            optimizer.zero_grad()
            pred_priorities = model(batch_features)
            loss = criterion(pred_priorities, batch_priorities)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_priorities in val_loader:
                pred_priorities = model(batch_features)
                loss = criterion(pred_priorities, batch_priorities)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

    return model


if __name__ == "__main__":
    # Quick test
    print("Testing Priority Trainer...")

    from .mapf_utils import get_grid, get_scenario

    grid = get_grid("../../assets/random-32-32-10.map")

    # Generate multiple scenarios
    starts_list = []
    goals_list = []
    for i in range(10):
        starts, goals = get_scenario("../../assets/random-32-32-10-random-1.scen", 50)
        starts_list.append(starts)
        goals_list.append(goals)

    # Generate training data
    features, priorities = generate_training_data(grid, starts_list, goals_list, num_scenarios=10)

    print(f"\nDataset generated: {features.shape}, {priorities.shape}")

    # Train model (small test)
    model = train_priority_network(features, priorities, epochs=10, batch_size=4)

    print("\nTrainer test completed!")
