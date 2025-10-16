"""
Script to train Priority Learning network.

Usage:
    uv run python train_priority_network.py
"""

import numpy as np
from src.pypibt.mapf_utils import get_grid, get_scenario
from src.pypibt.priority_trainer import generate_training_data, train_priority_network
from src.pypibt.priority_net import save_model

print("="* 80)
print("Priority Network Training")
print("="* 80)

# Load map
grid = get_grid("assets/random-32-32-10.map")
print(f"\nMap loaded: {grid.shape}")

# Generate multiple scenarios with varying agent counts
print("\nGenerating scenarios...")
starts_list = []
goals_list = []

# Small scenarios (50 agents)
for i in range(20):
    starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 50)
    starts_list.append(starts)
    goals_list.append(goals)

# Medium scenarios (100 agents)
for i in range(20):
    starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 100)
    starts_list.append(starts)
    goals_list.append(goals)

# Large scenarios (150 agents)
for i in range(10):
    starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 150)
    starts_list.append(starts)
    goals_list.append(goals)

print(f"Generated {len(starts_list)} scenarios")

# Generate training data
print("\n" + "="* 80)
features, priorities = generate_training_data(
    grid,
    starts_list,
    goals_list,
    num_scenarios=50  # Use all scenarios
)

# Train model
print("\n" + "="* 80)
model = train_priority_network(
    features,
    priorities,
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    validation_split=0.2
)

# Save model
model_path = "models/priority_network.pth"
import os
os.makedirs("models", exist_ok=True)
save_model(model, model_path)
print(f"\nModel saved to: {model_path}")

print("\n" + "="* 80)
print("Training completed successfully!")
print("="* 80)
