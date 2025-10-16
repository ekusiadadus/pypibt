"""
Priority Learning Network for PIBT.

A lightweight neural network that learns optimal agent priorities from expert demonstrations.
Based on imitation learning principles.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class PriorityNet(nn.Module):
    """
    Lightweight MLP for predicting agent priorities.

    Input: Per-agent features (position, goal, distances, densities)
    Output: Priority score for each agent
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32):
        """
        Args:
            input_dim: Number of features per agent (default: 6)
                - current_x, current_y
                - goal_x, goal_y
                - distance_to_goal
                - local_agent_density
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Simple 3-layer MLP
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Single priority score per agent
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_agents, input_dim)
               or (num_agents, input_dim)

        Returns:
            Priority scores of shape (batch_size, num_agents) or (num_agents,)
        """
        if x.dim() == 2:
            # Single instance: (num_agents, input_dim)
            priorities = self.network(x).squeeze(-1)  # (num_agents,)
        else:
            # Batch: (batch_size, num_agents, input_dim)
            batch_size, num_agents, _ = x.shape
            # Reshape to (batch_size * num_agents, input_dim)
            x_flat = x.view(-1, self.input_dim)
            priorities_flat = self.network(x_flat).squeeze(-1)
            # Reshape back to (batch_size, num_agents)
            priorities = priorities_flat.view(batch_size, num_agents)

        return priorities

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict priorities from numpy features (inference mode).

        Args:
            features: NumPy array of shape (num_agents, input_dim)

        Returns:
            Priority scores as NumPy array of shape (num_agents,)
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            priorities = self.forward(x)
            return priorities.numpy()


def save_model(model: PriorityNet, path: str | Path):
    """Save trained model to disk."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
    }, path)


def load_model(path: str | Path) -> PriorityNet:
    """Load trained model from disk."""
    checkpoint = torch.load(path, weights_only=True)
    model = PriorityNet(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing PriorityNet...")

    # Create model
    model = PriorityNet(input_dim=6, hidden_dim=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test single agent
    features_single = np.random.randn(10, 6).astype(np.float32)
    priorities_single = model.predict(features_single)
    print(f"Single batch test: input shape={features_single.shape}, output shape={priorities_single.shape}")

    # Test batch
    features_batch = torch.randn(4, 10, 6)  # 4 scenarios, 10 agents each
    priorities_batch = model(features_batch)
    print(f"Batch test: input shape={features_batch.shape}, output shape={priorities_batch.shape}")

    # Test save/load
    save_model(model, "test_model.pth")
    loaded_model = load_model("test_model.pth")
    priorities_loaded = loaded_model.predict(features_single)
    assert np.allclose(priorities_single, priorities_loaded), "Model save/load failed!"
    print("Save/load test passed!")

    import os
    os.remove("test_model.pth")

    print("All tests passed!")
