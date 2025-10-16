"""
Train MAPF-GPT Transformer

Trains the Transformer model using expert demonstration data.
Uses imitation learning (supervised learning) with cross-entropy loss.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path

from pypibt.mapf_transformer import MAPFTransformer, save_model
from pypibt.expert_data_generator import ExpertDataGenerator


class MAPFDataset(Dataset):
    """PyTorch Dataset for MAPF training data."""

    def __init__(self, observations: list[list[int]], actions: list[int]):
        """
        Initialize dataset.

        Args:
            observations: List of observation token sequences
            actions: List of action tokens
        """
        self.observations = torch.tensor(observations, dtype=torch.long)
        self.actions = torch.tensor(actions, dtype=torch.long)

        assert len(self.observations) == len(
            self.actions
        ), "Observations and actions must have same length"

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]


def train_epoch(
    model: MAPFTransformer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Args:
        model: MAPF Transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for observations, actions in dataloader:
        observations = observations.to(device)
        actions = actions.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(observations)

        # Compute loss
        loss = criterion(logits, actions)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(
    model: MAPFTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on validation set.

    Args:
        model: MAPF Transformer model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for observations, actions in dataloader:
            observations = observations.to(device)
            actions = actions.to(device)

            # Forward pass
            logits = model(observations)
            loss = criterion(logits, actions)

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == actions).sum().item()
            total += actions.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train MAPF-GPT Transformer")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data file (.pkl)"
    )
    parser.add_argument(
        "--output", type=str, default="models/mapf_gpt.pth", help="Output model path"
    )
    parser.add_argument("--d-model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument(
        "--num-layers", type=int, default=6, help="Number of Transformer layers"
    )
    parser.add_argument(
        "--dim-feedforward", type=int, default=1024, help="Feedforward dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu, cuda, mps, auto)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MAPF-GPT Transformer Training")
    print("=" * 80)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    observations, actions, metadata = ExpertDataGenerator.load_dataset(args.data)
    vocab_size = metadata["vocab_size"]
    seq_len = metadata["sequence_length"]
    print(f"  Total pairs: {len(observations)}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")

    # Create dataset
    full_dataset = MAPFDataset(observations, actions)

    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    print(f"\nModel configuration:")
    print(f"  d_model: {args.d_model}")
    print(f"  nhead: {args.nhead}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  dim_feedforward: {args.dim_feedforward}")
    print(f"  dropout: {args.dropout}")

    model = MAPFTransformer(
        vocab_size=vocab_size,
        num_actions=6,  # PAD + 5 actions
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=seq_len,
    ).to(device)

    num_params = model.count_parameters()
    print(f"\nModel parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_val_loss = float("inf")
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.2f}%, "
                f"time={epoch_time:.2f}s"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.output)

    training_time = time.time() - training_start

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {training_time:.2f}s")
    print(f"Model saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
