"""
Train MAPF-GPT Transformer with Loguru Profiling

Trains the Transformer model using expert demonstration data.
Uses imitation learning (supervised learning) with cross-entropy loss.
Includes comprehensive logging and profiling with loguru.
"""

import argparse
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from loguru import logger

from pypibt.mapf_transformer import MAPFTransformer, save_model
from pypibt.expert_data_generator import ExpertDataGenerator


# Configure loguru
logger.remove()
logger.add(
    "train_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="DEBUG",
)
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")


class MAPFDataset(Dataset):
    """PyTorch Dataset for MAPF training data."""

    def __init__(self, observations: list[list[int]], actions: list[int]):
        """
        Initialize dataset.

        Args:
            observations: List of observation token sequences
            actions: List of action tokens
        """
        logger.debug(f"Creating dataset with {len(observations)} samples")
        self.observations = torch.tensor(observations, dtype=torch.long)
        self.actions = torch.tensor(actions, dtype=torch.long)

        assert len(self.observations) == len(
            self.actions
        ), "Observations and actions must have same length"

        logger.debug(f"Dataset created: observations shape={self.observations.shape}, actions shape={self.actions.shape}")

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
    epoch: int,
) -> float:
    """
    Train for one epoch.

    Args:
        model: MAPF Transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average training loss
    """
    logger.debug(f"Starting training epoch {epoch}")
    model.train()
    total_loss = 0.0
    num_batches = 0

    batch_times = []
    forward_times = []
    backward_times = []

    for batch_idx, (observations, actions) in enumerate(dataloader):
        batch_start = time.time()

        observations = observations.to(device)
        actions = actions.to(device)

        # Forward pass
        forward_start = time.time()
        optimizer.zero_grad()
        logits = model(observations)
        forward_time = time.time() - forward_start
        forward_times.append(forward_time)

        # Compute loss
        loss = criterion(logits, actions)

        # Backward pass
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        backward_times.append(backward_time)

        total_loss += loss.item()
        num_batches += 1

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Log every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            avg_batch_time = np.mean(batch_times[-100:])
            logger.info(
                f"Epoch {epoch} Batch {batch_idx+1}/{len(dataloader)}: "
                f"loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, "
                f"batch_time={avg_batch_time:.3f}s"
            )

    avg_loss = total_loss / num_batches
    avg_forward_time = np.mean(forward_times)
    avg_backward_time = np.mean(backward_times)
    avg_batch_time = np.mean(batch_times)

    logger.debug(
        f"Epoch {epoch} training complete: avg_loss={avg_loss:.4f}, "
        f"avg_forward_time={avg_forward_time:.3f}s, "
        f"avg_backward_time={avg_backward_time:.3f}s, "
        f"avg_batch_time={avg_batch_time:.3f}s"
    )

    return avg_loss


def evaluate(
    model: MAPFTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Evaluate model on validation set.

    Args:
        model: MAPF Transformer model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        epoch: Current epoch number

    Returns:
        Tuple of (average loss, accuracy)
    """
    logger.debug(f"Starting validation for epoch {epoch}")
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

    logger.debug(f"Validation complete: avg_loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train MAPF-GPT Transformer with profiling")
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

    logger.info("=" * 80)
    logger.info("MAPF-GPT Transformer Training with Profiling")
    logger.info("=" * 80)

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

    logger.info(f"Device: {device}")

    # Load dataset
    logger.info(f"\nLoading dataset from {args.data}...")
    load_start = time.time()
    observations, actions, metadata = ExpertDataGenerator.load_dataset(args.data)
    load_time = time.time() - load_start

    vocab_size = metadata["vocab_size"]
    seq_len = metadata["sequence_length"]

    logger.info(f"Dataset loaded in {load_time:.2f}s")
    logger.info(f"  Total pairs: {len(observations)}")
    logger.info(f"  Vocabulary size: {vocab_size}")
    logger.info(f"  Sequence length: {seq_len}")
    logger.debug(f"  Observation dtype: {type(observations[0][0])}")
    logger.debug(f"  Action dtype: {type(actions[0])}")

    # Create dataset
    logger.info("\nCreating PyTorch dataset...")
    dataset_start = time.time()
    full_dataset = MAPFDataset(observations, actions)
    dataset_time = time.time() - dataset_start
    logger.info(f"Dataset created in {dataset_time:.2f}s")

    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logger.info(f"\nDataset split:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")

    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")

    # Create model
    logger.info(f"\nModel configuration:")
    logger.info(f"  d_model: {args.d_model}")
    logger.info(f"  nhead: {args.nhead}")
    logger.info(f"  num_layers: {args.num_layers}")
    logger.info(f"  dim_feedforward: {args.dim_feedforward}")
    logger.info(f"  dropout: {args.dropout}")

    logger.info("\nInitializing model...")
    model_start = time.time()
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
    model_time = time.time() - model_start

    num_params = model.count_parameters()
    logger.info(f"Model initialized in {model_time:.2f}s")
    logger.info(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Training")
    logger.info("=" * 80)

    best_val_loss = float("inf")
    training_start = time.time()

    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Estimate remaining time
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = args.epochs - epoch
        estimated_remaining = avg_epoch_time * remaining_epochs

        # Print progress
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.2f}%, "
            f"time={epoch_time:.2f}s, "
            f"est_remaining={estimated_remaining/60:.1f}min"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  New best validation loss! Saving model to {args.output}")
            save_model(model, args.output)

    training_time = time.time() - training_start

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Total training time: {training_time/60:.2f} minutes")
    logger.info(f"Average epoch time: {np.mean(epoch_times):.2f}s")
    logger.info(f"Model saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
