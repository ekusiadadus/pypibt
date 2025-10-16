"""
MAPF-GPT: Transformer Model for Multi-Agent Path Finding

Based on "MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale" (2024).
Implements a GPT-style decoder-only Transformer for action prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds positional information to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class MAPFTransformer(nn.Module):
    """
    GPT-style Transformer for MAPF action prediction.

    Architecture:
    - Token embedding
    - Positional encoding
    - Multi-layer Transformer decoder
    - Output projection to action space
    """

    def __init__(
        self,
        vocab_size: int = 128,
        num_actions: int = 6,  # PAD + 5 actions
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        """
        Initialize MAPF Transformer.

        Args:
            vocab_size: Size of token vocabulary
            num_actions: Number of action classes (PAD + WAIT + UP + DOWN + LEFT + RIGHT)
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layers: Number of Transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, feature)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output projection to action space
        self.output_projection = nn.Linear(d_model, num_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Source observation tokens (batch_size, seq_len)
            tgt: Target tokens for teacher forcing (optional, not used in decoder-only)

        Returns:
            Action logits (batch_size, num_actions)
        """
        # Embedding
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)

        # Create causal mask (for auto-regressive generation, if needed)
        # For single-step prediction, we don't need causal mask
        # But keep it for potential multi-step extension
        seq_len = src.size(1)
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        # Transformer decoder (using src as both memory and target for decoder-only)
        # In decoder-only architecture, we use self-attention only
        memory = torch.zeros_like(src_emb)  # Dummy memory for decoder interface
        output = self.transformer_decoder(
            tgt=src_emb,
            memory=memory,
            tgt_mask=causal_mask,
        )

        # Take the last token's output (aggregate sequence information)
        output = output[:, -1, :]  # (batch_size, d_model)

        # Project to action space
        logits = self.output_projection(output)  # (batch_size, num_actions)

        return logits

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask for autoregressive generation.

        Args:
            sz: Sequence length

        Returns:
            Causal mask (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def predict(self, observation_tokens: torch.Tensor) -> int:
        """
        Predict action for a single observation.

        Args:
            observation_tokens: Observation token sequence (seq_len,)

        Returns:
            Predicted action token (int)
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            obs = observation_tokens.unsqueeze(0)  # (1, seq_len)

            # Forward pass
            logits = self(obs)  # (1, num_actions)

            # Get predicted action (argmax)
            action = torch.argmax(logits, dim=-1).item()

        return action

    def predict_batch(self, observation_tokens: torch.Tensor) -> torch.Tensor:
        """
        Predict actions for a batch of observations.

        Args:
            observation_tokens: Batch of observation sequences (batch_size, seq_len)

        Returns:
            Predicted action tokens (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self(observation_tokens)  # (batch_size, num_actions)
            actions = torch.argmax(logits, dim=-1)  # (batch_size,)
        return actions

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_model(model: MAPFTransformer, path: str) -> None:
    """
    Save model to file.

    Args:
        model: Model to save
        path: Output file path
    """
    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save model state dict and config
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'vocab_size': model.vocab_size,
            'num_actions': model.num_actions,
            'd_model': model.d_model,
            'max_seq_len': model.max_seq_len,
        },
        path,
    )


def load_model(path: str) -> MAPFTransformer:
    """
    Load model from file.

    Args:
        path: Input file path

    Returns:
        Loaded model
    """
    checkpoint = torch.load(path, map_location='cpu')

    # Create model with saved config
    model = MAPFTransformer(
        vocab_size=checkpoint['vocab_size'],
        num_actions=checkpoint['num_actions'],
        d_model=checkpoint['d_model'],
        max_seq_len=checkpoint['max_seq_len'],
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def test_transformer():
    """Test Transformer model."""
    print("=" * 80)
    print("Testing MAPF Transformer")
    print("=" * 80)

    # Model configuration
    vocab_size = 128
    num_actions = 6
    d_model = 256
    seq_len = 148  # From tokenizer

    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Number of actions: {num_actions}")
    print(f"  Embedding dimension: {d_model}")
    print(f"  Sequence length: {seq_len}")

    # Create model
    model = MAPFTransformer(
        vocab_size=vocab_size,
        num_actions=num_actions,
        d_model=d_model,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
    )

    print(f"\nModel Parameters: {model.count_parameters():,}")

    # Test forward pass
    print("\n[Test 1] Forward Pass")
    batch_size = 4
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, num_actions), "Output shape mismatch"

    # Test prediction
    print("\n[Test 2] Single Prediction")
    single_input = torch.randint(0, vocab_size, (seq_len,))
    action = model.predict(single_input)
    print(f"  Input shape: {single_input.shape}")
    print(f"  Predicted action: {action}")
    assert 0 <= action < num_actions, "Invalid action prediction"

    # Test batch prediction
    print("\n[Test 3] Batch Prediction")
    batch_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    actions = model.predict_batch(batch_input)
    print(f"  Input shape: {batch_input.shape}")
    print(f"  Predicted actions: {actions}")
    assert actions.shape == (batch_size,), "Batch prediction shape mismatch"

    # Test save/load
    print("\n[Test 4] Save and Load")
    save_path = "models/mapf_transformer_test.pth"
    save_model(model, save_path)
    print(f"  Model saved to: {save_path}")

    loaded_model = load_model(save_path)
    print(f"  Model loaded from: {save_path}")

    # Verify loaded model produces same output
    output_original = model.predict_batch(batch_input)
    output_loaded = loaded_model.predict_batch(batch_input)
    assert torch.equal(output_original, output_loaded), "Loaded model output mismatch"
    print(f"  ✓ Loaded model produces identical output")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_transformer()
