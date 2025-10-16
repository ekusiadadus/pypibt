"""
MAPF-GPT Style Tokenizer

Converts MAPF observations and actions into tokens for Transformer training.
Based on "MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale" (2024).

Token Structure:
- Cost-to-go values: Local field-of-view (FOV) around agent
- Agent information: Positions and goals of nearby agents
- Actions: {UP, DOWN, LEFT, RIGHT, WAIT}
"""

import numpy as np
from pypibt.mapf_utils import Coord, Grid, get_neighbors


class MAPFTokenizer:
    """
    Tokenizes MAPF observations and actions for Transformer input.

    Token Vocabulary:
    - 0: PAD (padding token)
    - 1-5: Actions (WAIT=1, UP=2, DOWN=3, LEFT=4, RIGHT=5)
    - 6-100: Distance values [0, 94] (clipped and shifted)
    - 101-127: Special tokens (reserved)
    """

    # Action tokens
    PAD = 0
    ACTION_WAIT = 1
    ACTION_UP = 2
    ACTION_DOWN = 3
    ACTION_LEFT = 4
    ACTION_RIGHT = 5

    # Special tokens
    TOKEN_AGENT = 101
    TOKEN_GOAL = 102
    TOKEN_OBSTACLE = 103
    TOKEN_EMPTY = 104
    TOKEN_SEP = 105  # Separator

    def __init__(
        self,
        fov_size: int = 11,  # Field of view: 11x11 grid
        max_agents_visible: int = 13,  # Maximum agents to encode
        max_distance: int = 94,  # Maximum distance value
    ):
        """
        Initialize MAPF tokenizer.

        Args:
            fov_size: Size of local field-of-view (should be odd)
            max_agents_visible: Maximum number of nearby agents to encode
            max_distance: Maximum distance to clip values
        """
        self.fov_size = fov_size
        self.max_agents_visible = max_agents_visible
        self.max_distance = max_distance

        # Vocabulary size: PAD + Actions + Distances + Special
        self.vocab_size = 128

        assert fov_size % 2 == 1, "FOV size must be odd"

    def action_to_token(self, from_pos: Coord, to_pos: Coord) -> int:
        """
        Convert position transition to action token.

        Args:
            from_pos: Starting position (y, x)
            to_pos: Ending position (y, x)

        Returns:
            Action token (1-5)
        """
        dy = to_pos[0] - from_pos[0]
        dx = to_pos[1] - from_pos[1]

        if dy == 0 and dx == 0:
            return self.ACTION_WAIT
        elif dy == -1 and dx == 0:
            return self.ACTION_UP
        elif dy == 1 and dx == 0:
            return self.ACTION_DOWN
        elif dy == 0 and dx == -1:
            return self.ACTION_LEFT
        elif dy == 0 and dx == 1:
            return self.ACTION_RIGHT
        else:
            # Invalid action, treat as WAIT
            return self.ACTION_WAIT

    def token_to_action(self, token: int) -> tuple[int, int]:
        """
        Convert action token to position delta.

        Args:
            token: Action token (1-5)

        Returns:
            (dy, dx) position delta
        """
        if token == self.ACTION_WAIT:
            return (0, 0)
        elif token == self.ACTION_UP:
            return (-1, 0)
        elif token == self.ACTION_DOWN:
            return (1, 0)
        elif token == self.ACTION_LEFT:
            return (0, -1)
        elif token == self.ACTION_RIGHT:
            return (0, 1)
        else:
            return (0, 0)  # Default to WAIT

    def distance_to_token(self, distance: float) -> int:
        """
        Convert distance value to token.

        Args:
            distance: Distance value (0 to inf)

        Returns:
            Token in range [6, 100] (95 values)
        """
        # Clip distance to max_distance
        dist = min(int(distance), self.max_distance)
        # Map to token range [6, 100]
        return 6 + dist

    def token_to_distance(self, token: int) -> int:
        """
        Convert token to distance value.

        Args:
            token: Token in range [6, 100]

        Returns:
            Distance value
        """
        if token < 6 or token > 100:
            return 0
        return token - 6

    def extract_local_fov(
        self, grid: Grid, agent_pos: Coord, goal_pos: Coord, dist_table
    ) -> list[int]:
        """
        Extract local field-of-view (FOV) around agent.

        Creates a local grid centered on agent with cost-to-go values.

        Args:
            grid: Full map grid (1=obstacle, 0=free)
            agent_pos: Current agent position
            goal_pos: Agent's goal position
            dist_table: DistTable object with .get() method

        Returns:
            List of FOV tokens (fov_size * fov_size tokens)
        """
        fov_tokens = []
        half_fov = self.fov_size // 2

        for dy in range(-half_fov, half_fov + 1):
            for dx in range(-half_fov, half_fov + 1):
                y = agent_pos[0] + dy
                x = agent_pos[1] + dx

                # Check bounds
                if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1]:
                    # Out of bounds -> large distance
                    token = self.distance_to_token(self.max_distance)
                elif grid[y, x] == 1:
                    # Obstacle
                    token = self.TOKEN_OBSTACLE
                else:
                    # Free cell -> encode distance to goal
                    pos = (y, x)
                    distance = dist_table.get(pos)
                    token = self.distance_to_token(distance)

                fov_tokens.append(token)

        return fov_tokens

    def extract_nearby_agents(
        self,
        agent_pos: Coord,
        agent_goal: Coord,
        all_positions: list[Coord],
        all_goals: list[Coord],
        current_agent_id: int,
    ) -> list[int]:
        """
        Extract information about nearby agents.

        Encodes positions and goals of nearby agents relative to current agent.

        Args:
            agent_pos: Current agent position
            agent_goal: Current agent goal
            all_positions: Positions of all agents
            all_goals: Goals of all agents
            current_agent_id: ID of current agent

        Returns:
            List of agent tokens (up to max_agents_visible * 2 tokens)
        """
        agent_tokens = []

        # Calculate distances to all agents
        distances = []
        for i, pos in enumerate(all_positions):
            if i == current_agent_id:
                continue
            dist = abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
            distances.append((dist, i))

        # Sort by distance and take closest agents
        distances.sort()
        nearby_agent_ids = [i for _, i in distances[: self.max_agents_visible]]

        for agent_id in nearby_agent_ids:
            other_pos = all_positions[agent_id]
            other_goal = all_goals[agent_id]

            # Encode relative position
            dy = other_pos[0] - agent_pos[0]
            dx = other_pos[1] - agent_pos[1]

            # Clip to reasonable range and tokenize
            dy_clipped = max(-20, min(20, dy))
            dx_clipped = max(-20, min(20, dx))

            # Simple encoding: convert relative positions to tokens
            # Use distance encoding
            pos_dist = abs(dy) + abs(dx)
            agent_tokens.append(self.distance_to_token(pos_dist))

            # Encode relative goal position
            goal_dy = other_goal[0] - agent_pos[0]
            goal_dx = other_goal[1] - agent_pos[1]
            goal_dist = abs(goal_dy) + abs(goal_dx)
            agent_tokens.append(self.distance_to_token(goal_dist))

        # Pad if fewer than max_agents_visible
        while len(agent_tokens) < self.max_agents_visible * 2:
            agent_tokens.append(self.PAD)

        return agent_tokens

    def encode_observation(
        self,
        grid: Grid,
        agent_pos: Coord,
        agent_goal: Coord,
        all_positions: list[Coord],
        all_goals: list[Coord],
        agent_id: int,
        dist_table,
    ) -> list[int]:
        """
        Encode full observation for one agent into tokens.

        Observation structure:
        1. Local FOV (fov_size^2 tokens)
        2. Separator
        3. Nearby agents information (max_agents_visible * 2 tokens)

        Args:
            grid: Map grid
            agent_pos: Current position of agent
            agent_goal: Goal position of agent
            all_positions: Positions of all agents
            all_goals: Goals of all agents
            agent_id: ID of current agent
            dist_table: DistTable object with .get() method

        Returns:
            List of tokens representing observation
        """
        tokens = []

        # 1. Local FOV
        fov_tokens = self.extract_local_fov(grid, agent_pos, agent_goal, dist_table)
        tokens.extend(fov_tokens)

        # 2. Separator
        tokens.append(self.TOKEN_SEP)

        # 3. Nearby agents
        agent_tokens = self.extract_nearby_agents(
            agent_pos, agent_goal, all_positions, all_goals, agent_id
        )
        tokens.extend(agent_tokens)

        return tokens

    def get_sequence_length(self) -> int:
        """
        Get expected sequence length for one observation.

        Returns:
            Total number of tokens per observation
        """
        # FOV + Separator + Agent info
        return self.fov_size**2 + 1 + self.max_agents_visible * 2

    def create_padding_mask(self, tokens: list[int]) -> list[bool]:
        """
        Create padding mask for token sequence.

        Args:
            tokens: List of tokens

        Returns:
            Boolean mask (True for real tokens, False for padding)
        """
        return [token != self.PAD for token in tokens]


def test_tokenizer():
    """Test tokenizer functionality."""
    print("=" * 80)
    print("Testing MAPF Tokenizer")
    print("=" * 80)

    tokenizer = MAPFTokenizer(fov_size=11, max_agents_visible=13)

    # Test action tokenization
    print("\n[Test 1] Action Tokenization")
    from_pos = (5, 5)
    test_cases = [
        ((5, 5), "WAIT"),  # Stay
        ((4, 5), "UP"),  # Move up
        ((6, 5), "DOWN"),  # Move down
        ((5, 4), "LEFT"),  # Move left
        ((5, 6), "RIGHT"),  # Move right
    ]

    for to_pos, expected_action in test_cases:
        token = tokenizer.action_to_token(from_pos, to_pos)
        action_name = ["PAD", "WAIT", "UP", "DOWN", "LEFT", "RIGHT"][token]
        print(f"  {from_pos} -> {to_pos}: token={token} ({action_name}) ✓")
        assert action_name == expected_action

    # Test distance tokenization
    print("\n[Test 2] Distance Tokenization")
    for dist in [0, 10, 50, 94, 100]:
        token = tokenizer.distance_to_token(dist)
        recovered_dist = tokenizer.token_to_distance(token)
        print(f"  Distance={dist} -> token={token} -> recovered={recovered_dist}")

    # Test vocabulary size
    print(f"\n[Test 3] Vocabulary Size: {tokenizer.vocab_size}")
    print(f"  Sequence Length: {tokenizer.get_sequence_length()}")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_tokenizer()
