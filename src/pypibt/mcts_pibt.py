"""
Monte Carlo Tree Search Enhanced PIBT (Phase 8A).

Based on MATS-LP (AAAI 2024) research:
- Neural MCTS with 250 rollouts per decision
- CostTracer: 161,734 parameters, PPO-trained
- Performance: 103ms (32 agents), 300ms (192 agents)
- Expected improvement: +25% performance

This is a lightweight implementation without neural network components,
focusing on the core MCTS algorithm integrated with PIBT.

References:
- "Multi-Agent Path Finding with Monte Carlo Tree Search" (AAAI 2024)
- MATS-LP: https://github.com/mlvlab/mats-lp
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Configs, Coord, Grid, get_neighbors


@dataclass
class MCTSNode:
    """
    MCTS tree node representing a state (configuration + priorities).

    Attributes:
        config: Current configuration
        priorities: Current priority values
        parent: Parent node
        children: Child nodes
        visits: Number of times this node was visited
        value: Accumulated reward (lower cost = higher reward)
        untried_actions: Priority perturbations not yet explored
    """
    config: Config
    priorities: list[float]
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = None
    visits: int = 0
    value: float = 0.0
    untried_actions: list[tuple[int, float]] = None  # (agent_id, priority_delta)

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            # Generate priority perturbation actions
            self.untried_actions = []
            n_agents = len(self.priorities)
            # Try perturbing each agent's priority
            for agent_id in range(min(n_agents, 5)):  # Limit to top 5 for efficiency
                for delta in [-0.3, -0.1, 0.1, 0.3]:
                    self.untried_actions.append((agent_id, delta))

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self, goals: Config) -> bool:
        """Check if this is a goal configuration."""
        return self.config == goals

    def best_child(self, exploration_weight: float = 1.414) -> "MCTSNode":
        """
        Select best child using UCB1 formula.

        UCB1 = value/visits + exploration_weight * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_weight: Weight for exploration term (default: sqrt(2))

        Returns:
            Child node with highest UCB1 value
        """
        if not self.children:
            raise ValueError("No children to select from")

        def ucb1(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')
            # Inverse value since we minimize cost
            exploitation = -child.value / child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )
            return exploitation + exploration

        return max(self.children, key=ucb1)


class MCTSPIBT:
    """
    Monte Carlo Tree Search enhanced PIBT.

    Uses MCTS to explore priority orderings and find better solutions
    through simulated rollouts and backpropagation.
    """

    def __init__(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        seed: int = 0,
        num_rollouts: int = 250,
        rollout_depth: int = 50,  # Reduced from 100 for efficiency
        exploration_weight: float = 1.414,
        time_limit_ms: float = 5000.0,
    ):
        """
        Initialize MCTS-PIBT.

        Args:
            grid: Grid map
            starts: Starting positions
            goals: Goal positions
            seed: Random seed
            num_rollouts: Number of MCTS rollouts per iteration
            rollout_depth: Maximum depth for each rollout
            exploration_weight: UCB1 exploration weight
            time_limit_ms: Time limit in milliseconds
        """
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(starts)
        self.rng = np.random.default_rng(seed)

        # MCTS parameters
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.time_limit_ms = time_limit_ms

        # Distance tables
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # Cache
        self.NIL = self.N
        self.NIL_COORD: Coord = self.grid.shape

    def _initialize_priorities(self, config: Config) -> list[float]:
        """Initialize distance-based priorities for a configuration."""
        return [
            self.dist_tables[i].get(config[i]) / self.grid.size
            for i in range(self.N)
        ]

    def _run_pibt_single_step(
        self, Q_from: Config, priorities: list[float]
    ) -> Config:
        """Execute single PIBT step."""
        Q_to: list[Coord] = []
        occupied_now = np.full(self.grid.shape, self.NIL, dtype=int)
        occupied_nxt = np.full(self.grid.shape, self.NIL, dtype=int)

        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            occupied_now[v] = i

        # Sort by priority
        A = sorted(list(range(self.N)), key=lambda i: priorities[i], reverse=True)

        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self._func_pibt(Q_from, Q_to, i, occupied_now, occupied_nxt)

        # Cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            occupied_now[q_from] = self.NIL
            occupied_nxt[q_to] = self.NIL

        return tuple(Q_to)

    def _func_pibt(
        self,
        Q_from: Config,
        Q_to: list[Coord],
        i: int,
        occupied_now: np.ndarray,
        occupied_nxt: np.ndarray,
    ) -> bool:
        """PIBT function for single agent."""
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        for v in C:
            if occupied_nxt[v] != self.NIL:
                continue

            j = occupied_now[v]

            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            Q_to[i] = v
            occupied_nxt[v] = i

            if (
                j != self.NIL
                and Q_to[j] == self.NIL_COORD
                and not self._func_pibt(Q_from, Q_to, j, occupied_now, occupied_nxt)
            ):
                continue

            return True

        Q_to[i] = Q_from[i]
        occupied_nxt[Q_from[i]] = i
        return False

    def _rollout(self, node: MCTSNode, max_steps: int) -> float:
        """
        Perform random rollout from node to estimate value.

        Modified: Shorter rollouts with early termination for efficiency.

        Args:
            node: Starting node
            max_steps: Maximum rollout steps

        Returns:
            Estimated cost (negative of timesteps for maximization)
        """
        config = node.config
        priorities = list(node.priorities)

        # Count agents at goal initially
        agents_at_goal = sum(1 for i in range(self.N) if config[i] == self.goals[i])
        initial_at_goal = agents_at_goal

        for step in range(max_steps):
            # Check if all agents reached goal
            if agents_at_goal == self.N:
                # Reward: negative cost + bonus for reaching goal
                return -float(step) + 100.0

            # PIBT step
            config = self._run_pibt_single_step(config, priorities)

            # Update priorities and count progress
            agents_at_goal = 0
            for i in range(self.N):
                if config[i] == self.goals[i]:
                    priorities[i] -= np.floor(priorities[i])
                    agents_at_goal += 1
                else:
                    priorities[i] += 1.0

            # Early termination if no progress
            if step > max_steps // 2 and agents_at_goal <= initial_at_goal:
                # Penalize lack of progress
                return -float(max_steps) - 50.0

        # Partial credit based on agents at goal
        progress_bonus = (agents_at_goal / self.N) * 50.0
        return -float(max_steps) + progress_bonus

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand node by trying an untried action.

        Args:
            node: Node to expand

        Returns:
            New child node
        """
        if not node.untried_actions:
            raise ValueError("No untried actions available")

        # Pop an action
        agent_id, priority_delta = node.untried_actions.pop()

        # Create new priorities by perturbing one agent's priority
        new_priorities = list(node.priorities)
        new_priorities[agent_id] += priority_delta

        # Execute one PIBT step with new priorities
        new_config = self._run_pibt_single_step(node.config, new_priorities)

        # Update priorities after step
        for i in range(self.N):
            if new_config[i] != self.goals[i]:
                new_priorities[i] += 1.0
            else:
                new_priorities[i] -= np.floor(new_priorities[i])

        # Create child node
        child = MCTSNode(
            config=new_config,
            priorities=new_priorities,
            parent=node,
        )

        node.children.append(child)
        return child

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate value up the tree.

        Args:
            node: Starting node
            value: Value to backpropagate
        """
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _mcts_search(self, root: MCTSNode) -> MCTSNode:
        """
        Perform one iteration of MCTS (selection, expansion, simulation, backpropagation).

        Args:
            root: Root node

        Returns:
            Best leaf node found
        """
        # Selection: traverse tree using UCB1
        node = root
        while not node.is_terminal(self.goals) and node.is_fully_expanded():
            if not node.children:
                break
            node = node.best_child(self.exploration_weight)

        # Expansion: if not terminal and not fully expanded, expand
        if not node.is_terminal(self.goals) and not node.is_fully_expanded():
            node = self._expand(node)

        # Simulation: rollout from new node
        value = self._rollout(node, self.rollout_depth)

        # Backpropagation: update tree
        self._backpropagate(node, value)

        return node

    def _extract_solution_from_tree(self, root: MCTSNode, max_timestep: int) -> Configs:
        """
        Extract best solution using learned priorities from MCTS.

        Modified: Use best child's priorities instead of just executing from root.

        Args:
            root: Root node
            max_timestep: Maximum timesteps

        Returns:
            Solution configurations
        """
        # Find best path in tree (most visited or highest value)
        best_priorities = list(root.priorities)

        # If tree has children, use best child's priorities
        if root.children:
            # Select best child by value/visits ratio
            best_child = max(
                root.children,
                key=lambda c: (c.value / c.visits if c.visits > 0 else float('-inf'))
            )
            best_priorities = list(best_child.priorities)

        # Execute PIBT with learned best priorities
        configs = [self.starts]  # Start from initial config
        priorities = self._initialize_priorities(self.starts)

        # Apply learned priority adjustments
        for i in range(min(len(priorities), len(best_priorities))):
            priorities[i] = best_priorities[i]

        current_config = self.starts

        for _ in range(max_timestep):
            if current_config == self.goals:
                break

            # Execute PIBT step
            current_config = self._run_pibt_single_step(current_config, priorities)
            configs.append(current_config)

            # Update priorities
            for i in range(self.N):
                if current_config[i] != self.goals[i]:
                    priorities[i] += 1.0
                else:
                    priorities[i] -= np.floor(priorities[i])

        return configs

    def run(self, max_timestep: int = 1000) -> Configs:
        """
        Run MCTS-enhanced PIBT.

        Algorithm:
        1. Initialize root node with initial configuration
        2. Perform MCTS rollouts to build search tree
        3. Extract best solution from tree
        4. Return solution

        Args:
            max_timestep: Maximum timesteps

        Returns:
            Solution configurations
        """
        start_time = time.time()
        time_limit_sec = self.time_limit_ms / 1000.0

        print(f"MCTS-PIBT: Starting with {self.num_rollouts} rollouts")

        # Initialize root node
        initial_priorities = self._initialize_priorities(self.starts)
        root = MCTSNode(config=self.starts, priorities=initial_priorities)

        # MCTS iterations
        iteration = 0
        best_cost = float('inf')

        while iteration < self.num_rollouts:
            elapsed = time.time() - start_time
            if elapsed >= time_limit_sec:
                print(f"MCTS-PIBT: Time limit reached ({elapsed:.2f}s)")
                break

            # Perform MCTS search
            self._mcts_search(root)
            iteration += 1

            # Periodically extract and evaluate solution
            if iteration % 50 == 0:
                current_solution = self._extract_solution_from_tree(root, max_timestep)
                current_cost = len(current_solution)

                if current_cost < best_cost:
                    best_cost = current_cost
                    print(
                        f"MCTS-PIBT: Iteration {iteration}, "
                        f"found solution with {current_cost} steps "
                        f"({elapsed:.2f}s, {root.visits} tree visits)"
                    )

        # Extract final solution
        final_solution = self._extract_solution_from_tree(root, max_timestep)

        print(
            f"MCTS-PIBT: Completed {iteration} rollouts, "
            f"final cost = {len(final_solution)} steps, "
            f"tree size = {root.visits} nodes"
        )

        return final_solution
