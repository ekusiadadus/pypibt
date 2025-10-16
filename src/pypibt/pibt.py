import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Configs, Coord, Grid, get_neighbors


class PIBT:
    def __init__(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        seed: int = 0,
        enable_hindrance: bool = True,
        priority_increment: float = 1.0,
        hindrance_weight: float = 0.3,
        enable_regret_learning: bool = False,
        regret_learning_iterations: int = 3,
        regret_weight: float = 0.2,
    ):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)

        # distance table
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

        # optimization parameters
        self.enable_hindrance = enable_hindrance
        self.priority_increment = priority_increment
        self.hindrance_weight = hindrance_weight

        # regret learning parameters (2025 optimization)
        self.enable_regret_learning = enable_regret_learning
        self.regret_learning_iterations = regret_learning_iterations
        self.regret_weight = regret_weight
        # regret table: stores learned regret values for position-action pairs
        self.regret_table: dict[tuple[Coord, Coord], float] = {}

    def compute_hindrance(self, v: Coord, Q_from: Config, current_agent: int) -> float:
        """
        Compute hindrance term: evaluates if moving to v hinders neighboring agents.
        Based on 2025 research: "Lightweight and Effective Preference Construction in PIBT"

        Returns lower value (better) if v does NOT block neighboring agents' goals.
        Time complexity: O(Δ) where Δ is max degree.
        """
        if not self.enable_hindrance:
            return 0.0

        hindrance_score = 0.0

        # Check all neighbors of the candidate position v
        neighbors_of_v = get_neighbors(self.grid, v)

        for neighbor_pos in neighbors_of_v:
            # Find which agent (if any) is at this neighbor position
            j = self.occupied_now[neighbor_pos]
            if j == self.NIL or j == current_agent:
                continue

            # Check if v is on the path towards agent j's goal
            # If distance from v to j's goal < distance from neighbor_pos to j's goal,
            # then v is closer to j's goal, meaning we're blocking j
            dist_v_to_j_goal = self.dist_tables[j].get(v)
            dist_neighbor_to_j_goal = self.dist_tables[j].get(neighbor_pos)

            if dist_v_to_j_goal < dist_neighbor_to_j_goal:
                # We are moving towards j's goal, hindering agent j
                hindrance_score += 1.0

        return hindrance_score

    def compute_regret(self, from_pos: Coord, to_pos: Coord) -> float:
        """
        Compute regret value for moving from from_pos to to_pos.
        Based on 2025 research: "Lightweight and Effective Preference Construction in PIBT"

        Regret learning learns how each action affects other agents' cost gaps.
        Returns learned regret value (lower is better).
        """
        if not self.enable_regret_learning:
            return 0.0

        # Look up learned regret value
        key = (from_pos, to_pos)
        return self.regret_table.get(key, 0.0)

    def update_regret_table(self, configs: Configs) -> None:
        """
        Update regret table based on observed agent trajectories.
        Analyzes conflicts and suboptimal moves to build regret values.
        """
        if not self.enable_regret_learning or len(configs) < 2:
            return

        # Analyze each transition
        for t in range(len(configs) - 1):
            Q_from = configs[t]
            Q_to = configs[t + 1]

            for i in range(self.N):
                pos_from = Q_from[i]
                pos_to = Q_to[i]

                # Calculate regret: how much this move cost other agents
                regret = 0.0

                # Check if this agent stayed in place unnecessarily
                if pos_from == pos_to and pos_from != self.goals[i]:
                    # Agent didn't move - check if it was blocking others
                    neighbors = get_neighbors(self.grid, pos_from)
                    for neighbor_pos in neighbors:
                        # Check if any agent at neighbor wanted to move here
                        for j in range(self.N):
                            if i == j:
                                continue
                            if Q_from[j] == neighbor_pos:
                                # Agent j was at neighbor position
                                # Check if moving to pos_from would have been better for j
                                dist_current = self.dist_tables[j].get(Q_to[j])
                                dist_alternative = self.dist_tables[j].get(pos_from)
                                if dist_alternative < dist_current:
                                    # Agent j would have benefited from this position
                                    regret += (dist_current - dist_alternative) * 0.5

                # Update regret table
                key = (pos_from, pos_to)
                old_regret = self.regret_table.get(key, 0.0)
                # Exponential moving average
                self.regret_table[key] = 0.7 * old_regret + 0.3 * regret

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize

        # Sort by distance + hindrance term + regret (2025 optimization)
        C = sorted(
            C,
            key=lambda u: (
                self.dist_tables[i].get(u)
                + self.hindrance_weight * self.compute_hindrance(u, Q_from, i)
                + self.regret_weight * self.compute_regret(Q_from[i], u)
            ),
        )

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(self, Q_from: Config, priorities: list[float]) -> Config:
        # setup
        N = len(Q_from)
        Q_to: Config = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to

    def run(self, max_timestep: int = 1000) -> Configs:
        """
        Run PIBT algorithm with optional regret learning.

        If regret learning is enabled, runs PIBT multiple times to learn
        from previous executions and improve solution quality.
        """
        if self.enable_regret_learning:
            return self._run_with_regret_learning(max_timestep)
        else:
            return self._run_single(max_timestep)

    def _run_single(self, max_timestep: int) -> Configs:
        """Single PIBT execution without regret learning."""
        # define priorities
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        # main loop, generate sequence of configurations
        configs = [self.starts]
        while len(configs) <= max_timestep:
            # obtain new configuration
            Q = self.step(configs[-1], priorities)
            configs.append(Q)

            # update priorities & goal check (with configurable increment)
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += self.priority_increment
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break  # goal

        return configs

    def _run_with_regret_learning(self, max_timestep: int) -> Configs:
        """
        Run PIBT with regret learning.

        Executes PIBT multiple times, learning from each execution to improve
        subsequent runs. Based on 2025 research showing 40% throughput improvement.
        """
        best_configs = None
        best_timesteps = float("inf")

        for iteration in range(self.regret_learning_iterations):
            # Run single execution
            configs = self._run_single(max_timestep)

            # Update regret table based on this execution
            self.update_regret_table(configs)

            # Track best solution
            timesteps = len(configs)
            if timesteps < best_timesteps:
                best_timesteps = timesteps
                best_configs = configs

        return best_configs if best_configs is not None else self._run_single(max_timestep)
