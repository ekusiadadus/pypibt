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

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize

        # Sort by distance + hindrance term (2025 optimization)
        C = sorted(
            C,
            key=lambda u: (
                self.dist_tables[i].get(u)
                + self.hindrance_weight * self.compute_hindrance(u, Q_from, i)
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
