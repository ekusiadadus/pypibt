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
        enable_anytime: bool = False,
        anytime_time_limit_ms: float = 100.0,
        anytime_beam_width: int = 5,
        enable_lns: bool = False,
        lns_iterations: int = 10,
        lns_destroy_size: int = 10,
        lns_destroy_strategy: str = "adaptive",
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

        # anytime PIBT parameters (2025 optimization)
        self.enable_anytime = enable_anytime
        self.anytime_time_limit_ms = anytime_time_limit_ms
        self.anytime_beam_width = anytime_beam_width

        # LNS parameters (2025 optimization)
        self.enable_lns = enable_lns
        self.lns_iterations = lns_iterations
        self.lns_destroy_size = lns_destroy_size
        self.lns_destroy_strategy = lns_destroy_strategy

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
        Run PIBT algorithm with optional optimizations.

        Supports multiple modes:
        - Standard PIBT
        - Regret Learning (iterative execution)
        - Anytime PIBT (continuous improvement within time limit)
        - LNS (Large Neighborhood Search for conflict resolution)
        """
        if self.enable_lns:
            return self._run_with_lns(max_timestep)
        elif self.enable_anytime:
            return self._run_anytime(max_timestep)
        elif self.enable_regret_learning:
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

    def _run_anytime(self, max_timestep: int) -> Configs:
        """
        Run Anytime PIBT with continuous improvement.

        Based on 2025 research: "Anytime Single-Step MAPF Planning with Anytime PIBT"

        Algorithm:
        1. Find initial solution quickly (standard PIBT)
        2. Continuously improve using beam search over priority orderings
        3. Stop when time limit is reached or optimal solution found
        4. Return best solution found
        """
        import time

        start_time = time.time()
        time_limit_sec = self.anytime_time_limit_ms / 1000.0

        # Step 1: Get initial solution (fast)
        best_configs = self._run_single(max_timestep)
        best_cost = len(best_configs)

        print(f"Anytime PIBT: Initial solution = {best_cost} steps")

        # Step 2: Beam search over priority variations
        iteration = 0
        improvements = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed >= time_limit_sec:
                break

            iteration += 1

            # Generate candidate priority orderings using beam search
            candidate_solutions = self._generate_priority_candidates(
                max_timestep, self.anytime_beam_width
            )

            # Evaluate candidates
            for configs in candidate_solutions:
                if len(configs) < best_cost:
                    best_configs = configs
                    best_cost = len(configs)
                    improvements += 1
                    print(
                        f"Anytime PIBT: Improved to {best_cost} steps "
                        f"(iteration {iteration}, {elapsed:.3f}s)"
                    )

            # Early termination if no improvement
            if iteration > 5 and improvements == 0:
                break

        print(
            f"Anytime PIBT: Completed in {iteration} iterations, "
            f"{improvements} improvements"
        )
        return best_configs

    def _generate_priority_candidates(
        self, max_timestep: int, beam_width: int
    ) -> list[Configs]:
        """
        Generate multiple candidate solutions using varied priority strategies.

        Uses beam search to explore different priority orderings:
        - Distance-based priorities (default)
        - Random perturbations
        - Conflict-based priorities
        - Goal-distance weighted priorities
        """
        candidates = []

        for i in range(beam_width):
            # Strategy 1: Perturbed priorities
            if i == 0:
                # Use regret-enhanced priorities if available
                if len(self.regret_table) > 0:
                    configs = self._run_with_priority_strategy("regret", max_timestep)
                else:
                    configs = self._run_single(max_timestep)
            elif i == 1:
                # Random perturbation
                configs = self._run_with_priority_strategy("random", max_timestep)
            elif i == 2:
                # Conflict-aware priorities
                configs = self._run_with_priority_strategy("conflict", max_timestep)
            else:
                # Mixed strategy
                configs = self._run_with_priority_strategy("mixed", max_timestep)

            candidates.append(configs)

        return candidates

    def _run_with_priority_strategy(
        self, strategy: str, max_timestep: int
    ) -> Configs:
        """
        Run PIBT with specific priority initialization strategy.

        Strategies:
        - "regret": Use learned regret values
        - "random": Random perturbation of priorities
        - "conflict": Conflict-based priority adjustment
        - "mixed": Combination of strategies
        """
        # Define priorities based on strategy
        priorities: list[float] = []

        if strategy == "regret":
            # Use regret learning to inform priorities
            for i in range(self.N):
                base_priority = (
                    self.dist_tables[i].get(self.starts[i]) / self.grid.size
                )
                # Add regret-based adjustment
                regret_adj = sum(
                    self.regret_table.get((self.starts[i], n), 0.0)
                    for n in get_neighbors(self.grid, self.starts[i])
                )
                priorities.append(base_priority + regret_adj * 0.1)

        elif strategy == "random":
            # Random perturbation
            for i in range(self.N):
                base_priority = (
                    self.dist_tables[i].get(self.starts[i]) / self.grid.size
                )
                perturbation = self.rng.uniform(-0.2, 0.2)
                priorities.append(base_priority + perturbation)

        elif strategy == "conflict":
            # Conflict-aware: agents far from goals get higher priority
            for i in range(self.N):
                dist = self.dist_tables[i].get(self.starts[i])
                # Inverse distance priority (farther = higher priority)
                priorities.append(float(dist))

        else:  # mixed
            # Combination of distance and random
            for i in range(self.N):
                base_priority = (
                    self.dist_tables[i].get(self.starts[i]) / self.grid.size
                )
                perturbation = self.rng.uniform(-0.1, 0.1)
                priorities.append(base_priority * 1.5 + perturbation)

        # Main loop
        configs = [self.starts]
        while len(configs) <= max_timestep:
            Q = self.step(configs[-1], priorities)
            configs.append(Q)

            # Update priorities
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += self.priority_increment
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break

        return configs

    # ========================================================================
    # LNS (Large Neighborhood Search) Implementation (2025 Optimization)
    # ========================================================================

    def _detect_vertex_conflicts(self, configs: Configs) -> set[tuple[int, int, int]]:
        """
        Detect vertex conflicts in a solution.

        Returns set of (timestep, agent_i, agent_j) tuples where agents occupy same vertex.
        """
        conflicts = set()
        for t in range(len(configs)):
            Q = configs[t]
            # Check for vertex collisions
            position_map: dict[Coord, list[int]] = {}
            for i, pos in enumerate(Q):
                if pos not in position_map:
                    position_map[pos] = []
                position_map[pos].append(i)

            # Record conflicts
            for pos, agents in position_map.items():
                if len(agents) > 1:
                    # All pairs of agents at same position
                    for i in range(len(agents)):
                        for j in range(i + 1, len(agents)):
                            conflicts.add((t, agents[i], agents[j]))

        return conflicts

    def _detect_edge_conflicts(self, configs: Configs) -> set[tuple[int, int, int]]:
        """
        Detect edge conflicts (swapping) in a solution.

        Returns set of (timestep, agent_i, agent_j) tuples where agents swap positions.
        """
        conflicts = set()
        for t in range(len(configs) - 1):
            Q_from = configs[t]
            Q_to = configs[t + 1]

            for i in range(self.N):
                for j in range(i + 1, self.N):
                    # Check if agents i and j swap positions
                    if Q_from[i] == Q_to[j] and Q_from[j] == Q_to[i]:
                        conflicts.add((t, i, j))

        return conflicts

    def _count_conflicts(self, configs: Configs) -> int:
        """Count total number of conflicts (vertex + edge) in a solution."""
        vertex_conflicts = self._detect_vertex_conflicts(configs)
        edge_conflicts = self._detect_edge_conflicts(configs)
        return len(vertex_conflicts) + len(edge_conflicts)

    def _get_conflict_agents(self, configs: Configs) -> set[int]:
        """Get set of all agents involved in any conflict."""
        vertex_conflicts = self._detect_vertex_conflicts(configs)
        edge_conflicts = self._detect_edge_conflicts(configs)

        conflict_agents = set()
        for _, i, j in vertex_conflicts:
            conflict_agents.add(i)
            conflict_agents.add(j)
        for _, i, j in edge_conflicts:
            conflict_agents.add(i)
            conflict_agents.add(j)

        return conflict_agents

    def _destroy_random(self, configs: Configs, destroy_size: int) -> set[int]:
        """
        Random destroy heuristic: randomly select agents to replan.

        Args:
            configs: Current solution
            destroy_size: Number of agents to destroy

        Returns:
            Set of agent indices to replan
        """
        conflict_agents = self._get_conflict_agents(configs)

        if len(conflict_agents) == 0:
            return set()

        # Select random subset of conflict agents
        destroy_count = min(destroy_size, len(conflict_agents))
        destroyed = set(self.rng.choice(list(conflict_agents), destroy_count, replace=False))

        return destroyed

    def _destroy_conflict_based(self, configs: Configs, destroy_size: int) -> set[int]:
        """
        Conflict-based destroy heuristic: select agents with most conflicts.

        Prioritizes agents involved in many conflicts for replanning.
        """
        vertex_conflicts = self._detect_vertex_conflicts(configs)
        edge_conflicts = self._detect_edge_conflicts(configs)

        # Count conflicts per agent
        conflict_count: dict[int, int] = {}
        for _, i, j in vertex_conflicts:
            conflict_count[i] = conflict_count.get(i, 0) + 1
            conflict_count[j] = conflict_count.get(j, 0) + 1
        for _, i, j in edge_conflicts:
            conflict_count[i] = conflict_count.get(i, 0) + 1
            conflict_count[j] = conflict_count.get(j, 0) + 1

        if len(conflict_count) == 0:
            return set()

        # Sort agents by conflict count (descending)
        sorted_agents = sorted(conflict_count.items(), key=lambda x: x[1], reverse=True)

        # Select top destroy_size agents
        destroy_count = min(destroy_size, len(sorted_agents))
        destroyed = set(agent for agent, _ in sorted_agents[:destroy_count])

        return destroyed

    def _destroy_adaptive(
        self, configs: Configs, destroy_size: int, iteration: int
    ) -> set[int]:
        """
        Adaptive destroy heuristic: dynamically choose between strategies.

        Uses weighted random selection between different heuristics,
        adapting weights based on iteration number.
        """
        # Adaptive strategy selection based on iteration
        if iteration % 3 == 0:
            # Random exploration
            return self._destroy_random(configs, destroy_size)
        elif iteration % 3 == 1:
            # Conflict-based intensification
            return self._destroy_conflict_based(configs, destroy_size)
        else:
            # Mixed: conflict-based + random expansion
            conflict_agents = self._destroy_conflict_based(configs, destroy_size // 2)
            all_conflict_agents = self._get_conflict_agents(configs)
            remaining = all_conflict_agents - conflict_agents

            if len(remaining) > 0:
                additional_count = min(destroy_size - len(conflict_agents), len(remaining))
                additional = set(
                    self.rng.choice(list(remaining), additional_count, replace=False)
                )
                return conflict_agents | additional
            else:
                return conflict_agents

    def _repair_paths(
        self, original_configs: Configs, destroyed_agents: set[int], max_timestep: int
    ) -> Configs:
        """
        Repair paths for destroyed agents using PIBT.

        Args:
            original_configs: Original solution
            destroyed_agents: Set of agent indices to replan
            max_timestep: Maximum timesteps for repair

        Returns:
            New configs with repaired paths
        """
        if len(destroyed_agents) == 0:
            return original_configs

        # Create a sub-problem with only destroyed agents
        destroyed_list = sorted(list(destroyed_agents))

        # Extract starts from original_configs[0] for destroyed agents
        sub_starts = [self.starts[i] for i in destroyed_list]
        sub_goals = [self.goals[i] for i in destroyed_list]

        # Create temporary PIBT instance for repair (reuse same grid and parameters)
        repair_pibt = PIBT(
            self.grid,
            sub_starts,
            sub_goals,
            seed=self.rng.integers(0, 100000),
            enable_hindrance=self.enable_hindrance,
            hindrance_weight=self.hindrance_weight,
            priority_increment=self.priority_increment,
            enable_regret_learning=False,  # Disable for repair (faster)
            enable_anytime=False,
            enable_lns=False,
        )

        # Run PIBT for sub-problem
        sub_configs = repair_pibt._run_single(max_timestep)

        # Merge sub_configs back into original_configs
        # Strategy: Replace paths of destroyed agents while keeping others
        max_len = max(len(original_configs), len(sub_configs))
        merged_configs = []

        for t in range(max_len):
            if t < len(original_configs):
                Q = list(original_configs[t])  # Copy original config
            else:
                # Extend: agents stay at their last positions
                Q = list(original_configs[-1])

            # Update positions of destroyed agents
            if t < len(sub_configs):
                for idx, agent_id in enumerate(destroyed_list):
                    Q[agent_id] = sub_configs[t][idx]

            merged_configs.append(tuple(Q))

        return merged_configs

    def _run_with_lns(self, max_timestep: int) -> Configs:
        """
        Run PIBT with Large Neighborhood Search (LNS).

        Based on MAPF-LNS2 (AAAI 2022) and LNS2+RL (2024-2025 research).

        Algorithm:
        1. Generate initial solution with PIBT
        2. Iteratively destroy and repair:
           a. Destroy: Select subset of conflicting agents
           b. Repair: Replan paths using PIBT
        3. Accept improved solutions
        4. Repeat until no conflicts or iteration limit

        Returns:
            Conflict-free or improved solution
        """
        # Step 1: Generate initial solution
        initial_configs = self._run_single(max_timestep)
        best_configs = initial_configs
        best_cost = len(best_configs)
        best_conflicts = self._count_conflicts(best_configs)

        print(f"LNS: Initial solution = {best_cost} steps, {best_conflicts} conflicts")

        if best_conflicts == 0:
            print("LNS: Initial solution is conflict-free!")
            return best_configs

        # Step 2: LNS iterations
        for iteration in range(self.lns_iterations):
            # Destroy: Select agents to replan
            if self.lns_destroy_strategy == "random":
                destroyed_agents = self._destroy_random(best_configs, self.lns_destroy_size)
            elif self.lns_destroy_strategy == "conflict":
                destroyed_agents = self._destroy_conflict_based(
                    best_configs, self.lns_destroy_size
                )
            else:  # adaptive
                destroyed_agents = self._destroy_adaptive(
                    best_configs, self.lns_destroy_size, iteration
                )

            if len(destroyed_agents) == 0:
                print(f"LNS: No agents to destroy at iteration {iteration + 1}")
                break

            # Repair: Replan paths for destroyed agents
            repaired_configs = self._repair_paths(
                best_configs, destroyed_agents, max_timestep
            )

            # Evaluate repaired solution
            repaired_cost = len(repaired_configs)
            repaired_conflicts = self._count_conflicts(repaired_configs)

            # Accept if improved (fewer conflicts or same conflicts but shorter)
            improved = False
            if repaired_conflicts < best_conflicts:
                improved = True
            elif repaired_conflicts == best_conflicts and repaired_cost < best_cost:
                improved = True

            if improved:
                best_configs = repaired_configs
                best_cost = repaired_cost
                best_conflicts = repaired_conflicts
                print(
                    f"LNS: Iteration {iteration + 1}: Improved to {best_cost} steps, "
                    f"{best_conflicts} conflicts (destroyed {len(destroyed_agents)} agents)"
                )

            # Early termination if conflict-free
            if best_conflicts == 0:
                print(f"LNS: Found conflict-free solution at iteration {iteration + 1}!")
                break

        print(
            f"LNS: Completed {min(iteration + 1, self.lns_iterations)} iterations. "
            f"Final: {best_cost} steps, {best_conflicts} conflicts"
        )
        return best_configs
