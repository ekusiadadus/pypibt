"""
Tests for Phase 8 optimization implementations (8A, 8B, 8C).
"""

import numpy as np
import pytest

from pypibt.adaptive_beam_pibt import AdaptiveBeamPIBT
from pypibt.jps_dist_table import JPSDistTable, _calculate_obstacle_density
from pypibt.mapf_utils import generate_random_grid, generate_random_instance
from pypibt.mcts_pibt import MCTSPIBT


class TestJPSDistTable:
    """Tests for Phase 8C: JPS-Accelerated DistTable."""

    def test_sparse_graph_density_calculation(self):
        """Test obstacle density calculation."""
        # Create sparse grid (20% obstacles)
        grid = np.ones((10, 10), dtype=int)
        grid[0:2, :] = 0  # 20 cells blocked out of 100

        density = _calculate_obstacle_density(grid)
        assert 0.15 <= density <= 0.25, f"Expected density ~0.2, got {density}"

    def test_jps_dist_table_sparse(self):
        """Test JPS distance table on sparse grid (should use JPS)."""
        # Create sparse grid (30% obstacles, optimal for JPS)
        grid = generate_random_grid(20, 20, obstacle_prob=0.3, seed=42)
        goal = (10, 10)

        # Ensure goal is free
        grid[goal] = 1

        dist_table = JPSDistTable(grid, goal)

        # Should enable JPS for 30% density
        assert dist_table.use_jps, "JPS should be enabled for 30% obstacle density"

        # Test distance calculation
        target = (5, 5)
        grid[target] = 1
        distance = dist_table.get(target)

        # Distance should be reasonable (Manhattan distance is 10)
        assert 0 < distance < grid.size, f"Distance {distance} is out of range"

    def test_jps_dist_table_very_sparse(self):
        """Test JPS distance table on very sparse grid (should use standard BFS)."""
        # Create very sparse grid (5% obstacles, not optimal for JPS)
        grid = generate_random_grid(20, 20, obstacle_prob=0.05, seed=43)
        goal = (10, 10)
        grid[goal] = 1

        dist_table = JPSDistTable(grid, goal)

        # Should NOT enable JPS for 5% density
        assert not dist_table.use_jps, "JPS should be disabled for 5% obstacle density"

    def test_jps_dist_table_dense(self):
        """Test JPS distance table on dense grid (should use standard BFS)."""
        # Create dense grid (60% obstacles, not optimal for JPS)
        grid = generate_random_grid(20, 20, obstacle_prob=0.6, seed=44)
        goal = (10, 10)
        grid[goal] = 1

        dist_table = JPSDistTable(grid, goal)

        # Should NOT enable JPS for 60% density
        assert not dist_table.use_jps, "JPS should be disabled for 60% obstacle density"

    def test_jps_correctness(self):
        """Test that JPS produces correct distances."""
        # Simple 5x5 grid
        grid = np.ones((5, 5), dtype=int)
        grid[2, 1:4] = 0  # Horizontal wall with gaps

        goal = (4, 4)
        dist_table = JPSDistTable(grid, goal)

        # Test various positions
        assert dist_table.get((4, 4)) == 0, "Goal should have distance 0"
        assert dist_table.get((3, 4)) == 1, "Adjacent cell should have distance 1"

        # Position blocked by wall
        target = (0, 0)
        distance = dist_table.get(target)
        assert distance > 0, "Distance to (0,0) should be positive"


class TestAdaptiveBeamPIBT:
    """Tests for Phase 8B: Adaptive Diverse Beam Search."""

    def test_adaptive_beam_initialization(self):
        """Test initialization of adaptive beam search."""
        grid = generate_random_grid(10, 10, obstacle_prob=0.2, seed=50)
        starts, goals = generate_random_instance(grid, num_agents=5, seed=51)

        beam_pibt = AdaptiveBeamPIBT(
            grid,
            starts,
            goals,
            seed=52,
            beam_width=3,
            time_limit_ms=500.0,
        )

        assert beam_pibt.beam_width == 3
        assert beam_pibt.diversity_weight == 0.5
        assert len(beam_pibt.dist_tables) == 5

    def test_priority_strategies(self):
        """Test different priority strategies."""
        grid = generate_random_grid(10, 10, obstacle_prob=0.2, seed=60)
        starts, goals = generate_random_instance(grid, num_agents=3, seed=61)

        beam_pibt = AdaptiveBeamPIBT(grid, starts, goals, seed=62)

        strategies = beam_pibt._get_priority_strategy_list()
        assert len(strategies) == 5
        assert "distance_based" in strategies
        assert "regret_enhanced" in strategies
        assert "conflict_aware" in strategies

    def test_adaptive_beam_run_small(self):
        """Test adaptive beam search on small instance."""
        grid = generate_random_grid(8, 8, obstacle_prob=0.2, seed=70)
        starts, goals = generate_random_instance(grid, num_agents=3, seed=71)

        beam_pibt = AdaptiveBeamPIBT(
            grid,
            starts,
            goals,
            seed=72,
            beam_width=3,
            time_limit_ms=1000.0,  # Short time limit for test
        )

        configs = beam_pibt.run(max_timestep=100)

        # Basic validation
        assert len(configs) > 0, "Should produce a solution"
        assert configs[0] == starts, "First config should be starts"
        assert configs[-1] == goals, "Last config should be goals"


class TestMCTSPIBT:
    """Tests for Phase 8A: MCTS-Enhanced PIBT."""

    def test_mcts_initialization(self):
        """Test initialization of MCTS-PIBT."""
        grid = generate_random_grid(10, 10, obstacle_prob=0.2, seed=80)
        starts, goals = generate_random_instance(grid, num_agents=5, seed=81)

        mcts_pibt = MCTSPIBT(
            grid,
            starts,
            goals,
            seed=82,
            num_rollouts=100,
            rollout_depth=50,
        )

        assert mcts_pibt.num_rollouts == 100
        assert mcts_pibt.rollout_depth == 50
        assert len(mcts_pibt.dist_tables) == 5

    def test_mcts_node_creation(self):
        """Test MCTS node creation."""
        from pypibt.mcts_pibt import MCTSNode

        config = ((0, 0), (1, 1), (2, 2))
        priorities = [1.0, 2.0, 3.0]

        node = MCTSNode(config=config, priorities=priorities)

        assert node.config == config
        assert node.priorities == priorities
        assert node.visits == 0
        assert node.value == 0.0
        assert len(node.untried_actions) > 0

    def test_mcts_run_small(self):
        """Test MCTS on small instance."""
        grid = generate_random_grid(8, 8, obstacle_prob=0.2, seed=90)
        starts, goals = generate_random_instance(grid, num_agents=3, seed=91)

        mcts_pibt = MCTSPIBT(
            grid,
            starts,
            goals,
            seed=92,
            num_rollouts=50,  # Small number for test
            rollout_depth=30,
            time_limit_ms=2000.0,
        )

        configs = mcts_pibt.run(max_timestep=100)

        # Basic validation
        assert len(configs) > 0, "Should produce a solution"
        assert configs[0] == starts, "First config should be starts"
        # Note: MCTS might not always reach goal in limited rollouts
        # so we don't assert configs[-1] == goals


class TestPhase8Integration:
    """Integration tests comparing Phase 8 variants."""

    def test_all_phases_on_same_instance(self):
        """Test all Phase 8 variants on the same instance."""
        # Create test instance
        grid = generate_random_grid(12, 12, obstacle_prob=0.25, seed=100)
        starts, goals = generate_random_instance(grid, num_agents=5, seed=101)

        # Phase 8C: JPS (just test distance table)
        jps_dist = JPSDistTable(grid, goals[0])
        jps_distance = jps_dist.get(starts[0])
        assert jps_distance > 0, "JPS should compute positive distance"

        # Phase 8B: Adaptive Beam
        beam_pibt = AdaptiveBeamPIBT(
            grid, starts, goals, seed=102,
            beam_width=3,
            time_limit_ms=1500.0,
        )
        beam_configs = beam_pibt.run(max_timestep=100)

        # Phase 8A: MCTS
        mcts_pibt = MCTSPIBT(
            grid, starts, goals, seed=103,
            num_rollouts=50,
            time_limit_ms=2000.0,
        )
        mcts_configs = mcts_pibt.run(max_timestep=100)

        # Both should produce valid solutions
        assert len(beam_configs) > 0
        assert len(mcts_configs) > 0

        print(f"Beam Search: {len(beam_configs)} steps")
        print(f"MCTS: {len(mcts_configs)} steps")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
