"""
Optuna-based hyperparameter optimization for PIBT.
Based on 2025 research and optimization best practices.
"""

import optuna
from pypibt import PIBT, get_grid, get_scenario, is_valid_mapf_solution


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    Returns the number of timesteps (lower is better).
    """
    # Hyperparameters to optimize
    enable_hindrance = trial.suggest_categorical("enable_hindrance", [True, False])
    hindrance_weight = (
        trial.suggest_float("hindrance_weight", 0.0, 2.0, log=False)
        if enable_hindrance
        else 0.0
    )
    priority_increment = trial.suggest_float("priority_increment", 0.5, 2.0)

    # Load test scenario
    grid = get_grid("assets/random-32-32-10.map")
    starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 200)

    # Run PIBT with suggested parameters
    try:
        pibt = PIBT(
            grid,
            starts,
            goals,
            seed=0,
            enable_hindrance=enable_hindrance,
            hindrance_weight=hindrance_weight,
            priority_increment=priority_increment,
        )
        plan = pibt.run(max_timestep=1000)

        # Verify solution validity
        if not is_valid_mapf_solution(grid, starts, goals, plan):
            return float("inf")  # Invalid solution

        timesteps = len(plan)
        return float(timesteps)

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float("inf")


def main():
    print("=" * 60)
    print("PIBT Hyperparameter Optimization with Optuna")
    print("=" * 60)
    print()

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="pibt_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run optimization
    print("Starting optimization (this may take a few minutes)...")
    print()

    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print()
    print("=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print()

    print(f"Number of finished trials: {len(study.trials)}")
    print()

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (timesteps): {trial.value}")
    print()

    print("  Best parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print()

    # Run baseline for comparison
    print("Baseline (default parameters):")
    grid = get_grid("assets/random-32-32-10.map")
    starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 200)
    pibt_baseline = PIBT(
        grid, starts, goals, seed=0, enable_hindrance=False, priority_increment=1.0
    )
    plan_baseline = pibt_baseline.run(max_timestep=1000)
    baseline_timesteps = len(plan_baseline)
    print(f"  Timesteps: {baseline_timesteps}")
    print()

    improvement = ((baseline_timesteps - trial.value) / baseline_timesteps) * 100
    print(f"Improvement: {improvement:.2f}%")
    print()

    # Save best parameters to file
    with open("best_params.txt", "w") as f:
        f.write("# Best PIBT hyperparameters found by Optuna\n")
        f.write(f"# Baseline: {baseline_timesteps} timesteps\n")
        f.write(f"# Optimized: {trial.value} timesteps\n")
        f.write(f"# Improvement: {improvement:.2f}%\n\n")
        for key, value in trial.params.items():
            f.write(f"{key} = {value}\n")

    print("Best parameters saved to best_params.txt")


if __name__ == "__main__":
    main()
