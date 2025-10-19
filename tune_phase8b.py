"""
Optuna hyperparameter tuning for Phase 8B (Adaptive Beam Search PIBT).

Optimizes:
- beam_width: 3-15
- diversity_weight: 0.0-1.0
- min_beam_width: 2-5
- max_beam_width: 8-20
"""

import optuna
from optuna.samplers import TPESampler

from pypibt import get_grid, get_scenario
from pypibt.adaptive_beam_pibt import AdaptiveBeamPIBT


def objective(trial):
    """Optuna objective function."""
    # Hyperparameters to optimize
    beam_width = trial.suggest_int("beam_width", 3, 15)
    diversity_weight = trial.suggest_float("diversity_weight", 0.0, 1.0)
    min_beam_width = trial.suggest_int("min_beam_width", 2, 5)
    max_beam_width = trial.suggest_int("max_beam_width", 8, 20)

    # Ensure max >= beam >= min
    if max_beam_width < beam_width:
        max_beam_width = beam_width + 2
    if min_beam_width > beam_width:
        min_beam_width = max(2, beam_width - 1)

    # Test on medium instance (100 agents)
    grid = get_grid("assets/random-32-32-10.map")
    starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 100)

    try:
        beam_pibt = AdaptiveBeamPIBT(
            grid,
            starts,
            goals,
            seed=trial.number,  # Different seed per trial
            beam_width=beam_width,
            time_limit_ms=3000.0,  # 3 seconds
            diversity_weight=diversity_weight,
            adaptive_beam=True,
            min_beam_width=min_beam_width,
            max_beam_width=max_beam_width,
        )

        configs = beam_pibt.run(max_timestep=2000)

        # Objective: minimize timesteps
        timesteps = len(configs)

        # Check if solution is valid
        if configs[-1] != goals:
            # Penalize invalid solutions
            return 10000

        return timesteps

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 10000


def main():
    """Run Optuna study."""
    print("=" * 80)
    print("Phase 8B (Adaptive Beam Search) Hyperparameter Tuning")
    print("=" * 80)
    print()
    print("Optimizing on: Medium instance (100 agents)")
    print("Number of trials: 30")
    print("Timeout per trial: ~3-5 seconds")
    print("Total estimated time: ~2-3 minutes")
    print()

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="phase8b_optimization",
    )

    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print()
    print("=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print()
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best timesteps: {study.best_value}")
    print()
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print()

    # Save results
    import json

    results = {
        "best_trial": study.best_trial.number,
        "best_timesteps": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
            }
            for trial in study.trials
        ],
    }

    with open("phase8b_optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to: phase8b_optuna_results.json")


if __name__ == "__main__":
    main()
