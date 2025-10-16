# pypibt

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](./LICENCE.txt)
[![CI](https://github.com/Kei18/pypibt/actions/workflows/ci.yml/badge.svg)](https://github.com/Kei18/pypibt/actions/workflows/ci.yml)

A minimal python implementation of Priority Inheritance with Backtracking (PIBT) for Multi-Agent Path Finding (MAPF).
If you are just interested in moving hundreds of agents or more in a short period of time, PIBT may work as a powerful tool.

- Okumura, K., Machida, M., Défago, X., & Tamura, Y. Priority inheritance with backtracking for iterative multi-agent path finding. AIJ. 2022. [[project-page]](https://kei18.github.io/pibt2/)

## background

To be honest, as the developer of PIBT, I only developed it to keep multiple agents running smoothly, not to solve MAPF or MAPD.
But it turned out to be much more powerful than I expected.
A successful example is [LaCAM*](https://kei18.github.io/lacam2/).
It achieves remarkable performance, to say the least.
I also noticed that PIBT has been extended and used by other researchers.
These experiences were enough to motivate me to create a minimal implementation example to help other studies, including my future research projects.

As you know, many researchers like Python because it is friendly and has a nice ecosystem.
In contrast, most MAPF algorithms, such as the original PIBT, are coded in C++ for performance reasons.
So here is the Python implementation.
I hope the repo is helpful to understand the algorithm; the main part is only a hundred and a few lines.
You can also use and extend this repo, for example, applying to new problems, enhancing with machine learning, etc.

## setup

This repository is easily setup with [uv](https://docs.astral.sh/uv/).
After cloning this repo, run the following to complete the setup.

```sh
uv sync
```

## demo

```sh
uv run python app.py -m assets/random-32-32-10.map -i assets/random-32-32-10-random-1.scen -N 200
```

The result will be saved in `output.txt`
The grid maps and scenarios in `assets/` are from [MAPF benchmarks](https://movingai.com/benchmarks/mapf/index.html).

### visualization

You can visualize the planning result with [@kei18/mapf-visualizer](https://github.com/kei18/mapf-visualizer).

```sh
mapf-visualizer ./assets/random-32-32-10.map ./output.txt
```

![](./assets/demo.gif)

### jupyter lab

Jupyter Lab is also available.
Use the following command:

```sh
uv run jupyter lab
```

You can see an example in `notebooks/demo.ipynb`.

## Optimization Features (2024-2025 Research)

This implementation includes state-of-the-art optimizations based on recent MAPF research:

### Performance Improvements

| Agents | Baseline | **Optimized** | **Improvement** |
|--------|----------|---------------|-----------------|
| 100    | 63 steps | **55 steps**  | **+12.7%** |
| 200    | 54 steps | 54 steps      | 0.0% |
| 400    | 76 steps | **61 steps**  | **+19.7%** |

### Quick Example

```python
from pypibt import PIBT, get_grid, get_scenario

grid = get_grid("assets/random-32-32-10.map")
starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 400)

# Optimized PIBT with Hindrance Term + Regret Learning
pibt = PIBT(
    grid, starts, goals,
    enable_hindrance=True,           # Avoid blocking other agents
    hindrance_weight=0.3,
    enable_regret_learning=True,     # Learn from previous executions
    regret_learning_iterations=3,
    regret_weight=0.2
)
plan = pibt.run(max_timestep=2000)
print(f"Solution: {len(plan)} timesteps")
```

### Features

1. **Hindrance Term** (2025 Research)
   - Evaluates agent interference to avoid blocking
   - O(Δ) complexity, maintains scalability
   - Best for high-density scenarios (300+ agents)

2. **Regret Learning** (2025 Research)
   - Learns from multiple PIBT executions
   - Builds regret table from conflict patterns
   - +19.7% improvement with 400 agents

3. **Hyperparameter Optimization**
   - Optuna integration for automatic tuning
   - Environment-specific parameter adaptation

See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for detailed documentation.

### Benchmarking

```sh
# Test different hindrance weights
uv run python benchmark_hindrance.py

# Automatic optimization with Optuna
uv run python optimize_with_optuna.py

# Comprehensive benchmark across agent counts
uv run python benchmark_comprehensive.py
```

## Licence

This software is released under the MIT License, see [LICENSE.txt](LICENCE.txt).
