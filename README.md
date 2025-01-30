# Functional NSGA-II Implementation

A Python implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II) for multi-objective optimization problems.

## Features

- Supports 2D and 3D objective optimization problems
- Multiple crossover methods:
  - Single Point Crossover
  - Simulated Binary Crossover (SBX)
- Natural and random mutation operators
- Memory functionality to avoid redundant calculations
- Visualization capabilities for Pareto fronts
- Includes benchmark test functions (Binh-Korn)

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- NumPy
- Pandas
- Matplotlib

## Usage

### Basic Example

```python
from nsga2 import NSGAII

# Define your objective functions
def objective1(x):
    return x[0]**2 + x[1]**2

def objective2(x):
    return (x[0]-5)**2 + (x[1]-5)**2

# Create optimizer
optimizer = NSGAII(
    population_size=100,
    num_generations=50,
    num_objectives=2,
    variables_range=[(0,5), (0,3)]
)

# Run optimization
results = optimizer.run()
```

### Visualization

```python
# Plot Pareto front
optimizer.plot_pareto_front()
```

## Project Structure

```
nsga-ii/
├── src/
│   ├── __init__.py
│   ├── nsga2.py           # Core implementation
│   ├── utils.py           # Helper functions
│   └── benchmarks.py      # Test functions
├── examples/
│   └── binh_korn.ipynb    # Example notebook
├── tests/
│   └── test_nsga2.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.
