# NSGA-II Implementation with Memory

A Python implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II) with memory functionality for multi-objective optimization problems.

## Features

- Efficient implementation of NSGA-II algorithm
- Memory functionality to avoid redundant calculations
- Support for both 2D and 3D objective optimization
- Multiple crossover methods:
  - Single Point Crossover (SP)
  - Simulated Binary Crossover (SBX)
- Natural and random mutation operators
- Built-in benchmark problems:
  - Binh-Korn
  - Viennet
- Real-time visualization of Pareto fronts

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
# Define parameters
population = 300
offspring_pct = 0.40
obj_head = ['f1', 'f2']
input_head = ['x', 'y']
inp_domain = [[0, 5], [0, 3]]
cases = [False, False]
generations = 4
archive = 'archive.xlsx'

# Run optimization
df_final = nsga_ii_m(inp_gen_biko, input_inputs, input_head, 
                     inp_domain, solver_biko, solver_inputs, 
                     obj_head, cases, opt_par_val, generations, 
                     archive, True)
```

### Examples in Jupyter

Implementation of NSGA-II using provided benchmark function. 

### Visualization
```python
plot_pareto_front(df_final, obj_head)
```

## Key Components

- `nsga_ii`: Standard NSGA-II implementation
- `nsga_ii_m`: NSGA-II with memory functionality
- `rank_calc`: Non-dominated sorting implementation
- `crdist_calc`: Crowding distance calculator
- `plot_pareto_front`: Visualization function

## License

This project is licensed under the MIT License.

## Citation

If you use this implementation in your research, please cite:
```
@misc{nsga2_implementation,
  author = {Alfonsus Rahmadi Putranto},
  title = {NSGA-II Implementation with Memory},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/username/functional-nsga2}
}
```
