Looking at your code, I can identify the issue in the memory implementation. The problem lies in the `gen_n_mem` function's handling of solved and unsolved solutions. Here's the fix:

## Memory Implementation Fix

```python
def gen_n_mem(df, opt_par, inp_head, inp_domain, solver, solver_inputs, obj_head, obj_cases, archive):
    # Previous code remains the same until df_new creation
    
    # Modified archive handling
    try:
        df_archive = pd.read_excel(archive)
        
        # Create a merge key from input columns
        df_new['merge_key'] = df_new[inp_head].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        df_archive['merge_key'] = df_archive[inp_head].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        
        # Split into solved and unsolved
        df_new_solved = df_new[df_new['merge_key'].isin(df_archive['merge_key'])]
        df_new_unsolved = df_new[~df_new['merge_key'].isin(df_archive['merge_key'])]
        
        # Get full solved solutions from archive
        if not df_new_solved.empty:
            df_new_solved = df_new_solved.merge(
                df_archive.drop('merge_key', axis=1),
                on=inp_head,
                how='left'
            )
        
        # Solve unsolved solutions
        if not df_new_unsolved.empty:
            df_new_unsolved = solver(df_new_unsolved.drop('merge_key', axis=1), 
                                   inp_head, obj_head, *solver_inputs)
        
        # Combine solutions
        df_new = pd.concat([df_new_solved, df_new_unsolved], ignore_index=True)
        
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # If archive doesn't exist or is empty, solve everything
        df_new = solver(df_new, inp_head, obj_head, *solver_inputs)

    # Rest of the function remains the same
```

## README.md Update

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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

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
