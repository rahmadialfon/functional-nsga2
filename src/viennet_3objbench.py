import numpy as np
import pandas as pd
import math

# Define input generator of the Viennet Benchmark functions
def inp_gen_vie(pop_num, inp_head, inp_domain):
    df = pd.DataFrame(columns=inp_head)
    for index, head in enumerate(inp_head):
        domain_min, domain_max = inp_domain[index]viennet_3objbench.py
        df[head] = np.random.uniform(domain_min, domain_max, pop_num)
    return df

# Define the Vienna Benchmark function solver
def viennet(x, y):
    sq_sum = x**2 + y**2
    f1 = 0.5 * (sq_sum) + math.sin(sq_sum)
    f2 = 0.125 * (3 * x - 2 * y + 4)**2 + 0.0370370370370 * (x - y + 1)**2 + 15
    f3 = 1 / (sq_sum + 1) - 1.1 * math.exp(-sq_sum)
    return f1, f2, f3

def solver_vie(df, inp_head, obj_head):
    inputs = df[inp_head].values
    results = np.apply_along_axis(lambda row: viennet(*row), 1, inputs)
    df[obj_head] = pd.DataFrame(results, columns=obj_head)
    return df
