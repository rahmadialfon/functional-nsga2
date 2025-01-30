import numpy as np
import pandas as pd
import math

# Define input generator of the Binh-Korn Benchmark Function
def inp_gen_biko(pop_num, inp_head, inp_domain) :
    # Create an empty DataFrame with the specified headers
    df = pd.DataFrame(columns=inp_head)
    # Populate the DataFrame with random values within the specified domain for each header
    for index, head in enumerate(inp_head):
        domain_min, domain_max = inp_domain[index]
        df[head] = np.random.uniform(domain_min, domain_max, pop_num)    
    return df

# Define Binh-Korn Benchmark function solver
def biko(x,y) :
    f1 = 4*x**2 + 4*y**2
    f2 = (x-5)**2 + (y-5)**2
    return f1, f2

def solver_biko(df, inp_head, obj_head)  :
    inputs = df[inp_head].values
    results = np.apply_along_axis(lambda row: biko(*row), 1, inputs)
    df[obj_head] = pd.DataFrame(results, columns=obj_head)
    return df
