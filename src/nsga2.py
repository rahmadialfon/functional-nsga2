# Functional NSGA-II implementation using Viennet Benchmark functions
# Designed for two or three objectives, not advised to be extended
# For MOO using more than three objectives, use the NSGA-III algorithm instead
# Author: Alfonsus Rahmadi Putranto

# Import libraries

import numpy as np
import pandas as pd
import random as rnd 
from random import choices
import matplotlib.pyplot as plt
import warnings
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Define subfunctions for the sorting process

# Define rank calculator function
def rank_calc(df, obj_head, cases) :
    # Create a copy of the dataframe
    df_copy = df.copy()
    # Check the length of obj_head function
    obj_num = len(obj_head)
    # Create a list to store the ranks of the individuals
    rank_list = []
    # Get the number of rowns in the DataFrame
    row_num = df_copy.shape[0]
    # Get the column numbers of the objectives
    if obj_num == 2 :
        cols = [df_copy.columns.get_loc(obj_head[0]), df_copy.columns.get_loc(obj_head[1])]
    elif obj_num == 3 :
        cols = [df_copy.columns.get_loc(obj_head[0]), df_copy.columns.get_loc(obj_head[1]), df_copy.columns.get_loc(obj_head[2])]
    # Apply sorting according to the number of objectives
    if obj_num < 1 or obj_num > 3 :
        warnings.warn("The number of objectives is not supported by this function")
    if obj_num == 2 :
        # Start iterating through the rows of the DataFrame
        for i in range(row_num) :
            # Initialize the rank of the individual
            rank = 0
            for j in range(row_num) :
                # Get the condition for the first objective
                # For maximization
                if cases[0] :
                    cond1 = df_copy.iloc[i, cols[0]] >= df_copy.iloc[j, cols[0]]
                # For minimization
                if not cases[0] :
                    cond1 = df_copy.iloc[i, cols[0]] <= df_copy.iloc[j, cols[0]]
                # Get the condition for the second objective
                # For maximization
                if cases[1] :
                    cond2 = df_copy.iloc[i, cols[1]] >= df_copy.iloc[j, cols[1]]
                # For minimization
                if not cases[1] :
                    cond2 = df_copy.iloc[i, cols[1]] <= df_copy.iloc[j, cols[1]]
                # Get the rank based on the conditions
                if not cond1 and not cond2 :
                    rank += 1
                else : 
                    continue
            rank_list.append(rank)
    elif obj_num == 3 :
        # Start iterating through the rows of the DataFrame
        for i in range(row_num) :
            # Initialize the rank of the individual
            rank = 0
            for j in range(row_num) :
                # Get the condition for the first objective
                # For maximization
                if cases[0] :
                    cond1 = df_copy.iloc[i, cols[0]] >= df_copy.iloc[j, cols[0]]
                # For minimization
                if not cases[0] :
                    cond1 = df_copy.iloc[i, cols[0]] <= df_copy.iloc[j, cols[0]]
                # Get the condition for the second objective
                # For maximization
                if cases[1] :
                    cond2 = df_copy.iloc[i, cols[1]] >= df_copy.iloc[j, cols[1]]
                # For minimization
                if not cases[1] :
                    cond2 = df_copy.iloc[i, cols[1]] <= df_copy.iloc[j, cols[1]]
                # Get the condition for the third objective
                # For maximization
                if cases[2] :
                    cond3 = df_copy.iloc[i, cols[2]] >= df_copy.iloc[j, cols[2]]
                # For minimization
                if not cases[2] :
                    cond3 = df_copy.iloc[i, cols[2]] <= df_copy.iloc[j, cols[2]]
                # Get the rank based on the conditions
                if not cond1 and not cond2 and not cond3 :
                    rank += 1
                else : 
                    continue
            rank_list.append(rank)
    # Concatenate the rank to the initial DataFrame
    df_copy['Rank'] = rank_list
    # Sort the DataFrame by the rank
    df_copy.sort_values(by = 'Rank', inplace = True)
    # Return the new DataFrame
    return df_copy

# Rank splitter function
def rank_split(df):
    df_class = df.groupby('Rank')
    return [df_class.get_group(i) for i in df_class.groups]

# Crowd distance calculator function
def crdist_calc(df, obj_head, cases):
    # Get the copy of the DataFrame
    df_copy = df.copy()
    # Split the DataFrame according to the Rank value
    df_classes = rank_split(df_copy)
    # Get the objective number
    obj_num = len(obj_head)
    # Iterate thorugh the classes
    for df_class in df_classes :
        # Sort the class by the first objective, refresh the index
        df_o1 = df_class.sort_values(by = obj_head[0], ignore_index = True, ascending = not cases[0])
        df_o1.index = range(df_o1.shape[0])
        # Initialize the crowding distance list
        crdist_list = [0]*df_o1.shape[0]
        # Check the maximum and minimum values of the first objective
        # If the maximum and minimum values are the same, assign zero to the crowding distance
        if df_o1[obj_head[0]].max() == df_o1[obj_head[0]].min() :
            crdist_list = [0]*df_o1.shape[0]
        else :
            for i in range(df_o1.shape[0]) :
                # Check the first and last individuals
                if i == 0 or i == df_o1.shape[0] - 1 :
                    crdist_list[i] = 90000 #Representing infinity
                else :
                    # Calculate the crowding distance
                    crdist_list[i] = crdist_list[i] + (df_o1.loc[i+1, obj_head[0]] - df_o1.loc[i-1, obj_head[0]])/(df_o1[obj_head[0]].max() - df_o1[obj_head[0]].min())
        # Assign the crowding distance to the DataFrame
        df_o1['Crowding Distance'] = crdist_list
        # Sort the class by the second objective, refresh the index
        df_o2 = df_o1.sort_values(by = obj_head[1], ignore_index = True, ascending = not cases[1])
        df_o2.index = range(df_o2.shape[0])
        # Check the maximum and minimum values of the second objective
        # If the maximum and minimum values are the same, add zero to the crowding distance
        if df_o2[obj_head[1]].max() == df_o2[obj_head[1]].min() :
            df_o2['Crowding Distance'] = df_o2['Crowding Distance'] + 0
        else :
            for i in range(df_o2.shape[0]) :
                # Check the first and last individuals
                if i == 0 or i == df_o2.shape[0] - 1 :
                    df_o2.loc[i, ('Crowding Distance')] += 90000 #Representing infinity
                else :
                    # Calculate the crowding distance
                    df_o2.loc[i, ('Crowding Distance')] += (df_o2.loc[i+1, obj_head[1]] - df_o2.loc[i-1, obj_head[1]])/(df_o2[obj_head[1]].max() - df_o2[obj_head[1]].min())
        # Check if the number of objectives is three
        if obj_num == 3 :
            # Sort the class by the third objective, refresh the index
            df_o3 = df_o2.sort_values(by = obj_head[2], ignore_index = True, ascending = not cases[2])
            df_o3.index = range(df_o3.shape[0])
            # Check the maximum and minimum values of the third objective
            # If the maximum and minimum values are the same, add zero to the crowding distance
            if df_o3[obj_head[2]].max() == df_o3[obj_head[2]].min() :
                df_o3['Crowding Distance'] = df_o3['Crowding Distance'] + 0
            else :
                for i in range(df_o3.shape[0]) :
                    # Check the first and last individuals
                    if i == 0 or i == df_o3.shape[0] - 1 :
                        df_o3.loc[i, ('Crowding Distance')] += 90000
                    else :
                        # Calculate the crowding distance
                        df_o3.loc[i, ('Crowding Distance')] += (df_o3.loc[i+1, obj_head[2]] - df_o3.loc[i-1, obj_head[2]])/(df_o3[obj_head[2]].max() - df_o3[obj_head[2]].min())
            # Sort the class by the crowding distance, descending
            df_o3 = df_o3.sort_values(by = 'Crowding Distance', ignore_index = True, ascending = False)   
        else :
            df_o3 = df_o2.sort_values(by = 'Crowding Distance', ignore_index = True, ascending = False)
        # Concatenate the classes
        if df_class is df_classes[0] :
            df_final = df_o3
            # Refresh the index
            df_final.index = range(df_final.shape[0])
        else :
            df_final = pd.concat([df_final, df_o3])
            # Refresh the index
            df_final.index = range(df_final.shape[0])
    # Return the final DataFrame
    return df_final

# Define the crossover function
# Simple crossover function, using random shuffling
def simp_crossover(df_parent, inp_head):
    df_copy = df_parent.copy()
    df_child = df_copy[inp_head].sample(frac=1).reset_index(drop=True)
    return df_child

# Advanced crossover function, using Single Poinnt and Simulated Binary Crossover
# Single Point Crossover function
def sp_crossover(df_parent, inp_head):
    # Create an empty DataFrame to store offspring
    df_offspring = pd.DataFrame(columns=inp_head)

    # Randomly select pairs of parents for crossover
    parent_indices = np.random.choice(df_parent.index, size=(df_parent.shape[0] // 2, 2), replace=False)

    # Perform crossover for each pair of parents
    for parent1_idx, parent2_idx in parent_indices:
        parent1 = df_parent.loc[parent1_idx, inp_head]
        parent2 = df_parent.loc[parent2_idx, inp_head]

        # Randomly select a crossover point
        crossover_point = np.random.randint(1, len(inp_head))

        # Create offspring by combining parents' genes
        offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        # Add offspring to DataFrame
        if df_offspring.empty:
            df_offspring = pd.DataFrame([offspring1], columns=inp_head)
        else :
            df_offspring = pd.concat([df_offspring, pd.DataFrame([offspring1], columns=inp_head)], ignore_index=True)
        df_offspring = pd.concat([df_offspring, pd.DataFrame([offspring2], columns=inp_head)], ignore_index=True)

    return df_offspring

# SBX crossover function
def sbx_crossover_single(parent1, parent2, eta=2):
    child1 = parent1
    child2 = parent2
    if np.abs(parent1 - parent2) > 1e-10:
        if parent1 < parent2:
            beta = 1 + (2 * (parent1 - parent1.min())) / (parent2 - parent1)
        else:
            beta = 1 + (2 * (parent2 - parent2.min())) / (parent1 - parent2)
        alpha = 2 - beta**(-(eta + 1))
        u = np.random.rand()
        if u <= 1 / alpha:
            beta_q = (u * alpha)**(1 / (eta + 1))
        else:
            beta_q = (1 / (2 - u * alpha))**(1 / (eta + 1))
        child1 = 0.5 * ((parent1 + parent2) - beta_q * (parent2 - parent1))
        child2 = 0.5 * ((parent1 + parent2) + beta_q * (parent2 - parent1))
    return child1, child2

def sbx_crossover(df_parent, inp_head, eta=2, prob_crossover=0.9):
    df_copy = df_parent.copy()
    for head in inp_head:
        if np.random.rand() <= prob_crossover:
            parent1 = df_copy.sample()[head].iloc[0]
            parent2 = df_copy.sample()[head].iloc[0]
            child1, child2 = sbx_crossover_single(parent1, parent2, eta)
            df_copy.at[0, head] = child1
            df_copy.at[1, head] = child2
    return df_copy

# Define the random mutation function
def mut_rnd(df_parent, inp_head, inp_domain):
    df_copy = df_parent.copy()
    for head in inp_head:
        df_copy[head] += np.random.uniform(-1, 1, size=len(df_copy)) * (inp_domain[inp_head.index(head)][1] - inp_domain[inp_head.index(head)][0]) * 0.1
        df_copy[head] = np.clip(df_copy[head], inp_domain[inp_head.index(head)][0], inp_domain[inp_head.index(head)][1])
    return df_copy

# Define the natural mutation function
def mut_nat(df_parent, inp_head):
    df_copy = df_parent.copy()
    for head in inp_head:
        std = df_copy[head].std()
        df_copy[head] += np.random.normal(scale=std, size=len(df_copy))
    return df_copy

# Functional implementation of the NSGA-II algorithm
# Define optimization function

def opt_par(population, offs_pct):
    pop_num = population
    off_num = int(population * offs_pct)
    nat_mut_num = (pop_num - off_num) // 2
    rnd_mut_num = pop_num - off_num - nat_mut_num
    return pop_num, off_num, nat_mut_num, rnd_mut_num

# Define functions for generation

# Important note: The input generator and solver functions must be defined before the generation functions

# Pattern for inp_gen :
# def input_function(num, *other_inputs):
#    ...
#   return df

# Pattern for solver :
# def solver_function(df, inp_head, obj_head, *other_inputs):
#    ...
#   return df

def gen_0(opt_par, inp_gen, inp_inputs, inp_head, solver, solver_inputs, obj_head, obj_cases):
    pop_num, off_num, nat_mut_num, rnd_mut_num = opt_par
    # Generate initial population using input generator
    df = inp_gen(pop_num, *inp_inputs)
    # Solve using provided solver function
    df = solver(df, inp_head, obj_head, *solver_inputs)
    # Calculate rank and crowding distance
    df = rank_calc(df, obj_head, obj_cases)
    df = crdist_calc(df, obj_head, obj_cases)
    return df

def gen_n(df, opt_par, inp_head, inp_domain, solver, solver_inputs, obj_head, obj_cases):
    # Get the number of offspring, natural mutation, and random mutation
    off_num, nat_mut_num, rnd_mut_num = opt_par[1], opt_par[2], opt_par[3]
    # Get the top n of the parent dataframe, with n = off_num
    df_parent = df.head(off_num)
    drop_col = obj_head + ['Crowding Distance', 'Rank']
    # Drop the objective, crowding distance, and rank columns of the parent DataFrame
    df_parent = df_parent.drop(columns=drop_col)
    # Get the offspring DataFrame
    # Choose the crossover function, uncomment the desired one
    df_off = sp_crossover(df_parent, inp_head)
    # df_off = sbx_crossover(df_parent, inp_head)
    # df_off = simp_crossover(df_parent, inp_head)

    # Get the top n of the parent dataframe, with n = nat_mut_num
    df_nat = df_parent.head(nat_mut_num)
    # Get the natural mutation DataFrame
    df_nat = mut_nat(df_nat, inp_head)
    # Get the top n of the parent dataframe, with n = rnd_mut_num
    df_rnd = df_parent.head(rnd_mut_num)
    # Get the random mutation DataFrame
    df_rnd = mut_rnd(df_rnd, inp_head, inp_domain)
    # Concatenate the three DataFrames
    df_new = pd.concat([df_off, df_nat, df_rnd])
    # Save df_new to a excel file
    df_new.to_excel('df_new_inp.xlsx')
    # Solve the problem using provided solver function
    df_new = solver(df_new, inp_head, obj_head, *solver_inputs)
    # Save df_new to a excel file
    df_new.to_excel('df_new.xlsx')
    # Concatenate top n parent with n = population - off_num - nat_mut_num - rnd_mut_num offspring and the new one
    df_final = pd.concat([df_parent.tail(df.shape[0] - off_num - nat_mut_num - rnd_mut_num), df_new])
    # Refresh the index
    df_final.index = range(df_final.shape[0])
    df_final.to_excel('df_final.xlsx')
    # Calculate rank and crowding distance
    df_final = rank_calc(df_final, obj_head, obj_cases)
    df_final = crdist_calc(df_final, obj_head, obj_cases)
    
    # Return the final DataFrame
    return df_final

# Add memory function to the algorithm
# Define archiver function that append all populations into the archive file of .xlsx format
import os
import pandas as pd

def archiver(df, archive):
    # Check if the archive file exists
    if os.path.exists(archive):
        # Append the new population to the archive file
        with pd.ExcelWriter(archive, mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, index=False, header=True)
    else:
        # Create a new archive file
        df.to_excel(archive, index=False)

def gen_n_mem(df, opt_par, inp_head, inp_domain, solver, solver_inputs, obj_head, obj_cases, archive):
    # Get the number of offspring, natural mutation, and random mutation
    off_num, nat_mut_num, rnd_mut_num = opt_par[1], opt_par[2], opt_par[3]
    
    # Get the top n of the parent dataframe, with n = off_num
    df_parent = df.head(off_num)
    drop_col = obj_head + ['Crowding Distance', 'Rank']
    df_parent = df_parent.drop(columns=drop_col)
    
    # Generate offspring through crossover
    df_off = sp_crossover(df_parent, inp_head)
    
    # Generate mutations
    df_nat = mut_nat(df_parent.head(nat_mut_num), inp_head)
    df_rnd = mut_rnd(df_parent.head(rnd_mut_num), inp_head, inp_domain)
    
    # Combine all new solutions
    df_new = pd.concat([df_off, df_nat, df_rnd], ignore_index=True)
    
    # Handle memory/archive functionality
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
        # If no archive exists, solve everything
        df_new = solver(df_new, inp_head, obj_head, *solver_inputs)
    
    # Combine with remaining parents
    df_final = pd.concat([
        df_parent.tail(df.shape[0] - off_num - nat_mut_num - rnd_mut_num), 
        df_new
    ], ignore_index=True)
    
    # Calculate rank and crowding distance
    df_final = rank_calc(df_final, obj_head, obj_cases)
    df_final = crdist_calc(df_final, obj_head, obj_cases)
    
    # Archive solutions
    df_add_archive = df_final.drop(columns=['Rank', 'Crowding Distance'])
    archiver(df_add_archive, archive)
    
    return df_final

# Define the graphing function
def plot_pareto_front(df, obj_head):
    ranks = df['Rank'].unique()
    good_ranks = [r for r in ranks if r < 20]
    
    if len(obj_head) == 2:
        plt.figure(figsize=(8, 6))
        for i in good_ranks:
            pareto_front = df[df['Rank'] == i]
            plt.scatter(pareto_front[obj_head[0]], pareto_front[obj_head[1]], label=f'Rank {i}')
        plt.xlabel(obj_head[0])
        plt.ylabel(obj_head[1])
        plt.title('Pareto Front')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif len(obj_head) == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i in good_ranks:
            pareto_front = df[df['Rank'] == i]
            ax.scatter(pareto_front[obj_head[0]], pareto_front[obj_head[1]], pareto_front[obj_head[2]], label=f'Rank {i}')
        ax.set_xlabel(obj_head[0])
        ax.set_ylabel(obj_head[1])
        ax.set_zlabel(obj_head[2])
        ax.set_title('Pareto Front')
        ax.legend()
        plt.show()
    else:
        print("Plotting for more than 3 objectives is not supported.")

# Define nsga_ii function without memory
# Need to add savepoint for the following functions to keep track of the individuals in each generation
def nsga_ii(inp_gen, inp_inputs, inp_head, inp_domain, solver, solver_inputs, obj_head, obj_cases, opt_par, gen_num, graph_bool):
    # Generate the initial population
    df = gen_0(opt_par, inp_gen, inp_inputs, inp_head, solver, solver_inputs, obj_head, obj_cases)
    if graph_bool:
        plot_pareto_front(df, obj_head)
    # Iterate through the generations
    for i in range(gen_num):
        # Generate the new population
        df = gen_n(df, opt_par, inp_head, inp_domain, solver, solver_inputs, obj_head, obj_cases)
        # Plot the Pareto front
        if graph_bool:
            plot_pareto_front(df, obj_head)
        # Print the generation number
        print(f"Generation {i+1} completed.")
    # Return the final DataFrame
    return df

# Define nsga_ii function with memory
def nsga_ii_m(inp_gen, inp_inputs, inp_head, inp_domain, solver, solver_inputs, obj_head, obj_cases, opt_par, gen_num, archive, graph_bool):
    # Generate the initial population
    df = gen_0(opt_par, inp_gen, inp_inputs, inp_head, solver, solver_inputs, obj_head, obj_cases)
    if graph_bool:
        plot_pareto_front(df, obj_head)
    # Archive the initial population
    archiver(df, archive)
    # Iterate through the generations
    for i in range(gen_num):
        # Generate the new population
        df = gen_n_mem(df, opt_par, inp_head, inp_domain, solver, solver_inputs, obj_head, obj_cases, archive)
        # Plot the Pareto front
        if graph_bool:
            plot_pareto_front(df, obj_head)
        # Print the generation number
        print(f"Generation {i+1} completed.")
    # Return the final DataFrame
    return df
