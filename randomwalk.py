#!/bin/python3
"""
Module containing set-up for calculations of given tasks.

MIT License

Copyright (c) 2025 Sean McGeoghegan

See LICENSE.txt for details

"""
"""
Created on Wed May 14 03:53:15 2025

@author: seanm
"""

import numpy as np
from mpi4py import MPI
from montecarlo import MonteCarlo

# MPI Initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

seed = 52648 # Random seed generation

n = 101 # Grid needs to be odd to have a center grid point
h = 10e-1/((n-1)/2) # As step size is 0.1cm

grid = np.zeros([n+2, n+2]) # To form grid, +2 points must be added to account for boundaries

def grid_boundary(n, i_zero, i_n, j_zero, j_n):
    """
    This function generates the grid boundaries at which the walkers must stop at
    """
    full_grid = np.zeros([n, n])
    
    row_zero = np.repeat(i_zero, n)
    row_n = np.repeat(i_n, n)
    
    column_zero = np.repeat(j_zero, n+2)
    column_n = np.repeat(j_n, n+2)
    
    rows = np.vstack((row_zero, full_grid, row_n))
    grid_boundary = np.column_stack((column_zero, rows, column_n))
    return grid_boundary

def randomwalkgen(i, j):
    """Generates a random walk from (i, j) without grid being passed explicitly."""
    
    # Define grid inside the function
    global grid  

    n_walkers = int(100) # Number of walkers in grid
    walk_grid = np.zeros_like(grid)
    position = np.array([i, j], dtype=int)
    path = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # Corrected shape
    max_i, max_j = grid.shape
    steps = 0

    while 0 <= position[0] < max_i and 0 <= position[1] < max_j and steps < 500:
        walk_grid[position[0], position[1]] += 1
        walk = path[np.random.randint(0, len(path))]
        position += walk
        steps += 1
    
    return walk_grid / n_walkers

# start_i/j represent the starting points for each walker along the x & y axis (i & j)
start_i = np.array([int((n-1)/2), int((n-1)/4), int((n-1)/100), int((n-1)/100)])
start_j = np.array([int((n-1)/2), int((n-1)/4), int((n-1)/4), int((n-1)/100)])

variables = np.array([start_i, start_j], dtype=object)
boundary = np.ones_like(grid)
boundary[1:-1, 1:-1] = 0

# Task 4.1a set up for a charge of 10/n^2, while every boundary has a voltage of 1V
grid_boundary_1 = grid_boundary(n, 1, 1, 1, 1)
grid_1 = np.zeros_like(grid)
grid_1[1:-1, 1:-1] = 10/((n-2)**2)


grid_boundary_2 = grid_boundary(n, 1, 1, -1, -1)
def apply_charge_gradient(grid, h):
    """
    This function sets up a charge gradient which is drawn vertically along the grid
    """
    grid_2 = np.zeros_like(grid)
    i, j = grid.shape
    charge_gradient = np.linspace(1, 0, i-2)[:, np.newaxis]
    
    grid_2[1:-1, 1:-1] = charge_gradient * (h**2)
    return grid_2
grid_2 = apply_charge_gradient(grid_boundary_2, h)

grid_boundary_3 = grid_boundary(n, 2, 0, 2, -4)

def exponential_grid(grid, h):
    """
    This function generates an exponential decaying charge distribution at the center of the grid
    """
    grid_3 = np.zeros_like(grid)
    rows, cols = grid.shape
    
    center_x, center_y = (rows-1)/2, (cols-1)/2
    x, y = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    
    r = np.sqrt(((x - center_x) * h)**2 + ((y - center_y) * h)**2)
    grid_3[1:-1, 1:-1] = np.exp(-2000 * r[1:-1, 1:-1])
    return grid_3
grid_3 = exponential_grid(grid_boundary_3, h)

for idx in range(len(start_i)):  # Iterate through valid indices
    i, j = start_i[idx], start_j[idx]
    

    variables = np.array([i, j])
    montecarlo = MonteCarlo([0, 0], [n, n], n, lambda i_j: randomwalkgen(i_j[0],i_j[1]), *variables)
    mc = MonteCarlo.integral(montecarlo)
    
    if rank == 0:
        print(f"When i = {(10 * i) / (n - 1)}cm and j = {(10 * j) / (n - 1)}cm")
        print(mc[0]*boundary)
        print(f"Error: {np.mean(mc[2])}:4f")
        print(f"Potential for boundary conditions [1,1,1,1]V = {np.sum(mc[0]*grid_boundary_1):4f}")
        print(f"Potential for boundary conditions [1,1,-1,-1]V  = {np.sum(mc[0]*grid_boundary_2):4f}")
        print(f"Potential for boundary conditions [2,0,2,-4]V  = {np.sum(mc[0]*grid_boundary_3):4f}")
        print()
        print(f"Potential for uniform grid w/ boundary conditions [1,1,1,1]V = {np.sum(mc[0]*grid_boundary_1) + np.sum(h**2 * mc[0] * grid_1):4f}")
        print(f"Potential for uniform grid w/ boundary conditions [1,1,-1,-1]V = {np.sum(mc[0]*grid_boundary_2) + np.sum(h**2 * mc[0] * grid_1):4f}")
        print(f"Potential for uniform grid w/ boundary conditions [2,0,2,-4]V = {np.sum(mc[0]*grid_boundary_3) + np.sum(h**2 * mc[0] * grid_1):4f}")
        print()
        print(f"Potential for charge gradient w/ boundary conditions [1,1,1,1]V = {np.sum(mc[0]*grid_boundary_1) + np.sum(h**2 * mc[0] * grid_2):4f}")
        print(f"Potential for charge gradient w/ boundary conditions [1,1,-1,-1]V = {np.sum(mc[0]*grid_boundary_2) + np.sum(h**2 * mc[0] * grid_2):4f}")
        print(f"Potential for charge gradient w/ boundary conditions [2,0,2,-4]V = {np.sum(mc[0]*grid_boundary_3) + np.sum(h**2 * mc[0] * grid_2):4f}")
        print()
        print(f"Potential for exponential grid w/ boundary conditions [1,1,1,1]V = {np.sum(mc[0]*grid_boundary_1) + np.sum(h**2 * mc[0] * grid_3):4f}")
        print(f"Potential for exponential grid w/ boundary conditions [1,1,-1,-1]V = {np.sum(mc[0]*grid_boundary_2) + np.sum(h**2 * mc[0] * grid_3):4f}")
        print(f"Potential for exponential grid w/ boundary conditions [2,0,2,-4]V = {np.sum(mc[0]*grid_boundary_3) + np.sum(h**2 * mc[0] * grid_3):4f}")

MPI.Finalize()
