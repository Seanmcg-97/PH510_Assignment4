# -*- coding: utf-8 -*-
"""
Created on Wed May 14 03:53:15 2025

@author: seanm
"""

import numpy as np
from mpi4py import MPI
from class_1 import MonteCarlo

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

n_walkers = int(1000)
seed = 52648

n = 101
h = 10e-1/(n-1)

grid = np.zeros([n+2, n+2])

def grid_boundary(n, top, bottom, left, right):
    full_grid = np.zeros([n, n])
    
    row_zero = np.repeat(top, n)
    row_n = np.repeat(bottom, n)
    
    column_zero = np.repeat(left, n+2)
    column_n = np.repeat(right, n+2)
    
    rows = np.vstack((row_zero, full_grid, row_n))
    grid_boundary = np.hstack((column_zero, rows, column_n))
    return grid_boundary

def randomwalkgen(i, j, grid):
    walk_grid = np.zeros_like(grid)
    position = np.array([i, j])
    path = np.array([1, 0], [-1, 0], [0, 1], [0, -1])
    max_i, max_j = grid.shape
    steps = 0
    
    while 0 <= position[0] <= max_i and 0 <= position[1] <= max_j:
        walk_grid[position[0], position[1]] += 1
        walk = path[np.random.randint(0, len(path))]
        position += walk
        steps += 1
    return walk_grid

start_i = np.array([int((n-1)/2), int((n-1)/4), int((n-1)/100), int((n-1)/100)])
start_j = np.array([int((n-1)/2), int((n-1)/4), int((n-1)/4), int((n-1)/100)])

variables = np.array([start_i, start_j, grid], dtype=object)
boundary = np.ones_like(grid)
boundary[1:-1, 1:-1] = 0

grid_boundary_1 = grid_boundary(n, 1, 1, 1, 1)
grid_1 = np.zeros_like(grid)
grid_1[1:-1, 1:-1] = 10/(n**2)

grid_boundary_2 = grid_boundary(n, 1, 1, -1, -1)

def apply_charge_gradient(grid, h):
    grid_2 = np.zeros_like(grid)
    i, j = grid.shape
    charge_gradient = np.linspace(1, 0, i-2)[:, np.newaxis]
    
    grid_2[1:-1, 1:-1] = charge_gradient * (h**2)
    return grid_2
grid_2 = apply_charge_gradient(grid_boundary_2, h)

grid_boundary_3 = grid_boundary(n, 2, 0, 2, -4)

def exponential_grid(grid, h):
    grid_3 = np.zeros_like(grid)
    rows, cols = grid.shape
    
    center_x, center_y = (rows-1)/2, (cols-1)/2
    x, y = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    
    r = np.sqrt(((x - center_x) * h)**2 + ((y - center_y) * h)**2)
    grid_3[1:-1, 1:-1] = np.exp(-2000 * r[1:-1, 1:-1])
    return grid_3
grid_3 = exponential_grid(grid_boundary_3, h)

for i, j in zip(start_i, start_j):
    variables = np.array([i, j, grid])
    
    montecarlo = MonteCarlo([0], [1], randomwalkgen, var=variables)
    mc = MonteCarlo.integral(montecarlo)
    
    if rank == 0:
        print(f"When i = {(10*start_i[i])/(n-1)}cm and j = {(10*start_j[j])/(n-1)}cm")
        print(result[0]*boundary)
        print(f"Error: {np.mean(mc[2])}")
        print(f"Task 4.1a: Result for all boundaries at 1V: Potential = {np.sum(mc[0]*grid_boundary_1):4f}")
        print(f"Task 4.1b: Result for T & B = 1V, L & R = -1V: Potential = {np.sum(mc[0]*grid_boundary_2):4f}")
        print(f"Task 4.1c: Result for T = 2V, B = 0V, L = 2V, R = -4V: Potential = {np.sum(mc[0]*grid_boundary_3):4f}")
        print()
        print(f"Task 4.1a: Uniform grid w/ boundaries: {np.sum(mc[0]*grid_boundary_1) + np.sum(h**2 * mc[0] * grid_1)}:4f")
        print(f"Task 4.1b: Uniform grid w/ boundaries: {np.sum(mc[0]*grid_boundary_2) + np.sum(h**2 * mc[0] * grid_1)}:4f")
        print(f"Task 4.1c: Uniform grid w/ boundaries: {np.sum(mc[0]*grid_boundary_3) + np.sum(h**2 * mc[0] * grid_1)}:4f")
        print()
        print(f"Task 4.1a: Charge gradient w/ boundaries: {np.sum(mc[0]*grid_boundary_1) + np.sum(h**2 * mc[0] * grid_2)}:4f")
        print(f"Task 4.1b: Charge gradient w/ boundaries: {np.sum(mc[0]*grid_boundary_2) + np.sum(h**2 * mc[0] * grid_2)}:4f")
        print(f"Task 4.1c: Charge gradient w/ boundaries: {np.sum(mc[0]*grid_boundary_3) + np.sum(h**2 * mc[0] * grid_2)}:4f")
        print()
        print(f"Task 4.1a: Exponential grid w/ boundaries: {np.sum(mc[0]*grid_boundary_1) + np.sum(h**2 * mc[0] * grid_3)}:4f")
        print(f"Task 4.1a: Exponential grid w/ boundaries: {np.sum(mc[0]*grid_boundary_2) + np.sum(h**2 * mc[0] * grid_3)}:4f")
        print(f"Task 4.1a: Exponential grid w/ boundaries: {np.sum(mc[0]*grid_boundary_3) + np.sum(h**2 * mc[0] * grid_3)}:4f")