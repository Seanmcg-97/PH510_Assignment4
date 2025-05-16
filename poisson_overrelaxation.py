#!/bin/python3
"""
Module containing set-up for calculations of given tasks.

MIT License

Copyright (c) 2025 Sean McGeoghegan

See LICENSE.txt for details

"""
"""
Created on Wed May 14 03:53:15 2025

@author: Sean McGeoghegan
"""

import numpy as np

def greens_function(x_initial, y_initial, source_x, source_y):
    """Evaluates the 2D Green’s function at a given (x_initial, y_initial) point."""
    r = np.sqrt((x_initial - source_x)**2 + (y_initial - source_y)**2)
    return -1 / (2 * np.pi) * np.log(r) if r > 0 else 0  # Avoid singularity

def poissonrelaxation(n, h, N, boundaries, charge_distribution, initial_positions):
    Lx, Ly = 10, 10  # Define square grid of side length 10 cm
    nx, ny = n, n
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx, dy = h, h
    xx, yy = np.meshgrid(x, y)

    p = np.zeros((ny, nx))
    S = np.zeros((ny, nx))

    # Apply different charge distributions
    if charge_distribution == "uniform":
        S[:, :] = 10 / (nx * ny)  # 10 C spread uniformly

    elif charge_distribution == "gradient":
        charge_gradient = np.linspace(1, 0, ny)[:, np.newaxis]  # Top-to-bottom gradient
        S[:, :] = charge_gradient  

    elif charge_distribution == "exponential":
        center_x, center_y = Lx / 2, Ly / 2
        r = np.sqrt(((xx - center_x) * h)**2 + ((yy - center_y) * h)**2)
        S[:, :] = np.exp(-2000 * np.abs(r))  # Exponential decay centered in grid

    w = 2 / (1 + np.sin(np.pi / N))
    tol = 1e-3
    error = 1e10
    iteration = 0
    max_iteration = N

    while error > tol and iteration < max_iteration:
        p_k = p.copy()

        # Apply boundary conditions
        p[-1, :] = boundaries[0]  # Top wall
        p[0, :] = boundaries[1]   # Bottom wall
        p[:, 0] = boundaries[2]   # Left wall
        p[:, -1] = boundaries[3]  # Right wall

        # Relaxation process
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                p[j, i] = (
                    w * 1 / (2 * (dx**2 + dy**2)) * (
                        (dx**2) * (dy**2) * S[j, i] +
                        (dy**2) * (p[j, i+1] + p[j, i-1]) +
                        (dx**2) * (p[j+1, i] + p[j-1, i])
                    ) + (1 - w) * p_k[j, i]
                )

        diff = p - p_k
        error = np.linalg.norm(diff, 2)
        iteration += 1

    if iteration == max_iteration:
        print("Solution did not converge:", iteration, "Iterations")
        print("Error =", error)
    else:
        print("Solution has converged in:", iteration, "Iterations")

    # Extract numerical potential at query points
    extracted_potentials = []
    greens_potentials = []
    errors = []

    for in_x, in_y in initial_positions:
        idx_x = np.argmin(np.abs(x - in_x))  # Find nearest grid index for x
        idx_y = np.argmin(np.abs(y - in_y))  # Find nearest grid index for y
        
        numerical_potential = p[idx_y, idx_x]
        greens_potential = greens_function(in_x, in_y, Lx/2, Ly/2)  # Assume source at center
        error_value = np.abs(numerical_potential - greens_potential)

        extracted_potentials.append((in_x, in_y, f"{numerical_potential:.4g}"))
        greens_potentials.append((in_x, in_y, f"{greens_potential:.4g}"))
        errors.append((in_x, in_y,f"{error_value:.4g}"))

    return xx, yy, p, extracted_potentials, greens_potentials, errors

initial_positions = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]

boundaries_1 = np.array([1, 1, 1, 1])
boundaries_2 = np.array([1, 1, -1, -1])
boundaries_3 = np.array([2, 0, 2, -4])

xx_1a, yy_1a, potential_1a, extracted_potentials_1a, greens_potentials_1a, errors_1a = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[1, 1, 1, 1], charge_distribution="uniform", initial_positions=initial_positions
)

xx_1b, yy_1b, potential_1b, extracted_potentials_1b, greens_potentials_1b, errors_1b = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[1, 1, -1, -1], charge_distribution="uniform", initial_positions=initial_positions
)

xx_1c, yy_1c, potential_1c, extracted_potentials_1c, greens_potentials_1c, errors_1c = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[2, 0, 2, -4], charge_distribution="uniform", initial_positions=initial_positions
)

print("Numerical Potentials =", extracted_potentials_1a, "for all boundaries at 1V in uniform")
print("Green’s Function Potentials =", greens_potentials_1a, "for all boundaries at 1V in uniform")
print("Errors =", errors_1a, "for all boundaries at 1V in uniform")

print("Numerical Potentials:", extracted_potentials_1b, "for T/B = 1V, L/R = -1V in uniform")
print("Green’s Function Potentials:", greens_potentials_1b, "for T/B = 1V, L/R = -1V in uniform")
print("Errors:", errors_1b, "for T/B = 1V, L/R = -1V in uniform")

print("Numerical Potentials:", extracted_potentials_1c, "for T = 2V, B = 0V L = 2V & R = -4V in uniform")
print("Green’s Function Potentials:", greens_potentials_1c, "for T = 2V, B = 0V L = 2V & R = -4V in uniform")
print("Errors:", errors_1c, "for T = 2V, B = 0V L = 2V & R = -4V in uniform")

xx_2a, yy_2a, potential_2a, extracted_potentials_2a, greens_potentials_2a, errors_2a = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[1, 1, 1, 1], charge_distribution="gradient", initial_positions=initial_positions
)

xx_2b, yy_2b, potential_2b, extracted_potentials_2b, greens_potentials_2b, errors_2b = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[1, 1, -1, -1], charge_distribution="gradient", initial_positions=initial_positions
)

xx_2c, yy_2c, potential_2c, extracted_potentials_2c, greens_potentials_2c, errors_2c = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[2, 0, 2, -4], charge_distribution="gradient", initial_positions=initial_positions
)

print("Numerical Potentials:", extracted_potentials_2a, "for all boundaries at 1V in charge grad.")
print("Green’s Function Potentials:", greens_potentials_2a, "for all boundaries at 1V in charge grad.")
print("Errors:", errors_2a, "for all boundaries at 1V in charge grad.")

print("Numerical Potentials:", extracted_potentials_2b "for T/B = 1V, L/R = -1V in charge grad.")
print("Green’s Function Potentials:", greens_potentials_2b "for T/B = 1V, L/R = -1V in charge grad.")
print("Errors:", errors_2b "for T/B = 1V, L/R = -1V in charge grad.")

print("Numerical Potentials:", extracted_potentials_2c, "for T = 2V, B = 0V L = 2V & R = -4V in charge grad.")
print("Green’s Function Potentials:", greens_potentials_2c, "for T = 2V, B = 0V L = 2V & R = -4V in charge grad.")
print("Errors:", errors_2c, "for T = 2V, B = 0V L = 2V & R = -4V in charge grad.")

xx_3a, yy_3a, potential_3a, extracted_potentials_3a, greens_potentials_3a, errors_3a = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[1, 1, 1, 1], charge_distribution="exponential", initial_positions=initial_positions
)

xx_3b, yy_3b, potential_3b, extracted_potentials_3b, greens_potentials_3b, errors_3b = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[1, 1, -1, -1], charge_distribution="exponential", initial_positions=initial_positions
)

xx_3c, yy_3c, potential_3c, extracted_potentials_3c, greens_potentials_3c, errors_3c = poissonrelaxation(
    n=100, h=0.1, N=5000, boundaries=[2, 0, 2, -4], charge_distribution="exponential", initial_positions=initial_positions
)

print("Numerical Potentials:", extracted_potentials_3a, "for all boundaries at 1V in exponential grid")
print("Green’s Function Potentials:", greens_potentials_3a, "for all boundaries at 1V in exponential grid")
print("Errors:", errors_3a, "for all boundaries at 1V in exponential grid")

print("Numerical Potentials:", extracted_potentials_3b "for T/B = 1V, L/R = -1V in exponential grid")
print("Green’s Function Potentials:", greens_potentials_3b "for T/B = 1V, L/R = -1V in exponential grid")
print("Errors:", errors_3b "for T/B = 1V, L/R = -1V in exponential grid")

print("Numerical Potentials:", extracted_potentials_3c, "for T = 2V, B = 0V L = 2V & R = -4V in exponential grid")
print("Green’s Function Potentials:", greens_potentials_3c, "for T = 2V, B = 0V L = 2V & R = -4V in exponential grid")
print("Errors:", errors_3c, "for T = 2V, B = 0V L = 2V & R = -4V in exponential grid")
