#!/bin/python3
"""

Module containing framework for Monte Carlo simulations 
and assignment tasks

MIT License

Copyright (c) 2025 Sean McGeoghegan

See LICEnSE.txt for details

"""

from numpy.random import SeedSequence, default_rng
import numpy as np
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

class MonteCarlo:
    """Monte Carlo simulations for integral, variance, and error calculations."""
    
    def __init__(self, start, end, num, f, *varis):
        self.start = np.array(start)  # Ensure start is an array
        self.end = np.array(end)      # Ensure end is an array
        self.f = f
        self.num = num
        self.vars = varis
        self.values = 0

    def __str__(self):
        return f"(integral: {self.values[0]}, Var: {self.values[1]}, Err: {self.values[2]})"

    def integral(self):
        """Calculate integral using Monte Carlo method."""

        d = len(self.start)  # Dimension check

        # Setup random generator
        ss = SeedSequence(23456)
        nworkers_seed = ss.spawn(nworkers)
        random_gen = [default_rng(s) for s in nworkers_seed]
        r_num = random_gen[rank].random((self.num, d))  # Generates (num, d)

        # Initialize function arrays with correct shape
        s_func = np.zeros(((self.num)+2, (self.num)+2), dtype=np.float64)  
        s_func_sq = np.zeros(((self.num)+2, (self.num)+2), dtype=np.float64)
        final_func = np.zeros_like(s_func)
        final_func_sq = np.zeros_like(s_func_sq)

        # Adjust calculation to process 2D arrays properly
        for n in r_num:
            n = n * (self.end - self.start) + self.start  # Vectorized scaling
            f_values = self.f(n)  # Ensure function returns 2D-compatible values
            s_func += f_values
            s_func_sq += f_values**2

        # MPI reduction to aggregate results
        comm.Allreduce(s_func, final_func)
        comm.Allreduce(s_func_sq, final_func_sq)

        # Integral, variance, and error calculations
        ad_cb = np.prod(self.end - self.start)
        inv_n = 1 / (self.num * nworkers)

        integral = np.mean(ad_cb * inv_n * final_func)  # Use mean for 2D support
        variance = inv_n * np.mean(final_func_sq * inv_n - (final_func * inv_n) ** 2)
        error = ad_cb * np.sqrt(variance)

        self.values = np.array([round(integral, 5), round(variance, 10), round(error, 5)])
        return self.values
