"""Microbenchmark for apply_transfer_matrix_array vs NumPy.

Usage: run the script. It will import the built `pyegret` extension from
the repo build directory (ensure PYTHONPATH includes the repo root and build dir).

It compares:
- pyegret.apply_transfer_matrix_array(mats, vec)
- NumPy matmul for single vector: np.matmul(tmat.transpose(2,0,1), vec).T
- NumPy einsum for many vectors: np.einsum('ijk,jm->imk', tmat, vecs)

The script prints timings and speedups for representative N (slices) and M (vectors).
"""
from __future__ import annotations

import time
import numpy as np
import os, sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
build_dir = os.path.join(repo_root, 'build')
sys.path.insert(0, build_dir)
sys.path.insert(0, repo_root)

import pyegret


def time_fn(fn, repeats=5):
    # warmup
    for _ in range(2):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def bench_single_vector(tmat, vec, repeats=7):
    # helper
    def run_cpp():
        pyegret.apply_transfer_matrix_array(tmat, vec)

    def run_np():
        np.matmul(tmat.transpose(2, 0, 1), vec).T

    t_cpp = time_fn(run_cpp, repeats)
    t_np = time_fn(run_np, repeats)
    return t_cpp, t_np


def bench_many_vectors(tmat, vecs, repeats=7):
    # helper
    def run_cpp():
        pyegret.apply_transfer_matrix_array(tmat, vecs)

    def run_einsum():
        np.einsum('ijk,jm->imk', tmat, vecs)

    t_cpp = time_fn(run_cpp, repeats)
    t_einsum = time_fn(run_einsum, repeats)
    return t_cpp, t_einsum


def run_benchmarks():
    Ns = [100, 1000]
    Ms = [1, 16, 64]
    repeats = 7

    print('Benchmark apply_transfer_matrix_array vs NumPy')
    print('Repo root:', repo_root)
    print('Build dir:', build_dir)
    print()

    for N in Ns:
        print(f'--- N = {N} slices ---')
        # build random mats
        tmat = np.random.randn(4, 4, N).astype(np.float64)
        vec = np.random.randn(4).astype(np.float64)
        t_cpp, t_np = bench_single_vector(tmat, vec, repeats=repeats)
        print(f'Single vector:  pyegret median {t_cpp[0]:.6f}s, numpy matmul median {t_np[0]:.6f}s, speedup {t_np[0]/t_cpp[0]:.2f}x')

        for M in Ms:
            vecs = np.random.randn(4, M).astype(np.float64)
            t_cpp, t_einsum = bench_many_vectors(tmat, vecs, repeats=repeats)
            print(f'M={M:3d}: pyegret median {t_cpp[0]:.6f}s, numpy einsum median {t_einsum[0]:.6f}s, speedup {t_einsum[0]/t_cpp[0]:.2f}x')
        print()


if __name__ == '__main__':
    run_benchmarks()
