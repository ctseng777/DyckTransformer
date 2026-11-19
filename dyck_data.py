#!/usr/bin/env python3
"""
Generate a data/dyck_data_{n}.pkl file
"""

import sys
import os
from src.data.dyck_generator import generate_dyck_data
import argparse

def main(n, target_map="zeta", workers=1):
    print(f"Generating Dyck data for n={n}, target map={target_map}...")
    prefix = "dyck" if target_map == "zeta" else target_map
    data_path = f"data/{prefix}_data_{n}.pkl"
    try:
        generate_dyck_data(n, data_path=data_path, force_regenerate=False, target_map=target_map, workers=workers)
        print(f"Generated: {data_path}")
    except Exception as e:
        print(f"Error generating n={n}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Dyck data files.")
    parser.add_argument("--n", type=int, help="The value of n for Dyck data generation.")
    parser.add_argument("--map", type=str, default="zeta",
                        help="Target bijection: 'zeta' (SageMath area_dinv_to_bounce_area_map, default) "
                             "'sieve' (recursive involution sieve, n <= 11), "
                             "or a pure-Python map from src/utils/dyck_maps.py such as 'kappa'.")
    parser.add_argument("--workers", type=int, default=1, help="Processes for the sieve computation (n >= 8).")
    args = parser.parse_args()
    main(args.n, args.map, args.workers)