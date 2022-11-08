import ast
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from distutils.version import LooseVersion

"""
Script to take a look on the results from the experiments.
"""

def read_data(file_name):
    test_solutions = pd.read_csv(file_name)

    if "Unnamed: 0" in test_solutions.columns:
        test_solutions.rename(columns={"Unnamed: 0": "index"}, inplace=True)

    columns_to_process = ['iterations_log', 'time_pre_pf_log', 'time_pf_log', 'convergence_log']
    for _process_column in columns_to_process:
        test_solutions[_process_column] = test_solutions[_process_column].apply(lambda x: ast.literal_eval(x))

    return test_solutions

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

if __name__ == "__main__":
    file_name_ts_sweep = r"data\test_grid_sweep_two_time_steps_1_1000_DO_NOT_DELETE.csv"
    test_solutions_ts_sweep = read_data(file_name=file_name_ts_sweep)
    print_full(test_solutions_ts_sweep[["index", "time_pf", "grid_size", "time_steps"]])