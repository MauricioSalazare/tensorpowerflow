import pandas as pd
import warnings

"""
Computes the simulation times per algorithm power_flows/second.
It also creates a table that bounds the time sweep experiments by time. e.g., if you want to wait 3 min. per algorithm,
this script will automatically computes how many power flows per algorithm can fit in those 3 minutes.

WARNING!
The sweep_experiments.py should be run first, using the EXPERIMENT 0, in order to create the table found in the file
data\saved_solutions_STATS_GRID_SWEEP_100PF.csv, which is used in this script.
"""

def compute_time_per_iteration(tables_tests_concat: pd.DataFrame):
    # Compute the time per iteration with the FULL time of the algorithm i.e., time_pre_pf + time_pf
    # The dataframe must have a simulation with 100 time steps for all grid sizes.

    tables_tests_concat["pf_per_iter"] = tables_tests_concat["time_algorithm"] / tables_tests_concat["time_steps"]
    idx = tables_tests_concat["time_steps"] == 1000
    table_time_per_iter = tables_tests_concat.loc[idx].pivot(index="index", columns="grid_size", values="pf_per_iter")

    return table_time_per_iter

def compute_worst_time_algorithm(table_time_per_iter: pd.DataFrame, minutes: int = 23):
    # minutes = 23  # Maximum I am willing to wait for the slowest algorithm.
    max_iterations = (minutes * 60) / table_time_per_iter
    print(f"Worst alg. can make {max_iterations.min(axis=0).min():.0f} iterations in {minutes} min.")


def compute_bound_table_for_experiment_tests(table_time_per_iter: pd.DataFrame,
                                             minutes: float = 3,
                                             save_results=True):
    """
    For the time step sweep. Bound each algorithm based on the table
    minutes: int: Time in minutes that i am willing to wait for a test (combination grid size/time steps)
    """

    max_iterations = (minutes * 60) / table_time_per_iter

    if save_results:
        max_iterations.to_csv("data\iterations_bound_per_algorithm.csv")
        warnings.warn("File saved!!")
        # max_iterations_recovered = pd.read_csv("data\iterations_bound_per_algorithm.csv", index_col=[0])

    return max_iterations


if __name__ == "__main__":
    # tables_tests_concat = pd.read_csv("data\saved_solutions_STATS_GRID_SWEEP_100PF.csv")
    tables_tests_concat = pd.read_csv("data\merged_data.csv")
    print(tables_tests_concat.sort_values(by=["grid_size", "time_pf"]))

    table_time_per_iteration = compute_time_per_iteration(tables_tests_concat)

    # Experiment 1: Grid sweep from 10 to 5000 nodes and run 1 and 1000 time steps:
    total_iterations = 1000
    experiment_hours = (table_time_per_iteration * (total_iterations + 1)).sum().sum() / (60 * 60)
    print(f"For the following grid sizes:\n{table_time_per_iteration.columns.values}")
    print(f"For the following algorithms:\n{table_time_per_iteration.index.to_list()}")
    print(f"Experiment 1 time: {experiment_hours:.2f} hours. for {total_iterations} iter. for all algorithms.")


    # Experiment 2: Parameter fitting. Grid sweep from 10 to 5000 nodes, from  1 to X time steps bounded by time.
    # minutes => How much I am willing to wait PER ALGORITHM combination? (grid_size, time_steps)
    # Fixed grid size, time step sweep
    max_waiting = 3
    max_iterations_bounded = compute_bound_table_for_experiment_tests(table_time_per_iteration,
                                                                      minutes=max_waiting,
                                                                      save_results=True)
    algo_n = max_iterations_bounded.shape[0]
    grid_n = max_iterations_bounded.shape[1]

    print(f"Maximum number of iterations with a bound of {max_waiting} min. per algorithm")
    print(f"Maximum waiting time for the experiment {algo_n * grid_n * max_waiting} min.")

    # Experiment 3: Grid fixed to 500 nodes and run from 1 to 100_000 time steps (or bounded by time, whatever comes first)
    # i.e., backward and forward can not do 100_0000 time steps.
    max_waiting = 3
    max_iterations_bounded_fixed_grid = compute_bound_table_for_experiment_tests(table_time_per_iteration,
                                                                                 minutes=max_waiting,
                                                                                 save_results=True)
    print(f"Maximum iterations per algorithm, grid size of 500 waiting time of {max_waiting} min. per algorithm.")
    print(max_iterations_bounded_fixed_grid[1000])
    print(f"Experiment 3 time: {max_waiting * max_iterations_bounded_fixed_grid.shape[0]} min.")

    #%%
    hourly = 365 * 24
    halvely = hourly * 2
    quarterly = halvely * 2
    minutely = quarterly * 15

    grid_size = 100
    table100 = pd.concat([table_time_per_iteration[grid_size] * hourly,
                    table_time_per_iteration[grid_size] * halvely,
                    table_time_per_iteration[grid_size] * quarterly,
                    table_time_per_iteration[grid_size] * minutely], axis=1) / (60)
    table100.columns = [f"1_hr@{hourly}", f"30_min@{halvely}", f"15_min@{quarterly}", f"1_min@{minutely}"]

    table100.index.name = "algorithm"
    table100["grid"] = 100
    table100 = table100[[table100.columns[-1]] + table100.columns[:-1].tolist()]
    table100 = table100.loc[["bfsw", "nr", "hp-tensor", "sam", "gpu-tensor", "tensor"]]
    print(f"Grid size: {grid_size}")
    print(table100.round(2))
    print("\n")

    grid_size = 5000
    table5000 = pd.concat([table_time_per_iteration[grid_size]*hourly,
                    table_time_per_iteration[grid_size] *  halvely,
                    table_time_per_iteration[grid_size] * quarterly,
                    table_time_per_iteration[grid_size] * minutely], axis=1) / (60)
    table5000.columns = [f"1_hr@{hourly}", f"30_min@{halvely}", f"15_min@{quarterly}", f"1_min@{minutely}"]
    table5000.index.name = "algorithm"
    table5000["grid"] = 5000
    table5000 = table5000[[table5000.columns[-1]] + table5000.columns[:-1].tolist()]
    # table5000 = table5000.loc[["bfsw", "sam", "tensor", "nr", "hp-tensor"]]
    table5000 = table5000.loc[["bfsw", "nr", "hp-tensor", "sam", "gpu-tensor", "tensor"]]
    print(f"Grid size: {grid_size}")
    print(table5000.round(2))

    table_results = pd.concat([table100, table5000], axis=0).round(2)
    table_results.to_excel(r"data\table_results.xlsx")

    # Max iterations table:
    # | grid_set | x |  ts_set | x minutes = maximum time whole experiment
    # e.g. 10 x 10 x 1.5 min = 150 min = 2.5 hrs
