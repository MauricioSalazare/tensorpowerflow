import sys
print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
current_directory = Path().absolute()
print(f"Current directory: {current_directory}")
sys.path.extend([str(current_directory.parent)])

import numpy as np
import pandas as pd
from utils_experiments import run_test, pandapower_pf, load_grid_from_file
import time
from time import perf_counter
import argparse


# Warning:
# LIMIT_ALGORITHM is a flag that reads a .csv file. The .csv file was crated from running an experiment of running
# all algorithms from grid sizes [  10   30  100  200  300  500  800 1000 3000 5000] for time_steps [1, 10, 100]

def runt_tests(N_POWER_FLOWS, NETWORK_SIZE, LIMIT_ALGORITHMS, FILE_NAME):
    start_whole_experiment = perf_counter()
    SAMPLING_SIZE = 10  # Max number of statistics per iteration want to be saved (UNUSED)

    # Configuration parameters for each one of the algorithms.
    test_config_SAM = dict(method_name= "sam",
                           is_single_step=True,
                           is_pandapower_function=False,
                           network=None,
                           numba_enable=False,
                           sampling_size=SAMPLING_SIZE)
    test_config_tensorpowerflow = dict(method_name= "tensor",
                                       is_single_step=False,
                                       is_pandapower_function=False,
                                       network=None,
                                       numba_enable=True,
                                       sampling_size=SAMPLING_SIZE)
    test_config_HP_tensor = dict(method_name="hp-tensor",
                                 is_single_step=False,
                                 is_pandapower_function=False,
                                 network=None,
                                 numba_enable=True,
                                 sampling_size=SAMPLING_SIZE)
    test_config_GPU_tensor = dict(method_name="gpu-tensor",
                                  is_single_step=False,
                                  is_pandapower_function=False,
                                  network=None,
                                  numba_enable=True,
                                  sampling_size=SAMPLING_SIZE)
    test_config_newton_raphson = dict(method_name= "nr",
                                      is_single_step=True,
                                      is_pandapower_function=True,
                                      network=None,
                                      numba_enable=True,
                                      sampling_size=SAMPLING_SIZE)
    test_config_backward_forward = dict(method_name= "bfsw",
                                        is_single_step=True,
                                        is_pandapower_function=True,
                                        network=None,
                                        numba_enable=True,
                                        sampling_size=SAMPLING_SIZE)

    if LIMIT_ALGORITHMS:
        table_with_limits = pd.read_csv("data\iterations_bound_per_algorithm.csv", index_col=[0])


    tables_tests = []
    for N_S in NETWORK_SIZE:
        print(f"\nNetwork size: {N_S}")
        network_info = load_grid_from_file(file_name=f"case_nodes_{N_S}",
                                           delete_temp_folder=True)
        for N_PF in N_POWER_FLOWS:
            assert network_info["active_power"].shape[0] >= N_PF, "Power flows requested exceed the case from file."
            network_t_numba = network_info["network"]
            active_ns = network_info["active_power"][:N_PF]
            reactive_ns = network_info["reactive_power"][:N_PF]

            pf_methods = [
                          (network_t_numba.run_pf, test_config_SAM),
                          # (pandapower_pf, test_config_newton_raphson),  # Pandapower on steroids
                          (pandapower_pf, test_config_backward_forward),
                          # (network_t_numba.run_pf, test_config_tensorpowerflow),  # Tensor Dense
                          # (network_t_numba.run_pf, test_config_HP_tensor),  # Tensor with sparse formulation
                          # (network_t_numba.run_pf, test_config_GPU_tensor),  # Tensor with sparse formulation
                          ]

            test_results = {}
            last_voltage_results = []
            at_least_one_solution_flag = False
            algorithms_ran = 0  # Number of algorithms with the same configuration on N_S and N_PF (To compare)
            for pf_method, test_config in pf_methods:
                if LIMIT_ALGORITHMS:
                    n_pf_max_algorithm = table_with_limits[str(N_S)][test_config["method_name"]].astype(int)
                    if N_PF > n_pf_max_algorithm:
                        print(f"Skipping {test_config['method_name']} on PF: {N_PF} on number of nodes: {N_S}")
                        continue # skip the test if its too long.
                try:
                    # Valid network object (in order to enable/disable numba, and pass the grid to pandapower)
                    test_config["network"] = network_t_numba
                    test_result = run_test(pf_method=pf_method,
                                           active_power_data=active_ns,
                                           reactive_power_data=reactive_ns,
                                           power_flow_settings=test_config
                                           )
                    algorithms_ran += 1

                    # Solve conflict with naming convention (Methods that have the same name but different solvers/enhancers)
                    if "sparse_solver" in test_config.keys():
                        solver_name = "-" + test_config["sparse_solver"]

                    else:
                        solver_name = ""

                    method_name_dict = test_config["method_name"] + solver_name

                    if not at_least_one_solution_flag:
                        last_voltage_results = test_result[method_name_dict]["v"]
                        at_least_one_solution_flag = True

                    elif at_least_one_solution_flag and algorithms_ran >= 2:
                        current_voltage_results = test_result[method_name_dict]["v"]
                        assert np.allclose(last_voltage_results.squeeze(), current_voltage_results.squeeze(), rtol=1e-04, atol=1e-07)
                        last_voltage_results = current_voltage_results

                    test_result[method_name_dict].pop("v")  # TODO: Need to save this to plot somewhere?
                    test_results = test_results | test_result  # Merge the results
                except Exception as e:
                    print(e)
                    print("Probably a memory error. Jumping to the next iteration.")


            table = pd.DataFrame.from_dict(test_results, orient="index")
            tables_tests.append(table)

            table_temp = pd.concat(tables_tests, axis=0)
            time_stamp_str = time.strftime("%Y%m%d-%H-%M-%S")
            table_temp.to_csv(f"temp_log\check_point_{time_stamp_str}.csv", index=True)

    tables_tests_concat = pd.concat(tables_tests, axis=0)
    tables_tests_concat = tables_tests_concat.reset_index()

    tables_tests_concat.to_csv(FILE_NAME, index=False)
    # print(tables_tests_concat.sort_values(by=["grid_size", "time_pf"]))
    end_whole_experiment = perf_counter()
    print(f"Whole experiment: {(end_whole_experiment - start_whole_experiment):.3f} sec.")

if __name__ == "__main__":
    # ==================================================================================================================
    # EXPERIMENT 0:
    # For all grid sizes compute 1, 10 and 100 PF. In order to find the iterations/sec.
    # IMPORTANT!:
    # This experiment will create the file  "data\iterations_bound_per_algorithm.csv". You need to rename it and change
    # to the "data\" folder. The table is used to compute the boundaries of powerflows for the time sweep experiments.
    # the objective is not to wait forever. The table allow us to compute the iterations/sec per algorithm.
    # NETWORK_SIZE = [10, 30, 100, 200, 300, 500, 800, 1_000, 3_000, 5_000]
    # N_POWER_FLOWS = [1, 10, 100]
    # LIMIT_ALGORITHMS = False
    # -> Command in terminal
    # $ python sweep_experiments.py -t 1 10 100 -g 10 30 100 200 300 500 800 1000 3000 5000 -f test_grid_boundary_values.csv
    # Real time:  1162.943 sec -> 19.38 min

    # ==================================================================================================================
    # EXPERIMENT 1:
    # For all grid sizes (GRID SIZE EXPERIMENT)
    # NETWORK_SIZE = [10, 30, 100, 200, 300, 500, 800, 1_000, 3_000, 5_000]
    # N_POWER_FLOWS = [1, 1000]
    # LIMIT_ALGORITHMS = False
    # -> Command in terminal
    # $ python sweep_experiments.py -t 1 1000 -g 10 30 100 200 300 500 800 1000 3000 5000 -f test_grid_sweep_two_time_steps_1_1000.csv
    # $ python sweep_experiments.py -t 1000 -g 10 30 100 200 300 500 -f sam_test_no_numba.csv
    # Time run: 1.06 hrs // Real time:  3794.8 sec -> 1.05 hrs DONE!!

    # ==================================================================================================================
    # EXPERIMENT 2:  (Bounded by time) -> will take max iterations from: "data\iterations_bound_per_algorithm.csv"
    # Only one grid size. (TIME SWEEP EXPERIMENT)
    # N_POWER_FLOWS = [1, 10, 100, 300, 1_000, 3_000, 10_000, 30_000, 50_000, 100_000]
    # NETWORK_SIZE = [500]
    # LIMIT_ALGORITHMS = False
    # -> Command in terminal
    # $ python sweep_experiments.py -t 1 10 100 300 1000 3000 10000 30000 50000 100000 -g 500 -f test_ts_fixed_grid_500.csv -c
    # Time run: 50 min. (max 10 min per algorithm)  // Real time:  3748.373 sec. -> 62.47 min.

    # ==================================================================================================================
    # EXPERIMENT 3:  (Bounded by time) -> will take max iterations from: "data\iterations_bound_per_algorithm.csv"
    # Parameter calculation (TIME SWEEP EXPERIMENT - ALL GRIDS)
    # N_POWER_FLOWS = [1, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 50_000, 100_000]
    # NETWORK_SIZE = [10, 30, 100, 200, 300, 500, 800, 1_000, 3_000, 5_000]
    # LIMIT_ALGORITHMS = True
    # -> Command in terminal
    # $ python sweep_experiments.py -t 1 10 30 100 300 1000 3000 10000 30000 50000 100000 -g 10 30 100 200 300 500 800 1000 3000 5000 -f test_parameter_calculation.csv -c
    # Time run: 2.5 hrs  // Real time:  8914.051 sec. -> 148.5 min -> 2.47 hrs  no!!

    time_stamp_str = time.strftime("%Y%m%d-%H-%M-%S")

    parser = argparse.ArgumentParser(
        description="Simulate all the combinations of grids and step size requested by user.")
    parser.add_argument('-t', '--times', nargs='+', type=int, default=[1],
                        help="List of time steps that are going to be simulated. e.g. "
                             "--times 10 100 1000 it will simulate 10, 100 and 1000 time steps.")
    parser.add_argument('-g', '--grids', nargs='+', type=int, default=[100],
                        help="List of grid sizes that are going to be simulated. e.g. "
                             "--grids 10 100 1000 it will simulate grids with 10, 100 and 1000 nodes.")
    parser.add_argument('-c', '--constrained', action="store_true",
                        help="Enable the time limitation per algorithm (used only in the time step sweep tests).")
    parser.add_argument('-f', '--file', type=str, default=f"run_{time_stamp_str}.csv",
                        help="File name for saving the test results.")

    args, unknown = parser.parse_known_args()
    print(args)

    runt_tests(N_POWER_FLOWS=args.times,
               NETWORK_SIZE=args.grids,
               LIMIT_ALGORITHMS=args.constrained,
               FILE_NAME=args.file)

    # runt_tests(N_POWER_FLOWS=[1000],
    #            NETWORK_SIZE=[5000],
    #            LIMIT_ALGORITHMS=False,
    #            FILE_NAME="delete_me_numba.csv")


