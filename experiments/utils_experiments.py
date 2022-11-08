import numpy as np
from tqdm import trange
from time import perf_counter
from tensorpowerflow import create_pandapower_net
import pandapower as pp
import psutil
import warnings
from tensorpowerflow import GridTensor
import pandas as pd
from pathlib import Path
import shutil

"""
Methods to make the experiments.
"""


def fit_in_memory(n_power_flows: int, n_network_nodes: int, verbose: bool=False):
    bytes_per_element = 16  # 16 bytes for complex numbers a 8 bytes for real numbers (Active and reactive power)
    overhead_operations = 3.3  # Multiplier for extra operations in the grid program. Bus matrices, graphs, numba, etc

    memory_request_load = float(n_power_flows) * float(n_network_nodes - 1) * bytes_per_element * 9.31e-10  # GiB
    memory_request_network = float(n_network_nodes) * float(n_network_nodes) * bytes_per_element * 9.31e-10  # GiB
    worst_case_memory = (memory_request_network + 2 * memory_request_load) * overhead_operations

    memory_available = psutil.virtual_memory().available * 9.31e-10  # GiB

    if verbose:
        print(f"Requested: {worst_case_memory:.3f} - Available: {memory_available:.3f}")

    if worst_case_memory < memory_available * 0.90:  # Use 90% of memory available
        return True
    else:
        print(f"Requested: {worst_case_memory:.3f} GiB - Available: {memory_available:.3f} GiB")
        warnings.warn(f"DO NOT FIT IN MEMORY!!: Combination: n_power_flows: {n_power_flows},"
                      f" n_network_nodes: {n_network_nodes} omitted")

        return False

# def fit_in_memory_sparse():
#     n_bytes_complex = 16
#     n_bytes_int = 8
#
#     non_zero_elements_size = network_t_numba.Ydd_sparse.nnz * N_TIME_STEPS * n_bytes_complex * 9.31e-10
#     row_index_size = network_t_numba.Ydd_sparse.nnz * N_TIME_STEPS * n_bytes_int * 9.31e-10
#     column_index_size = network_t_numba.Ydd_sparse.nnz * N_TIME_STEPS * n_bytes_int * 9.31e-10
#     total_memory = non_zero_elements_size + row_index_size + column_index_size
#     memory_available = psutil.virtual_memory().available * 9.31e-10  # GiB
#
#     print(f"HP-TENSOR: Requested: {total_memory:.3f} - Available: {memory_available:.3f}")
#
#     solutions_hp_tensor = network_t_numba.run_pf(active_power=active_ns, reactive_power=reactive_ns,
#                                                  sparse_solver="pardiso", algorithm="hp-tensor")
#     print(f"HP-TENSOR FINISHED the PF in: {solutions_hp_tensor['time_pf']} sec")
#     assert solutions_hp_tensor['convergence']
#     print(solutions_hp_tensor['v'])


def build_valid_network(n_nodes: int,
                        n_power_flows: int,
                        lambda_f: float=6.0,
                        try_outs:int=100,
                        truncate_to: int = None,
                        seed: int = None,
                        dtype: str= "float64",
                        enable_numba=False) -> (GridTensor, np.ndarray, np.ndarray):
    """
    Creates a radial network with the specific number of nodes. Also, creates synthetic consumption profiles for the
    generated network, checking that it is a solvable case (it has at least one solution).

    Parameters:
    -----------
        n_nodes: int: Nodes for the distribution network
        n_power_flows: int: Time steps to be simulated for the network.
        lambda_f: float: Parameter to control the "loadibility" of the grid.
        try_outs: int: Maximum number of attempts to create the distribution network (For a valid case)
        truncate_to: int or None: Number of power flows to be evaluated on the created network, to consider it "valid".
            If it is "None", the number of power flows to be run will be from the parameter n_power_flows. If it is an
            int value, then it will run "truncate_to" number of power flows.
            The purpose of this parameter is to avoid running unnecessarily high amount of power flows to consider
            the grid to be valid.
        seed: int: Seed for the numpy random generator. In order to have reproducible results.
        dtype: str: Not implemented yet, but used in the case to reduce the size in memory of the problem.


    Returns:
        Grid: GridTensor: Object with the created grid.
        Active_power: np.ndarray: Active power numpy array of type float64. The size is "n_power_flows x n_nodes".
        Reactive_power: np.ndarray: Reactive power numpy array of type float64. The size is "n_power_flows x n_nodes".

        WARNING: Be aware that even though the validation of the net is done with "truncate_to" parameter. The number
                 of active and reactive power values will have the size of "n_power_flows".

    """

    if seed is not None and isinstance(seed, int):
        np.random.seed(seed)

    if truncate_to is not None:
        n_power_flows_to_test = truncate_to
    else:
        n_power_flows_to_test = n_power_flows

    active_ns = np.random.normal(50 * lambda_f,
                                 scale=10,
                                 size=(n_power_flows, n_nodes - 1)).round(3) # Assume 1 slack variable
    reactive_ns = (active_ns * .1).round(3)

    # Build network
    print("Building network from graph...")
    network_t_numba = GridTensor.generate_from_graph(nodes=n_nodes,
                                                     child=3,
                                                     plot_graph=False,
                                                     load_factor=lambda_f,
                                                     line_factor=3,
                                                     numba=enable_numba)
    print("Built!")

    not_solvable_case = True
    try_number = 0
    solutions = None

    while not_solvable_case and try_number < try_outs:
        # TODO: If the size of the network is very big. Solve with HP tensor?
        print(f"Solving {n_power_flows_to_test} power flows...")
        # solutions = network_t_numba.run_pf_tensor(active_power=active_ns[:n_power_flows_to_test],
        #                                           reactive_power=reactive_ns[:n_power_flows_to_test])
        solutions = network_t_numba.run_pf(active_power=active_ns[:n_power_flows_to_test],
                                           reactive_power=reactive_ns[:n_power_flows_to_test],
                                           algorithm="hp-tensor")
        if solutions["convergence"]:
            print(f"Grid is goood!... Good Anakin! Goood!")
            not_solvable_case = False
            break

        # Re-sample with a lower loading factor and solve again
        warnings.warn("Random graph generation created a non-solvable case. Resampling consumption.")
        try_number += 1
        lambda_f = lambda_f * 0.9  # Reduce the load by 10% and try again.
        active_ns = np.random.normal(50 * lambda_f, scale=10, size=(n_power_flows, n_nodes - 1))
        reactive_ns = active_ns * .1

        # TODO: Need to update the base case to make it solvable

    if not_solvable_case and try_number == try_outs:
        raise RuntimeError("Random generation on graphs created an overloaded case and no way to fix it.")

    network_info = {"network": network_t_numba,
                    "active_power": active_ns,
                    "reactive_power": reactive_ns,
                    "solutions": solutions,
                    "truncated_test": truncate_to}

    return network_info


def pandapower_pf(active_power_data: np.ndarray,
                  reactive_power_data: np.ndarray,
                  pandapower_settings: dict,
                  ):
    """
    Param:
    ------
        active_power_data: np.ndarray: Must be a 2-D array (Matrix) with shape (n_time_steps, n_nodes-1).
                                        The input is non in p.u. The values must be in kW.
        reactive_power_data: np.ndarray: Must be a 2-D array (Matrix) with shape (n_time_steps, n_nodes-1)
                                        The input is non in p.u. The values must be in kVAr.
        pandapower_settings: dict: Configuration dictionary for the power flow.

    """

    assert active_power_data.ndim == 2, "The active power array must be 2D."
    assert reactive_power_data.ndim == 2, "The active power array must be 2D."

    method_name = pandapower_settings["method_name"]
    sampling_size = pandapower_settings["sampling_size"]

    assert method_name in ["nr", "bfsw"], "Incorrect power flow algorithm for pandapower"

    net = create_pandapower_net(branch_info_=pandapower_settings["network"].branch_info,
                                bus_info_=pandapower_settings["network"].bus_info)

    s_base = pandapower_settings["network"].s_base
    numba_enable = pandapower_settings["numba_enable"]
    n_time_steps = active_power_data.shape[0]

    assert n_time_steps >= 1, "No power data (At least one time step is necessary)."
    active_power_data_pu = active_power_data / s_base
    reactive_power_data_pu = reactive_power_data / s_base

    time_total_pf_pandas = []
    times_pf_pandas = []
    iterations_pf_pandas = []
    v_solutions = []
    convergence_pandas = []

    # Run at least one power flow to unstuck the solver of pandapower
    i = 0
    net.load.p_mw[1:] = active_power_data_pu[i, :]  # Assume 1 is slack bus
    net.load.q_mvar[1:] = reactive_power_data_pu[i, :]  # Assume 1 is slack bus

    pp.runpp(net,
             algorithm=method_name,
             numba=numba_enable,
             v_debug=True,
             VERBOSE=False,
             tolerance_mva=1e-6,
             )

    for i in trange(n_time_steps, desc=f'Method: {method_name}' + "-pandapower"):
        net.load.p_mw[1:] = active_power_data_pu[i, :]  # Assume 1 is slack bus
        net.load.q_mvar[1:] = reactive_power_data_pu[i, :]  # Assume 1 is slack bus

        start = perf_counter()
        pp.runpp(net,
                 algorithm=method_name,
                 numba=numba_enable,
                 v_debug=True,
                 VERBOSE=False,
                 tolerance_mva=1e-6,
                 )
        time_total_pf_pandas.append(perf_counter() - start)
        times_pf_pandas.append(net._ppc['et'])
        iterations_pf_pandas.append(net._ppc['iterations'])
        convergence_pandas.append(net._ppc['success'])

        v_real = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
        v_img = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
        v_result = v_real + 1j * v_img
        v_solutions.append(v_result[1:])

    idx = reduced_index(n_time_steps, sampling_size)  # Consistent indexing for the subsampling

    solution_method = {
                        "v": np.array(v_solutions),
                        "time_pre_pf": np.sum(time_total_pf_pandas) - np.sum(times_pf_pandas),
                        "time_pf": np.sum(times_pf_pandas),
                        "time_pf_mean": np.mean(times_pf_pandas),
                        "time_algorithm": np.sum(time_total_pf_pandas),
                        "iterations": np.mean(iterations_pf_pandas),

                        "convergence": np.alltrue(convergence_pandas),
                        "iterations_log": list(np.array(iterations_pf_pandas)[idx]),
                        "time_pre_pf_log": list(np.array(time_total_pf_pandas)[idx] - np.array(times_pf_pandas)[idx]),
                        "time_pf_log": list(np.array(times_pf_pandas)[idx]),
                        "convergence_log": list(np.array(convergence_pandas)[idx])
                        }

    return solution_method


def run_test(pf_method,
             active_power_data: np.ndarray,
             reactive_power_data: np.ndarray,
             power_flow_settings: dict):
    """Helper method to run different power flow algorihtms, could be based on pandapower of my own."""


    is_single_step = power_flow_settings["is_single_step"]
    is_pandapower_function = power_flow_settings["is_pandapower_function"]
    sampling_size = power_flow_settings["sampling_size"]

    if "sparse_solver" in power_flow_settings.keys():
        kwargs = dict(sparse_solver = power_flow_settings["sparse_solver"])
        solver_name = "-" + power_flow_settings["sparse_solver"]
    else:
        kwargs = dict()
        solver_name = ""

    kwargs.update(algorithm=power_flow_settings["method_name"])

    method_name = power_flow_settings["method_name"] + solver_name  # To differentiate the same method different solvers
    n_time_steps = active_power_data.shape[0]
    grid_size = active_power_data.shape[1]

    if is_single_step and not is_pandapower_function:
        voltage_solutions = []
        times_pf = []
        times_pre_pf = []
        times_total = []
        iterations = []
        convergence = []

        if power_flow_settings["network"].is_numba_enabled and not power_flow_settings["numba_enable"]:
            # warnings.warn("Disabling Numba. This will reduce the performance of the algorithm.")
            power_flow_settings["network"].disable_numba()
        elif not power_flow_settings["network"].is_numba_enabled and power_flow_settings["numba_enable"]:
            # warnings.warn("Enabling Numba. This will take a moment.")
            power_flow_settings["network"].enable_numba()
        else:
            # Network was created with numba_enabled, and the settings requested to enable the numba.
            pass

        # Run at least one power flow to unstuck the numba for the Laurent and Pandapower
        i = 0
        scrap_this = pf_method(active_power=np.atleast_2d(active_power_data[i, :]),
                               reactive_power=np.atleast_2d(reactive_power_data[i, :]),
                               **kwargs)

        for i in trange(n_time_steps, desc=f'Method: {method_name}', position=1):
            solution = pf_method(active_power=np.atleast_2d(active_power_data[i, :]),
                                 reactive_power=np.atleast_2d(reactive_power_data[i, :]),
                                 **kwargs)
            voltage_solutions.append(solution["v"])
            times_pf.append(solution["time_pf"])
            times_pre_pf.append(solution["time_pre_pf"])
            times_total.append(solution["time_algorithm"])
            iterations.append(solution["iterations"])
            convergence.append(solution["convergence"])

        idx = reduced_index(n_time_steps, sampling_size)  # Consistent indexing for the subsampling

        # print(f"Ran sequential timesteps: {n_time_steps}, time pre-pf: {np.sum(times_pre_pf)}" )

        solution_method = {
                           "v": np.array(voltage_solutions),
                           "time_pre_pf": np.sum(times_pre_pf),
                           "time_pf": np.sum(times_pf),
                           "time_pf_mean": np.mean(times_pf),
                           "time_algorithm": np.sum(times_total),
                           "iterations": np.mean(iterations),

                           "convergence": np.alltrue(convergence),
                           "iterations_log": list(np.array(iterations)[idx]),
                           "time_pre_pf_log": list(np.array(times_pre_pf)[idx]),
                           "time_pf_log": list(np.array(times_pf)[idx]),
                           "convergence_log": list(np.array(convergence)[idx]),
                           }
    elif is_single_step:  # The power flow method is from Pandapower

        solution_method = pf_method(active_power_data=active_power_data,  # This is NOT in p.u.
                                    reactive_power_data=reactive_power_data,  # This is NOT in p.u.
                                    pandapower_settings=power_flow_settings
                                    )

    else:  # Tensor power flow (HP or dense)
        solution_method = None
        assert active_power_data.ndim == 2, "Power must have 2 dimensions."

        if active_power_data.shape[0] == 1:
            # Make sure that numba is pre-compiled and or hp-tensor call functions are done at least once
            solution = pf_method(active_power=np.atleast_2d(active_power_data[0:2]),
                                 reactive_power=np.atleast_2d(reactive_power_data[0:2]),
                                 **kwargs)

        if active_power_data.shape[0] > 1:
            # Make sure that numba is pre-compiled
            solution = pf_method(active_power=np.atleast_2d(active_power_data[0:2]),
                                 reactive_power=np.atleast_2d(reactive_power_data[0:2]),
                                 **kwargs)

        for i in trange(1, desc=f'Method: {method_name}'):  # Place holder for tqdm
            solution = pf_method(active_power=active_power_data,
                                 reactive_power=reactive_power_data,
                                 **kwargs)
            solution_method = solution

    # Dict with the results to create table with pandas
    test_results = {method_name: solution_method}
    test_results[method_name]["time_steps"] = n_time_steps
    test_results[method_name]["grid_size"] = grid_size + 1  # Assume only 1 slack variable

    return test_results

def reduced_index(n_time_steps: int, sampling_size: int):
    """Generates a sequences of indexes to subsample an array bigger than sampling_size"""

    if n_time_steps > sampling_size:
        idx = np.random.choice(n_time_steps, sampling_size, replace=False)
    else:
        idx = np.arange(n_time_steps)

    return idx



def save_grid_to_file(file_name: str,  # Does not require the extension
                      line_data: pd.DataFrame,
                      node_data: pd.DataFrame,
                      active_power: np.ndarray,
                      reactive_power: np.ndarray,
                      folder_path: str=None,
                      voltage_solutions: np.ndarray=None):

    # TODO: Add folder path if you want to save the cases somewhere else
    current_path = Path().absolute()
    temp_dir = "temp"
    temp_dir_path = current_path / temp_dir

    if Path.is_file(current_path / (file_name + ".zip")):
        raise FileExistsError(f"The case was not saved. The file: '{file_name}.zip' exists. Delete it and try again)")

    if not Path.is_dir(temp_dir_path):
        Path.mkdir(temp_dir_path)

    index = [str(ii) for ii in range(active_power.shape[0])]
    columns = [str(ii) for ii in range(active_power.shape[1])]

    real_consumption = pd.DataFrame(active_power, index=index, columns=columns)
    imag_consumption = pd.DataFrame(reactive_power, index=index, columns=columns)

    real_consumption.to_parquet(temp_dir_path / "power_real.parquet.gzip", compression="gzip")
    imag_consumption.to_parquet(temp_dir_path / "power_imag.parquet.gzip", compression="gzip")

    line_data.to_parquet(temp_dir_path / "lines.parquet.gzip", compression="gzip")
    node_data.to_parquet(temp_dir_path / "nodes.parquet.gzip", compression="gzip")

    if voltage_solutions is not None:
        index = [str(ii) for ii in range(voltage_solutions.shape[0])]
        columns = [str(ii) for ii in range(voltage_solutions.shape[1])]

        real_voltage = pd.DataFrame(voltage_solutions.real, index=index, columns=columns)
        imag_voltage = pd.DataFrame(voltage_solutions.imag, index=index, columns=columns)

        real_voltage.to_parquet(temp_dir_path / "voltage_real.parquet.gzip", compression="gzip")
        imag_voltage.to_parquet(temp_dir_path / "voltage_imag.parquet.gzip", compression="gzip")

    # Zip the case
    file_name_zip = file_name
    file_path_zip = current_path / file_name_zip
    shutil.make_archive(file_path_zip, 'zip', temp_dir_path)

    # Delete folder
    if Path.is_dir(temp_dir_path):
        shutil.rmtree(temp_dir_path)

def load_grid_from_file(file_name,
                        folder_path: str=None,
                        delete_temp_folder:bool=True):

    current_path = Path().absolute()  #TODO: Be able to select the folder where the cases are
    file_path_recovered_zip = current_path / "recovered"

    if Path.is_dir(file_path_recovered_zip):  # Create temporary folder to unpack
        shutil.rmtree(file_path_recovered_zip)

    if not Path.is_file(current_path / (file_name + ".zip")):
        raise FileNotFoundError(f"File: '{file_name}.zip' does not exist.")

    shutil.unpack_archive(f'{file_name}.zip', current_path / "recovered")

    real_consumption_recovered = pd.read_parquet(file_path_recovered_zip / 'power_real.parquet.gzip')
    imag_consumption_recovered = pd.read_parquet(file_path_recovered_zip / 'power_imag.parquet.gzip')

    lines_recovered = pd.read_parquet(file_path_recovered_zip / 'lines.parquet.gzip')
    nodes_recovered = pd.read_parquet(file_path_recovered_zip / 'nodes.parquet.gzip')

    network_data = {}
    # print("Building grid...")
    grid_recovered = GridTensor(node_file_path="",
                                lines_file_path="",
                                from_file=False,
                                nodes_frame=nodes_recovered,
                                lines_frame=lines_recovered)
    # print("Grid built!")

    active_power = real_consumption_recovered.values
    reactive_power = imag_consumption_recovered.values

    network_data["network"] = grid_recovered
    network_data["active_power"] = active_power
    network_data["reactive_power"] = reactive_power

    if Path.is_file(file_path_recovered_zip /'voltage_real.parquet.gzip') and \
        Path.is_file(file_path_recovered_zip / 'voltage_imag.parquet.gzip'):
        real_voltage_recovered = pd.read_parquet(file_path_recovered_zip / 'voltage_real.parquet.gzip')
        imag_voltage_recovered = pd.read_parquet(file_path_recovered_zip / 'voltage_imag.parquet.gzip')
        network_data["v"] = real_voltage_recovered.values + 1j*imag_voltage_recovered.values

    if delete_temp_folder:
        shutil.rmtree(file_path_recovered_zip)

    return network_data
