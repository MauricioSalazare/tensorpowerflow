from tensorpowerflow import GridTensor
import pandapower as pp
import numpy as np
from time import perf_counter
import pandas as pd
from tqdm import trange
from tensorpowerflow import create_pandapower_net, net_test


#%%
def pandapower_pf(active_power_data,
                  reactive_power_data,
                  method_name,
                  pandapower_settings):

    assert method_name in ["nr", "bfsw"], "Incorrect power flow algorithm for pandapower"

    net = create_pandapower_net(branch_info_=pandapower_settings["network"].branch_info,
                                bus_info_=pandapower_settings["network"].bus_info)
    s_base = pandapower_settings["network"].s_base

    N_TS = active_power_data.shape[0]
    active_power_data_pu = active_power_data / s_base
    reactive_power_data_pu = reactive_power_data / s_base

    time_total_pf_pandas = []
    times_pf_pandas = []
    iterations_pf_pandas = []
    v_solutions = []
    for i in trange(N_TS, desc=f'Method: {method_name}' + "-pandapower"):
        start = perf_counter()
        net.load.p_mw[1:] = active_power_data_pu[i, :]  # Assume 1 is slack bus
        net.load.q_mvar[1:] = reactive_power_data_pu[i, :]  # Assume 1 is slack bus

        pp.runpp(net, algorithm=method_name, numba=True, v_debug=True, VERBOSE=False, tolerance_mva=1e-6)
        time_total_pf_pandas.append(perf_counter() - start)
        # print(f"Iterations: {net._ppc['iterations']}" )
        # print(f"Time: {net._ppc['et']}")
        times_pf_pandas.append(net._ppc['et'])
        iterations_pf_pandas.append(net._ppc['iterations'])
        # print(f"NR. Iterations: {net._ppc['iterations']}. PF time: {net._ppc['et']}")

        v_real = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
        v_img = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
        v_result = v_real + 1j * v_img
        v_solutions.append(v_result[1:])

    solution_method = {
                        "v": np.array(v_solutions),
                        "time_pre_pf": np.sum(time_total_pf_pandas) - np.sum(times_pf_pandas),
                        "time_pf": np.sum(times_pf_pandas),
                        "time_pf_mean": np.mean(times_pf_pandas),
                        "time_algorithm": np.sum(time_total_pf_pandas),
                        "iterations": np.mean(iterations_pf_pandas),
                        "convergence": np.nan
    }

    return solution_method


#%%

if __name__ == "__main__":
    branch_info = pd.read_csv("../grid_data/Lines_34.csv")
    bus_info = pd.read_csv("../grid_data/Nodes_34.csv")

    net = create_pandapower_net(branch_info, bus_info)
    net_test(net)  # Only works with 34 node network.

    #%% Time series simulation:
    active = np.random.normal(300, scale=10, size=(10_000, 33))  # kW
    reactive = active * .1
    
    #%%
    time_total_pf_pandas = []
    times_nr_pandas = []
    iterations_nr_pandas = []
    v_solutions = []
    N_POWER_FLOWS = 1000
    for i in trange(N_POWER_FLOWS):
        start = perf_counter()
        net.load.p_mw[1:] = active[i, :] / 1000  # Assume 1 is slack bus
        net.load.q_mvar[1:] = reactive[i, :] / 1000  # Assume 1 is slack bus
        pp.runpp(net, algorithm="nr", numba=True, v_debug=True, VERBOSE=False, tolerance_mva=1e-6)
        time_total_pf_pandas.append(perf_counter() - start)
        # print(f"Iterations: {net._ppc['iterations']}" )
        # print(f"Time: {net._ppc['et']}")
        times_nr_pandas.append(net._ppc['et'])
        iterations_nr_pandas.append(net._ppc['iterations'])
        # print(f"NR. Iterations: {net._ppc['iterations']}. PF time: {net._ppc['et']}")
    
        v_real = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
        v_img = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
        v_result = v_real + 1j * v_img
        v_solutions.append(v_result[1:])
    
    print(f"Total pf times (iterations): {np.sum(times_nr_pandas)}")
    print(f"Total pf times (all steps): {np.sum(time_total_pf_pandas)}")
    v_solutions = np.array(v_solutions)

    #%% Same that before, but using the function
    network_t = GridTensor(node_file_path="../grid_data/Nodes_34.csv",
                           lines_file_path="../grid_data/Lines_34.csv",
                           numba=False)
    pandapower_settings = {}
    pandapower_settings["network"] = network_t

    N_POWER_FLOWS = 100
    # method_name = "bfsw"
    method_name = "nr"
    solution_pandapower =  pandapower_pf(active_power_data = active[:N_POWER_FLOWS, :],  # This is NOT in p.u.
                                         reactive_power_data= reactive[:N_POWER_FLOWS, :],
                                         method_name=method_name,
                                         pandapower_settings=pandapower_settings)

    print(f"Method: {method_name}")
    print(f"Sum time pf: {solution_pandapower['time_pf']}")
    print(f"Iterations: {solution_pandapower['iterations']}")
    print(f"Time per iteration (mean): {solution_pandapower['time_pf_mean'] / solution_pandapower['iterations']}")

    
    #%%
    network_t = GridTensor(node_file_path="../grid_data/Nodes_34.csv",
                           lines_file_path="../grid_data/Lines_34.csv")

    #%%
    start = perf_counter()
    solution = network_t.run_pf_tensor(active_power=active[:N_POWER_FLOWS, :],
                                       reactive_power=reactive[:N_POWER_FLOWS, :])
    print(f"Done: {perf_counter() - start}")
    
    #%% Compare with pandapower
    np.allclose(v_solutions, solution["v"])
    
    #%%
    N_S = 20  # Network size
    network_t = GridTensor.generate_from_graph(nodes=N_S,
                                               child=3,
                                               plot_graph=False,
                                               load_factor=2,
                                               line_factor=3,
                                               numba=True)
    solutions = network_t.run_pf_tensor()
    assert solutions["convergence"]

    #%%
    net = create_pandapower_net(branch_info_=network_t.branch_info,
                                bus_info_=network_t.bus_info)

    #%% Time series simulation:
    active = np.random.normal(50, scale=10, size=(10_000, N_S-1))  # kW
    reactive = active * .1

    time_total_pf_pandas = []
    times_nr_pandas = []
    iterations_nr_pandas = []
    v_solutions = []
    N_POWER_FLOWS = 100
    for i in trange(N_POWER_FLOWS):
        start = perf_counter()
        net.load.p_mw[1:] = active[i, :] / 1000  # Assume 1 is slack bus
        net.load.q_mvar[1:] = reactive[i, :] / 1000  # Assume 1 is slack bus
        pp.runpp(net, algorithm="nr", numba=True, v_debug=True, VERBOSE=False, tolerance_mva=1e-6)
        time_total_pf_pandas.append(perf_counter() - start)
        print(f"Iterations: {net._ppc['iterations']}" )
        print(f"Time: {net._ppc['et']}")
        times_nr_pandas.append(net._ppc['et'])
        iterations_nr_pandas.append(net._ppc['iterations'])
        # print(f"NR. Iterations: {net._ppc['iterations']}. PF time: {net._ppc['et']}")

        v_real = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
        v_img = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
        v_result = v_real + 1j * v_img
        v_solutions.append(v_result[1:])

    print(f"Total pf times (iterations): {np.sum(times_nr_pandas)}")
    print(f"Total pf times (all steps): {np.sum(time_total_pf_pandas)}")
    v_solutions = np.array(v_solutions)

    solutions = network_t.run_pf_tensor(active_power=active[:N_POWER_FLOWS, :],
                                        reactive_power=reactive[:N_POWER_FLOWS, :])
    print(f"Laurent total algorithm: {solutions['time_algorithm']}")

