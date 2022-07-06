from tensorpowerflow import GridTensor
import pandapower as pp
import numpy as np
from time import perf_counter
import pandas as pd
from tqdm import trange
from tensorpowerflow import create_pandapower_net, net_test

if __name__ == "__main__":
    branch_info = pd.read_csv("grid_data/Lines_34.csv")
    bus_info = pd.read_csv("grid_data/Nodes_34.csv")

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
    N_POWER_FLOWS = 1
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
    
    #%%
    network_t = GridTensor(node_file_path="grid_data/Nodes_34.csv",
                           lines_file_path="grid_data/Lines_34.csv")

    #%%
    start = perf_counter()
    solution = network_t.run_pf_tensor(active_power=active[:N_POWER_FLOWS, :],
                                       reactive_power=reactive[:N_POWER_FLOWS, :])
    print(f"Done: {perf_counter() - start}")
    
    #%% Compare with pandapower
    np.allclose(v_solutions, solution["v"])
    
    #%%
    N_S = 20
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

