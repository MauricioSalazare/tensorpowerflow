from tensorpowerflow import GridTensor
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils_experiments import run_test, pandapower_pf
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import psutil
# plt.switch_backend('Qt5Agg')

# v_solution = [0.98965162 + 0.00180549j, 0.98060256 + 0.00337785j, 0.96828145 + 0.00704551j,
#               0.95767051 + 0.01019764j, 0.94765203 + 0.01316654j, 0.94090964 + 0.01600068j,
#               0.93719984 + 0.01754998j, 0.93283877 + 0.01937559j, 0.93073823 + 0.02026054j,
#               0.9299309 + 0.02058985j, 0.92968994 + 0.02068728j, 0.98003142 + 0.00362498j,
#               0.97950885 + 0.00385019j, 0.97936712 + 0.00391065j, 0.97935604 + 0.0039148j,
#               0.93971131 + 0.01547898j, 0.93309482 + 0.01739656j, 0.92577912 + 0.01988823j,
#               0.91988489 + 0.02188907j, 0.91475251 + 0.02362566j, 0.90888169 + 0.02596304j,
#               0.90404908 + 0.02788248j, 0.89950353 + 0.02968449j, 0.89731375 + 0.03055177j,
#               0.89647201 + 0.03088507j, 0.89622055 + 0.03098473j, 0.94032081 + 0.01625577j,
#               0.93992817 + 0.01642583j, 0.93973182 + 0.01651086j, 0.9301316 + 0.02052908j,
#               0.92952481 + 0.02079761j, 0.92922137 + 0.02093188j, 0.92912022 + 0.02097663j]
# v_solution = np.array(v_solution, dtype="complex128")

# ======================================================================================================================
#%% TEST FOR NETWORK SIZE:
# ======================================================================================================================
# network_t = GridTensor(node_file_path="grid_data/Nodes_34.csv",
#                        lines_file_path="grid_data/Lines_34.csv")
# V1 = network_t.run_pf_sequential()
# V2 = network_t.run_pf_tensor()
# assert np.allclose(V1["v"], v_solution)
# assert np.allclose(V2["v"], v_solution)
#
# network_t = GridTensor.generate_from_graph(nodes=100, child=3, plot_graph=False)
# V1 = network_t.run_pf_sequential()
# V2 = network_t.run_pf_tensor()
# assert np.allclose(V1["v"], V2["v"])


#%%
def fits_in_memory(n_power_flows: int, n_network_nodes: int, verbose: bool=False):
    memory_request = float(n_power_flows) * float(n_network_nodes - 1) * 8 * 9.31e-10  # GiB
    memory_available = psutil.virtual_memory().available * 9.31e-10  # GiB

    if verbose:
        print(f"Requested: {memory_request:.3f} - Available: {memory_available:.3f}")

    if memory_request < memory_available * 0.90:
        return True
    else:
        return False





#%%
# NETWORK_SIZE = [100]
# N_PF = 1
# NETWORK_SIZE = [3000]
# NETWORK_SIZE = [10]
# NETWORK_SIZE = [1_500, 3_000]
NETWORK_SIZE = [10, 100, 500, 1_000, 2_000, 3_000, 5_000, 10_000]
# NETWORK_SIZE = [1_000_000]
N_PF = 1
# N_PF = 100

NETWORK_SIZE_ = [10, 100, 500, 1_000, 2_000, 3_000, 5_000, 10_000]
POWER_FLOWS_ = [10, 100, 500, 1_000, 10_000, 100_000, 500_000]

testes = 0
for netw in NETWORK_SIZE_:
    for powf in POWER_FLOWS_:
        if fits_in_memory(powf, netw, verbose=True):
            testes += 1
            print("OK")
        else:
            print("NOT OK")

#%%

print(f"Memory consumption: {(N_PF * (NETWORK_SIZE[-1] - 1) ) * 8 * 9.31e-10:.2f} Gb")
print(f"Memory available: {(psutil.virtual_memory().available * 9.31e-10):.2f} Gb")

lambda_f = 1
active_ns = np.random.normal(50 * lambda_f, scale=10, size=(N_PF, NETWORK_SIZE[-1] - 1))
reactive_ns = active_ns * .1



#%%
tables_tests = []
for N_S in tqdm(NETWORK_SIZE, desc="Network size"):
    print("\n")
    print("=" * 200)
    print(f"Network size: {N_S}")
    print("=" * 200)
    # Build network
    # network_t = GridTensor.generate_from_graph(nodes=N_S, child=3, plot_graph=False, numba=False)
    network_t_numba = GridTensor.generate_from_graph(nodes=N_S,
                                                     child=3,
                                                     plot_graph=False,
                                                     load_factor=2,
                                                     line_factor=3,
                                                     numba=True)

    nl = N_S - 1
    not_solvable_case = True
    try_outs = 100
    try_number = 1
    while not_solvable_case and try_number < try_outs:
        solutions = network_t_numba.run_pf_tensor(active_power=active_ns[:N_PF, :nl],
                                                  reactive_power=reactive_ns[:N_PF, :nl])
        if solutions["convergence"]:
            not_solvable_case = False
            break

        warnings.warn("Random graph generation created a non-solvable case. Resampling")
        try_number += 1

        network_t_numba = GridTensor.generate_from_graph(nodes=N_S,
                                                         child=3,
                                                         plot_graph=False,
                                                         load_factor=2,
                                                         line_factor=3,
                                                         numba=True)
        active_ns = np.random.normal(25, scale=10, size=(N_PF, NETWORK_SIZE[-1] - 1))
        reactive_ns = active_ns * .1

    if try_number == try_outs:
        raise RuntimeError("Random generation on graphs created an overloaded case.")

    # Structure of the tuple: (pf_method: describes the power flow technique to use,
    #                          single_step: [bool] Flag to tell the runt_test method if it need to add a for loop,
    #                          method_name: [string] The name of the technique for plotting purposes,
    #                          pandapower_function: [bool] Flag to tell if the power flow function is from pandapower)

    pf_methods = [
                    # (network_t.run_pf_sequential, True, "seq"),
                    # (network_t.run_pf_tensor, False, "tensor"),
                    # (network_t.run_pf_sequential, True, "Laurent"),
                    # (network_t.run_pf_tensor, False, "tensor"),

                    # (network_t_numba.run_pf_sam_sequential, True, "SAM", False),
                    # (network_t_numba.run_pf_sam_sequential_juan, True, "SAM-Juan", False),

                    (network_t_numba.run_pf_sequential, True, "Laurent-Seq", False),
                    (network_t_numba.run_pf_hp_laurent, True, "HP-Seq-SCIPY", False),
                    (network_t_numba.run_pf_hp_laurent_pardiso, True, "HP-Seq-PARDISO", False),
                    (pandapower_pf, True, "nr", True),
                    # (pandapower_pf, True, "bfsw", True),
                    (network_t_numba.run_pf_tensor, False, "Laurent-Tensor", False),


                    # (network_t_numba.run_pf_sequential, True, "seq-Numba"),
                    # (network_t_numba.run_pf_tensor, False, "tensor-Numba")
                 ]

    test_results = {}
    last_voltage_results = []
    for i, (pf_method, single_step_method, method_name, pandapower_function) in enumerate(pf_methods):
        if pandapower_function:
            pandapower_settings = {}
            pandapower_settings["network"] = network_t_numba
            pandapower_settings["numba_enable"] = True
        else:
            pandapower_settings = None

        test_result = run_test(pf_method=pf_method,
                               active_power_data=active_ns[:N_PF, :nl],
                               reactive_power_data=reactive_ns[:N_PF, :nl],
                               method_name=method_name,
                               single_step=single_step_method,
                               pandapower_settings=pandapower_settings
                               )
        test_result[method_name]["grid_size"] = N_S

        if i == 0:
            last_voltage_results = test_result[method_name]["v"]
        else:
            current_voltage_results = test_result[method_name]["v"]
            print(f"Results assert: {np.allclose(last_voltage_results.squeeze(), current_voltage_results.squeeze())}")
            assert np.allclose(last_voltage_results.squeeze(), current_voltage_results.squeeze(), rtol=1e-04, atol=1e-07)
            last_voltage_results = current_voltage_results

        test_result[method_name].pop("v")
        test_results = test_results | test_result

    table = pd.DataFrame.from_dict(test_results, orient="index")
    tables_tests.append(table)

tables_tests_concat = pd.concat(tables_tests, axis=0)
tables_tests_concat = tables_tests_concat.reset_index()

#%%


fig, ax = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = ax[0,:].flatten()
ax2 = ax[1,:].flatten()
times_columns = ['time_pre_pf', 'time_pf', 'time_algorithm']

for time_column, ax1_, ax2_ in zip(times_columns, ax1, ax2):
    sns.lineplot(data=tables_tests_concat, x="grid_size", y=time_column, hue="index", style="index", ax=ax1_,
                 markers=True)
    ax1_.set_xscale('log')
    ax1_.set_yscale('log')
    ax1_.set_title(time_column)

    sns.lineplot(data=tables_tests_concat, x="grid_size", y=time_column, hue="index", style="index", ax=ax2_,
                 markers=True)
    ax2_.set_title(time_column)

    ax1_.legend(fontsize="x-small")
    ax2_.legend(fontsize="x-small")

    ax1_.grid(True, which='major', linewidth=1.8)
    ax1_.grid(True, which = 'minor', linestyle = '--')

    ax2_.grid(True, which='major', linewidth=1.8)
    ax1_.grid(True, which='minor', linestyle='--')

fig.suptitle(f"Total Power Flows: {N_PF}")
# plt.savefig("figures/grid_size_sweep.pdf")
