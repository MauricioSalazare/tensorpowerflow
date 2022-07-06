from tensorpowerflow import GridTensor
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils_experiments import run_test
import seaborn as sns
import matplotlib.pyplot as plt
# plt.switch_backend('Qt5Agg')

v_solution = [0.98965162 + 0.00180549j, 0.98060256 + 0.00337785j, 0.96828145 + 0.00704551j,
              0.95767051 + 0.01019764j, 0.94765203 + 0.01316654j, 0.94090964 + 0.01600068j,
              0.93719984 + 0.01754998j, 0.93283877 + 0.01937559j, 0.93073823 + 0.02026054j,
              0.9299309 + 0.02058985j, 0.92968994 + 0.02068728j, 0.98003142 + 0.00362498j,
              0.97950885 + 0.00385019j, 0.97936712 + 0.00391065j, 0.97935604 + 0.0039148j,
              0.93971131 + 0.01547898j, 0.93309482 + 0.01739656j, 0.92577912 + 0.01988823j,
              0.91988489 + 0.02188907j, 0.91475251 + 0.02362566j, 0.90888169 + 0.02596304j,
              0.90404908 + 0.02788248j, 0.89950353 + 0.02968449j, 0.89731375 + 0.03055177j,
              0.89647201 + 0.03088507j, 0.89622055 + 0.03098473j, 0.94032081 + 0.01625577j,
              0.93992817 + 0.01642583j, 0.93973182 + 0.01651086j, 0.9301316 + 0.02052908j,
              0.92952481 + 0.02079761j, 0.92922137 + 0.02093188j, 0.92912022 + 0.02097663j]
v_solution = np.array(v_solution, dtype="complex128")

# ======================================================================================================================
#%% TEST FOR NETWORK SIZE:
# ======================================================================================================================
network_t = GridTensor(node_file_path="grid_data/Nodes_34.csv",
                       lines_file_path="grid_data/Lines_34.csv")
V1 = network_t.run_pf_sequential()
V2 = network_t.run_pf_tensor()
assert np.allclose(V1["v"], v_solution)
assert np.allclose(V2["v"], v_solution)

network_t = GridTensor.generate_from_graph(nodes=100, child=3, plot_graph=False)
V1 = network_t.run_pf_sequential()
V2 = network_t.run_pf_tensor()
assert np.allclose(V1["v"], V2["v"])

#%%
# NETWORK_SIZE = [200]
# N_PF = 1
# NETWORK_SIZE = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1_000, 3_000]
NETWORK_SIZE = [10, 50, 100, 110, 120, 130, 140, 150, 200]
N_PF = 1_000
active_ns = np.random.normal(50, scale=1, size=(N_PF, NETWORK_SIZE[-1]))
reactive_ns = active_ns * .1

#%%
tables_tests = []
for N_S in tqdm(NETWORK_SIZE, desc="Network size"):
    print("\n")
    print("=" * 200)
    print(f"Network size flows: {N_S}")
    print("=" * 200)
    # Build network
    # network_t = GridTensor.generate_from_graph(nodes=N_S, child=3, plot_graph=False, numba=False)
    network_t_numba = GridTensor.generate_from_graph(nodes=N_S, child=3, plot_graph=False, numba=True)

    solution  = network_t_numba.run_pf_tensor()
    assert solution["convergence"], "Random generation on graphs created an overloaded case."

    nl = N_S - 1
    pf_methods = [
                    # (network_t.run_pf_sequential, True, "seq"),
                    # (network_t.run_pf_tensor, False, "tensor"),
                    # (network_t_numba.run_pf_sequential, True, "seq-Numba"),
                    (network_t_numba.run_pf_tensor, False, "tensor-Numba")
                 ]

    test_results = {}
    last_voltage_results = []
    for i, (pf_method, single_step_method, method_name) in enumerate(pf_methods):
        test_result = run_test(pf_method=pf_method,
                               active_power_data=active_ns[:N_PF, :nl],
                               reactive_power_data=reactive_ns[:N_PF, :nl],
                               method_name=method_name,
                               single_step=single_step_method)
        test_result[method_name]["grid_size"] = N_S

        if i == 0:
            last_voltage_results = test_result[method_name]["v"]
        else:
            current_voltage_results = test_result[method_name]["v"]
            print(f"Results assert: {np.allclose(last_voltage_results.squeeze(), current_voltage_results.squeeze())}")
            assert np.allclose(last_voltage_results.squeeze(), current_voltage_results.squeeze())
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
fig.suptitle(f"Total Power Flows: {N_PF}")
