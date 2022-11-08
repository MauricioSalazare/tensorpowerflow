import sys
print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
current_directory = Path().absolute()
print(f"Current directory: {current_directory}")
sys.path.extend([str(current_directory.parent)])

from tensorpowerflow import GPUPowerFlow
from utils_experiments import build_valid_network
import numpy as np
from time import perf_counter

def func(F, W, v0, iterations, tolerance):
    """This only works for one time step"""
    iteration = 0
    tol = np.inf

    while (iteration < iterations) & (tol >= tolerance):
        print(f"Iteration: {iteration}")
        v =  F @  (1 / np.conj(v0)).reshape(-1,1) + W
        tol = np.max(np.abs(np.abs(v) - np.abs(v0)))
        v0 = v  # Voltage at load buses
        iteration += 1

    return v0, iteration

def func_tensor(K, W, S, v0, ts, nb, iterations, tolerance):
    """Old implementation of the dense tensor power flow"""
    iteration = 0
    tol = np.inf

    while (iteration < iterations) & (tol >= tolerance):
        v = np.zeros((ts, nb - 1), dtype="complex128")  # TODO: Test putting this outside of while loop
        for i in range(ts):  # This is inefficient but the only way to fit in memory
            v[i] = (K @ (np.conj(S[i]) * (1 / np.conj(v0[i]))).reshape(-1, 1) + W).T
            # v[k+1] = (K @ S) v[k]^(*(-1)) + W
            # v[k+1] = (F) v[k]^(*(-1)) + W
        tol = np.max(np.abs(np.abs(v) - np.abs(v0)))
        v0 = v  # Voltage at load buses
        iteration += 1

    return v0, iteration

def func_tensor_gpu(K, W, S, v0, ts, nb, iterations, tolerance):
    """
    This is the mathematical implementation of the dense tensor in the GPU but... here, in the CPU.
    It's just to compare
    """

    iteration = 0
    tol = np.inf

    while iteration < iterations and tol >= tolerance:
        LAMBDA = np.conj(S.T * (1 / v0.T))
        Z = K @ LAMBDA
        voltage_k = []
        for j in range(ts):
            voltage_k.append(Z[:, j].reshape(-1, 1) + W)
        voltage_k = np.hstack(voltage_k)

        tol = np.max(np.abs(np.abs(voltage_k.T) - np.abs(v0)))

        v0 = voltage_k.T.copy()
        iteration += 1

    return v0, iteration

#%%
n_nodes = 3000
n_power_flows = 50_000

memory_s = n_nodes * n_power_flows
memory_k = n_nodes ** 2
memory_w = n_nodes * 1
memory_v = n_nodes * n_power_flows
memory_v_holder = n_nodes * n_power_flows
memory_z_holder = n_nodes * n_power_flows
memory_euclidean_dist = n_nodes * n_power_flows

formula = (2 * (memory_s + memory_k + memory_w + memory_z_holder + memory_v + memory_v_holder) + memory_euclidean_dist ) * 8 * 9.3132*1e-10

network_data = build_valid_network(n_nodes=n_nodes,
                                   n_power_flows=n_power_flows,
                                   lambda_f=0.5,
                                   truncate_to=100,
                                   seed=12345,
                                   enable_numba=True)
network = network_data["network"]
active_power = network_data["active_power"]
reactive_power = network_data["reactive_power"]

start = perf_counter()
solution = network.run_pf(active_power=active_power,
                          reactive_power=reactive_power,
                          algorithm="gpu-tensor")
print(f"\n\nTotal time GPU: {perf_counter() - start}")

# start = perf_counter()
# solution_cpu = network.run_pf(active_power=active_power,
#                               reactive_power=reactive_power,
#                               algorithm="tensor")
# print(f"\n\nTotal time CPU: {perf_counter() - start}")

# print("Expected CPU:")
# print(solution_cpu["v"])
print("GPU")
print(solution["v"])
print(f"Iterations GPU: {solution['iterations']}")
# assert np.allclose(solution_cpu["v"], solution["v"])