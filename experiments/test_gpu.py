import sys
print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
current_directory = Path().absolute()
print(f"Current directory: {current_directory}")
sys.path.extend([str(current_directory.parent)])

from numba import njit, prange
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


def func_tensor_gpu_for_numba(K, W, S, v0, ts, nb, iterations, tolerance):
    """
    This is the mathematical implementation of the dense tensor in the GPU but... here, in the CPU.
    It's just to compare
    """

    iteration = 0
    tol = np.inf
    # S = S.T
    # v0 = v0.T

    LAMBDA = np.zeros((nb - 1, ts)).astype(np.complex128)
    Z = np.zeros((nb - 1, ts)).astype(np.complex128)
    voltage_k = np.zeros((nb - 1, ts)).astype(np.complex128)

    voltage_k = voltage_k.T
    L = W.ravel()

    while iteration < iterations and tol >= tolerance:
        LAMBDA = np.conj(S.T * (1 / v0.T))  #  Hadamard product ( (nb-1) x ts)
        Z = K @ LAMBDA  # Matrix ( (nb-1) x ts )
        # voltage_k = Z + W  # This is a brodcasted sum ( (nb-1) x ts  +  (nb-1) x 1 => (nb-1) x ts )
        Z = Z.T
        for j in prange(ts):
            voltage_k[j] = Z[j] + L

        tol = np.max(np.abs(np.abs(voltage_k) - np.abs(v0)))
        v0 = voltage_k
        iteration += 1

    # S = S.T  # Recover the original shape of the power
    # v0 = v0.T  # Recover the original shape of the power

    return v0, iteration

#%%
n_nodes = 800
n_power_flows = 50_000
network_data = build_valid_network(n_nodes=n_nodes,
                                   n_power_flows=n_power_flows,
                                   lambda_f=0.5,
                                   truncate_to=100,
                                   seed=12345)
network = network_data["network"]
active_power = network_data["active_power"]
reactive_power = network_data["reactive_power"]

#%% Implementation on CPU, tensorized
S = active_power + 1j * reactive_power
S = S / network.s_base
ts = S.shape[0]
nb = S.shape[1] + 1


start = perf_counter()
V0_init = np.ones((ts, nb - 1)) + 1j * np.zeros((ts, nb - 1))
v_cpu , iteration_cpu = func_tensor(K=network._K_,  # This is the minus inverse of Ydd
                                        W=network._L_,
                                        S=S,
                                        v0=V0_init,
                                        ts=ts,
                                        nb=nb,
                                        iterations=100,
                                        tolerance=1e-5)
print(f"Total iterations: {iteration_cpu}")
print(f"Total time CPU: {perf_counter() - start}")


# #%% Implementation on CPU like the GPU formulation
start = perf_counter()
V0_init = np.ones((ts, nb - 1)) + 1j * np.zeros((ts, nb - 1))
v_cpu_like_gpu , iteration_cpu_like_gpu = func_tensor_gpu(K=network._K_,  # This is the minus inverse of Ydd
                                        W=network._L_,
                                        S=S,
                                        v0=V0_init,
                                        ts=ts,
                                        nb=nb,
                                        iterations=100,
                                        tolerance=1e-5)
print(f"Total iterations: {iteration_cpu_like_gpu}")
print(f"Total time CPU like GPU: {perf_counter() - start}")
assert np.allclose(v_cpu, v_cpu_like_gpu)

# #%% Implementation on CPU like the GPU formulation but with numba
start = perf_counter()
V0_init = np.ones((ts, nb - 1)) + 1j * np.zeros((ts, nb - 1))
v_cpu_numba , iteration_cpu_numba = func_tensor_gpu_for_numba(K=network._K_,  # This is the minus inverse of Ydd
                                                              W=network._L_,
                                                              S=S,
                                                              v0=V0_init,
                                                              ts=ts,
                                                              nb=nb,
                                                              iterations=100,
                                                              tolerance=1e-5)
print(f"Total iterations: {iteration_cpu_numba}")
print(f"Total time CPU for numba: {perf_counter() - start}")
assert np.allclose(v_cpu, v_cpu_numba)

func_tensor_numbarized = njit(func_tensor_gpu_for_numba, parallel=True)
v_cpu_numbarized , iteration_cpu_numbarized = func_tensor_numbarized(K=network._K_,  # This is the minus inverse of Ydd
                                                                    W=network._L_,
                                                                    S=S,
                                                                    v0=V0_init,
                                                                    ts=ts,
                                                                    nb=nb,
                                                                    iterations=100,
                                                                    tolerance=1e-5)

start = perf_counter()
v_cpu_numbarized , iteration_cpu_numbarized = func_tensor_numbarized(K=network._K_,  # This is the minus inverse of Ydd
                                                                     W=network._L_,
                                                                     S=S,
                                                                     v0=V0_init,
                                                                     ts=ts,
                                                                     nb=nb,
                                                                     iterations=100,
                                                                     tolerance=1e-5)
print(f"Total iterations: {iteration_cpu_numba}")
print(f"Total time CPU for numbarized: {perf_counter() - start}")
assert np.allclose(v_cpu, v_cpu_numbarized)

#
# print("\n")
# start = perf_counter()
# gpu_solver = GPUPowerFlow()
# v_gpu_real, iter_gpu_real = gpu_solver.power_flow_gpu(K=network._K_,  # (m x m) == (nodes x nodes)
#                                                       L=network._L_,  # (p x 1) == (time_steps x 1) or just p
#                                                       S=S,  # (p x m) == (time_steps x nodes)
#                                                       v0=V0_init, # (p x m) == (time_steps x nodes)
#                                                       ts=ts,
#                                                       nb=nb,
#                                                       iterations=100,
#                                                       tolerance=1e-10)
# print(f"Total time GPU: {perf_counter() - start}")
#
#
# # print("Expected CPU:")
# # print(v_cpu.T)
# print("GPU")
# print(v_gpu_real.T)
# print(f"Iterations GPU: {iter_gpu_real}")
# # assert np.allclose(v_cpu, v_gpu_real)