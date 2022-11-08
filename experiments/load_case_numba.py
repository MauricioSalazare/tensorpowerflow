from utils_experiments import load_grid_from_file
import numpy as np
from time import perf_counter
import ctypes as ctypes
import os

os.add_dll_directory(r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug")

# A = K
# B = L
# S = S
# v = Kv0 + L  == v = Av0 + B

#%%
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
    """ This is the implementation in the GPU"""
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
        iteration = + 1

    return v0, iteration

def load_grid_and_some_time_steps(ts=10, nodes=500):
    recovered_network_data = load_grid_from_file(file_name=f"case_nodes_{nodes}",
                                                 delete_temp_folder=True)

    network = recovered_network_data["network"]
    active_power = recovered_network_data["active_power"]
    reactive_power = recovered_network_data["reactive_power"]

    return (network, active_power[:ts].copy(), reactive_power[:ts].copy())

#%%
network, active_power, reactive_power = load_grid_and_some_time_steps(ts=1000, nodes=300)

#%% Implementation on CPU, tensorized
S = active_power + 1j*reactive_power
S = S / network.s_base

ts = S.shape[0]
nb = S.shape[1] + 1

start = perf_counter()
V0_init = np.ones((ts, nb - 1)) + 1j * np.zeros((ts, nb - 1))
v , iteration= func_tensor(K=network._K_,  # This is the minus inverse of Ydd
                           W=network._L_,
                           S=S,
                           v0=V0_init,
                           ts=ts,
                           nb=nb,
                           iterations=100,
                           tolerance=1e-5)
print(f"Total time: {perf_counter() - start}")


#%% Implementation on CPU like the GPU formulation
start = perf_counter()
V0_init = np.ones((ts, nb - 1)) + 1j * np.zeros((ts, nb - 1))
v_gpu , iteration_gpu = func_tensor_gpu(K=network._K_,  # This is the minus inverse of Ydd
                                        W=network._L_,
                                        S=S,
                                        v0=V0_init,
                                        ts=ts,
                                        nb=nb,
                                         iterations=100,
                                        tolerance=1e-5)
print(f"Total time: {perf_counter() - start}")
assert np.allclose(v, v_gpu)


#%%
# This is only one time step!!!!!!!!!!!!!!!!!
F = network._K_ @ np.diag(np.conj(S[0]))
start = perf_counter()
# This only works for one time step
v1 , iteration1 = func(F=F,  # This is the minues de inverse of Ydd multiply by power
                       W=network._L_,
                       v0=np.ones((nb - 1, 1)) + 1j * np.zeros((nb - 1, 1)),
                       iterations=100,
                       tolerance=1e-5)
print(f"Total time tensor with one time step: {perf_counter() - start}")

assert np.allclose(v[0].reshape(-1,1), v1)

