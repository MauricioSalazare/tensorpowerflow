import sys
print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
current_directory = Path().absolute()
print(f"Current directory: {current_directory}")
sys.path.extend([str(current_directory.parent)])



from utils_experiments import load_grid_from_file
import numpy as np
from time import perf_counter
from ctypes import CDLL, POINTER, c_int, c_float, byref, Structure, c_bool, c_double, cast
from numpy import ctypeslib
import os
from utils_experiments import build_valid_network, save_grid_to_file
from tqdm import tqdm
from time import perf_counter

def create_case(n_nodes, n_power_flows):
    network_data = build_valid_network(n_nodes=n_nodes,
                                       n_power_flows=n_power_flows,
                                       lambda_f=0.5,
                                       truncate_to=100,
                                       seed=12345)

    save_grid_to_file(file_name=f"case_nodes_{n_nodes}",  # Does not require the extension
                      line_data=network_data["network"].branch_info,
                      node_data=network_data["network"].bus_info,
                      active_power=network_data["active_power"],
                      reactive_power=network_data["reactive_power"],
                      voltage_solutions=network_data["solutions"]["v"])


#%% Prepare the dynamic library to run the GPU code:


class Complex64(Structure):
    _fields_ = [("real", c_float), ("imag", c_float)]

    @property
    def value(self):
        return self.real + 1j * self.imag  # fields declared above

ctypes_dtype_complex = ctypeslib.ndpointer(np.complex64)

# # Visual Studio: Generated DLL
# os.add_dll_directory(r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug")
# dynamic_library = r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug\gpu_windows.dll"
# my_functions = CDLL(dynamic_library, winmode=0)


# CLI: Generated DLL

from ctypes import CDLL
import os

# os.add_dll_directory(r"C:\Users\20175334\Documents\PycharmProjects\tensorpowerflow\experiments\dll")
# dynamic_library = r"C:\Users\20175334\Documents\PycharmProjects\tensorpowerflow\experiments\dll\shared_library_complex.dll"

os.add_dll_directory(r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug")
dynamic_library = r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug\gpu_windows.dll"

my_functions = CDLL(dynamic_library, winmode=0)


tensor_power_flow = my_functions.tensorPowerFlow
tensor_power_flow.restype = None
tensor_power_flow.argtypes = [
                               POINTER(ctypes_dtype_complex),  # Matrix S, dimensions: S(m x m)
                               POINTER(ctypes_dtype_complex),  # Matrix K, dimensions: S(m x m)
                               POINTER(ctypes_dtype_complex),  # Matrix V0, dimensions: V0(m x p)\
                               POINTER(ctypes_dtype_complex),  # Matrix W, dimensions: W(m x 1)
                               POINTER(c_int),      # m
                               POINTER(c_int),      # p
                               POINTER(c_double),   # tolerance
                               POINTER(c_int),      # iterations
                               POINTER(c_bool)      # convergence
                              ]


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
        print(f"Tol: {tol} -- iteration: {iteration}")

        v0 = voltage_k.T.copy()
        iteration += 1

    return v0, iteration

def load_grid_and_some_time_steps(ts=10, nodes=500):
    recovered_network_data = load_grid_from_file(file_name=f"case_nodes_{nodes}",
                                                 delete_temp_folder=True)

    network = recovered_network_data["network"]
    active_power = recovered_network_data["active_power"]
    reactive_power = recovered_network_data["reactive_power"]

    return (network, active_power[:ts].copy(), reactive_power[:ts].copy())

#%%
# network, active_power, reactive_power = load_grid_and_some_time_steps(ts=3, nodes=10)

n_nodes = 100
n_power_flows = 2
network_data = build_valid_network(n_nodes=n_nodes,
                                   n_power_flows=n_power_flows,
                                   lambda_f=0.5,
                                   truncate_to=100,
                                   seed=12345)
network = network_data["network"]
active_power = network_data["active_power"]
reactive_power = network_data["reactive_power"]


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
print(f"Total iterations: {iteration_gpu}")
# assert np.allclose(v, v_gpu)

#
# #%% GPU CODE:
# S_host = S.T.copy()
# S_host = S_host.astype(np.complex64)
#
# K_host = network._K_.copy()
# K_host = K_host.astype(np.complex64)
#
# V0_host = (np.ones((ts, nb - 1)).T + 1j * np.zeros((ts, nb - 1)).T).astype(np.complex64)
#
# W_host = network._L_.copy()
# W_host = W_host.astype(np.complex64)
#
# m = int(nb-1)
# p = int(ts)
# tolerance = float(1e-10)
# iterations = 100
# convergence = c_bool()
#
#
# m_int = byref(c_int(m))
# p_int = byref(c_int(p))
# iterations_int = byref(c_int(iterations))
# # iterations_x = cast(iterations, POINTER(c_int))
# tolerance_int = byref(c_double(tolerance))
# convergence_int = byref(convergence)
#
# S_c = S_host.ravel(order="F")
# S_ca = S_c.ctypes.data_as(POINTER(ctypes_dtype_complex))
#
# K_c = K_host.ravel(order="F")
# K_ca = K_c.ctypes.data_as(POINTER(ctypes_dtype_complex))
#
# V0_c = V0_host.ravel(order="F")
# V0_ca = V0_c.ctypes.data_as(POINTER(ctypes_dtype_complex))
#
# W_c = W_host.ravel(order="F")
# W_ca = W_c.ctypes.data_as(POINTER(ctypes_dtype_complex))
#
# tensor_power_flow(S_ca,
#                   K_ca,
#                   V0_ca,
#                   W_ca,
#                   m_int,
#                   p_int,
#                   tolerance_int,
#                   iterations_int,
#                   convergence_int)
# print("Expected CPU:")
# print(v_gpu.T)
#
# print("GPU")
# print(V0_c.reshape(m,p, order="F"))
#
# print(f"Convergence: {convergence.value}")
# print(f"Iterations: {iterations_int._obj.value}")
#
#
#
# #
# # #%%
# # # This is only onte time step!!!!!!!!!!!!!!!!!
# # F = network._K_ @ np.diag(np.conj(S[0]))
# # start = perf_counter()
# # # This only works for one time step
# # v1 , iteration1 = func(F=F,  # This is the minues de inverse of Ydd multiply by power
# #                        W=network._L_,
# #                        v0=np.ones((nb - 1, 1)) + 1j * np.zeros((nb - 1, 1)),
# #                        iterations=100,
# #                        tolerance=1e-5)
# # print(f"Total time tensor with one time step: {perf_counter() - start}")
# #
# # assert np.allclose(v[0].reshape(-1,1), v1)
# #
# # #%%
# #
#
#
#
#
#
#
# #%%
# # np.savetxt(fr'caso_edgar\F_real_{F.real.shape[0]}x{F.real.shape[1]}.csv', F.real.flatten(), delimiter=',')
# # np.savetxt(fr'caso_edgar\F_imag_{F.imag.shape[0]}x{F.imag.shape[1]}.csv', F.imag.flatten(), delimiter=',')
# # np.savetxt(fr'caso_edgar\W_real_{network._L_.real.shape[0]}x{network._L_.real.shape[1]}.csv', network._L_.real.flatten(), delimiter=',')
# # np.savetxt(fr'caso_edgar\W_imag_{network._L_.imag.shape[0]}x{network._L_.imag.shape[1]}.csv', network._L_.imag.flatten(), delimiter=',')
# # np.savetxt(fr'caso_edgar\v_real_{v1.real.shape[0]}x{v1.real.shape[1]}.csv', v1.real.flatten(), delimiter=',')
# # np.savetxt(fr'caso_edgar\v_imag_{v1.imag.shape[0]}x{v1.imag.shape[1]}.csv', v1.imag.flatten(), delimiter=',')
# # np.savetxt(fr'caso_edgar\S_real_{S.real.shape[0]}x{S.real.shape[1]}.csv', S.real.flatten(), delimiter=',')
# # np.savetxt(fr'caso_edgar\S_iamg_{S.imag.shape[0]}x{S.imag.shape[1]}.csv', S.imag.flatten(), delimiter=',')
# #
# # np.savetxt('K_real.csv', network._K_.real, delimiter=',')
# # np.savetxt('K_imag.csv', network._K_.imag, delimiter=',')
# #
# #
# #
