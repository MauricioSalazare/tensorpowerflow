import os
import numpy as np
from ctypes import CDLL, POINTER, c_int, byref, c_bool, c_double
from numpy import ctypeslib
import sys
from time import perf_counter

def load_library():
    platform = sys.platform
    my_functions = None

    if platform == "linux" or platform == "linux2":
        # linux (Hard coded in the meantime)
        so_file = r"/home/mauricio/PycharmProjects/gpu_tensorpf/shared_library_complex.so"
        my_functions = CDLL(so_file)

    elif platform == "darwin":
        raise NotImplementedError("MacOS is not currently supported.")

    elif platform == "win32":
        # TODO: Fallback if the gpu library is not found.
        # Windows.
        # os.add_dll_directory(r"C:\Users\20175334\Documents\PycharmProjects\tensorpowerflow\experiments\dll")
        # dynamic_library = r"C:\Users\20175334\Documents\PycharmProjects\tensorpowerflow\experiments\dll\shared_library_complex.dll"

        # Hardcode the directory for debugging purposes.
        os.add_dll_directory(r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug")
        dynamic_library = r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug\gpu_windows.dll"

        my_functions = CDLL(dynamic_library, winmode=0)
    else:
        raise ValueError("OS not recognized.")

    tensor_power_flow = my_functions.tensorPowerFlow
    tensor_power_flow.restype = None
    ctypes_dtype_complex = ctypeslib.ndpointer(np.complex64)
    tensor_power_flow.argtypes = [
                                  POINTER(ctypes_dtype_complex),  # Matrix S, dimensions: S(m x m)
                                  POINTER(ctypes_dtype_complex),  # Matrix K, dimensions: S(m x m)
                                  POINTER(ctypes_dtype_complex),  # Matrix V0, dimensions: V0(m x p)\
                                  POINTER(ctypes_dtype_complex),  # Matrix W, dimensions: W(m x 1)
                                  POINTER(c_int),  # m
                                  POINTER(c_int),  # p
                                  POINTER(c_double),  # tolerance
                                  POINTER(c_int),  # iterations
                                  POINTER(c_bool)  # convergence
                                 ]

    return tensor_power_flow


class GPUPowerFlow(object):

    def __init__(self):
        self.gpu_solver = load_library()

    def power_flow_gpu(self,
                       K: np.ndarray,  # (m x m) == (nodes-1 x nodes-1)
                       L: np.ndarray,  # (p x 1) == (time_steps x 1) or just p
                       S: np.ndarray,  # (p x m) == (time_steps x nodes)
                       v0: np.ndarray, # (p x m) == (time_steps x nodes)
                       ts: np.ndarray,
                       nb: int,
                       iterations: int = 100,
                       tolerance: float = None):
        """
        Wrapper function for the execution of the shared library function.

        The function was implemented with the matrices dimension of the paper
        """

        if tolerance is None:
            tolerance_gpu = 1e-10
        else:
            tolerance_gpu = tolerance ** 2  # Heuristic, this match to the CPU tolerance.

        # ======================================================================
        # Reshape/casting and making sure that the complex matrices are 32 bits.
        # S_host = S.T.copy()
        S_host = S.T
        S_host = S_host.astype(np.complex64)

        # K_host = K.copy()
        K_host = K
        K_host = K_host.astype(np.complex64)

        # V0_host = V0.T.copy()
        V0_host = v0.T
        V0_host = V0_host.astype(np.complex64)

        # W_host =  L.copy()
        W_host = L
        W_host = W_host.astype(np.complex64)

        m = int(nb - 1)
        p = int(ts)

        tolerance_gpu = float(tolerance_gpu)
        iterations = int(iterations)
        convergence = c_bool()

        # ======================================================================
        # Pointers for the dynamic library function.
        m_int = byref(c_int(m))
        p_int = byref(c_int(p))
        iterations_int = byref(c_int(iterations))
        tolerance_int = byref(c_double(tolerance_gpu))
        convergence_int = byref(convergence)

        ctypes_dtype_complex = ctypeslib.ndpointer(np.complex64)

        S_c = S_host.ravel(order="F")
        S_ca = S_c.ctypes.data_as(POINTER(ctypes_dtype_complex))

        K_c = K_host.ravel(order="F")
        K_ca = K_c.ctypes.data_as(POINTER(ctypes_dtype_complex))

        V0_c = V0_host.ravel(order="F")
        V0_ca = V0_c.ctypes.data_as(POINTER(ctypes_dtype_complex))

        W_c = W_host.ravel(order="F")
        W_ca = W_c.ctypes.data_as(POINTER(ctypes_dtype_complex))

        # start = perf_counter()
        self.gpu_solver(S_ca,
                        K_ca,
                        V0_ca,
                        W_ca,
                        m_int,
                        p_int,
                        tolerance_int,
                        iterations_int,
                        convergence_int)
        # print(f"GPU Dynamic library execution: {perf_counter() - start} sec.")
        # print(f"Convergence: {convergence.value}")
        v_solution = V0_c.reshape(m, p, order="F")
        iter_solution = iterations_int._obj.value

        # Voltage solution is a matrix with dimensions (time_steps x (n_nodes-1))-> Including the transpose.
        return v_solution.T, iter_solution