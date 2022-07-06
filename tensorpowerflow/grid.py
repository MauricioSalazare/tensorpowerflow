import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from time import perf_counter
from numba import njit
import warnings
from .utils import generate_network, _load_default_34_node_case
from .numbarize import (pre_power_flow_sequential,
                        power_flow_sequential,
                        power_flow_sequential_constant_power,
                        pre_power_flow_tensor,
                        power_flow_tensor,
                        power_flow_tensor_constant_power)

class GridTensor:
    def __init__(self,
                 node_file_path: str = None,
                 lines_file_path: str = None,
                 *,
                 s_base: int = 1000,  # kVA - 1 phase
                 v_base: float = 11,  # kV - 1 phase
                 iterations: int = 100,
                 tolerance: float = 1e-6,
                 from_file=True,
                 nodes_frame: pd.DataFrame = None,
                 lines_frame: pd.DataFrame = None,
                 numba=True):

        self.s_base = s_base
        self.v_base = v_base
        self.z_base = (self.v_base ** 2 * 1000) / self.s_base
        self.i_base = self.s_base / (np.sqrt(3) * self.v_base)

        self.iterations = iterations
        self.tolerance = tolerance

        if node_file_path is None and lines_file_path is None:
            _nodes_frame, _lines_frame = _load_default_34_node_case()
            self.branch_info = _nodes_frame
            self.bus_info = _lines_frame
            print("Default case loaded.")

        elif node_file_path is not None and lines_file_path is not None and from_file:
            self.branch_info = pd.read_csv(lines_file_path)
            self.bus_info = pd.read_csv(node_file_path)

        elif nodes_frame is not None and lines_frame is not None:
            self.branch_info = lines_frame
            self.bus_info = nodes_frame
        else:
            raise ValueError("Wrong input configuration")

        self.P_file = self.bus_info[self.bus_info.Tb != 1].PD.values  # Vector with all active power except slack
        self.Q_file = self.bus_info[self.bus_info.Tb != 1].QD.values  # Vector with all reactive power except slack

        self._make_y_bus()
        self._compute_alphas()
        self.v_0 = None
        self._F_ = None
        self._W_ = None

        if np.all(self.alpha_P) and not np.any(self.alpha_Z) and not np.any(self.alpha_I):
            # From ZIP model, only P is equal to one, the rest are zero.
            self.constant_power_only = True
            self._K_ = -np.linalg.inv(self.Ydd)  # Reduced version of -B^-1 (Reduced version of _F_)
            self._L_ = self._K_ @ self.Yds  # Reduced version of _W_
        else:
            self.constant_power_only = False

        if numba:
            print("Numba mode enabled")
            self._power_flow_sequential_constant_power = njit(power_flow_sequential_constant_power)
            self._pre_power_flow_sequential = njit(pre_power_flow_sequential)
            self._power_flow_sequential = njit(power_flow_sequential)

            self._power_flow_tensor_constant_power = njit(power_flow_tensor_constant_power, parallel=True)
            self._pre_power_flow_tensor = njit(pre_power_flow_tensor, parallel=True)
            self._power_flow_tensor = njit(power_flow_tensor, parallel=True)
            self._compile_numba()
        else:
            warnings.warn("Numba NOT enabled. Performance is greatly reduced.", RuntimeWarning)
            self._power_flow_sequential_constant_power = power_flow_sequential_constant_power
            self._pre_power_flow_sequential = pre_power_flow_sequential
            self._power_flow_sequential = power_flow_sequential

            self._power_flow_tensor_constant_power = power_flow_tensor_constant_power
            self._pre_power_flow_tensor = pre_power_flow_tensor
            self._power_flow_tensor = power_flow_tensor

    @classmethod
    def generate_from_graph(cls, *, nodes=100, child=2, plot_graph=True, load_factor=2, line_factor=3, **kwargs):
        """
        Constructor of a synthetic grid using networkX package
        """

        nodes_frame, lines_frame = generate_network(nodes=nodes,
                                                    child=child,
                                                    plot_graph=plot_graph,
                                                    load_factor=load_factor,
                                                    line_factor=line_factor)

        return cls(node_file_path="",
                   lines_file_path="",
                   from_file=False,
                   nodes_frame=nodes_frame,
                   lines_frame=lines_frame,
                   **kwargs)

    def reset_start(self):
        self.v_0 = np.ones((self.nb - 1, 1), dtype="complex128")  # Flat start  #2D array

    def _make_y_bus(self) -> None:
        """
        Compute Y_bus submatrices

        For each branch, compute the elements of the branch admittance matrix where
              | Is |   | Yss  Ysd |   | Vs |
              |    | = |          | * |    |
              |-Id |   | Yds  Ydd |   | Vd |
        """

        self.nb = self.bus_info.shape[0]  # number of buses
        self.nl = self.branch_info.shape[0]  # number of lines

        sl = self.bus_info[self.bus_info['Tb'] == 1]['NODES'].tolist()  # Slack node(s)

        stat = self.branch_info.iloc[:, 5]  # ones at in-service branches
        Ys = stat / ((self.branch_info.iloc[:, 2] + 1j * self.branch_info.iloc[:, 3]) / (
                    self.v_base ** 2 * 1000 / self.s_base))  # series admittance
        Bc = stat * self.branch_info.iloc[:, 4] * (self.v_base ** 2 * 1000 / self.s_base)  # line charging susceptance
        tap = stat * self.branch_info.iloc[:, 6]  # default tap ratio = 1

        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / tap
        Yft = - Ys / tap
        Ytf = Yft

        # build connection matrices
        f = self.branch_info.iloc[:, 0] - 1  # list of "from" buses
        t = self.branch_info.iloc[:, 1] - 1  # list of "to" buses

        # connection matrix for line & from buses
        Cf = csr_matrix((np.ones(self.nl), (range(self.nl), f)), (self.nl, self.nb))

        # connection matrix for line & to buses
        Ct = csr_matrix((np.ones(self.nl), (range(self.nl), t)), (self.nl, self.nb))

        # build Yf and Yt such that Yf * V is the vector of complex branch currents injected
        # at each branch's "from" bus, and Yt is the same for the "to" bus end
        i = np.r_[range(self.nl), range(self.nl)]  # double set of row indices

        Yf = csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])))
        Yt = csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])))

        # build Ybus
        Ybus = Cf.T * Yf + Ct.T * Yt  # Full Ybus

        # Dense matrices
        self._Ybus = Ybus.toarray()
        self.Yss = csr_matrix(Ybus[sl[0] - 1, sl[0] - 1], shape=(len(sl), len(sl))).toarray()
        self.Ysd = Ybus[0, 1:].toarray()  # TODO: Here assume the slack is the first one?
        self.Yds = self.Ysd.T
        self.Ydd = Ybus[1:, 1:].toarray()

    def _compute_alphas(self):
        self.alpha_P = self.bus_info[self.bus_info.Tb != 1].Pct.values.reshape(-1, )
        self.alpha_I = self.bus_info[self.bus_info.Tb != 1].Ict.values.reshape(-1, )
        self.alpha_Z = self.bus_info[self.bus_info.Tb != 1].Zct.values.reshape(-1, )

        self.flag_all_constant_impedance_is_zero = not np.any(self.alpha_Z)
        self.flag_all_constant_current_is_zero = not np.any(self.alpha_I)
        self.flag_all_constant_powers_are_ones = np.all(self.alpha_P)

    def _compile_numba(self):
        # Compile JIT code running the base case PF at least once
        print("Compiling JIT functions")
        start = perf_counter()
        _ = self.run_pf_sequential()
        _ = self.run_pf_tensor()
        print(f"Compile time: {(perf_counter() - start):.4f} seconds")

    def run_pf_sequential(self,
                          active_power: np.ndarray = None,
                          reactive_power: np.ndarray = None,
                          flat_start: bool = True,
                          start_value: np.array = None) -> dict:
        """Single time step power flow with numba performance increase"""

        if (active_power is not None) and (reactive_power is not None):
            assert len(active_power.shape) == 1, "Array should be one dimensional."
            assert len(reactive_power.shape) == 1, "Array should be one dimensional."
            assert len(active_power) == len(reactive_power) == self.nb - 1, "All load nodes must have power values."
        else:
            active_power = self.P_file
            reactive_power = self.Q_file

        if flat_start:
            self.v_0 = np.ones((self.nb - 1, 1), dtype="complex128")  # 2D-Vector
        elif start_value is not None:
            # TODO: Check the dimensions of the flat start
            self.v_0 = start_value  # User's start value

        if self.constant_power_only:
            start_time_pre_pf = perf_counter()
            # No precomputing, the minimum matrix multiplication is done in the powerflow.
            end_time_pre_pf = perf_counter()

            start_time_pf = perf_counter()
            V, iteration = self._power_flow_sequential_constant_power(active_power,
                                                                      reactive_power,
                                                                      self.s_base,
                                                                      -self._K_,
                                                                      self._L_,
                                                                      v_0=self.v_0,
                                                                      iterations=self.iterations,
                                                                      tolerance=self.tolerance)
            end_time_pf = perf_counter()

        else:
            start_time_pre_pf = perf_counter()
            _W, _F = self._pre_power_flow_sequential(active_power,
                                                     reactive_power,
                                                     s_base = self.s_base,
                                                     alpha_Z=self.alpha_Z,
                                                     alpha_I=self.alpha_I,
                                                     alpha_P=self.alpha_P,
                                                     Yds=self.Yds,
                                                     Ydd=self.Ydd,
                                                     nb=self.nb)
            end_time_pre_pf = perf_counter()

            start_time_pf = perf_counter()
            V, iteration = self._power_flow_sequential(_W,
                                                       _F,
                                                       v_0=self.v_0,
                                                       iterations=self.iterations,
                                                       tolerance=self.tolerance)
            end_time_pf = perf_counter()

        if iteration == self.iterations:
            flag_convergence = False
        else:
            flag_convergence = True

        total_time_pre_pf = end_time_pre_pf - start_time_pre_pf
        total_time_pf = end_time_pf - start_time_pf
        total_time_algorithm = total_time_pre_pf + total_time_pf

        solution = {"v": V.flatten(),  # 1D-Vector. Solution of voltage in complex numbers
                    "time_pre_pf": total_time_pre_pf,
                    "time_pf": total_time_pf,
                    "time_algorithm": total_time_algorithm,
                    "iterations": iteration,
                    "convergence": flag_convergence}

        return solution

    def run_pf_tensor(self,
                      active_power: np.ndarray = None,
                      reactive_power: np.ndarray = None,
                      *,
                      iterations: int = 100,
                      tolerance: float = 1e-6,
                      flat_start: bool = True) -> dict:
        """Run power flow for an array of active and reactive power consumption"""

        if (active_power is not None) and (reactive_power is not None):
            assert len(active_power.shape) == 2, "Array should be two dimensional."
            assert len(reactive_power.shape) == 2, "Array should be two dimensional."
            assert active_power.shape[1] == reactive_power.shape[1] == self.nb - 1, "All load nodes must have power values."
            # rows are time steps, columns are nodes
        else:
            active_power = self.P_file[np.newaxis, :]
            reactive_power = self.Q_file[np.newaxis, :]

        self.ts_n = active_power.shape[0]  # Time steps to be simulated
        if flat_start:
            self.v_0 = np.ones((self.ts_n, self.nb - 1)) + 1j * np.zeros((self.ts_n, self.nb - 1))  # Flat start

        active_power_pu = active_power / self.s_base  # Vector with all active power except slack
        reactive_power_pu = reactive_power / self.s_base  # Vector with all reactive power except slack
        S_nom = active_power_pu + 1j * reactive_power_pu  # (ts x nodes)

        if self.constant_power_only:
            start_time_pre_pf = perf_counter()
            # No pre-computing (Already done when creating the object)
            end_time_pre_pf = perf_counter()
            start_time_pf = perf_counter()
            self.v_0, t_iterations = self._power_flow_tensor_constant_power(_K_=self._K_,
                                                                            _L_=self._L_,
                                                                            S_nom=S_nom,
                                                                            v_0=self.v_0,
                                                                            ts_n=self.ts_n,
                                                                            nb=self.nb,
                                                                            iterations=iterations,
                                                                            tolerance=tolerance)
            end_time_pf = perf_counter()

        else:
            start_time_pre_pf = perf_counter()
            self._F_, self._W_ = self._pre_power_flow_tensor(flag_all_constant_impedance_is_zero=self.flag_all_constant_impedance_is_zero,
                                                             flag_all_constant_current_is_zero=self.flag_all_constant_current_is_zero,
                                                             flag_all_constant_powers_are_ones=self.flag_all_constant_powers_are_ones,
                                                             ts_n=self.ts_n,
                                                             nb=self.nb,
                                                             S_nom=S_nom,
                                                             alpha_Z=self.alpha_Z,
                                                             alpha_I=self.alpha_I,
                                                             alpha_P=self.alpha_P,
                                                             Yds=self.Yds,
                                                             Ydd=self.Ydd)
            end_time_pre_pf = perf_counter()

            start_time_pf = perf_counter()
            self.v_0, t_iterations = self._power_flow_tensor(_F_=self._F_,
                                                             _W_=self._W_,
                                                             v_0=self.v_0,
                                                             ts_n=self.ts_n,
                                                             nb=self.nb,
                                                             iterations=iterations,
                                                             tolerance=tolerance)
            end_time_pf = perf_counter()

        if t_iterations == iterations:
            flag_convergence = False
        else:
            flag_convergence = True

        total_time_pre_pf = end_time_pre_pf - start_time_pre_pf
        total_time_pf = end_time_pf - start_time_pf
        total_time_algorithm = total_time_pre_pf + total_time_pf

        solution = {"v": self.v_0,  # 2D-Vector. Solution of voltage in complex numbers
                    "time_pre_pf": total_time_pre_pf,
                    "time_pf": total_time_pf,
                    "time_algorithm": total_time_algorithm,
                    "iterations": t_iterations,
                    "convergence": flag_convergence}

        return solution

    def line_currents(self, volt_solutions=None):
        raise NotImplementedError
