from numba import prange
import numpy as np

def pre_power_flow_sequential(active_power,
                              reactive_power,
                              s_base,
                              alpha_Z,
                              alpha_I,
                              alpha_P,
                              Yds,
                              Ydd,
                              nb
                              ):
    active_power_pu = active_power / s_base  # Vector with all active power except slack
    reactive_power_pu = reactive_power / s_base  # Vector with all reactive power except slack

    S_nom = (active_power_pu + 1j * reactive_power_pu).reshape(-1,)
    if not np.any(alpha_Z):  # \alpha_z is 0
        B_inv = np.linalg.inv(Ydd)
    else:
        B = np.diag(np.multiply(alpha_Z, np.conj(S_nom))) + Ydd
        B_inv = np.linalg.inv(B)

    if not np.any(alpha_I):
        C = Yds  # Constant
    else:
        C = Yds + np.multiply(alpha_I, np.conj(S_nom)).reshape(nb - 1, 1)  # Constant

    _W = -B_inv @ C

    if np.all(alpha_P):
        _F = -B_inv @ np.diag(np.conj(S_nom))
    else:
        _F = -B_inv @ np.diag(np.multiply(alpha_P, np.conj(S_nom)))

    return _W, _F

def power_flow_sequential(_W,
                          _F,
                          v_0,
                          iterations,
                          tolerance,
                          ):
    iteration = 0
    tol = np.inf
    while (iteration < iterations) & (tol >= tolerance):
        v = _W + (_F @ np.reciprocal(np.conj(v_0)))
        tol = np.max(np.abs(np.abs(v) - np.abs(v_0)))
        v_0 = v  # Voltage at load buses
        iteration += 1

    return v_0, iteration  # Solution of voltage in complex numbers

def power_flow_sequential_constant_power(active_power,
                                         reactive_power,
                                         s_base,
                                         B_inv,
                                         _W,
                                         v_0,
                                         iterations,
                                         tolerance,
                                         ):
    active_power_pu = active_power / s_base  # Vector with all active power except slack
    reactive_power_pu = reactive_power / s_base  # Vector with all reactive power except slack
    S_nom = (active_power_pu + 1j * reactive_power_pu).reshape(-1, )

    iteration = 0
    tol = np.inf
    while (iteration < iterations) & (tol >= tolerance):
        v = -B_inv @ (np.conj(S_nom / v_0.flatten())).reshape(-1, 1) + _W
        tol = np.max(np.abs(np.abs(v) - np.abs(v_0)))
        v_0 = v
        iteration += 1

    return v_0, iteration

def pre_power_flow_tensor(flag_all_constant_impedance_is_zero,
                          flag_all_constant_current_is_zero,
                          flag_all_constant_powers_are_ones,
                          ts_n,
                          nb,
                          S_nom,
                          alpha_Z,
                          alpha_I,
                          alpha_P,
                          Yds,
                          Ydd):
    if not flag_all_constant_impedance_is_zero:
        _alpha_z_power = np.multiply(np.conj(S_nom), alpha_Z)  # (ts x nodes)
    else:
        _alpha_z_power = np.zeros((ts_n, nb - 1))  # (ts x nodes)

    if not flag_all_constant_current_is_zero:
        _alpha_i_power = np.multiply(np.conj(S_nom), alpha_I)  # (ts x nodes)
    else:
        _alpha_i_power = np.zeros((ts_n, nb - 1))  # (ts x nodes)

    if flag_all_constant_powers_are_ones:
        _alpha_p_power = np.conj(S_nom)  # (ts x nodes)
    else:
        _alpha_p_power = np.multiply(np.conj(S_nom), alpha_P)  # (ts x nodes)

    _B_inv2 = np.zeros((ts_n, nb - 1, nb - 1), dtype="complex128")
    _F_2 = np.zeros((ts_n, nb - 1, nb - 1), dtype="complex128")
    _W_2 = np.zeros((ts_n, nb - 1), dtype="complex128")

    _C2 = _alpha_i_power + Yds.reshape(-1)  # (ts x nodes)  Sum is broadcasted to all rows of _alpha_i_power

    for i in prange(ts_n):
        _B_inv2[i] = np.linalg.inv(np.diag(_alpha_z_power[i]) + Ydd)
        _F_2[i] = -_B_inv2[i] * _alpha_p_power[i].reshape(1, -1)  # Broadcast multiplication
        _W_2[i] = (-_B_inv2[i] @ _C2[i].reshape(-1, 1)).reshape(-1)

    return _F_2, _W_2

def power_flow_tensor(_F_,
                      _W_,
                      v_0,
                      ts_n,
                      nb,
                      iterations,
                      tolerance,
                      ):
    iteration = 0
    tol = np.inf
    while (iteration < iterations) & (tol >= tolerance):
        v_recp_conj = np.reciprocal(np.conj(v_0))
        RT2 = np.zeros((ts_n, nb - 1), dtype="complex128")
        for i in prange(ts_n):  # This is critical as it makes a lot of difference
            RT2[i] = _F_[i] @ v_recp_conj[i]
        v = _W_ + RT2
        tol = np.max(np.abs(np.abs(v) - np.abs(v_0)))
        v_0 = v
        iteration += 1

    return v_0, iteration



def power_flow_tensor_constant_power(_K_,
                                     _L_,
                                     S_nom,
                                     v_0,
                                     ts_n,
                                     nb,
                                     iterations,
                                     tolerance
                                     ):

    iteration = 0
    tol = np.inf
    while (iteration < iterations) & (tol >= tolerance):
        v = np.zeros((ts_n, nb - 1), dtype="complex128")  # TODO: Test putting this outside of while loop
        for i in prange(ts_n):
            v[i] = (_K_ @ (np.conj(S_nom[i]) * (1 / np.conj(v_0[i]))).reshape(-1, 1) + _L_).T
        tol = np.max(np.abs(np.abs(v) - np.abs(v_0)))
        v_0 = v  # Voltage at load buses
        iteration += 1

    return v_0, iteration