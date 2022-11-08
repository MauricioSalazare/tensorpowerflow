import numpy as np
from tensorpowerflow import GridTensor
from tensorpowerflow import numbarize

def test_tensor_shape():
    network = GridTensor()
    nb = network.nb

    active_power = network.P_file
    reactive_power = network.Q_file

    s_base = 1000
    alpha_Z = np.zeros(nb)
    alpha_I = np.zeros(nb)

    Yds = network.Yds
    Ydd = network.Ydd

    B_inv, C = numbarize.pre_power_flow_sam_sequential(active_power,
                                                       reactive_power,
                                                       s_base,
                                                       alpha_Z,
                                                       alpha_I,
                                                       Yds,
                                                       Ydd,
                                                       nb
                                                       )
    assert B_inv.shape == (33,33)
    assert C.shape == (33, 1)

def test_compute_power_flow_sam():
    network = GridTensor()
    nb = network.nb

    active_power = network.P_file
    reactive_power = network.Q_file

    s_base = 1000
    alpha_Z = np.zeros(nb)
    alpha_I = np.zeros(nb)

    Yds = network.Yds
    Ydd = network.Ydd

    B_inv, C, _ = numbarize.pre_power_flow_sam_sequential(active_power,
                                                       reactive_power,
                                                       s_base,
                                                       alpha_Z,
                                                       alpha_I,
                                                       Yds,
                                                       Ydd,
                                                       nb
                                                       )

    iterations = 100
    tolerance = 1e-6
    alpha_P = np.ones(nb-1)
    v_0 = np.ones((nb - 1, 1), dtype="complex128")

    active_power_pu = network.P_file / s_base  # Vector with all active power except slack
    reactive_power_pu = network.Q_file / s_base  # Vector with all reactive power except slack

    s_n = (active_power_pu + 1j * reactive_power_pu).reshape(-1, )

    iteration = 0
    tol = np.inf
    while (iteration < iterations) & (tol >= tolerance):
        # Update matrices A and D:
        A = np.diag(alpha_P * (1 / np.conj(v_0.flatten()) ** 2) * np.conj(s_n))
        D = 2 * alpha_P * (1 / np.conj(v_0.flatten())) * np.conj(s_n)
        D = D.reshape(-1,1)

        v = B_inv @ (A @ np.conj(v_0) - C - D)
        tol = np.max(np.abs(np.abs(v) - np.abs(v_0)))
        v_0 = v  # Voltage at load buses
        iteration += 1

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

    assert np.allclose(v_0.ravel(), v_solution)

