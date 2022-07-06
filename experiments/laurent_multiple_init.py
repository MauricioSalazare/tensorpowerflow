from tensorpowerflow import GridTensor
import numpy as np
import matplotlib.pyplot as plt

#%%
network_t = GridTensor(node_file_path="grid_data/Nodes_2.csv",
                       lines_file_path="grid_data/Lines_2.csv",
                       s_base=1000,  # kVA - 1 phase
                       v_base=1)
V1 = network_t.run_pf()

print(f"Convergence: {V1['convergence']}")
print(f"Iterations: {V1['iterations']}")
print(f"Volt. mag: {np.abs(V1['v'])}")

v_pu_sol = 0.9209
ang = -5.2335

second_v_solution = 0.1180
ang = -45.3858

inv_z = np.abs(1/network_t.Ydd)
eta = np.abs(inv_z * (network_t.P_file / 1000 + 1j* network_t.Q_file / 1000) /  V1['v'] ** 2)
print(f"Eta: {eta}")

#%%

init_nr_jodido = np.array([3.0 + 1j*3])

init = [np.array([0.5 + 1j*0])/2,
        np.array([-0.311 + 1j*1.0])/2,
        np.array([-3.0 - 1j*3])/2,
        np.array([-3.0 + 1j*3])/2,
        np.array([3.0 - 1j*3])/2,
        np.array([0.69 - 1j * 0.11]),
        np.array([0.5 - 1j * 0.1]),
        np.array([0.39 - 1j * 0.33]),
        np.array([0.47 - 1j * 0.21]),
        np.array([0.5 - 1j * 0.5]),
        V1['v']*0.999]

soluciones_list = []

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for init_ in init:
    V1 = network_t.run_pf(flat_start=False, start_value=init_)
    print(f"Convergence: {V1['convergence']}")
    print(f"Iterations: {V1['iterations']}")
    print(f"Volt. mag: {np.abs(V1['v'])}")
    print(V1["voltages_iterations"])
    soluciones = np.array(V1["voltages_iterations"])
    ax.plot(soluciones.real, soluciones.imag, "-o")
    ax.scatter(init_.real, init_.imag, color='r', marker="x", zorder=3)

#%%
plt.figure()
plt.quiver([0, 0, 3], [0, 0, 3], [1, -2, 4], [1, 2, -7], angles='xy', scale_units='xy', scale=1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()