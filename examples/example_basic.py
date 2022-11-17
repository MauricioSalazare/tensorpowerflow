from tensorpowerflow import GridTensor
import numpy as np
from time import perf_counter

#%% Solve base case (34 node bus)
network = GridTensor()
solution = network.run_pf()
print(solution["v"])

#%% Solve 10_000 power flows on the 34 node bus case.
network_size = network.nb - 1  # Size of network without slack bus.
active_ns = np.random.normal(50, scale=1, size=(10_000, network_size)) # Power in kW
reactive_ns = active_ns * np.tan(np.arccos(.99))  # kVAr, Constant PF of 0.99
solution_tensor = network.run_pf(active_power=active_ns, reactive_power=reactive_ns)
print(solution_tensor["v"])

#%% Generate random radial network of 100 nodes and a maximum of 1 to 3 branches per node.
network_rnd = GridTensor.generate_from_graph(nodes=100, child=3, plot_graph=True)
solution_rnd = network_rnd.run_pf()
print(solution_rnd["v"])

#%% Solve a tensor power flow. For 10 scenarios, 8_760 time steps (one year - 1 hr res), for the 33 PQ nodes.
# Meaning that the dimensions of the tensor is (10, 8_760, 33)

network = GridTensor(numba=True)  # Loads the basic 34 bus node network.
active_ns = np.random.normal(50,  # Power in kW
                             scale=10,
                             size=(10, 8_760, 33)).round(3)  # Assume 1 slack variable
reactive_ns = (active_ns * np.tan(np.arccos(.99))).round(3)  # kVAr, Constant PF of 0.99

start_tensor_dense = perf_counter()
solution = network.run_pf(active_power=active_ns, reactive_power=reactive_ns, algorithm="tensor")
t_tensor_dense = perf_counter() - start_tensor_dense
assert solution["convergence"], "Algorithm did not converge."
assert solution["v"].shape == active_ns.shape
print(f"Time tensor dense: {t_tensor_dense:.3f} sec.")

