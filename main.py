from tensorpowerflow import GridTensor
import numpy as np


#%% Solve base case
network = GridTensor()
solution = network.run_pf_sequential()

#%% Solve tensor
network_size = network.nb - 1  # Remove slack node
active_ns = np.random.normal(50, scale=1, size=(10_000, network_size))
reactive_ns = active_ns * 0.1
solution_tensor = network.run_pf_tensor(active_power=active_ns, reactive_power=reactive_ns)

#%% Generate random network
network_rnd = GridTensor.generate_from_graph(nodes=100, child=3, plot_graph=True)
solution_rnd = network_rnd.run_pf_sequential()
print(solution_rnd["v"])