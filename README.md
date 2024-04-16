[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MauricioSalazare/tensorpowerflow/master?urlpath=lab/tree/examples)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MauricioSalazare/tensorpowerflow/blob/master/LICENSE)
[![Python versions supported](https://img.shields.io/pypi/pyversions/tensorpowerflow.svg)](https://pypi.python.org/pypi/tensorpowerflow/)
[![Downloads per month](https://img.shields.io/pypi/dm/tensorpowerflow.svg)](https://pypi.python.org/pypi/tensorpowerflow/)
[![Code size](https://img.shields.io/github/languages/code-size/MauricioSalazare/tensorpowerflow)](https://github.com/MauricioSalazare/tensorpowerflow)
[![PyPI - Python Version](https://img.shields.io/pypi/v/tensorpowerflow)](https://pypi.python.org/pypi/tensorpowerflow/)
[![Build (GithubActions)](https://img.shields.io/github/workflow/status/MauricioSalazare/tensorpowerflow/Python%20package/master)](https://github.com/MauricioSalazare/tensorpowerflow/actions)
[![Test (GithubActions)](https://img.shields.io/github/workflow/status/MauricioSalazare/tensorpowerflow/Python%20package/master?label=tests)](https://github.com/MauricioSalazare/tensorpowerflow/actions)


# TensorPowerFlow

## What is TensorPowerFlow?
An ultra-fast power flow based on a fixed-point iteration algorithm. The power flow is intended for applications where massive
amounts of power flow computations are required. e.g., electrical load time series, metaheuristics, electrical grid
environments for reinforcement learning.

## How to install

The package can be installed via pip using:

```shell
pip install tensorpowerflow
```

## Example:

Here we show four examples of what can be done with the packages:
1. *Example 1:* The package comes with a preloaded grid case of 34 nodes that can be solved by default. 
2. *Example 2:* Solve 10.000 PF for the default case. The active power for the nodes is generated by sampling a Normal 
    distribution.
3. *Example 3:* The GridTensor class can generate a random grid using the `GridTensor.generate_from_graph()` method.
    The number of nodes and branching per node can be controlled by parameters `nodes` and `child`, respectively.
4. *Example 4:* Here we test the grid with a tensor of 3 dimensions. The first dimension is the number of scenarios 
    (10 in the example). The second dimension, the number of time steps (8.760 to simulate a 30 min. resolution 
    consumption for one year). The third dimension is the number of PQ nodes in the grid (33 PQ nodes for the based 
    grid case).

```python
from tensorpowerflow import GridTensor
import numpy as np
from time import perf_counter

#%% Example 1: Solve base case (34 node bus)
network = GridTensor(gpu_mode=False)
solution = network.run_pf()
print(solution["v"])

#%% Example 2: Solve 10_000 power flows on the 34 node bus case.
network_size = network.nb - 1  # Size of network without slack bus.
active_ns = np.random.normal(50, scale=1, size=(10_000, network_size)) # Power in kW
reactive_ns = active_ns * 0.1  # kVAr
solution_tensor = network.run_pf(active_power=active_ns, reactive_power=reactive_ns)
print(solution_tensor["v"])

#%% Example 3: Generate random radial network of 100 nodes and a maximum of 1 to 3 branches per node.
network_rnd = GridTensor.generate_from_graph(nodes=100, child=3, plot_graph=True)
solution_rnd = network_rnd.run_pf()
print(solution_rnd["v"])

#%% Example 4: Solve a tensor power flow. For 10 scenarios, 8_760 time steps (one year - 1 hr res), for the 33 PQ nodes.
# Meaning that the dimensions of the tensor is (10, 8_760, 33)

network = GridTensor(numba=True)  # Loads the basic 34 bus node network.
active_ns = np.random.normal(50,  # Power in kW
                             scale=10,
                             size=(10, 8_760, 33)).round(3)  # Assume 1 slack variable
reactive_ns = (active_ns * .1).round(3)  # Constant PF of 0.1

start_tensor_dense = perf_counter()
solution = network.run_pf(active_power=active_ns, reactive_power=reactive_ns, algorithm="tensor")
t_tensor_dense = perf_counter() - start_tensor_dense
assert solution["convergence"], "Algorithm did not converge."
assert solution["v"].shape == active_ns.shape
print(f"Time tensor dense: {t_tensor_dense:.3f} sec.")


```

More examples can be found in the [examples folder](examples) (under development).
Also, you can try the package via jupyter lab clicking in the binder icon:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MauricioSalazare/tensorpowerflow/master?urlpath=lab/tree/examples)

## Reading and citations:

The mathematical formulation of the power flow can be found at:

> *"Tensor Power Flow Formulations for Multidimensional Analyses in Distribution Systems."* E.M. Salazar Duque,
Juan S. Giraldo, Pedro P. Vergara, Phuong H. Nguyen, and Han (J.G.) Slootweg. [arXiv:2403.04578 (2024)](https://arxiv.org/pdf/2403.04578).

## How to contact us

Any questions, suggestions or collaborations contact Juan S. Giraldo at <jnse@ieee.org>
