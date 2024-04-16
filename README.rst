.. Binder
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/MauricioSalazare/tensorpowerflow/master?urlpath=lab/tree/examples
   :alt: binder

.. License
.. image:: https://img.shields.io/github/license/MauricioSalazare/tensorpowerflow
   :target: https://github.com/MauricioSalazare/tensorpowerflow/blob/master/LICENSE

.. Python versions supported
.. image:: https://img.shields.io/pypi/pyversions/tensorpowerflow.svg
   :target: https://pypi.python.org/pypi/tensorpowerflow/

.. Downloads per month
.. image:: https://img.shields.io/pypi/dm/tensorpowerflow.svg
   :target: https://pypi.python.org/pypi/tensorpowerflow/

.. Code size
.. image:: https://img.shields.io/github/languages/code-size/MauricioSalazare/tensorpowerflow
   :target: https://github.com/MauricioSalazare/tensorpowerflow

.. PyPi version
.. image:: https://img.shields.io/pypi/v/tensorpowerflow
   :target: https://pypi.python.org/pypi/tensorpowerflow/

.. Build (GithubActions)
.. image:: https://img.shields.io/github/workflow/status/MauricioSalazare/tensorpowerflow/Python%20package/master
   :target: https://github.com/MauricioSalazare/tensorpowerflow/actions

.. Test (GithubActions)
.. image:: https://img.shields.io/github/workflow/status/MauricioSalazare/tensorpowerflow/Python%20package/master?label=tests
   :target: https://github.com/MauricioSalazare/tensorpowerflow/actions




TensorPowerFlow
===============


What is TensorPowerFlow?
------------------------

An ultra-fast power flow based on a fixed-point iteration algorithm. The power flow is intended for applications where massive
amounts of power flow computations are required. e.g., electrical load time series, metaheuristics, electrical grid
environments for reinforcement learning.

How to install
--------------
The package can be installed via pip using:

.. code:: shell

    pip install tensorpowerflow

Example:
--------
Run the load base case as:

.. code-block:: python

    from tensorpowerflow import GridTensor
    import numpy as np
    from time import perf_counter

    #%% Solve base case (34 node bus)
    network = GridTensor(gpu_mode=False)
    solution = network.run_pf()
    print(solution["v"])

    #%% Solve 10_000 power flows on the 34 node bus case.
    network_size = network.nb - 1  # Size of network without slack bus.
    active_ns = np.random.normal(50, scale=1, size=(10_000, network_size)) # Power in kW
    reactive_ns = active_ns * 0.1  # kVAr
    solution_tensor = network.run_pf(active_power=active_ns, reactive_power=reactive_ns)
    print(solution_tensor["v"])

    #%% Generate random radial network of 100 nodes and a maximum of 3 branches per node.
    network_rnd = GridTensor.generate_from_graph(nodes=100, child=3, plot_graph=True, gpu_mode=False)
    solution_rnd = network_rnd.run_pf()
    print(solution_rnd["v"])

    #%% Solve a tensor power flow. For 10 scenarios, 8_760 time steps (one year - 1 hr res), for the 33 PQ nodes.
    # Meaning that the dimensions of the tensor is (10, 8_760, 33)

    network = GridTensor(numba=True, gpu_mode=False)  # Loads the basic 34 bus node network.
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

More examples can be found in the `examples folder <https://github.com/MauricioSalazare/tensorpowerflow/tree/master/examples>`_ (under development).
Also, you can try the package via jupyter lab clicking in the binder icon:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/MauricioSalazare/tensorpowerflow/master?urlpath=lab/tree/examples
   :alt: binder

Reading and citations:
----------------------
The mathematical formulation of the power flow can be found at:

*"Tensor Power Flow Formulations for Multidimensional Analyses in Distribution Systems."* E.M. Salazar Duque,
Juan S. Giraldo, Pedro P. Vergara, Phuong H. Nguyen, and Han (J.G.) Slootweg. 	arXiv:2403.04578 (2024). `link <https://arxiv.org/pdf/2403.04578>`_


How to contact us
-----------------
Any questions, suggestions or collaborations contact Juan S. Giraldo at <jnse@ieee.org>
