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

An ultra-fast power flow based on Laurent series expansion. The power flow is intended for applications where massive
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

   #%% Solve base case (34 node bus)
   network = GridTensor()
   solution = network.run_pf_sequential()
   print(solution["v"])

   #%% Solve 10_000 power flows on the 34 node bus case.
   network_size = network.nb - 1  # Remove slack node
   active_ns = np.random.normal(50, scale=1, size=(10_000, network_size)) # Power in kW
   reactive_ns = active_ns * 0.1  # kVAr
   solution_tensor = network.run_pf_tensor(active_power=active_ns, reactive_power=reactive_ns)
   print(solution_tensor["v"])

   #%% Generate random radial network of 100 nodes and a maximum of 1 to 3 branches per node.
   network_rnd = GridTensor.generate_from_graph(nodes=100, child=3, plot_graph=True)
   solution_rnd = network_rnd.run_pf_sequential()
   print(solution_rnd["v"])

More examples can be found in the `examples folder <https://github.com/MauricioSalazare/tensorpowerflow/tree/master/examples>`_ (under development).
Also, you can try the package via jupyter lab clicking in the binder icon:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/MauricioSalazare/tensorpowerflow/master?urlpath=lab/tree/examples
   :alt: binder

Reading and citations:
----------------------
The mathematical formulation of the power flow can be found at:

*"A Fixed-Point Current Injection Power Flow for Electric Distribution Systems using Laurent Series."* J.S. Giraldo,
O.D. Montoya, P.P. Vergara, F. Milano. Power Systems Computational Conference (PSCC) 2022. `link <http://faraday1.ucd.ie/archive/papers/laurent.pdf>`_


How to contact us
-----------------
Any questions, suggestions or collaborations contact Juan S. Giraldo at <jnse@ieee.org>