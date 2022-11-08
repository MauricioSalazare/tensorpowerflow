import sys
print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
current_directory = Path().absolute()
print(f"Current directory: {current_directory}")
sys.path.extend([str(current_directory.parent.parent)])
# sys.path.extend(['C:\\Users\\20175334\\Documents\\PycharmProjects\\tensorpowerflow',
#                  'C:/Users/20175334/Documents/PycharmProjects/tensorpowerflow'])

import unittest
import numpy as np
import scipy.sparse as sp
from tensorpowerflow.pyMKL import pardisoSolver
from time import perf_counter
from scipy.sparse.linalg import spsolve

print(pardisoSolver.get_version())


#%%
nSize = 3000
A = sp.rand(nSize, nSize, 0.001, format='csr', random_state=100)
A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
A.data = A.data + 1j*np.random.rand(A.nnz)
A = A.tocsr()

np.random.seed(1)
xTrue = np.random.rand(nSize) + 1j*np.random.rand(nSize)
rhs = A.dot(xTrue)

start_init_pardiso = perf_counter()
pSolve = pardisoSolver(A, mtype=13)
pSolve.run_pardiso(12)
end_init_pardiso = perf_counter()
print(f"Init time: {(end_init_pardiso - start_init_pardiso) * 1000:.3f} msec")

start_pardiso = perf_counter()
x1 = pSolve.run_pardiso(33, rhs)
end_pardiso = perf_counter()
pSolve.clear()

print(f"Solve time: {(end_pardiso - start_pardiso) * 1000:.3f} msec")


start_time = perf_counter()
x2 = spsolve(A, rhs)
end_time = perf_counter()
print(f"SCIPY Total time: {(end_time - start_time) * 1000:.3f} msec")
# solver_times[n_size].update({"scipy": (end_time - start_time) * 1000})