from tensorpowerflow import GridTensor
import numpy as np

network = GridTensor(numba=False)
V1 = network.run_pf_hp_laurent()
# V2 = network.run_pf_sequential()