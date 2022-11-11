from tensorpowerflow import GridTensor
import numpy as np


# Solution of the base test case (34-bus system):
V_SOLUTION = [0.98965162+0.00180549j, 0.98060256+0.00337785j, 0.96828145+0.00704551j,
              0.95767051+0.01019764j, 0.94765203+0.01316654j, 0.94090964+0.01600068j,
              0.93719984+0.01754998j, 0.93283877+0.01937559j, 0.93073823+0.02026054j,
              0.9299309 +0.02058985j, 0.92968994+0.02068728j, 0.98003142+0.00362498j,
              0.97950885+0.00385019j, 0.97936712+0.00391065j, 0.97935604+0.0039148j,
              0.93971131+0.01547898j, 0.93309482+0.01739656j, 0.92577912+0.01988823j,
              0.91988489+0.02188907j, 0.91475251+0.02362566j, 0.90888169+0.02596304j,
              0.90404908+0.02788248j, 0.89950353+0.02968449j, 0.89731375+0.03055177j,
              0.89647201+0.03088507j, 0.89622055+0.03098473j, 0.94032081+0.01625577j,
              0.93992817+0.01642583j, 0.93973182+0.01651086j, 0.9301316+0.02052908j,
              0.92952481+0.02079761j, 0.92922137+0.02093188j, 0.92912022+0.02097663j]

def test_compute_correct_answer_gpu_tensor():
    network = GridTensor()
    V = network.run_pf(algorithm="gpu-tensor")
    assert np.allclose(V["v"], np.array(V_SOLUTION).astype(np.complex64))

def test_compute_correct_answer_sparse_pardiso():
    network = GridTensor()
    V = network.run_pf(algorithm="hp", sparse_solver="pardiso")
    assert np.allclose(V["v"], V_SOLUTION)



if __name__ == "__main__":
    test_compute_correct_answer_gpu_tensor()
    test_compute_correct_answer_sparse_pardiso()
