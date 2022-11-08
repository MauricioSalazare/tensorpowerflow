import numpy as np
import warnings

#%%
try:
    aa = np.random.rand(int(1e8), int(1e8))
except MemoryError:
    warnings.warn("Memory gigant")

#%%
def test_func(a=1,b=2,c=3,d=4):
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")
    print(f"d: {d}")


kwargs = dict()
test_func(**kwargs)
