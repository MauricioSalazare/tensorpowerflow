import numpy as np


def reshape_tensor(tensor_array):
    original_shape = tensor_array.shape
    tau = np.prod(original_shape[:-1])

    # # This makes a copy
    # reshaped = tensor_array.reshape((tau, original_shape[-1]))

    # This reshapes in place
    tensor_array.shape = (tau, original_shape[-1])

    return tensor_array, original_shape




aa1 = np.arange(1,21).reshape((5,4))
aa2 = aa1.copy() + 20

bb1 = np.stack([aa1,aa2]) # 3D
bb2 = bb1.copy() + 40

cc1 = np.concatenate([bb1,bb2], axis=0)  #3D
cc2 = cc1.copy() + 80

dd1 = np.stack([cc1, cc2])  # 4D


final_tensor = dd1.copy()

if final_tensor.ndim > 2:
    reshaped, original_shape = reshape_tensor(tensor_array=final_tensor)
else:
    reshaped = final_tensor

if final_tensor.ndim > 2:
    original_shape.shape = original_shape
