import torch
import numpy as np

#numpy-torch
# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
# print('\nnumpy',np_data,
#       '\ntorch',torch_data,
#       '\ntensor2array',tensor2array)

#data = [-1,-2,1,2]
#abs
# tensor = torch.FloatTensor(data)
# print('tensor',torch.abs(tensor),
#       'numpy',np.abs(tensor))

#sin
# tensor = torch.FloatTensor(data)
# print('tensor',torch.sin(tensor),
#       'numpy',np.sin(tensor))

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
data = np.array(data)
# print(
#     '\nnumpy',np.matmul(data,data),
#     '\ntorch',torch.mm(tensor,tensor)
# )
print(
     '\nnumpy',data.dot(data),
     '\ntorch',tensor.dot(tensor)
)

