import torch
from torch.autograd import Variable

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
variable = Variable(tensor,requires_grad=True)

# print(tensor)
# print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
# print(t_out)
# print(v_out)
v_out.backward()

print(variable.grad)
print(variable.data)
print(variable.data.numpy())