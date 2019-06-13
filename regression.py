import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#fake data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())


x,y = Variable(x),Variable(y)       #变量化
# plt.scatter(x.data, y.data)
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)#黑盒子里每个神经元处理的输入量和输出量
        self.predict = torch.nn.Linear(n_hidden,n_output)#输出神经元处理的输入量和输出量

    def forward(self,x):
        x = F.relu(self.hidden(x))      #将输入数据传入黑盒神经元并进行激励
        x = self.predict(x)             #输出数据
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

#优化模型
optimizer = torch.optim.SGD(net.parameters(),lr=0.2)
loss_func = torch.nn.MSELoss()
plt.ion()
# plt.show()


for t in range(200):
    prediction = net(x)#预测
    loss = loss_func(prediction,y)#计算误差
    optimizer.zero_grad()#梯度清零
    loss.backward()#反向传播
    optimizer.step()#前向传播

    if t % 5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5, 0, 'Loss=%.4f'% loss.item(), fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()



