import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adagrad

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义函数
def f(x):
    return 0.05*x[0]**2+x[1]**2

# 定义函数，实现梯度下降法
def gradient_descent(X,optimizer,num_iters):
    # 拷贝X的值，放入列表
    X_arr = X.detach().numpy().copy()
    for i in range(num_iters):
        # 1. 前向传播（得到损失值）
        y = f(X)
        # 2. 反向传播（得到梯度）
        y.backward()
        # 3. 参数更新
        optimizer.step()
        # 4. 梯度清零
        optimizer.zero_grad()

        # 保存X
        X_arr = np.vstack((X_arr,X.detach().numpy()))
    return X_arr

# 主流程
# 1. 参数初始化
X=torch.tensor([-7.0,2.0],requires_grad=True)

# 2. 定义超参数
lr=0.1
num_iters=100

# 3. 优化器对比
# 3.1 SGD
X_clone=X.clone().detach().requires_grad_(True)
optimizer=torch.optim.SGD([X_clone],lr=lr)
# 梯度下降
X_arr1=gradient_descent(X_clone,optimizer,num_iters)
# 画图
plt.plot(X_arr1[:,0],X_arr1[:,1],'r',label='SGD')

# 3.2 RMSprop
X_clone=X.clone().detach().requires_grad_(True)
optimizer_rmsprop=torch.optim.RMSprop([X_clone],lr=lr,alpha=0.9)
X_arr2=gradient_descent(X_clone,optimizer_rmsprop,num_iters)

plt.plot(X_arr2[:,0],X_arr2[:,1],'b',label='RMSprop')

# 等高线
x1_grid,x2_grid = np.meshgrid(np.linspace(-7,7,100),np.linspace(-2,2,100))
y_grid=0.05*x1_grid**2+x2_grid**2
plt.contour(x1_grid,x2_grid,y_grid,levels=30,colors='gray')

plt.legend()
plt.show()












