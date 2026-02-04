import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.optim.lr_scheduler import StepLR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义函数
def f(x):
    return 0.05*x[0]**2+x[1]**2
# 主流程
# 1. 参数X初始化
X=torch.tensor([-7.0,2.0],requires_grad=True)

# 2. 定义超参数
lr=0.9
num_iters=500

# 3. 定义优化器SGD
optimizer=torch.optim.SGD([X],lr=lr)

# 4.定义学习率衰减策略
lr_scheduler=ExponentialLR(optimizer,gamma=0.99)

# 拷贝X的值，放入列表
X_arr = X.detach().numpy().copy()
lr_list=[]
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
    lr_list.append(optimizer.param_groups[0]['lr'])
    # 5. 学习率衰减
    lr_scheduler.step()
# 画图
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, stop=7, num=100), np.linspace(-2, stop=2, num=100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
# 画等高线和点轨迹
ax[0].contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')
ax[0].plot(X_arr[:, 0], X_arr[:, 1], 'r')
ax[0].set_title('梯度下降过程')
# 画出学习率的变化
ax[1].plot(lr_list,'k')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('学习率衰减')
plt.show()




