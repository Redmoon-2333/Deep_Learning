import torch
import matplotlib.pyplot as plt

x=torch.linspace(-5,5,1000,requires_grad=True)
y=x.relu()

# 画图
fig,ax=plt.subplots(1,2,figsize=(12,5))
# 原函数图像
ax[0].plot(x.data,y.data,"purple")
ax[0].set_xlabel("x",fontsize=12)
ax[0].set_ylabel("y",fontsize=12)
ax[0].set_title("ReLU Function",fontsize=14)
ax[0].axhline(y=-1,color="gray",alpha=0.5)
ax[0].axhline(y=1,color="gray",alpha=0.5)

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["left"].set_position("zero")
ax[0].spines["bottom"].set_position("zero")


# 反向传播，计算x的梯度
y.sum().backward()

# 导数图像
ax[1].plot(x.data,x.grad,"orange")
ax[1].set_xlabel("x",fontsize=12)
ax[1].set_ylabel("dy/dx",fontsize=12)
ax[1].set_title("Derivative of ReLU Function",fontsize=14)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["left"].set_position("zero")
ax[1].spines["bottom"].set_position("zero")


plt.show()










