# 导入必要的库
import torch  # PyTorch深度学习框架
import matplotlib.pyplot as plt  # 绘图库

from torch import nn  # 神经网络模块：包含各种层和损失函数
from torch import optim  # 优化器模块：包含SGD、Adam等优化算法

from torch.utils.data import TensorDataset, DataLoader # 数据处理模块

# 设置matplotlib中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 使用楷体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 1. 构建数据
# 生成100个样本的线性回归数据集
X=torch.randn(100,1)  # 生成100个服从标准正态分布的随机数作为输入特征
# 定义真实的线性关系参数
w=torch.tensor([2.5])  # 真实权重：斜率
b=torch.tensor([5.2])  # 真实偏置：截距
# 添加高斯噪声模拟真实数据的不确定性
noise=torch.randn(100,1)*0.5  # 生成噪声并缩放(标准差为0.5)
y=X*w+b+noise  # 根据线性模型 y = wx + b + noise 生成目标值

# 构建数据集和数据加载器
# TensorDataset将特征和标签打包成数据集
# 每个样本包含一对(X[i], y[i])
dataset=TensorDataset(X,y)
# DataLoader提供批量加载和数据打乱功能
# batch_size=10: 每次训练使用10个样本
# shuffle=True: 每个epoch随机打乱数据顺序，提高训练效果
dataloader=DataLoader(dataset,batch_size=10,shuffle=True)

# 2. 构建模型
# 使用PyTorch内置的线性层nn.Linear
# in_features=1: 输入特征维度为1
# out_features=1: 输出维度为1
# 模型形式: y = wx + b，其中w和b是待学习的参数
model=nn.Linear(in_features=1,out_features=1)
# 3. 定义损失函数和优化器
# 损失函数：均方误差(Mean Squared Error)
# 衡量预测值与真实值之间的平方差
loss=nn.MSELoss()
# 优化器：随机梯度下降(SGD)
# model.parameters(): 获取模型中所有可学习参数(w和b)
# lr=0.001: 学习率，控制参数更新的步长
optimizer=optim.SGD(model.parameters(),lr=0.001)
# 4. 训练模型
epoch_num=1000  # 训练轮次：遍历完整数据集的次数
loss_list=[]  # 记录每个epoch的平均损失值，用于绘制损失曲线

# 开始训练循环
for epoch in range(epoch_num):
    # 一个训练轮次(epoch)的迭代过程
    total_loss=0  # 累计本轮次所有批次的总损失
    
    # 遍历每个批次的数据
    for x_train,y_train in dataloader:
        # 前向传播：输入数据通过模型得到预测值
        y_pred=model(x_train)
        
        # 计算损失：比较预测值y_pred与真实值y_train
        loss_value=loss(y_pred,y_train)
        
        # 反向传播：自动计算梯度
        # PyTorch的autograd机制会自动计算loss对所有requires_grad=True参数的梯度
        loss_value.backward()
        
        # 更新参数：根据梯度和学习率调整模型参数
        # w = w - lr * dw, b = b - lr * db
        optimizer.step()
        
        # 梯度清理：清空参数的梯度缓存
        # 必须在每次迭代后清零，否则梯度会累积
        optimizer.zero_grad()
        
        # 累计损失：loss_value.item()获取标量值
        # y_train.size(0)获取当前批次的样本数
        total_loss+=loss_value.item()*y_train.size(0)

    # 计算本轮次的平均损失并记录
    avg_loss = total_loss/len(dataset)
    loss_list.append(avg_loss)
    # 每轮次打印训练进度和损失值
    print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {avg_loss:.4f}')

# 5. 打印训练后的模型参数
# model.weight: 线性层的权重参数(形状为[1,1])
# model.bias: 线性层的偏置参数(形状为[1])
# .item()将单元素张量转换为Python标量
print(f'训练得到的参数 - 权重w: {model.weight.item():.4f}, 偏置b: {model.bias.item():.4f}')
print(f'真实参数 - 权重w: 2.5000, 偏置b: 5.2000')

# 6. 可视化训练结果
# 创建1行2列的子图
fig,ax=plt.subplots(1,2,figsize=(12,5))

# 左图：绘制训练损失曲线
ax[0].plot(loss_list, linewidth=2)
ax[0].set_xlabel('训练轮次(Epoch)', fontsize=12)
ax[0].set_ylabel('损失值(Loss)', fontsize=12)
ax[0].set_title('训练损失收敛曲线', fontsize=14)
ax[0].grid(True, alpha=0.3)  # 添加网格线

# 右图：绘制数据散点图和拟合直线
ax[1].scatter(X,y, alpha=0.6, label='训练数据')  # 原始数据点
# 使用训练得到的参数绘制拟合直线
y_pred=model.weight.item()*X+model.bias.item()
ax[1].plot(X,y_pred,'r-', linewidth=2, label=f'拟合直线: y={model.weight.item():.2f}x+{model.bias.item():.2f}')
ax[1].set_xlabel('输入特征 X', fontsize=12)
ax[1].set_ylabel('目标值 y', fontsize=12)
ax[1].set_title('线性回归拟合结果', fontsize=14)
ax[1].legend()  # 显示图例
ax[1].grid(True, alpha=0.3)

# 调整布局并显示图形
plt.tight_layout()
plt.show()











