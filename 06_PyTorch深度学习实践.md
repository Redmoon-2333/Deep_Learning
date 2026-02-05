# 第6章 PyTorch深度学习实践

## 本章导读

在前一章掌握了PyTorch核心概念的基础上，本章将深入探讨如何使用PyTorch构建和训练深度学习模型。我们将从激活函数、参数初始化、网络构建、损失函数到优化算法，系统地介绍深度学习实践中的关键技术。

**学习目标**：
- 理解各种激活函数的特点和适用场景
- 掌握参数初始化方法和正则化技术
- 学会使用PyTorch构建神经网络模型
- 理解不同损失函数的选择原则
- 掌握各种优化算法的原理和应用

**学习路线**：
```
激活函数 → 参数初始化 → 网络构建 → 损失函数 → 优化算法 → 实战应用
(非线性变换) (权重初始化) (模型搭建)   (误差衡量)  (参数更新)  (房价预测)
```

**核心概念**：
- 激活函数：引入非线性，增强模型表达能力
- 参数初始化：影响训练稳定性和收敛速度
- 正则化：防止过拟合，提高泛化能力
- 损失函数：衡量预测与真实值的差距
- 优化算法：高效更新模型参数

---

## 6.1 激活函数

激活函数是神经网络中引入非线性变换的关键组件。没有激活函数，多层神经网络将等价于单层线性模型，无法学习复杂的非线性关系。

### 6.1.1 Sigmoid函数

Sigmoid函数将输入映射到(0,1)区间，常用于二分类问题的输出层。

**数学定义**：
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**特点**：
- 输出范围：(0, 1)
- 平滑可导
- 存在梯度消失问题（当|x|较大时，梯度趋近于0）
- 输出不以0为中心

**代码实现**（来自 `activation_functions/1_sigmoid.py`）：

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10, 1000, requires_grad=True)
y = x.sigmoid()

# 画图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# 原函数图像
ax[0].plot(x.data, y.data, "purple")
ax[0].set_xlabel("x", fontsize=12)
ax[0].set_ylabel("y", fontsize=12)
ax[0].set_title("Sigmoid Function", fontsize=14)
ax[0].axhline(y=0.5, color="gray", alpha=0.5)
ax[0].axhline(y=1, color="gray", alpha=0.5)

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["left"].set_position("zero")
ax[0].spines["bottom"].set_position("zero")

# 反向传播，计算x的梯度
y.sum().backward()

# 导数图像
ax[1].plot(x.data, x.grad, "orange")
ax[1].set_xlabel("x", fontsize=12)
ax[1].set_ylabel("dy/dx", fontsize=12)
ax[1].set_title("Derivative of Sigmoid Function", fontsize=14)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["left"].set_position("zero")
ax[1].spines["bottom"].set_position("zero")

plt.show()
```

### 6.1.2 Tanh函数

Tanh函数将输入映射到(-1,1)区间，输出以0为中心，相比Sigmoid具有更好的梯度特性。

**数学定义**：
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**特点**：
- 输出范围：(-1, 1)
- 以0为中心，有利于梯度下降
- 仍存在梯度消失问题
- 常用于循环神经网络（RNN）

**代码实现**（来自 `activation_functions/2_tanh.py`）：

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 1000, requires_grad=True)
y = x.tanh()

# 画图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# 原函数图像
ax[0].plot(x.data, y.data, "purple")
ax[0].set_xlabel("x", fontsize=12)
ax[0].set_ylabel("y", fontsize=12)
ax[0].set_title("Tanh Function", fontsize=14)
ax[0].axhline(y=-1, color="gray", alpha=0.5)
ax[0].axhline(y=1, color="gray", alpha=0.5)

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["left"].set_position("zero")
ax[0].spines["bottom"].set_position("zero")

# 反向传播，计算x的梯度
y.sum().backward()

# 导数图像
ax[1].plot(x.data, x.grad, "orange")
ax[1].set_xlabel("x", fontsize=12)
ax[1].set_ylabel("dy/dx", fontsize=12)
ax[1].set_title("Derivative of Tanh Function", fontsize=14)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["left"].set_position("zero")
ax[1].spines["bottom"].set_position("zero")

plt.show()
```

### 6.1.3 ReLU函数

ReLU（Rectified Linear Unit）是目前最常用的激活函数，因其简单高效而广受欢迎。

**数学定义**：
$$
\text{ReLU}(x) = \max(0, x)
$$

**特点**：
- 计算简单，收敛速度快
- 缓解梯度消失问题（正区间梯度为1）
- 存在"神经元死亡"问题（负区间梯度为0）
- 常用于卷积神经网络（CNN）和全连接层

**代码实现**（来自 `activation_functions/3_relu.py`）：

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 1000, requires_grad=True)
y = x.relu()

# 画图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# 原函数图像
ax[0].plot(x.data, y.data, "purple")
ax[0].set_xlabel("x", fontsize=12)
ax[0].set_ylabel("y", fontsize=12)
ax[0].set_title("ReLU Function", fontsize=14)
ax[0].axhline(y=-1, color="gray", alpha=0.5)
ax[0].axhline(y=1, color="gray", alpha=0.5)

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["left"].set_position("zero")
ax[0].spines["bottom"].set_position("zero")

# 反向传播，计算x的梯度
y.sum().backward()

# 导数图像
ax[1].plot(x.data, x.grad, "orange")
ax[1].set_xlabel("x", fontsize=12)
ax[1].set_ylabel("dy/dx", fontsize=12)
ax[1].set_title("Derivative of ReLU Function", fontsize=14)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["left"].set_position("zero")
ax[1].spines["bottom"].set_position("zero")

plt.show()
```

### 6.1.4 Softmax函数

Softmax函数将输入转换为概率分布，常用于多分类问题的输出层。

**数学定义**：
$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

**特点**：
- 输出范围：(0, 1)，且和为1
- 放大差异（指数函数特性）
- 常用于多分类问题的最后一层
- 与交叉熵损失函数配合使用

**代码实现**（来自 `activation_functions/4_softmax.ipynb`）：

```python
import matplotlib.pyplot as plt
import torch

x = torch.randn(3, 5)
print(x)

y = torch.softmax(x, dim=1)
print(y)
```

**激活函数选择建议**：

| 激活函数 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| Sigmoid | 二分类输出层 | 输出概率 | 梯度消失 |
| Tanh | 循环神经网络 | 零中心化 | 梯度消失 |
| ReLU | 隐藏层默认选择 | 计算快、缓解梯度消失 | 神经元死亡 |
| Softmax | 多分类输出层 | 概率分布 | 计算量大 |

---

## 6.2 参数初始化和正则化

良好的参数初始化对神经网络的训练至关重要。不恰当的初始化可能导致梯度消失或爆炸，使模型难以收敛。

### 6.2.1 全连接层（nn.Linear）

PyTorch中使用`nn.Linear`定义全连接层（也称为线性层或密集层）。

**数学定义**：
$$
y = xW^T + b
$$

其中：
- $x$：输入特征，形状为(batch_size, in_features)
- $W$：权重矩阵，形状为(out_features, in_features)
- $b$：偏置向量，形状为(out_features,)
- $y$：输出，形状为(batch_size, out_features)

**代码示例**（来自 `1_init.ipynb`）：

```python
import torch
import torch.nn as nn

# 定义全连接层
linear = nn.Linear(5, 2)
```

### 6.2.2 常数初始化

常数初始化将所有参数设置为相同的常数值。这种方法简单但不推荐用于训练，因为会破坏对称性。

**代码实现**（来自 `1_init.ipynb`）：

```python
# 常数初始化
nn.init.zeros_(linear.weight)
print(linear.weight)

nn.init.ones_(linear.weight)
print(linear.weight)

nn.init.constant_(linear.weight, 10)
print(linear.weight)
```

### 6.2.3 秩初始化

秩初始化使用单位矩阵初始化权重，适用于输入输出维度相同的情况。

**代码实现**（来自 `1_init.ipynb`）：

```python
# 秩初始化
nn.init.eye_(linear.weight)
print(linear.weight)
```

### 6.2.4 正态分布初始化

从正态分布（高斯分布）中采样初始化权重。

**代码实现**（来自 `1_init.ipynb`）：

```python
# 正态分布初始化
nn.init.normal_(linear.weight, mean=0, std=0.01)
print(linear.weight)
```

### 6.2.5 均匀分布初始化

从均匀分布中采样初始化权重。

**代码实现**（来自 `1_init.ipynb`）：

```python
# 均匀分布初始化
nn.init.uniform_(linear.weight, a=-3, b=3)
print(linear.weight)
```

### 6.2.6 Xavier初始化（Glorot初始化）

Xavier初始化根据输入输出维度自适应地调整初始化范围，适用于Sigmoid和Tanh激活函数。

**原理**：
- 保持前向和反向传播中梯度的方差一致
- 权重从均匀分布或正态分布中采样

**代码实现**（来自 `1_init.ipynb`）：

```python
# Xavier初始化
nn.init.xavier_uniform_(linear.weight)
print(linear.weight)

nn.init.xavier_normal_(linear.weight)
print(linear.weight)
```

### 6.2.7 He初始化（Kaiming初始化）

He初始化专为ReLU激活函数设计，考虑了ReLU的负值截断特性。

**原理**：
- 针对ReLU激活函数的改进初始化方法
- 使用更大的初始化方差

**代码实现**（来自 `1_init.ipynb`）：

```python
# He初始化
nn.init.kaiming_uniform_(linear.weight, a=0)
print(linear.weight)

nn.init.kaiming_normal_(linear.weight, a=0.01)
print(linear.weight)
```

**初始化方法选择建议**：

| 初始化方法 | 适用激活函数 | 特点 |
|-----------|-------------|------|
| Xavier | Sigmoid、Tanh | 保持梯度方差 |
| He | ReLU、LeakyReLU | 考虑负值截断 |
| 正态/均匀 | 通用 | 需要手动设置范围 |

### 6.2.8 Dropout随机失活

Dropout是一种有效的正则化技术，通过在训练时随机丢弃部分神经元来防止过拟合。

**原理**：
- 训练时以概率$p$随机丢弃神经元
- 测试时使用所有神经元，输出按比例缩放
- 相当于训练多个子网络的集成

**代码实现**（来自 `2_dropout.ipynb`）：

```python
import torch
import torch.nn as nn

x = torch.randint(1, 10, (10,), dtype=torch.float32)

# 定义一个dropout层
dropout = nn.Dropout(0.5)

y = dropout(x)
print(x)
print(y)
```

**使用建议**：
- Dropout概率通常设置为0.2-0.5
- 仅在训练时使用，测试时自动关闭
- 常用于全连接层，卷积层使用较少

---

## 6.3 搭建神经网络

PyTorch提供了灵活的方式来构建神经网络，包括自定义模型类和使用Sequential容器。

### 6.3.1 自定义模型

通过继承`nn.Module`类，可以定义自己的神经网络模型。

**代码实现**（来自 `3_nn_test.py`）：

```python
import torch
import torch.nn as nn

# 自定义神经网络类
class Model(nn.Module):
    # 初始化方法
    def __init__(self, device):
        super().__init__()
        # 定义三个线性层
        self.linear1 = nn.Linear(3, 4, device=device)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(4, 4, device=device)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.out = nn.Linear(4, 2, device='cuda')
    
    # 前向传播
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        
        x = self.linear2(x)
        x = torch.relu(x)
        
        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x

# 全局变量device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试
# 1. 定义输入数据
x = torch.randn(10, 3, device=device)

# 2. 创建模型
model = Model(device=device)

# 3. 前向传播
output = model(x)
print("神经网络输出为:", output)

# 调用 parameters() 方法
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param)

# 调用state_dict() 方法，得到模型的参数
print()
print(model.state_dict())

from torchsummary import summary
summary(model, (3,), batch_size=10, device='cuda')
```

### 6.3.2 查看模型结构和参数数量

PyTorch提供了多种方法来查看模型结构和参数信息。

**常用方法**：
- `model.parameters()`：返回模型所有可学习参数
- `model.named_parameters()`：返回带名称的参数
- `model.state_dict()`：返回模型状态字典（包含所有参数和缓冲区）
- `torchsummary.summary()`：可视化模型结构和参数量

### 6.3.3 使用Sequential构建模型

`nn.Sequential`提供了一种简洁的方式来构建顺序执行的神经网络。

**代码实现**（来自 `4_nn_sequential.py`）：

```python
import torch
import torch.nn as nn
from torchsummary import summary

# 1. 定义数据
x = torch.randn(10, 3)

# 2. 构建模型
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1)
)

# 3. 参数初始化
def init_params(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.1)

model.apply(init_params)

# 4. 前向传播
output = model(x)
print("神经网络输出为:", output)

summary(model, (3,), batch_size=10, device='cpu')
```

**Sequential vs 自定义类**：

| 特性 | Sequential | 自定义类 |
|------|-----------|---------|
| 简洁性 | 高 | 中 |
| 灵活性 | 低（仅顺序执行） | 高（任意结构） |
| 适用场景 | 简单网络 | 复杂网络、多分支 |
| 前向传播 | 自动 | 手动定义 |

---

## 6.4 损失函数

损失函数衡量模型预测值与真实值之间的差距，是指导模型学习的核心指标。

### 6.4.1 分类任务损失函数

#### 二元交叉熵损失（BCELoss）

用于二分类问题，衡量预测概率与真实标签之间的差异。

**数学定义**：
$$
L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \cdot \log(\hat{y}_i) + (1-y_i) \cdot \log(1-\hat{y}_i)]
$$

**代码实现**（来自 `loss_functions/1_bce.ipynb`）：

```python
import torch
import torch.nn as nn

# 输入数据
input = torch.randn(3, 2)
print(input)

# 得到预测概率
pred = torch.sigmoid(input)
print(pred)

# 目标值
target = torch.tensor([[0, 1], [1, 0], [0, 1]]).float()

# 定义损失函数
loss = nn.BCELoss()
print(loss(pred, target))
```

#### 交叉熵损失（CrossEntropyLoss）

用于多分类问题，结合了LogSoftmax和NLLLoss。

**代码实现**（来自 `loss_functions/2_cross_entropy.ipynb`）：

```python
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

input = torch.randn(6, 8)
print(input)

# 目标值，六个数据的类型标签
target = torch.tensor([1, 0, 3, 7, 5, 2])
print(target)

# 定义损失函数
loss = CrossEntropyLoss()
print(loss(input, target))
```

**注意**：`CrossEntropyLoss`的输入是原始logits（未经过softmax），内部会自动应用softmax。

### 6.4.2 回归任务损失函数

#### 均方误差损失（MSELoss）

用于回归问题，计算预测值与真实值之间差的平方的平均值。

**数学定义**：
$$
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

#### 平均绝对误差损失（L1Loss）

计算预测值与真实值之间绝对差的平均值，对离群点更鲁棒。

**数学定义**：
$$
L = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

#### Smooth L1损失

结合MSE和MAE的优点，在误差较小时使用MSE，误差较大时使用MAE。

**代码实现**（来自 `loss_functions/3_regression_loss.ipynb`）：

```python
import torch
import torch.nn as nn

input = torch.randn(3, 5)
target = torch.randn(3, 5)
print(input)
print(target)

# MSE: L2 loss
mse_loss = nn.MSELoss()
print(mse_loss(input, target))

# MAE: L1 Loss
mae_loss = nn.L1Loss()
print(mae_loss(input, target))

# Smooth L1
smooth_l1_loss = nn.SmoothL1Loss()
print(smooth_l1_loss(input, target))
```

**损失函数选择建议**：

| 任务类型 | 推荐损失函数 | 说明 |
|---------|-------------|------|
| 二分类 | BCELoss | 配合Sigmoid使用 |
| 多分类 | CrossEntropyLoss | 内部包含Softmax |
| 回归 | MSELoss | 对离群点敏感 |
| 回归（鲁棒） | L1Loss/SmoothL1Loss | 对离群点鲁棒 |

**完整训练流程示例**（来自 `loss_functions/4_loss_test.py`）：

```python
import torch
import torch.nn as nn

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 只定义一个全连接层
        self.linear = nn.Linear(in_features=5, out_features=3)
        # 权重初始化
        self.linear.weight.data = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.2, 0.3, 0.2, 0.1, 0.5]
        ])
        self.linear.bias.data = torch.tensor([0.1, 0.2, 0.3])
    
    # 前向传播
    def forward(self, x):
        return self.linear(x)

# 主流程
# 1. 定义数据 2*5
x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=torch.float)

# 目标值 2*3
y = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float)

# 2. 模型定义
model = Model()

# 3. 损失函数定义
criterion = nn.MSELoss()

# 4. 损失计算
loss = criterion(model(x), y)

# 5. 反向传播，计算梯度
loss.backward()

# 6. 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 7. 参数更新
optimizer.step()
optimizer.zero_grad()

# 8. 输出参数
for param in model.state_dict():
    print(param)
    print(model.state_dict()[param])
```

---

## 6.5 参数更新方法

优化算法决定了如何根据梯度更新模型参数，不同的优化算法具有不同的收敛特性和适用场景。

### 6.5.1 Momentum

动量法通过引入速度变量，累积历史梯度方向，加速收敛并减少震荡。

**原理**：
- 模拟物理中的动量概念
- 在梯度方向一致时加速，不一致时减速
- 有助于逃离局部最优和鞍点

**代码实现**（来自 `optimizer/1_momentum.py`）：

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(X):
    return 0.05 * X[0]**2 + X[1]**2

# 定义函数，实现梯度下降法
def gradient_descent(X, optimizer, num_iters):
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
        X_arr = np.vstack((X_arr, X.detach().numpy()))
    print(X_arr)
    return X_arr

# 主流程
# 1. 参数初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.01
num_iters = 1000

# 3. 优化器对比
# 3.1 SGD
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr)
# 梯度下降
X_arr1 = gradient_descent(X_clone, optimizer, num_iters)
# 画图
plt.plot(X_arr1[:, 0], X_arr1[:, 1], 'r', label='SGD')

# 3.2 动量法
X_clone = X.clone().detach().requires_grad_(True)
optimizer_momentum = torch.optim.SGD([X_clone], lr=lr, momentum=0.9)
X_arr2 = gradient_descent(X_clone, optimizer_momentum, num_iters)
plt.plot(X_arr2[:, 0], X_arr2[:, 1], 'b', label='Momentum')

# 等高线
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
plt.contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')

plt.legend()
plt.show()
```

### 6.5.2 学习率衰减

学习率衰减策略在训练过程中逐渐降低学习率，有助于模型更好地收敛。

#### StepLR

每隔固定步数，将学习率乘以衰减系数。

**代码实现**（来自 `optimizer/2_step_lr.py`）：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义函数
def f(x):
    return 0.05 * x[0]**2 + x[1]**2

# 主流程
# 1. 参数X初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.9
num_iters = 500

# 3. 定义优化器SGD
optimizer = torch.optim.SGD([X], lr=lr)

# 4. 定义学习率衰减策略
lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.7)

# 拷贝X的值，放入列表
X_arr = X.detach().numpy().copy()
lr_list = []
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
    X_arr = np.vstack((X_arr, X.detach().numpy()))
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
ax[1].plot(lr_list, 'k')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('学习率衰减')
plt.show()
```

#### MultiStepLR

在指定的里程碑步数处衰减学习率。

**代码实现**（来自 `optimizer/3_multistep_lr.py`）：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义函数
def f(x):
    return 0.05 * x[0]**2 + x[1]**2

# 主流程
# 1. 参数X初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.9
num_iters = 500

# 3. 定义优化器SGD
optimizer = torch.optim.SGD([X], lr=lr)

# 4. 定义学习率衰减策略
lr_scheduler = MultiStepLR(optimizer, milestones=[10, 50, 200], gamma=0.7)

# 拷贝X的值，放入列表
X_arr = X.detach().numpy().copy()
lr_list = []
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
    X_arr = np.vstack((X_arr, X.detach().numpy()))
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
ax[1].plot(lr_list, 'k')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('学习率衰减')
plt.show()
```

#### ExponentialLR

按指数方式衰减学习率。

**代码实现**（来自 `optimizer/4_exp_lr.py`）：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义函数
def f(x):
    return 0.05 * x[0]**2 + x[1]**2

# 主流程
# 1. 参数X初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.9
num_iters = 500

# 3. 定义优化器SGD
optimizer = torch.optim.SGD([X], lr=lr)

# 4. 定义学习率衰减策略
lr_scheduler = ExponentialLR(optimizer, gamma=0.99)

# 拷贝X的值，放入列表
X_arr = X.detach().numpy().copy()
lr_list = []
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
    X_arr = np.vstack((X_arr, X.detach().numpy()))
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
ax[1].plot(lr_list, 'k')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('学习率衰减')
plt.show()
```

### 6.5.3 AdaGrad

AdaGrad（Adaptive Gradient）为每个参数自适应地调整学习率，对稀疏特征特别有效。

**原理**：
- 累积历史梯度的平方
- 频繁更新的参数学习率降低，稀疏参数学习率保持较高
- 适合处理稀疏数据

**代码实现**（来自 `optimizer/5_adagrad.py`）：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义函数
def f(x):
    return 0.05 * x[0]**2 + x[1]**2

# 定义函数，实现梯度下降法
def gradient_descent(X, optimizer, num_iters):
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
        X_arr = np.vstack((X_arr, X.detach().numpy()))
    return X_arr

# 主流程
# 1. 参数初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.9
num_iters = 500

# 3. 优化器对比
# 3.1 SGD
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr)
# 梯度下降
X_arr1 = gradient_descent(X_clone, optimizer, num_iters)
# 画图
plt.plot(X_arr1[:, 0], X_arr1[:, 1], 'r', label='SGD')

# 3.2 Adagrad
X_clone = X.clone().detach().requires_grad_(True)
optimizer_adagrad = torch.optim.Adagrad([X_clone], lr=lr)
X_arr2 = gradient_descent(X_clone, optimizer_adagrad, num_iters)

plt.plot(X_arr2[:, 0], X_arr2[:, 1], 'b', label='Adagrad')

# 等高线
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
plt.contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')

plt.legend()
plt.show()
```

### 6.5.4 RMSProp

RMSProp改进了AdaGrad，使用指数移动平均来累积梯度平方，避免学习率过早下降。

**原理**：
- 使用指数衰减平均代替累积和
- 更适合处理非平稳目标
- 在循环神经网络中表现良好

**代码实现**（来自 `optimizer/6_rmsprop.py`）：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义函数
def f(x):
    return 0.05 * x[0]**2 + x[1]**2

# 定义函数，实现梯度下降法
def gradient_descent(X, optimizer, num_iters):
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
        X_arr = np.vstack((X_arr, X.detach().numpy()))
    return X_arr

# 主流程
# 1. 参数初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.1
num_iters = 100

# 3. 优化器对比
# 3.1 SGD
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr)
# 梯度下降
X_arr1 = gradient_descent(X_clone, optimizer, num_iters)
# 画图
plt.plot(X_arr1[:, 0], X_arr1[:, 1], 'r', label='SGD')

# 3.2 RMSprop
X_clone = X.clone().detach().requires_grad_(True)
optimizer_rmsprop = torch.optim.RMSprop([X_clone], lr=lr, alpha=0.9)
X_arr2 = gradient_descent(X_clone, optimizer_rmsprop, num_iters)

plt.plot(X_arr2[:, 0], X_arr2[:, 1], 'b', label='RMSprop')

# 等高线
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
plt.contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')

plt.legend()
plt.show()
```

### 6.5.5 Adam

Adam（Adaptive Moment Estimation）结合了Momentum和RMSProp的优点，是目前最常用的优化算法。

**原理**：
- 一阶矩估计：动量（梯度均值）
- 二阶矩估计：自适应学习率（梯度平方的指数平均）
- 偏差修正：解决初始阶段估计偏差问题

**代码实现**（来自 `optimizer/7_adam.py`）：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义函数
def f(x):
    return 0.05 * x[0]**2 + x[1]**2

# 定义函数，实现梯度下降法
def gradient_descent(X, optimizer, num_iters):
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
        X_arr = np.vstack((X_arr, X.detach().numpy()))
    return X_arr

# 主流程
# 1. 参数初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.1
num_iters = 500

# 3. 优化器对比
# 3.1 SGD
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr)
# 梯度下降
X_arr1 = gradient_descent(X_clone, optimizer, num_iters)
# 画图
plt.plot(X_arr1[:, 0], X_arr1[:, 1], 'r', label='SGD')

# 3.2 Adam
X_clone = X.clone().detach().requires_grad_(True)
optimizer_adam = torch.optim.Adam([X_clone], lr=lr, betas=(0.9, 0.999))
X_arr2 = gradient_descent(X_clone, optimizer_adam, num_iters)

plt.plot(X_arr2[:, 0], X_arr2[:, 1], 'b', label='Adam')

# 等高线
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
plt.contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')

plt.legend()
plt.show()
```

**优化算法选择建议**：

| 优化算法 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| SGD | 简单问题 | 计算简单 | 收敛慢、易震荡 |
| Momentum | 梯度方向一致时 | 加速收敛 | 需要调参 |
| AdaGrad | 稀疏数据 | 自适应学习率 | 学习率过早下降 |
| RMSProp | 非平稳目标 | 解决AdaGrad问题 | 需要调参 |
| Adam | 通用首选 | 结合Momentum和RMSProp | 可能泛化稍差 |

---

## 6.6 应用案例：房价预测

本节将综合运用前面所学的知识，实现一个完整的房价预测模型。

### 6.6.1 导入所需的模块

```python
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
```

### 6.6.2 特征工程

特征工程是机器学习中的关键步骤，包括数据清洗、特征转换和数据标准化。

**代码实现**（来自 `5_house_price.py`）：

```python
# 创建数据集
def create_dataset():
    # 1. 从文件读取数据
    data = pd.read_csv('../data/house_prices.csv')
    # 2. 数据预处理，去除无关列
    data = data.drop(['Id'], axis=1)
    # 3. 划分特征和目标
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 5. 特征转换
    # 5.1 按照特征数据类型划分为数值型和类别型
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
    # 5.2 创建特征转换器
    # 5.2.1 数值型特征：平均值填充，然后标准化
    numeric_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='mean')),
            ('std', StandardScaler())
        ]
    )
    # 5.2.2 类别特征：用默认值填充，然后独热编码
    categorical_transformer = Pipeline(steps=[
        ('fillna', SimpleImputer(strategy='constant', fill_value='NaN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # 5.2.3 组合列转换器
    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    # 5.3 进行特征转换
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    X_train = pd.DataFrame(X_train.toarray(), columns=transformer.get_feature_names_out())
    X_test = pd.DataFrame(X_test.toarray(), columns=transformer.get_feature_names_out())
    # 6. 构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), 
                                   torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), 
                                  torch.tensor(y_test.values, dtype=torch.float32))
    # 7. 返回数据集以及特征数量
    return train_dataset, test_dataset, X_train.shape[1]
```

### 6.6.3 搭建模型

使用Sequential构建神经网络模型，包含批归一化和Dropout正则化。

**代码实现**（来自 `5_house_price.py`）：

```python
# 测试
train_dataset, test_dataset, feature_num = create_dataset()

# 创建模型
model = nn.Sequential(
    nn.Linear(feature_num, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1)
)
```

### 6.6.4 损失函数

定义自定义的RMSE损失函数，用于评估回归任务的性能。

**代码实现**（来自 `5_house_price.py`）：

```python
# 自定义损失函数
def loss_rmse(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1, float("inf"))
    mse = nn.MSELoss()
    return torch.sqrt(mse(torch.log(y_pred + 1e-10), torch.log(y_true + 1e-10)))
```

### 6.6.5 模型训练

实现完整的训练循环，包括训练阶段和验证阶段。

**代码实现**（来自 `5_house_price.py`）：

```python
# 模型训练和测试
def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    # 1. 初始化相关操作
    def init_params(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
    # 1.1 参数初始化
    model.apply(init_params)
    # 1.2 将模型加载到设备
    model = model.to(device)
    # 1.3 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 1.4 定义训练误差和测试误差变化列表
    train_loss_list = []
    test_loss_list = []
    # 2. 模型训练
    for epoch in range(epoch_num):
        model.train()
        # 2.1 创建数据加载器
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss_total = 0
        # 2.2 迭代训练
        for batch_idx, (X, y) in enumerate(dataloader):
            # 加载数据到设备
            X = X.to(device)
            y = y.to(device)
            # 2.3.1 前向传播
            y_pred = model(X)
            # 2.3.2 计算损失
            loss = loss_rmse(y_pred.squeeze(), y)
            # 2.3.3 反向传播
            loss.backward()
            # 2.3.4 参数更新
            optimizer.step()
            optimizer.zero_grad()
            # 累加损失
            train_loss_total += loss.item() * X.shape[0]
        train_loss_list.append(train_loss_total / len(train_dataset))
        # 3. 测试
        test_train_loss = 0
        model.eval()
        # 3.1 创建数据加载器
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # 3.2 计算测试误差
        test_loss_total = 0
        with torch.no_grad():  # 禁用梯度计算
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss_value = loss_rmse(y_pred.squeeze(), y)
                test_loss_total += loss_value.item() * X.shape[0]
        this_test_loss = test_loss_total / len(test_dataset)
        test_loss_list.append(this_test_loss)
        
        print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {train_loss_total/len(train_dataset):.4f}, Test Loss: {this_test_loss:.4f}')
    return train_loss_list, test_loss_list

# 超参数
lr = 0.1
epoch_num = 200
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device)

# 画图
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

## 本章小结

### 核心概念回顾

1. **激活函数**：
   - Sigmoid：二分类输出，存在梯度消失
   - Tanh：零中心化，适用于RNN
   - ReLU：计算高效，缓解梯度消失
   - Softmax：多分类概率分布

2. **参数初始化**：
   - Xavier：适用于Sigmoid/Tanh
   - He：适用于ReLU
   - 良好的初始化加速收敛

3. **正则化技术**：
   - Dropout：随机失活防止过拟合
   - 批归一化：稳定训练过程

4. **损失函数**：
   - 分类：BCELoss、CrossEntropyLoss
   - 回归：MSELoss、L1Loss、SmoothL1Loss

5. **优化算法**：
   - SGD + Momentum：基础优化
   - AdaGrad/RMSProp：自适应学习率
   - Adam：通用首选

6. **学习率调度**：
   - StepLR：固定步数衰减
   - MultiStepLR：里程碑衰减
   - ExponentialLR：指数衰减

### 最佳实践

1. **激活函数选择**：
   - 隐藏层默认使用ReLU
   - 输出层根据任务选择Sigmoid/Softmax/Linear

2. **参数初始化**：
   - 使用Xavier或He初始化
   - 避免使用常数初始化

3. **优化器选择**：
   - 一般使用Adam
   - 需要精细调参时尝试SGD+Momentum

4. **学习率设置**：
   - 使用学习率衰减策略
   - 监控训练过程调整初始学习率

5. **正则化应用**：
   - 全连接层使用Dropout
   - 批归一化加速训练

### 下一步学习

掌握了本章内容后，建议继续学习：
- 卷积神经网络（CNN）
- 循环神经网络（RNN/LSTM/GRU）
- 注意力机制和Transformer
- 生成对抗网络（GAN）
- 模型部署和推理优化

通过系统学习这些内容，你将能够使用PyTorch构建和训练各种复杂的深度学习模型。
