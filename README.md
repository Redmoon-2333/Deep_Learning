# 深度学习入门教程

本项目是一个从零开始学习深度学习的实践教程，包含理论文档和代码实现。

## 项目结构

```
深度学习/
├── 01_神经网络基础.md          # 第1章：神经网络基础理论文档
├── 02_神经网络的学习.md        # 第2章：神经网络学习理论文档
├── 03_反向传播算法.md          # 第3章：反向传播算法理论文档
├── ch02_nn_base/               # 第1章配套代码：神经网络基础实现
│   ├── 1_simple_network.py     # 简单三层网络实现
│   ├── 2_digit_recognizer.py   # 手写数字识别（单样本）
│   └── 3_digit_recognizer_batch.py  # 手写数字识别（批量处理）
├── ch03_train/                 # 第2章配套代码：神经网络训练（数值微分）
│   ├── 1_tangent_line.py       # 数值微分：绘制函数切线
│   ├── 2_simple_net_grad.py    # 简单网络梯度计算示例
│   ├── 3_gradient_descent.py   # 梯度下降法可视化
│   ├── 4_digit_recognizer_nn_train.py  # 完整的手写数字训练
│   └── two_layer_net.py        # 两层网络类实现
├── ch04_backward/              # 第3章配套代码：反向传播算法
│   ├── 1_digit_recognizer_nn_train.py  # 使用反向传播的完整训练
│   └── two_layer_net.py        # 基于层结构的两层网络
├── ch05_optim/                 # 第4章配套代码：优化算法对比
│   └── 1_optimiazer_compare.py # 优化器对比实验（SGD/Momentum/AdaGrad/Adam）
├── common/                     # 公共模块
│   ├── __init__.py
│   ├── functions.py            # 激活函数和损失函数
│   ├── gradient.py             # 数值梯度计算
│   ├── layers.py               # 神经网络层实现（ReLU/Sigmoid/Affine/SoftmaxWithLoss）
│   ├── optimizer.py            # 优化器实现（SGD/Momentum/AdaGrad/RMSProp/Adam）
│   └── load_data.py            # 数据加载和预处理
└── data/                       # 数据集目录
    ├── nn_sample               # 预训练网络参数
    ├── train.csv               # MNIST训练数据
    ├── heart.csv               # 心脏病预测数据集
    ├── fashion-mnist_test.csv  # Fashion-MNIST测试集
    └── ...
```

## 学习进度

### 第1章 神经网络基础 ✅

**理论内容**：
- [x] 神经网络的基本结构（输入层、隐藏层、输出层）
- [x] 感知机回顾
- [x] 激活函数详解（Sigmoid、ReLU、Softmax等8种）
- [x] 激活函数选择指南
- [x] 三层神经网络的信号传递

**代码实现**：
- [x] 简单三层网络 (`ch02_nn_base/1_simple_network.py`)
- [x] 手写数字识别 - 单样本预测 (`ch02_nn_base/2_digit_recognizer.py`)
- [x] 手写数字识别 - 批量处理 (`ch02_nn_base/3_digit_recognizer_batch.py`)

### 第2章 神经网络的学习 ✅

**理论内容**：
- [x] 损失函数（MSE、MAE、交叉熵、Smooth L1）
- [x] 数值微分与梯度
- [x] 神经网络的梯度计算
- [x] 梯度下降法与SGD优化器
- [x] Epoch、Batch Size、Learning Rate等核心概念

**代码实现**：
- [x] 数值微分求导 (`common/gradient.py`)
- [x] 数值梯度计算 (`common/gradient.py`)
- [x] 绘制函数切线 (`ch03_train/1_tangent_line.py`)
- [x] 简单网络梯度计算 (`ch03_train/2_simple_net_grad.py`)
- [x] 梯度下降法可视化 (`ch03_train/3_gradient_descent.py`)
- [x] 两层网络类 (`ch03_train/two_layer_net.py`)
- [x] 数据加载模块 (`common/load_data.py`)
- [x] 完整训练流程 (`ch03_train/4_digit_recognizer_nn_train.py`)
- [x] 激活函数实现 (`common/functions.py`)
- [x] 损失函数实现 (`common/functions.py`)

### 第3章 反向传播算法 ✅

**理论内容**：
- [x] 计算图和链式法则
- [x] 反向传播原理（加法/乘法节点）
- [x] 激活层的反向传播（ReLU/Sigmoid）
- [x] Affine层的反向传播
- [x] Softmax-with-Loss层的反向传播
- [x] 梯度检查方法

**代码实现**：
- [x] ReLU 层 (`common/layers.py`)
- [x] Sigmoid 层 (`common/layers.py`)
- [x] Affine 层 (`common/layers.py`)
- [x] SoftmaxWithLoss 层 (`common/layers.py`)
- [x] 基于层结构的两层网络 (`ch04_backward/two_layer_net.py`)
- [x] 反向传播完整训练 (`ch04_backward/1_digit_recognizer_nn_train.py`)

### 第4章 神经网络训练技巧 ✅

**理论内容**：
- [x] 深度神经网络及其问题（梯度消失/梯度爆炸）
- [x] 优化算法（SGD、Momentum、AdaGrad、RMSProp、Adam）
- [x] 学习率衰减策略
- [x] 权重初始化（Xavier、He初始化）
- [x] 正则化技术（Batch Normalization、权值衰减、Dropout）

**代码实现**：
- [x] SGD 优化器 (`common/optimizer.py`)
- [x] Momentum 优化器 (`common/optimizer.py`)
- [x] AdaGrad 优化器 (`common/optimizer.py`)
- [x] RMSProp 优化器 (`common/optimizer.py`)
- [x] Adam 优化器 (`common/optimizer.py`)
- [x] 优化器对比实验 (`ch05_optim/1_optimiazer_compare.py`)

## 公共模块说明

### common/functions.py

包含神经网络常用函数：

| 函数 | 说明 | 用途 |
|------|------|------|
| `step_function` | 阶跃函数 | 二分类 |
| `sigmoid` | Sigmoid函数 | 隐藏层激活/二分类输出 |
| `relu` | ReLU函数 | 隐藏层激活（推荐） |
| `softmax` | Softmax函数 | 多分类输出层 |
| `identity` | 恒等函数 | 回归输出层 |
| `leaky_relu` | Leaky ReLU | 避免神经元死亡 |
| `swish` | Swish函数 | 深度网络 |
| `softplus` | Softplus函数 | 平滑ReLU |
| `mean_squared_error` | 均方误差 | 回归损失 |
| `cross_entropy` | 交叉熵误差 | 分类损失 |

### common/gradient.py

包含梯度计算函数：

| 函数 | 说明 |
|------|------|
| `numerical_diff` | 数值微分求导（中心差分） |
| `_numerical_gradient` | 单样本梯度计算 |
| `numerical_gradient` | 支持矩阵输入的梯度计算 |

### common/layers.py

包含神经网络层实现：

| 类 | 说明 |
|------|------|
| `Relu` | ReLU激活层，支持前向和反向传播 |
| `Sigmoid` | Sigmoid激活层，支持前向和反向传播 |
| `Affine` | 仿射层（全连接层），执行 y = xW + b |
| `SoftmaxWithLoss` | Softmax输出层+交叉熵损失，组合计算 |

### common/optimizer.py

包含各种优化算法实现：

| 类 | 说明 | 适用场景 |
|------|------|---------|
| `SGD` | 随机梯度下降 | 基准算法 |
| `Momentum` | 动量优化器 | 减少振荡，加速收敛 |
| `AdaGrad` | 自适应梯度 | 稀疏数据 |
| `RMSProp` | 均方根传播 | 解决AdaGrad学习率衰减问题 |
| `Adam` | 自适应矩估计 | 综合性能最好，默认推荐 |

### common/load_data.py

数据加载和预处理：

| 函数 | 说明 |
|------|------|
| `get_data` | 加载MNIST数据，划分训练/测试集，归一化 |

## 快速开始

### 运行手写数字识别示例

```bash
cd ch02_nn_base
python 3_digit_recognizer_batch.py
```

### 运行梯度下降示例

```bash
cd ch03_train
python 3_gradient_descent.py
```

### 运行完整训练

```bash
cd ch03_train
python 4_digit_recognizer_nn_train.py
```

### 运行反向传播训练

```bash
cd ch04_backward
python 1_digit_recognizer_nn_train.py
```

**效率对比**：
- `ch03_train/4_digit_recognizer_nn_train.py`：使用数值微分（慢）
- `ch04_backward/1_digit_recognizer_nn_train.py`：使用反向传播（快 **500+倍**）

### 运行优化器对比实验

```bash
cd ch05_optim
python 1_optimiazer_compare.py
```

该实验可视化对比 SGD、Momentum、AdaGrad、Adam 四种优化器在相同损失函数上的收敛轨迹。

## 环境要求

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## 后续计划

- [x] 第1章：神经网络基础
- [x] 第2章：神经网络的学习
- [x] 第3章：反向传播算法
- [x] 第4章：神经网络训练技巧（优化算法、权重初始化、Batch Normalization、正则化等）
- [ ] 第5章：卷积神经网络（CNN）
- [ ] 第6章：循环神经网络（RNN）

## 理论文档索引

| 章节 | 文档 | 内容概要 |
|------|------|---------|
| 第1章 | [01_神经网络基础.md](01_神经网络基础.md) | 神经网络结构、激活函数、信号传递 |
| 第2章 | [02_神经网络的学习.md](02_神经网络的学习.md) | 损失函数、数值微分、梯度下降法 |
| 第3章 | [03_反向传播算法.md](03_反向传播算法.md) | 计算图、链式法则、各层反向传播 |
| 第4章 | [04_学习的技巧.md](04_学习的技巧.md) | 优化算法、权重初始化、正则化技术 |

## 参考资料

- 《深度学习入门：基于Python的理论与实现》
- 尚硅谷大模型技术之深度学习课程

---

**最后更新**：2026年2月1日
