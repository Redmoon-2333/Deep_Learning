# 深度学习入门教程

本项目是一个从零开始学习深度学习的实践教程，包含理论文档和代码实现。内容涵盖从神经网络基础到卷积神经网络的完整知识体系。

## 项目结构

```
深度学习/
├── 01_神经网络基础.md          # 第1章：神经网络基础理论文档
├── 02_神经网络的学习.md        # 第2章：神经网络学习理论文档
├── 03_反向传播算法.md          # 第3章：反向传播算法理论文档
├── 04_学习的技巧.md            # 第4章：神经网络训练技巧
├── 05_PyTorch简介.md           # 第5章：PyTorch基础入门
├── 05_PyTorch核心概念详解.md   # 第6章：PyTorch核心概念
├── 06_PyTorch深度学习实践.md   # 第7章：PyTorch深度学习实践
├── 07_卷积神经网络.md          # 第8章：卷积神经网络（CNN）
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
├── ch06_pytorch_base/          # 第5-6章配套代码：PyTorch基础
│   ├── 1_tensor_create.ipynb   # 张量创建
│   ├── 2_tensor_conversion.ipynb # 张量类型转换
│   ├── 3_tensor_calculation.ipynb # 张量运算
│   ├── 4_tensor_stats.ipynb    # 张量统计函数
│   ├── 5_tensor_index.ipynb    # 张量索引操作
│   ├── 6_tensor_shape.ipynb    # 张量形状操作
│   ├── activation_functions/   # 激活函数
│   │   ├── 1_sigmoid.py
│   │   ├── 2_tanh.py
│   │   ├── 3_relu.py
│   │   └── 4_softmax.ipynb
│   ├── loss_functions/         # 损失函数
│   │   ├── 1_bce.ipynb
│   │   ├── 2_cross_entropy.ipynb
│   │   ├── 3_regression_loss.ipynb
│   │   └── 4_loss_test.py
│   └── optimizer/              # 优化器
│       ├── 1_momentum.py
│       ├── 2_step_lr.py
│       ├── 3_multistep_lr.py
│       ├── 4_exp_lr.py
│       ├── 5_adagrad.py
│       ├── 6_rmsprop.py
│       └── 7_adam.py
├── ch07_pytorch_dl/            # 第7章配套代码：PyTorch深度学习
│   ├── 1_init.ipynb            # 参数初始化
│   ├── 2_dropout.ipynb         # Dropout正则化
│   ├── 3_nn_test.py            # 自定义神经网络
│   ├── 4_nn_sequential.py      # Sequential构建模型
│   └── 5_house_price.py        # 房价预测案例
├── ch08_cnn/                   # 第8章配套代码：卷积神经网络
│   ├── 1_conv_test.ipynb       # 卷积层测试
│   ├── 2_pooling_test.ipynb    # 池化层测试
│   ├── 3_deep_cnn.ipynb        # 深度CNN架构
│   └── 4.fashion_category.py   # Fashion-MNIST服装分类
├── ch09_rnn/                   # 第9章配套代码：循环神经网络
│   ├── 1_embedding_test.ipynb  # 词嵌入层测试
│   ├── 2_rnn_test.ipynb        # RNN基础用法
│   └── 3_poems_generation.py   # 古诗生成系统
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
    ├── fashion-mnist_train.csv # Fashion-MNIST训练集
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

### 第5章 PyTorch简介 ✅

**理论内容**：
- [x] PyTorch安装与环境配置
- [x] 张量（Tensor）基础
- [x] 张量创建方法
- [x] 张量类型转换
- [x] 张量运算（基本运算、矩阵乘法）
- [x] 张量索引与形状操作
- [x] GPU加速（CUDA）

**代码实现**：
- [x] 张量创建 (`ch06_pytorch_base/1_tensor_create.ipynb`)
- [x] 张量类型转换 (`ch06_pytorch_base/2_tensor_conversion.ipynb`)
- [x] 张量运算 (`ch06_pytorch_base/3_tensor_calculation.ipynb`)
- [x] 张量统计函数 (`ch06_pytorch_base/4_tensor_stats.ipynb`)
- [x] 张量索引操作 (`ch06_pytorch_base/5_tensor_index.ipynb`)
- [x] 张量形状操作 (`ch06_pytorch_base/6_tensor_shape.ipynb`)

### 第6章 PyTorch核心概念详解 ✅

**理论内容**：
- [x] 张量的数据类型与设备
- [x] Tensor与ndarray的转换
- [x] 自动微分机制（Autograd）
- [x] 计算图与梯度计算
- [x] 梯度管理（detach、zero_grad）
- [x] 线性回归实战

**代码实现**：
- [x] 张量转换操作
- [x] 自动微分示例
- [x] 梯度计算与管理
- [x] 线性回归完整实现

### 第7章 PyTorch深度学习实践 ✅

**理论内容**：
- [x] 激活函数（Sigmoid、Tanh、ReLU、Softmax）
- [x] 参数初始化方法（Xavier、He初始化等）
- [x] 正则化技术（Dropout）
- [x] 神经网络构建（自定义模型、Sequential）
- [x] 损失函数（分类、回归）
- [x] 优化算法（Momentum、AdaGrad、RMSProp、Adam）
- [x] 学习率调度策略

**代码实现**：
- [x] 激活函数可视化 (`ch07_pytorch_dl/activation_functions/`)
- [x] 参数初始化 (`ch07_pytorch_dl/1_init.ipynb`)
- [x] Dropout正则化 (`ch07_pytorch_dl/2_dropout.ipynb`)
- [x] 自定义神经网络 (`ch07_pytorch_dl/3_nn_test.py`)
- [x] Sequential构建模型 (`ch07_pytorch_dl/4_nn_sequential.py`)
- [x] 损失函数 (`ch07_pytorch_dl/loss_functions/`)
- [x] 优化器对比 (`ch07_pytorch_dl/optimizer/`)
- [x] 房价预测案例 (`ch07_pytorch_dl/5_house_price.py`)

### 第8章 卷积神经网络（CNN） ✅

**理论内容**：
- [x] CNN核心思想（局部连接、权重共享、层次化特征）
- [x] 卷积运算原理
- [x] 填充（Padding）与步幅（Stride）
- [x] 多通道卷积
- [x] 池化层（Max Pooling、Average Pooling）
- [x] 经典CNN架构（LeNet、AlexNet、VGG、ResNet）
- [x] 现代CNN设计趋势

**代码实现**：
- [x] 卷积层测试 (`ch08_cnn/1_conv_test.ipynb`)
- [x] 池化层测试 (`ch08_cnn/2_pooling_test.ipynb`)
- [x] 深度CNN架构 (`ch08_cnn/3_deep_cnn.ipynb`)
- [x] Fashion-MNIST服装分类 (`ch08_cnn/4.fashion_category.py`)

### 第9章 循环神经网络（RNN） ✅

**理论内容**：
- [x] 自然语言处理概述（同义词词典、共现矩阵、Word2Vec）
- [x] 词嵌入技术（Embedding层、One-hot vs Embedding）
- [x] RNN基础结构与工作原理
- [x] 梯度消失/爆炸与长期依赖问题
- [x] LSTM长短期记忆网络（门控机制）
- [x] GRU门控循环单元
- [x] 双向RNN与深度RNN

**代码实现**：
- [x] 词嵌入实战 (`ch09_rnn/1_embedding_test.ipynb`)
- [x] RNN基础用法 (`ch09_rnn/2_rnn_test.ipynb`)
- [x] 古诗生成系统 (`ch09_rnn/3_poems_generation.py`)

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

### 运行PyTorch房价预测

```bash
cd ch07_pytorch_dl
python 5_house_price.py
```

### 运行CNN服装分类

```bash
cd ch08_cnn
python 4.fashion_category.py
```

### 运行古诗生成

```bash
cd ch09_rnn
python 3_poems_generation.py
```

该示例使用RNN模型训练古诗数据，然后生成符合格律的七言绝句。

## 环境要求

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PyTorch
- TorchVision
- Joblib

## 学习路线建议

```
第1章 → 第2章 → 第3章 → 第4章 → 第5章 → 第6章 → 第7章 → 第8章 → 第9章
(基础)   (学习)   (反向)   (技巧)   (PyTorch) (核心)  (实践)  (CNN)   (RNN)
```

## 理论文档索引

| 章节 | 文档 | 内容概要 |
|------|------|---------|
| 第1章 | [01_神经网络基础.md](01_神经网络基础.md) | 神经网络结构、激活函数、信号传递 |
| 第2章 | [02_神经网络的学习.md](02_神经网络的学习.md) | 损失函数、数值微分、梯度下降法 |
| 第3章 | [03_反向传播算法.md](03_反向传播算法.md) | 计算图、链式法则、各层反向传播 |
| 第4章 | [04_学习的技巧.md](04_学习的技巧.md) | 优化算法、权重初始化、正则化技术 |
| 第5章 | [05_PyTorch简介.md](05_PyTorch简介.md) | PyTorch基础、张量操作、GPU加速 |
| 第6章 | [05_PyTorch核心概念详解.md](05_PyTorch核心概念详解.md) | 自动微分、梯度管理、线性回归 |
| 第7章 | [06_PyTorch深度学习实践.md](06_PyTorch深度学习实践.md) | 激活函数、初始化、损失函数、优化器 |
| 第8章 | [07_卷积神经网络.md](07_卷积神经网络.md) | CNN原理、卷积层、池化层、经典架构 |
| 第9章 | [08_循环神经网络.md](08_循环神经网络.md) | RNN/LSTM/GRU、词嵌入、古诗生成 |

## 参考资料

- 《深度学习入门：基于Python的理论与实现》
- 尚硅谷大模型技术之深度学习课程
- PyTorch官方文档

---

**最后更新**：2026年2月6日
