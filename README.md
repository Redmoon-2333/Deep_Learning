# 深度学习入门教程

本项目是一个从零开始学习深度学习的实践教程，包含理论文档和代码实现。

## 项目结构

```
深度学习/
├── 01_神经网络基础.md          # 第1章：神经网络基础理论文档
├── 02_神经网络的学习.md        # 第2章：神经网络学习理论文档
├── ch02_nn_base/               # 第2章代码：神经网络基础实现
│   ├── 1_simple_network.py     # 简单三层网络实现
│   ├── 2_digit_recognizer.py   # 手写数字识别（单样本）
│   └── 3_digit_recognizer_batch.py  # 手写数字识别（批量处理）
├── ch03_train/                 # 第3章代码：神经网络训练
│   ├── 1_tangent_line.py       # 数值微分：绘制函数切线
│   └── 2_simple_net_grad.py    # 简单网络梯度计算示例
├── common/                     # 公共模块
│   ├── __init__.py
│   ├── functions.py            # 激活函数和损失函数
│   └── gradient.py             # 数值梯度计算
└── data/                       # 数据集目录
    ├── nn_sample               # 预训练网络参数
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
- [x] 激活函数实现 (`common/functions.py`)
- [x] 损失函数实现 (`common/functions.py`)

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

## 快速开始

### 运行手写数字识别示例

```bash
cd ch02_nn_base
python 3_digit_recognizer_batch.py
```

### 运行梯度计算示例

```bash
cd ch03_train
python 2_simple_net_grad.py
```

## 环境要求

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## 后续计划

- [ ] 第3章：反向传播算法
- [ ] 第4章：神经网络训练技巧
- [ ] 第5章：卷积神经网络（CNN）
- [ ] 第6章：循环神经网络（RNN）

## 参考资料

- 《深度学习入门：基于Python的理论与实现》
- 尚硅谷大模型技术之深度学习课程

---

**最后更新**：2026年1月30日
