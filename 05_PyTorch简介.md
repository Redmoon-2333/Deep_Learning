# 第5章 PyTorch简介

## 本章导读

在前面的章节中，我们使用NumPy从零开始实现了神经网络的各种组件。虽然这种方式有助于理解底层原理，但在实际项目中，我们通常会使用深度学习框架来简化开发流程。PyTorch是目前最流行的深度学习框架之一，以其动态计算图、简洁的API和强大的生态系统深受研究者喜爱。

**学习目标**：
- 理解PyTorch的核心概念和设计理念
- 掌握PyTorch张量（Tensor）的基本操作
- 理解自动微分（Autograd）机制
- 能够使用PyTorch实现简单的机器学习模型

**学习路线**：
```
PyTorch概述 → 张量基础 → 张量操作 → 自动微分 → 实战案例
(框架介绍)   (数据核心)  (运算变换)  (梯度计算)  (线性回归)
```

**核心概念**：
- 张量（Tensor）：PyTorch中的基本数据结构
- 计算图：动态图机制
- 自动微分：自动计算梯度
- GPU加速：CUDA支持

---

## 5.1 什么是PyTorch

### 5.1.1 PyTorch概述

**PyTorch**是由Facebook人工智能研究院（FAIR）开发的开源深度学习框架，于2016年发布。它基于Torch框架，使用Python作为前端接口，同时保留了C++的高性能后端。

**PyTorch的核心特点**：

1. **动态计算图（Define-by-Run）**：
   - 计算图在运行时动态构建
   - 支持动态网络结构（如变长序列）
   - 调试方便，可以使用标准Python调试工具

2. **Python优先设计**：
   - 与Python生态系统无缝集成
   - 代码简洁直观，易于学习
   - 支持NumPy风格的API

3. **强大的GPU加速**：
   - 原生支持CUDA
   - 张量可以在CPU和GPU之间无缝切换
   - 自动并行计算

4. **丰富的生态系统**：
   - TorchVision：计算机视觉工具包
   - TorchText：自然语言处理工具包
   - PyTorch Lightning：高级封装框架

**PyTorch vs NumPy**：

| 特性 | NumPy | PyTorch |
|------|-------|---------|
| 核心数据结构 | ndarray | Tensor |
| GPU支持 | 不支持 | 原生支持 |
| 自动微分 | 不支持 | Autograd支持 |
| 计算图 | 无 | 动态计算图 |
| 深度学习 | 需手动实现 | 内置支持 |

**PyTorch的应用场景**：
- 计算机视觉（图像分类、目标检测、语义分割）
- 自然语言处理（文本分类、机器翻译、预训练模型）
- 生成模型（GAN、VAE、扩散模型）
- 强化学习
- 科学计算

### 5.1.2 PyTorch的设计理念

**1. 简洁性**：
- API设计直观，符合Python习惯
- 学习曲线平缓，易于上手

**2. 灵活性**：
- 不强制特定的网络结构
- 支持研究和生产的各种需求

**3. 性能**：
- 底层使用C++和CUDA实现
- 提供与静态图框架相当的性能

---

## 5.2 PyTorch安装

### 5.2.1 CPU版本PyTorch安装

CPU版本适用于没有独立显卡或仅用于学习开发的场景。

**使用pip安装**：

```bash
# 安装最新版本
pip install torch torchvision torchaudio

# 安装指定版本
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0
```

**使用conda安装**：

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**验证安装**：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应返回False
```

### 5.2.2 GPU版本PyTorch安装

GPU版本可以利用NVIDIA显卡进行加速计算，大幅提升训练速度。

**前置要求**：
- NVIDIA显卡（计算能力3.5以上）
- CUDA Toolkit（推荐11.8或12.1）
- cuDNN（与CUDA版本匹配）

**使用pip安装（推荐）**：

```bash
# CUDA 12.1版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**使用conda安装**：

```bash
# CUDA 12.1版本
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CUDA 11.8版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**验证GPU可用性**：

```python
import torch

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 获取GPU数量
print(f"GPU数量: {torch.cuda.device_count()}")

# 获取GPU名称
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

**代码待补充**：
```python
# 此处预留GPU检测完整代码实现
```

---

## 5.3 张量创建

张量（Tensor）是PyTorch中的核心数据结构，类似于NumPy的ndarray，但可以在GPU上运行并支持自动微分。

### 5.3.1 基本张量创建

**从列表创建张量**：

```python
import torch

# 创建一维张量
x1 = torch.tensor([1, 2, 3, 4, 5])
print(x1)  # tensor([1, 2, 3, 4, 5])

# 创建二维张量（矩阵）
x2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x2)
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# 创建三维张量
x3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(x3.shape)  # torch.Size([2, 2, 2])
```

**使用特定函数创建张量**：

| 函数 | 说明 | 示例 |
|------|------|------|
| `torch.zeros()` | 创建全零张量 | `torch.zeros(3, 4)` |
| `torch.ones()` | 创建全一张量 | `torch.ones(2, 3)` |
| `torch.eye()` | 创建单位矩阵 | `torch.eye(3)` |
| `torch.empty()` | 创建未初始化张量 | `torch.empty(2, 2)` |
| `torch.full()` | 创建填充指定值的张量 | `torch.full((2, 3), 5)` |

**完整代码实现**：

```python
import torch
import numpy as np

# ========== (1) 按内容创建 ==========

# 创建标量(0维张量)
# torch.tensor()根据输入数据自动推断数据类型
# 输入整数→torch.int64, 输入浮点数→torch.float32
tensor1 = torch.tensor(10)
print("张量值:", tensor1)
print("张量形状:", tensor1.size())  # torch.Size([])表示0维标量
print("数据类型:", tensor1.dtype)   # torch.int64

# 从NumPy数组创建张量
# torch.tensor()会复制数据,修改tensor不影响原ndarray
# 对比: torch.from_numpy()共享内存,修改会互相影响
ndarray1 = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3的NumPy数组
tensor2 = torch.tensor(ndarray1)
print("张量值:\n", tensor2)
print("张量形状:", tensor2.size())  # torch.Size([2, 3])
print("数据类型:", tensor2.dtype)   # torch.int64(继承NumPy的int类型)

# ========== (2) 创建指定形状的张量 ==========

# torch.Tensor(维度参数): 创建指定形状的未初始化张量
# 注意: 默认类型为float32, 值为内存中的随机数(不是真随机,是未初始化内存值)
tensor1 = torch.Tensor(3, 2, 4)
print("张量值(未初始化,显示内存随机值):\n", tensor1)
print("张量形状:", tensor1.size())  # torch.Size([3, 2, 4])
print("数据类型:", tensor1.dtype)   # torch.float32(Tensor默认类型)

# torch.Tensor(列表/数组): 从数据创建张量并转换为float32
# 区别: torch.tensor()保持原数据类型, torch.Tensor()强制转为float32
tensor2 = torch.Tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
print("张量值(整数被转为浮点):\n", tensor2)
print("张量形状:", tensor2.size())  # torch.Size([2, 4])
print("数据类型:", tensor2.dtype)   # torch.float32(被强制转换)

# torch.Tensor(单个整数): 创建指定长度的一维未初始化张量
# 注意: 与torch.tensor(10)不同! torch.tensor(10)创建标量,torch.Tensor(10)创建长度10的向量
tensor3 = torch.Tensor(10)  # 创建长度为10的一维张量
print("张量值(未初始化):", tensor3)
print("张量形状:", tensor3.size())  # torch.Size([10])

# ========== (3) 创建指定类型的张量 ==========

# 创建整数类型张量的三种方式
tensor1 = torch.IntTensor([1, 2, 3])        # 方式1: 类型构造器→int32
tensor2 = torch.tensor([1, 2, 3], dtype=torch.int64)  # 方式2: dtype参数→int64
tensor3 = torch.LongTensor([1, 2, 3])      # 方式3: Long=int64别名
print("IntTensor类型:", tensor1.dtype)     # torch.int32
print("dtype=int64类型:", tensor2.dtype)   # torch.int64
print("LongTensor类型:", tensor3.dtype)    # torch.int64

# 创建短整数类型张量(int16)
tensor1 = torch.ShortTensor([1, 2, 3])
print("ShortTensor类型:", tensor1.dtype)   # torch.int16
tensor2 = torch.tensor([1, 2, 3], dtype=torch.short)
print("dtype=short类型:", tensor2.dtype)   # torch.int16

# 创建字节类型张量
tensor1 = torch.ByteTensor([1, 2, 3])      # 无符号uint8
print("ByteTensor类型:", tensor1.dtype)    # torch.uint8
tensor2 = torch.tensor([1, 2, 3], dtype=torch.int8)  # 有符号int8
print("dtype=int8类型:", tensor2.dtype)    # torch.int8

# 创建单精度浮点数张量(float32)
tensor1 = torch.FloatTensor([1, 2, 3])
print("FloatTensor类型:", tensor1.dtype)   # torch.float32
tensor2 = torch.tensor([1, 2, 3], dtype=torch.float32)
print("dtype=float32类型:", tensor2.dtype) # torch.float32

# 创建双精度浮点数张量(float64)
tensor1 = torch.DoubleTensor(2, 3)  # 未初始化的2x3双精度张量
tensor2 = torch.tensor([1, 2, 3], dtype=torch.float64)
print("DoubleTensor类型:", tensor1.dtype)  # torch.float64
print("dtype=float64类型:", tensor2.dtype) # torch.float64

# 创建半精度浮点数张量(float16)
tensor1 = torch.tensor([1, 2, 3], dtype=torch.float16)
print("dtype=float16类型:", tensor1.dtype) # torch.float16
tensor2 = torch.tensor([1, 2, 3], dtype=torch.half)  # half是float16别名
print("dtype=half类型:", tensor2.dtype)    # torch.float16

# 创建布尔类型张量
tensor1 = torch.BoolTensor([True, False, True])
print("BoolTensor类型:", tensor1.dtype)    # torch.bool
tensor2 = torch.tensor([True, False, True], dtype=torch.bool)
print("dtype=bool类型:", tensor2.dtype)    # torch.bool
```

### 5.3.2 指定区间的张量创建

**等差数列张量**：

```python
import torch

# torch.arange() - 类似Python的range
x1 = torch.arange(10)           # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x2 = torch.arange(1, 10, 2)     # tensor([1, 3, 5, 7, 9])

# torch.linspace() - 线性等分
x3 = torch.linspace(0, 1, 5)    # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# torch.logspace() - 对数等分
x4 = torch.logspace(0, 2, 5)    # 10^0到10^2之间等分5个数
```

**完整代码实现**：

```python
import torch

# torch.arange(start, end, step): 生成等差数列
# 参数: start(起始,包含), end(终止,不包含), step(步长)
# 类似Python的range(), 但返回张量而非列表
# 示例: arange(10,30,2)→[10,12,14,...,28] (不含30)
tensor1 = torch.arange(10, 30, 2)  # 从10到30(不含), 步长2
print("等差数列:", tensor1)  # tensor([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

# torch.arange(n): 生成0到n-1的整数序列
# 等价于arange(0, n, 1), 常用于生成索引
tensor1 = torch.arange(6)  # 生成[0,1,2,3,4,5]
print("索引序列:", tensor1)

# torch.linspace(start, end, steps): 在区间均匀生成指定数量的点
# 参数: start(起始,包含), end(终止,包含), steps(生成点数)
# 区别arange: linspace指定点数, arange指定步长
# 示例: linspace(10,30,5)在[10,30]间均匀取5个点→[10,15,20,25,30]
tensor1 = torch.linspace(10, 30, 5)  # 在[10,30]均匀取5个点
print("均匀采样点:", tensor1)  # tensor([10., 15., 20., 25., 30.])

# torch.logspace(start, end, steps, base): 在对数空间均匀生成点
# 参数: 生成base^start到base^end的steps个点
# 公式: [base^start, base^(start+Δ), ..., base^end], 其中Δ=(end-start)/(steps-1)
# 示例: logspace(1,3,3,base=2)→[2^1, 2^2, 2^3]=[2,4,8]
tensor1 = torch.logspace(1, 3, 3, 2)  # base=2, 从2^1到2^3取3个点
print("对数空间点:", tensor1)  # tensor([2., 4., 8.])
```

### 5.3.3 按数值填充张量

**创建特定值填充的张量**：

```python
import torch

# 创建全零张量
zeros = torch.zeros(3, 4)

# 创建全一张量
ones = torch.ones(2, 3, 4)

# 创建填充特定值的张量
full = torch.full((2, 3), 7.0)

# 创建与现有张量形状相同的新张量
x = torch.tensor([[1, 2], [3, 4]])
zeros_like = torch.zeros_like(x)    # 形状为(2, 2)的全零张量
ones_like = torch.ones_like(x)      # 形状为(2, 2)的全一张量
```

**完整代码实现**：

```python
import torch

# torch.zeros(*shape): 创建全0张量
# 应用: 偏置初始化为0、梯度清零、占位符张量
tensor1 = torch.zeros(2, 3)
print("全0张量:\n", tensor1)

# torch.full(shape, value): 创建全部填充指定值的张量
# 应用: 初始化LSTM遗忘门偏置为1、常量掩码、特定值填充
tensor1 = torch.full((3, 2), 6)  # 注意:shape用元组
print("全6张量:\n", tensor1)

# torch.eye(n): 创建nxn单位矩阵(对角线为1,其余为0)
# 应用: 矩阵初等变换、残差连接初始化、协方差矩阵初始化
tensor1 = torch.eye(3)
print("单位矩阵:\n", tensor1)

# torch.empty_like(input): 创建与input相同形状和数据类型的未初始化张量
# 区别zeros_like: empty_like不初始化(内存随机值), 创建速度更快
tensor2 = torch.empty_like(tensor1)  # 与tensor1同形状(3x3)
print("未初始化张量(随机值):\n", tensor2)
```

### 5.3.4 随机张量创建

**随机数张量**：

```python
import torch

# 均匀分布随机数 [0, 1)
uniform = torch.rand(3, 4)

# 标准正态分布随机数
normal = torch.randn(3, 4)

# 指定范围的整数随机数
randint = torch.randint(0, 10, (3, 4))  # [0, 10)范围内的整数

# 设置随机种子（保证可复现）
torch.manual_seed(42)
random_tensor = torch.rand(3, 3)
```

**随机分布参数化**：

```python
import torch

# 正态分布（指定均值和标准差）
normal_custom = torch.normal(mean=0.0, std=1.0, size=(3, 4))

# 均匀分布（指定范围）
uniform_custom = torch.uniform_(torch.empty(3, 4), 0, 10)
```

**完整代码实现**：

```python
import torch

# (1) rand: 生成[0, 1)区间均匀分布的随机数
# 用途:权重初始化、数据增强、dropout等需要均匀随机值的场景
tensor1 = torch.rand(2, 3)
print("rand均匀分布:", tensor1)
print("数据类型:", tensor1.dtype)

# (2) randn: 生成标准正态分布(均值0,方差1)的随机数
# 用途:神经网络权重初始化(Xavier/He初始化的基础)、噪声生成
tensor2 = torch.randn(3, 4)
print("randn标准正态分布:", tensor2)
print("数据类型:", tensor2.dtype)

# (3) randint: 生成指定范围[low, high)的整数随机数
# 用途:生成标签索引、随机采样、数据打乱等需要整数随机值的场景
tensor3 = torch.randint(0, 10, (3, 3))
print("randint整数随机数:", tensor3)
print("数据类型:", tensor3.dtype)

# (4) randperm: 生成0到n-1的随机排列
# 用途:数据打乱顺序、随机采样索引、交叉验证数据分割
tensor4 = torch.randperm(10)
print("randperm随机排列:", tensor4)
print("数据类型:", tensor4.dtype)

# (5) normal: 生成指定均值和标准差的正态分布随机数
# 用途:自定义权重初始化、生成特定分布的噪声
tensor5 = torch.normal(mean=0.0, std=1.0, size=(2, 4))
print("normal指定正态分布:", tensor5)
print("数据类型:", tensor5.dtype)

# (6) rand_like/randn_like: 生成与给定张量相同形状的随机张量
# 用途:保持张量形状一致的随机初始化、生成相同形状的噪声
x = torch.zeros(2, 3)
tensor6 = torch.rand_like(x)  # 生成与x相同形状的[0,1)均匀分布张量
print("rand_like结果:", tensor6)
tensor7 = torch.randn_like(x)  # 生成与x相同形状的标准正态分布张量
print("randn_like结果:", tensor7)

# (7) 设置随机种子:保证随机数可复现
# 用途:实验结果可重复、调试代码、对比不同模型在相同随机初始化下的表现
torch.manual_seed(42)
tensor8 = torch.randn(2, 2)
print("第一次生成:", tensor8)

torch.manual_seed(42)  # 重新设置相同种子
tensor9 = torch.randn(2, 2)
print("第二次生成(相同种子):", tensor9)
print("两次结果是否相等:", torch.equal(tensor8, tensor9))
```

---

## 5.4 张量转换

### 5.4.1 张量元素类型转换

PyTorch支持多种数据类型，可以根据需要转换张量的数据类型。

**常用数据类型**：

| 数据类型 | 说明 | 占用空间 |
|---------|------|---------|
| `torch.float32` / `torch.float` | 32位浮点数 | 4字节 |
| `torch.float64` / `torch.double` | 64位浮点数 | 8字节 |
| `torch.float16` / `torch.half` | 16位浮点数 | 2字节 |
| `torch.int32` / `torch.int` | 32位整数 | 4字节 |
| `torch.int64` / `torch.long` | 64位整数 | 8字节 |
| `torch.bool` | 布尔类型 | 1字节 |

**类型转换方法**：

```python
import torch

# 创建默认类型（float32）的张量
x = torch.tensor([1, 2, 3])
print(x.dtype)  # torch.int64

# 方法1：使用type()函数
x_float = x.type(torch.float32)

# 方法2：使用to()方法
x_double = x.to(torch.float64)

# 方法3：使用特定类型方法
x_float16 = x.half()
x_float32 = x.float()
x_float64 = x.double()
x_int32 = x.int()
x_int64 = x.long()
```

**完整代码实现**：

```python
import torch

# (1) 使用.type()转换数据类型
# 功能:将张量转换为指定类型,返回新张量(不修改原张量)
# 示例:int64→float32, 用于模型输入数据类型统一
tensor1 = torch.tensor([1, 2, 3])  # 默认int64
print("原始类型:", tensor1.dtype)  # torch.int64
tensor2 = tensor1.type(torch.float32)  # 转换为float32
print("转换后类型:", tensor2.dtype)  # torch.float32
print("转换后值:", tensor2)  # tensor([1., 2., 3.])

# (2) 使用快捷方法转换类型
# .int()→int32, .long()→int64, .float()→float32, .double()→float64, .half()→float16
# 应用:标签索引用.long(), 权重用.float(), 混合精度训练用.half()
tensor1 = torch.tensor([1.5, 2.7, 3.9])
print("原始float:", tensor1)
print("转int32:", tensor1.int())      # tensor([1, 2, 3])
print("转int64:", tensor1.long())     # tensor([1, 2, 3])
print("转float16:", tensor1.half())   # tensor([1.5000, 2.7000, 3.9000], dtype=float16)

# (3) 使用.to()转换类型和设备(推荐方法)
# 优势:同时支持类型转换和设备迁移(CPU↔GPU), 代码更统一
# 示例:.to(torch.float32)转类型, .to('cuda')转GPU, .to('cuda', dtype=torch.float16)同时转换
tensor1 = torch.tensor([1, 2, 3])
tensor2 = tensor1.to(torch.float32)  # 转换为float32
print("使用.to()转换:", tensor2, tensor2.dtype)

# 转化为复数
tensor1 = torch.tensor([1, 2, 3])
tensor2 = tensor1.to(torch.complex64)
print("转为复数:", tensor2)  # tensor([1.+0.j, 2.+0.j, 3.+0.j])
```

### 5.4.2 Tensor与ndarray转换

PyTorch张量与NumPy数组可以方便地互相转换，共享底层内存。

**Tensor转ndarray**：

```python
import torch
import numpy as np

# 创建Tensor
x_torch = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 转换为NumPy数组
x_numpy = x_torch.numpy()
print(type(x_numpy))  # <class 'numpy.ndarray'>

# 注意：CPU上的Tensor与NumPy共享内存
x_torch[0, 0] = 100
print(x_numpy[0, 0])  # 100（同步变化）
```

**ndarray转Tensor**：

```python
import torch
import numpy as np

# 创建NumPy数组
x_numpy = np.array([[1, 2, 3], [4, 5, 6]])

# 方法1：使用torch.from_numpy()（共享内存）
x_torch1 = torch.from_numpy(x_numpy)

# 方法2：使用torch.tensor()（复制数据）
x_torch2 = torch.tensor(x_numpy)

# 方法3：使用torch.as_tensor()（根据情况选择是否共享内存）
x_torch3 = torch.as_tensor(x_numpy)
```

**完整代码实现**：

```python
import torch
import numpy as np

# (1) 张量 ↔ NumPy数组
# torch.from_numpy(ndarray):共享内存,修改一个会影响另一个(仅CPU张量)
# tensor.numpy():转为NumPy数组,也共享内存
# 应用:与NumPy生态交互,如使用matplotlib绘图、sklearn预处理

# NumPy → Tensor (共享内存)
np_array = np.array([1, 2, 3])
tensor1 = torch.from_numpy(np_array)
print("从NumPy创建:", tensor1)
np_array[0] = 999  # 修改NumPy数组
print("修改NumPy后tensor:", tensor1)  # tensor也变了!tensor([999, 2, 3])

# Tensor → NumPy (共享内存)
tensor2 = torch.tensor([4, 5, 6])
np_array2 = tensor2.numpy()
print("转为NumPy:", np_array2)
tensor2[0] = 888  # 修改张量
print("修改tensor后NumPy:", np_array2)  # NumPy也变了!array([888, 5, 6])

# (2) 张量 ↔ Python列表
# tensor.tolist():转为Python原生列表,不共享内存
# torch.tensor(list):从列表创建张量,复制数据
# 应用:数据持久化(JSON存储)、小规模数据调试查看

# Tensor → List
tensor1 = torch.tensor([[1, 2], [3, 4]])
py_list = tensor1.tolist()
print("转为列表:", py_list)  # [[1, 2], [3, 4]]
print("列表类型:", type(py_list))  # <class 'list'>

# List → Tensor
tensor2 = torch.tensor(py_list)
print("从列表创建:", tensor2)
```

### 5.4.3 Tensor与标量转换

**Tensor转Python标量**：

```python
import torch

# 创建单元素张量
x = torch.tensor([3.14])

# 转换为Python标量
scalar = x.item()
print(type(scalar))  # <class 'float'>
print(scalar)        # 3.14

# 多元素张量转列表
y = torch.tensor([1, 2, 3, 4, 5])
value_list = y.tolist()
print(value_list)  # [1, 2, 3, 4, 5]
```

**完整代码实现**：

```python
import torch

# (1) 获取张量的标量值
# .item():仅适用于单元素张量,返回Python标量
# 应用:提取损失值、准确率等单个数值用于日志记录
# 注意:多元素张量调用.item()会报错
tensor1 = torch.tensor(3.14)  # 0维标量张量
value = tensor1.item()
print("标量值:", value)  # 3.14
print("值类型:", type(value))  # <class 'float'>

# 实际应用示例
loss = torch.tensor(0.523)  # 假设这是损失值
print(f"Epoch Loss: {loss.item():.4f}")  # 格式化输出

# (2) 多元素张量转列表
# .tolist():将张量转为Python列表,适用于任意形状
# 应用:保存结果、JSON序列化
tensor2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
value_list = tensor2.tolist()
print(value_list)  # [[1, 2, 3], [4, 5, 6]]
print(type(value_list))  # <class 'list'>
```

---

## 5.5 张量数值计算

### 5.5.1 基本运算

PyTorch支持丰富的数学运算，语法与NumPy类似。

**算术运算**：

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 加法
print(a + b)           # tensor([5, 7, 9])
print(torch.add(a, b))

# 减法
print(a - b)           # tensor([-3, -3, -3])

# 乘法（元素级）
print(a * b)           # tensor([4, 10, 18])

# 除法
print(b / a)           # tensor([4.0000, 2.5000, 2.0000])

# 幂运算
print(a ** 2)          # tensor([1, 4, 9])
print(torch.pow(a, 2))
```

**比较运算**：

```python
import torch

a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([5, 4, 3, 2, 1])

print(a > b)   # tensor([False, False, False,  True,  True])
print(a == b)  # tensor([False, False,  True, False, False])
print(a >= 3)  # tensor([False, False,  True,  True,  True])
```

**完整代码实现**：

```python
import torch

# 1. 四则运算
# PyTorch支持逐元素运算和广播机制
# 广播:当两个张量形状不同时,自动扩展维度进行运算
# 示例:shape(2,3)+shape(3,)会将后者扩展为(1,3)再广播为(2,3)
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 张量间加法(逐元素相加)
print(a + b)  # [[1+5, 2+6], [3+7, 4+8]] = [[6,8], [10,12]]

# 张量与标量运算(广播机制)
print(a+10)  # 每个元素+10: [[11,12], [13,14]]

# .add()方法:返回新张量,不修改原张量
print(a.add(10))  # 结果[[11,12], [13,14]], 但a仍是[[1,2], [3,4]]

# .add_()原地操作:带下划线后缀的方法会直接修改原张量,节省内存
# 应用:梯度累积、参数更新等需要原地修改的场景
print(a.add_(10))  # a被修改为[[11,12], [13,14]]
print(a)  # 验证a已被修改

# 减法运算
print(a.sub_(10))  # a从[[11,12],[13,14]]变回[[1,2],[3,4]]

# 幂运算
# .pow_(2)计算每个元素的平方
# 应用:计算L2范数、方差等统计量
print(a.pow_(2))  # [[1,4],[9,16]]

# 开方运算
# .sqrt_()计算每个元素的平方根
print(a.sqrt_())  # 恢复为[[1,2],[3,4]]

# 指数运算
# 使用**运算符计算e^x, 其中e≈2.7183(自然对数底数)
tensor1=torch.tensor([1.0,2,3])
print(2.7183**tensor1)  # [e^1, e^2, e^3]
print(tensor1.exp())  # [e^1, e^2, e^3]精确计算
```

### 5.5.2 哈达玛积（元素级乘法）

哈达玛积（Hadamard Product）是指两个相同形状张量的对应元素相乘。

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 元素级乘法（哈达玛积）
c = a * b
d = torch.mul(a, b)

print(c)
# tensor([[ 5, 12],
#         [21, 32]])
```

**完整代码实现**：

```python
import torch

# 哈达玛积(Hadamard Product):逐元素相乘
# 区别矩阵乘法:对应位置元素相乘,不是线性代数的矩阵乘法
# 示例:[[1,2],[3,4]] ⊙ [[5,6],[7,8]] = [[1*5,2*6],[3*7,4*8]] = [[5,12],[21,32]]
# 应用:注意力机制的mask、dropout、特征融合
tensor1 = torch.tensor([[1,2],[3,4]])
tensor2 = torch.tensor([[5,6],[7,8]])
print(tensor1 * tensor2)  # 运算符方式
print(torch.mul(tensor1,tensor2))  # 函数方式(等价)
# 输出:
# tensor([[ 5, 12],
#         [21, 32]])
```

### 5.5.3 矩阵乘法运算

矩阵乘法是深度学习中最核心的运算之一。

```python
import torch

a = torch.tensor([[1, 2], [3, 4], [5, 6]])  # 形状 (3, 2)
b = torch.tensor([[7, 8, 9], [10, 11, 12]])  # 形状 (2, 3)

# 方法1：使用@运算符
c1 = a @ b  # 形状 (3, 3)

# 方法2：使用torch.matmul()
c2 = torch.matmul(a, b)

# 方法3：使用mm()（仅适用于2D张量）
c3 = torch.mm(a, b)

print(c1)
# tensor([[27, 30, 33],
#         [61, 68, 75],
#         [95, 106, 117]])
```

**批量矩阵乘法**：

```python
import torch

# 批量矩阵乘法 (b, m, n) @ (b, n, p) = (b, m, p)
a = torch.randn(10, 3, 4)  # 10个3x4矩阵
b = torch.randn(10, 4, 5)  # 10个4x5矩阵
c = torch.bmm(a, b)        # 10个3x5矩阵
print(c.shape)  # torch.Size([10, 3, 5])
```

**完整代码实现**：

```python
import torch

# 矩阵乘法(Matrix Multiplication):线性代数中的矩阵乘法
# 规则:(m,n)@(n,p)→(m,p), 第一个的列数必须等于第二个的行数
# 计算:结果[i,j] = Σ(A[i,k] * B[k,j])
# 示例:[[1,2],[3,4]] @ [[5,6],[7,8]]
#       →[[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
# 应用:全连接层、注意力分数计算、线性变换
tensor1 = torch.tensor([[1,2],[3,4]])
tensor2 = torch.tensor([[5,6],[7,8]])
print(tensor1 @ tensor2)  # @运算符(推荐)
print(torch.matmul(tensor1,tensor2))  # matmul函数(等价)
# 输出:
# tensor([[19, 22],
#         [43, 50]])

# 批量矩阵乘法(Batched Matrix Multiplication)
# 三维张量:(batch, m, n) @ (batch, n, p) → (batch, m, p)
# 每个batch独立进行矩阵乘法
# 应用:Transformer中的批量注意力计算
tensor1 = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])  # shape:(2,2,2)
tensor2 = torch.tensor([[[9,10],[11,12]],[[13,14],[15,16]]])  # shape:(2,2,2)
print(tensor1 @ tensor2)  # batch0和batch1分别相乘
# 输出:
# tensor([[[ 31,  34],
#          [ 71,  78]],
#         [[155, 166],
#          [211, 226]]])
```

### 5.5.4 节省内存

在深度学习训练中，节省内存是一个重要的优化手段。

**原地操作**：

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 非原地操作（创建新张量）
c = a + b
print(id(a))  # 原地址
print(id(c))  # 新地址

# 原地操作（使用_后缀）
a.add_(b)  # a = a + b，不创建新张量
print(a)   # tensor([5, 7, 9])
```

**常用原地操作**：

| 非原地操作 | 原地操作 | 说明 |
|-----------|---------|------|
| `add()` | `add_()` | 加法 |
| `sub()` | `sub_()` | 减法 |
| `mul()` | `mul_()` | 乘法 |
| `div()` | `div_()` | 除法 |
| `zero_()` | - | 置零 |

**完整代码实现**：

```python
import torch

# 内存分配验证
# Python的id()返回对象的内存地址
# 非原地操作(如X=X+10)会创建新张量,内存地址改变
# 问题:大张量频繁运算会导致内存碎片和性能下降
X = torch.randint(1,10,(3,2,4))
print(id(X))  # 记录原始内存地址
X=X+10  # 创建新张量并重新赋值给X
print(id(X))  # 内存地址已改变(新对象)

# 原地操作节省内存
# 方法1: X = X @ Y (创建新张量,内存地址变)
# 方法2: X += 10 (语法糖,等价于X = X + 10,也创建新张量)
# 方法3: X[:] = X @ Y (切片赋值,原地修改,内存地址不变)
# 应用:训练大模型时减少显存占用
X = torch.randint(1, 10, (3, 2, 4))
Y = torch.randint(1, 10, (3, 4, 1))
print("原始地址:", id(X))

# 切片赋值,保持原地址
X[:]= X @ Y  # 原地操作
print("原地操作后地址:", id(X))  # 地址未变,证明是原地操作

# 常用原地操作函数
# add_(), sub_(), mul_(), div_(), pow_(), sqrt_()
# zero_(): 将张量所有元素置零
# 注意:原地操作会修改原张量,使用时需谨慎
```

---

## 5.6 张量运算函数

PyTorch提供了丰富的数学运算函数，涵盖各种科学计算需求。

**数学函数**：

```python
import torch
import math

x = torch.tensor([1.0, 2.0, 3.0])

# 指数和对数
print(torch.exp(x))      # e^x
print(torch.log(x))      # 自然对数
print(torch.log10(x))    # 常用对数
print(torch.log2(x))     # 二进制对数

# 三角函数
print(torch.sin(x))
print(torch.cos(x))
print(torch.tan(x))

# 平方根
print(torch.sqrt(x))

# 绝对值
print(torch.abs(torch.tensor([-1, -2, 3])))

# 取整
y = torch.tensor([1.2, 2.5, 3.7])
print(torch.round(y))    # 四舍五入
print(torch.floor(y))    # 向下取整
print(torch.ceil(y))     # 向上取整
```

**统计函数**：

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# 求和
print(torch.sum(x))          # 21.0
print(torch.sum(x, dim=0))   # 按行求和: tensor([5., 7., 9.])
print(torch.sum(x, dim=1))   # 按列求和: tensor([ 6., 15.])

# 均值
print(torch.mean(x))         # 3.5

# 最大值和最小值
print(torch.max(x))          # 6.0
print(torch.min(x))          # 1.0

# 获取最大值索引
print(torch.argmax(x))       # 5（展平后的索引）
print(torch.argmax(x, dim=1))  # 每行最大值的索引
```

**完整代码实现**：

```python
import torch

# 创建测试张量
# shape:(3,2,4) 可理解为3个2x4的矩阵
tensor1 = torch.randint(1, 10, (3, 2, 4)).float()
print(tensor1)

# sum求和运算
# .sum(): 对所有元素求和,返回标量
# .sum(dim=k): 沿着第k维求和,该维度被消除
# .sum(dim=(i,j)): 沿着多个维度求和
# 应用:计算总损失、batch内求和、注意力权重归一化
print(tensor1.sum())  # 全局求和
print(tensor1.sum(dim=0))  # 沿第0维求和,shape变为(2,4)
print(tensor1.sum(dim=(0, 2)))  # 沿0和2维求和,shape变为(2,)

# mean求均值
# 用法与sum相同,但返回的是平均值
# 公式:mean = sum / count
# 应用:batch平均损失、均值池化(Global Average Pooling)
print(tensor1.mean())  # 全局均值
print(tensor1.mean(dim=0))  # 沿0维求平均

# std求标准差
# 标准差衡量数据的离散程度
# 公式:std = sqrt(mean((x - mean(x))^2))
# 应用:Batch Normalization、梯度裁剪阈值设置、特征标准化
print(tensor1.std())  # 全局标准差
print(tensor1.std(dim=0))  # 沿0维求标准差

# max、min求最大最小值
# .max(): 返回单个标量(全局最大值)
# .max(dim=k): 返回(values, indices)两个张量
#   - values: 每个位置的最大值
#   - indices: 最大值在dim=k维度上的索引位置
# 应用:分类任务取预测类别、池化层、TopK准确率
print(tensor1.max())  # 全局最大值
print(tensor1.min())  # 全局最小值
print(tensor1.max(dim=0))  # 沿0维求最大值,返回values和indices

# argmin求最小值的索引位置
# 将张量展平为一维后,返回最小元素的位置索引
# 应用:查找极值位置、负样本挖掘
print(tensor1.argmin())  # 返回展平后的一维索引

# unique去重
# 返回排序后的唯一值列表
# 应用:统计类别数、检查标签范围、数据探索性分析
print(torch.unique(tensor1))  # 返回排序后的唯一值

# sort排序
# 默认沿最后一个维度升序排序
# 返回(values, indices):
#   - values: 排序后的张量
#   - indices: 原始元素在排序前的索引位置
# 应用:TopK选择、排序池化、中位数计算
print(tensor1.sort())  # 默认dim=-1(最后一维)
```

---

## 5.7 张量索引操作

索引操作是数据处理和特征提取的基础。

### 5.7.1 简单索引

```python
import torch

# 一维张量索引
x = torch.tensor([10, 20, 30, 40, 50])
print(x[0])   # tensor(10)
print(x[-1])  # tensor(50)

# 二维张量索引
y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(y[0, 0])    # tensor(1)
print(y[1, 2])    # tensor(6)
print(y[0])       # tensor([1, 2, 3])（第一行）
```

**完整代码实现**：

```python
import torch

# 创建测试张量
# shape:(3,5,4) 可理解为3个5x4的矩阵
tensor1= torch.randint(1,10,(3,5,4))
print(tensor1)

# 1. 基础索引(整数索引)
# 类似Python列表索引,但支持多维
# tensor[i,j,k]表示第i个矩阵的第j行第k列
# 冒号:表示该维度全选
# 应用:提取特定位置的元素、批次中的某一样本

# 提取单个元素:tensor1[2,1,3]→第2个矩阵第1行第3列
print(tensor1[2,1,3])  # 返回标量

# 提取一列:tensor1[:,2,3]→所有3个矩阵的第2行第3列
print(tensor1[:,2,3])  # 返回(3,)一维张量

# 提取一行:tensor1[:,3]→所有3个矩阵的第3行
print(tensor1[:,3])  # 返回(3,4)二维张量
```

### 5.7.2 范围索引

```python
import torch

x = torch.tensor([10, 20, 30, 40, 50, 60])

# 切片操作 [start:end:step]
print(x[1:4])     # tensor([20, 30, 40])
print(x[:3])      # tensor([10, 20, 30])
print(x[3:])      # tensor([40, 50, 60])
print(x[::2])     # tensor([10, 30, 50])
print(x[::-1])    # tensor([60, 50, 40, 30, 20, 10])

# 二维切片
y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(y[0:2, 1:3])
# tensor([[2, 3],
#         [5, 6]])
```

**完整代码实现**：

```python
import torch

tensor1= torch.randint(1,10,(3,5,4))

# 2. 范围索引(切片)
# 语法:start:end:step, 类似Python列表切片
# start缺省为0, end缺省为长度, step缺省为1
# 注意:end不包含在范围内([ , ))
# 应用:提取子序列、数据分块处理

# tensor1[1:]→从第1个矩阵开始到末尾
# 结果:(3,5,4)→(2,5,4)只保留后2个矩阵
print(tensor1[1:])  # shape:(2,5,4)

# 多维切片
# tensor1[-1:,1:4]→最后一个矩阵,取第1-3行
# -1表示最后一个,1:4表示索引[1,2,3](不含4)
# 结果:(3,5,4)→(1,3,4)只保留1个矩阵的3行
print(tensor1[-1:,1:4])  # shape:(1,3,4)
```

### 5.7.3 列表索引

```python
import torch

x = torch.tensor([10, 20, 30, 40, 50])

# 使用索引列表
indices = torch.tensor([0, 2, 4])
print(x[indices])  # tensor([10, 30, 50])

# 使用Python列表
print(x[[0, 2, 4]])  # tensor([10, 30, 50])

# 二维索引
y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 2])
print(y[rows, cols])  # tensor([2, 9])
```

**完整代码实现**：

```python
import torch

tensor1= torch.randint(1,10,(3,5,4))

# 3. 列表索引(花式索引)
# 使用列表指定要提取的索引位置
# tensor[[i1,i2],[j1,j2]]→提取(i1,j1)和(i2,j2)位置的元素
# 应用:不规则采样、根据索引列表提取数据

# 一维列表索引:tensor1[[1,2,0],[0,1,2]]
# 提取:tensor1[1,0], tensor1[2,1], tensor1[0,2]
# 即第1个矩阵的第0行,第2个矩阵的第1行,第0个矩阵的第2行
print(tensor1[[1,2,0],[0,1,2]])  # 返回(3,4)

# 二维列表索引:tensor1[[[0],[1]],[0,1,2]]
# 广播机制:[[0],[1]](2,1)与[0,1,2](3,)广播为(2,3)
# 提取:tensor1[0,[0,1,2]]和tensor1[1,[0,1,2]]
# 即第0个矩阵的第[0,1,2]行和第1个矩阵的第[0,1,2]行
print(tensor1[[[0],[1]],[0,1,2]])  # 返回(2,3,4)
```

### 5.7.4 布尔索引

```python
import torch

x = torch.tensor([1, 2, 3, 4, 5])

# 布尔条件索引
mask = x > 3
print(mask)       # tensor([False, False, False,  True,  True])
print(x[mask])    # tensor([4, 5])

# 直接条件索引
print(x[x > 3])   # tensor([4, 5])
print(x[x % 2 == 0])  # tensor([2, 4])

# 复合条件
print(x[(x > 2) & (x < 5)])  # tensor([3, 4])
```

**完整代码实现**：

```python
import torch

tensor1= torch.randint(1,10,(3,5,4))

# 4. 布尔索引(条件筛选)
# 使用布尔张量(True/False)作为索引
# tensor[mask]提取mask为True位置对应的元素
# 应用:条件筛选、异常值检测、数据清洗

# 场景1:选取符合条件的行
# 条件:每行的首元素([:,i,0])>5
mask = tensor1[:,:,0]>5  # shape:(3,5),对每个矩阵的5行判断
print(mask)  # True/False矩阵

# tensor1[mask]提取所有mask=True对应的行
# 结果:将符合条件的行拼接成(n,4),n为True的数量
print(tensor1[mask])  # shape:(n,4),n是True的个数

# 场景2:选取符合条件的矩阵
# 条件:第1行第2列的元素>5
mask = tensor1[:,1,2]>5  # shape:(3,),对3个矩阵判断
print(mask)  # [True, False, True]表示只有第0和第2个矩阵符合

# tensor1[mask]提取mask=True的矩阵
print(tensor1[mask])  # shape:(m,5,4),m为True的个数

# 场景3:选取所有大于5的元素
# 条件:每个元素>5
mask = tensor1>5  # shape:(3,5,4),每个元素对应一个True/False
print(tensor1[mask])  # 一维张量,包含所有>5的值
```

---

## 5.8 张量形状操作

### 5.8.1 交换维度

```python
import torch

# 创建3维张量 (2, 3, 4)
x = torch.randn(2, 3, 4)
print(x.shape)  # torch.Size([2, 3, 4])

# 转置（仅适用于2D）
y = torch.randn(3, 4)
print(y.t().shape)  # torch.Size([4, 3])

# 交换维度
z = x.transpose(0, 2)  # 交换维度0和2
print(z.shape)  # torch.Size([4, 3, 2])

# 置换维度
w = x.permute(2, 0, 1)  # 重新排列维度
print(w.shape)  # torch.Size([4, 2, 3])
```

**完整代码实现**：

```python
import torch

# 创建一个三维张量用于演示
# shape=[2,3,6]可理解为: 2个样本,每个样本3行6列的矩阵
tensor1 = torch.randint(1, 10, [2, 3, 6])
print("张量形状:", tensor1.shape)      # torch.Size([2, 3, 6])
print("张量维度数:", tensor1.ndim)     # 3 (三维张量)
print("张量元素总数:", tensor1.numel()) # 2*3*6=36

# transpose(dim0, dim1): 交换两个指定维度
# 应用: 矩阵转置、调整通道顺序(如NCHW↔NHWC)
# 示例: shape=[2,3,6]交换维度1和2→shape=[2,6,3]
# 注意: transpose只能交换两个维度,多维交换需用permute
tensor2 = tensor1.transpose(1, 2)  # 交换第1维(3)和第2维(6)
print("原始形状:", tensor1.shape)    # torch.Size([2, 3, 6])
print("交换后形状:", tensor2.shape)  # torch.Size([2, 6, 3])

# .T: 二维张量的转置快捷方式(等价于transpose(0,1))
# 应用: 矩阵转置、线性代数运算
# 注意: .T只适用于二维张量,高维张量会交换前两维
matrix = torch.randint(1, 10, [3, 4])  # 3x4矩阵
print("原始矩阵形状:", matrix.shape)    # torch.Size([3, 4])
print("转置后形状:", matrix.T.shape)    # torch.Size([4, 3])

# permute(*dims): 重新排列所有维度
# 参数: 新维度的排列顺序(必须包含所有维度)
# 应用: 图像格式转换(CHW→HWC)、调整批次维度顺序
# 示例: shape=[2,3,6], permute(2,0,1)→维度顺序变为[第2维,第0维,第1维]→shape=[6,2,3]
tensor3 = tensor1.permute(2, 0, 1)  # 将维度重排为[6,2,3]
print("permute后形状:", tensor3.shape) # torch.Size([6, 2, 3])
```

### 5.8.2 调整形状

```python
import torch

# 创建张量
x = torch.arange(12)
print(x.shape)  # torch.Size([12])

# view() - 共享内存，要求数据连续
y1 = x.view(3, 4)
print(y1.shape)  # torch.Size([3, 4])

# reshape() - 不共享内存，更灵活
y2 = x.reshape(3, 4)
print(y2.shape)  # torch.Size([3, 4])

# -1自动推断维度
y3 = x.view(2, -1)  # 自动计算为6
print(y3.shape)  # torch.Size([2, 6])

# 展平
z = torch.randn(2, 3, 4)
print(z.view(-1).shape)      # torch.Size([24])
print(z.flatten().shape)     # torch.Size([24])
```

**完整代码实现**：

```python
import torch

tensor1 = torch.randint(1, 10, [2, 3, 6])

# reshape(*shape): 改变张量形状(不改变元素顺序和总数)
# 规则: 新形状的元素总数必须等于原形状元素总数
# 应用: 全连接层输入展平、调整批次大小
# 示例: [2,3,6]共36个元素→可reshape为[2,18]、[6,6]、[36]等
# 技巧: 可用-1让PyTorch自动推断该维度大小
tensor4 = tensor1.reshape(2, 18)  # 2x3x6=36元素→reshape为2x18
print("原始形状:", tensor1.shape)   # torch.Size([2, 3, 6])
print("reshape后形状:", tensor4.shape) # torch.Size([2, 18])

# reshape使用-1自动推断维度
# -1表示: 根据其他维度和总元素数自动计算该维度大小
# 示例: [2,3,6]共36元素, reshape(3,-1)→PyTorch计算-1位置为36/3=12
tensor5 = tensor1.reshape(3, -1)  # 自动计算为[3,12]
print("reshape(3,-1)形状:", tensor5.shape)  # torch.Size([3, 12])
tensor6 = tensor1.reshape(-1)     # 展平为一维向量
print("reshape(-1)形状:", tensor6.shape)    # torch.Size([36])

# view(*shape): 类似reshape,但要求张量内存连续
# 区别: view要求内存连续(contiguous),reshape会在必要时复制数据
# 性能: view更快(不复制数据),但有内存连续性要求
# 检查: 用is_contiguous()检查内存是否连续
tensor5 = tensor1  # 直接赋值,内存连续
print("tensor5是否内存连续:", tensor5.is_contiguous())  # True
print("view操作成功:", tensor5.view(2, 18).shape)      # torch.Size([2, 18])

# flatten(start_dim, end_dim): 展平指定维度范围
# 参数: start_dim(起始维度), end_dim(结束维度,默认-1表示最后一维)
# 应用: 卷积层输出展平后输入全连接层
# 示例: [2,3,6]从dim=1开始展平→[2, 3*6]=[2,18]
tensor7 = tensor1.flatten(start_dim=1)  # 保持dim0,展平后续维度
print("flatten(1)后形状:", tensor7.shape)  # torch.Size([2, 18])
tensor8 = tensor1.flatten()  # 全部展平为一维
print("flatten()后形状:", tensor8.shape)   # torch.Size([36])
```

### 5.8.3 增加或删除维度

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape)  # torch.Size([2, 3])

# 增加维度
y1 = x.unsqueeze(0)  # 在位置0增加维度
print(y1.shape)  # torch.Size([1, 2, 3])

y2 = x.unsqueeze(-1)  # 在最后增加维度
print(y2.shape)  # torch.Size([2, 3, 1])

# 删除维度（只能删除大小为1的维度）
z = torch.randn(1, 3, 1, 4)
print(z.squeeze().shape)       # torch.Size([3, 4])
print(z.squeeze(0).shape)      # torch.Size([3, 1, 4])
print(z.squeeze(-2).shape)     # torch.Size([1, 3, 4])
```

**完整代码实现**：

```python
import torch

# unsqueeze(dim): 在指定位置增加一个大小为1的维度
# 参数: dim(插入位置,可为负数)
# 应用: 增加batch维度、扩展维度以匹配广播规则
# 示例: [3]→unsqueeze(0)→[1,3], unsqueeze(1)→[3,1]
tensor1 = torch.tensor([1, 2, 3])
print("原始形状:", tensor1.shape)  # torch.Size([3])

tensor2 = tensor1.unsqueeze(0)  # 在第0维插入
print("unsqueeze(0)形状:", tensor2.shape)  # torch.Size([1, 3]) - 行向量
print("unsqueeze(0)内容:", tensor2)

tensor3 = tensor1.unsqueeze(1)  # 在第1维插入
print("unsqueeze(1)形状:", tensor3.shape)  # torch.Size([3, 1]) - 列向量
print("unsqueeze(1)内容:\n", tensor3)

# unsqueeze_(): 原地操作版本(会修改原张量)
# 区别: unsqueeze()返回新张量, unsqueeze_()直接修改原张量
# 注意: 原地操作更节省内存,但会改变原张量
tensor4 = torch.tensor([1, 2, 3])
print("原地操作前:", tensor4.shape)      # torch.Size([3])
tensor4.unsqueeze_(dim=0)  # 直接修改tensor4
print("原地操作后:", tensor4.shape)      # torch.Size([1, 3])

# squeeze(dim): 移除大小为1的维度
# 参数: dim(可选,指定移除哪个维度;不指定则移除所有大小为1的维度)
# 应用: 移除冗余维度、降维操作
# 示例: [1,3,1]→squeeze()→[3], squeeze(0)→[3,1]
tensor5 = torch.tensor([[[1, 2, 3]]])  # shape=[1,1,3]
print("原始形状:", tensor5.shape)  # torch.Size([1, 1, 3])

tensor6 = tensor5.squeeze()  # 移除所有大小为1的维度
print("squeeze()形状:", tensor6.shape)  # torch.Size([3])

tensor7 = tensor5.squeeze(0)  # 只移除第0维
print("squeeze(0)形状:", tensor7.shape)  # torch.Size([1, 3])
```

---

## 5.9 张量拼接操作

张量拼接是将多个张量组合成一个张量的操作。

```python
import torch

# 创建张量
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# cat() - 按指定维度拼接
# 按行拼接（dim=0）
c1 = torch.cat([a, b], dim=0)
print(c1)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

# 按列拼接（dim=1）
c2 = torch.cat([a, b], dim=1)
print(c2)
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])

# stack() - 在新维度上堆叠
# 要求输入张量形状完全相同
c3 = torch.stack([a, b], dim=0)
print(c3.shape)  # torch.Size([2, 2, 2])

c4 = torch.stack([a, b], dim=2)
print(c4.shape)  # torch.Size([2, 2, 2])
```

**其他拼接函数**：

```python
import torch

# vstack() - 垂直堆叠（按行）
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.vstack([a, b]))
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# hstack() - 水平堆叠（按列）
print(torch.hstack([a, b]))  # tensor([1, 2, 3, 4, 5, 6])
```

**完整代码实现**：

```python
import torch

# torch.cat(tensors, dim): 沿指定维度拼接张量
# 参数: tensors(张量列表/元组), dim(拼接维度)
# 要求: 除拼接维度外,其他维度大小必须相同
# 应用: 合并批次数据、特征拼接
# 示例: 两个[2,3]沿dim=0拼接→[4,3], 沿dim=1拼接→[2,6]
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])    # shape=[2,3]
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]]) # shape=[2,3]

# 沿第0维(行)拼接 - 增加行数
tensor3 = torch.cat((tensor1, tensor2), dim=0)
print("沿dim=0拼接:\n", tensor3)
print("拼接后形状:", tensor3.shape)  # torch.Size([4, 3])

# 沿第1维(列)拼接 - 增加列数
tensor4 = torch.cat((tensor1, tensor2), dim=1)
print("沿dim=1拼接:\n", tensor4)
print("拼接后形状:", tensor4.shape)  # torch.Size([2, 6])

# torch.stack(tensors, dim): 沿新维度堆叠张量
# 区别cat: cat在现有维度拼接, stack创建新维度
# 要求: 所有张量形状必须完全相同
# 应用: 批次数据组合、时序数据堆叠
# 示例: 两个[4,3,5]沿dim=0堆叠→[2,4,3,5], 沿dim=1堆叠→[4,2,3,5]
tensor1 = torch.randint(1, 10, [4, 3, 5])  # shape=[4,3,5]
tensor2 = torch.randint(1, 10, [4, 3, 5])  # shape=[4,3,5]

# 在第0维堆叠 - 创建新的批次维度
tensor3 = torch.stack([tensor1, tensor2], dim=0)
print("沿dim=0堆叠形状:", tensor3.shape)  # torch.Size([2, 4, 3, 5])

# 在第1维堆叠 - 在第1维位置插入新维度
tensor4 = torch.stack([tensor1, tensor2], dim=1)
print("沿dim=1堆叠形状:", tensor4.shape)  # torch.Size([4, 2, 3, 5])

# cat vs stack 对比示例
# cat: 在现有维度上拼接,不增加维度数
# stack: 创建新维度进行堆叠,维度数+1
a = torch.tensor([1, 2, 3])  # shape=[3]
b = torch.tensor([4, 5, 6])  # shape=[3]

cat_result = torch.cat([a, b], dim=0)
print("cat结果:", cat_result)           # tensor([1,2,3,4,5,6])
print("cat结果形状:", cat_result.shape) # torch.Size([6])

stack_result = torch.stack([a, b], dim=0)
print("stack结果:\n", stack_result)     # tensor([[1,2,3],[4,5,6]])
print("stack结果形状:", stack_result.shape) # torch.Size([2, 3])
```

---

## 5.10 自动微分模块

自动微分（Autograd）是PyTorch最核心的特性之一，它能够自动计算张量的梯度。

### 5.10.1 计算图与梯度

**计算图基础**：

```python
import torch

# 创建需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 构建计算图
z = x ** 2 + y ** 3
print(z)  # tensor(31., grad_fn=<AddBackward0>)

# 反向传播计算梯度
z.backward()

# 查看梯度
print(x.grad)  # tensor(4.)  dz/dx = 2x = 4
print(y.grad)  # tensor(27.) dz/dy = 3y^2 = 27
```

**requires_grad属性**：

```python
import torch

# 默认不需要梯度
x = torch.tensor([1.0, 2.0, 3.0])
print(x.requires_grad)  # False

# 创建时需要梯度
y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(y.requires_grad)  # True

# 使用requires_grad_()方法切换
x.requires_grad_(True)
print(x.requires_grad)  # True
```

**代码待补充**：
```python
# 此处预留计算图与梯度完整代码实现
```

### 5.10.2 梯度计算示例

**标量梯度**：

```python
import torch

# 定义输入
x = torch.tensor(2.0, requires_grad=True)

# 定义函数 y = 3x^2 + 2x + 1
y = 3 * x ** 2 + 2 * x + 1

# 反向传播
y.backward()

# dy/dx = 6x + 2 = 14
print(x.grad)  # tensor(14.)
```

**向量梯度**：

```python
import torch

# 向量输入
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 定义函数 y = sum(x^2)
y = torch.sum(x ** 2)

# 反向传播
y.backward()

# dy/dx = 2x
print(x.grad)  # tensor([2., 4., 6.])
```

**多输出梯度**：

```python
import torch

# 输入
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 多输出函数
y = x ** 2  # y = [1, 4, 9]

# 需要传入与y同形状的权重
v = torch.tensor([1.0, 1.0, 1.0])
y.backward(v)

print(x.grad)  # tensor([2., 4., 6.])
```

**代码待补充**：
```python
# 此处预留梯度计算完整代码实现
```

### 5.10.3 梯度清零与关闭

**梯度清零**：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

# 第一次反向传播
y = x ** 2
y.backward()
print(x.grad)  # tensor(4.)

# 不清零再次反向传播
y = x ** 2
y.backward()
print(x.grad)  # tensor(8.)，梯度累积了！

# 正确做法：先清零
x.grad.zero_()
y = x ** 2
y.backward()
print(x.grad)  # tensor(4.)
```

**关闭梯度追踪**：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

# 方法1：使用torch.no_grad()
with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)  # False

# 方法2：使用detach()
y = x.detach()
print(y.requires_grad)  # False

# 方法3：使用torch.inference_mode()（推荐用于推理）
with torch.inference_mode():
    y = x ** 2
    print(y.requires_grad)  # False
```

**代码待补充**：
```python
# 此处预留梯度控制完整代码实现
```

---

## 5.11 机器学习案例：线性回归

本节将使用PyTorch实现一个简单的线性回归模型，综合运用本章所学的知识。

### 5.11.1 问题定义

**线性回归模型**：

$$
y = wx + b
$$

其中：
- $w$：权重参数
- $b$：偏置参数
- $x$：输入特征
- $y$：预测输出

**损失函数**（均方误差）：

$$
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

### 5.11.2 数据准备

```python
import torch
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)

# 生成合成数据
# 真实参数: w=2.0, b=1.0
N = 100
X = torch.randn(N, 1)
y_true = 2.0 * X + 1.0 + 0.5 * torch.randn(N, 1)  # 添加噪声

# 可视化数据
plt.scatter(X.numpy(), y_true.numpy(), alpha=0.5)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data for Linear Regression')
plt.show()
```

**代码待补充**：
```python
# 此处预留数据准备完整代码实现
```

### 5.11.3 模型定义

```python
import torch
import torch.nn as nn

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # 定义可学习的参数
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1
    
    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = LinearRegression()
print(model)

# 查看初始参数
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")
```

**代码待补充**：
```python
# 此处预留模型定义完整代码实现
```

### 5.11.4 训练过程

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 超参数
learning_rate = 0.01
num_epochs = 1000

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练循环
losses = []
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    
    # 计算损失
    loss = criterion(y_pred, y_true)
    losses.append(loss.item())
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    # 打印进度
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 查看训练后的参数
print("\n训练后的参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")
```

**代码待补充**：
```python
# 此处预留训练过程完整代码实现
```

### 5.11.5 结果可视化

```python
import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# 绘制拟合结果
plt.subplot(1, 2, 2)
plt.scatter(X.numpy(), y_true.numpy(), alpha=0.5, label='Data')
with torch.no_grad():
    y_pred = model(X)
plt.plot(X.numpy(), y_pred.numpy(), 'r-', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend()

plt.tight_layout()
plt.show()
```

**代码待补充**：
```python
# 此处预留结果可视化完整代码实现
```

### 5.11.6 完整代码汇总

```python
"""
PyTorch线性回归完整实现
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 数据准备
torch.manual_seed(42)
N = 100
X = torch.randn(N, 1)
y_true = 2.0 * X + 1.0 + 0.5 * torch.randn(N, 1)

# 2. 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y_true)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 查看结果
print("\n真实参数: w=2.0, b=1.0")
print(f"学习到的参数: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}")
```

**代码待补充**：
```python
# 此处预留完整代码实现
```

---

## 本章小结

### 核心概念回顾

1. **PyTorch基础**：
   - PyTorch是动态图深度学习框架
   - 核心数据结构是张量（Tensor）
   - 支持GPU加速和自动微分

2. **张量操作**：
   - 创建：从列表、随机、特定值创建
   - 转换：类型转换、与NumPy互转
   - 运算：算术运算、矩阵乘法、数学函数
   - 索引：简单索引、切片、布尔索引
   - 形状：调整形状、交换维度、增删维度
   - 拼接：cat、stack、vstack、hstack

3. **自动微分（Autograd）**：
   - 通过`requires_grad`追踪梯度
   - 使用`backward()`计算梯度
   - 使用`zero_grad()`清零梯度
   - 使用`no_grad()`关闭梯度追踪

4. **线性回归实战**：
   - 使用`nn.Module`定义模型
   - 使用`nn.MSELoss`定义损失
   - 使用`optim.SGD`进行优化
   - 完整的训练循环流程

### PyTorch vs NumPy对比

| 特性 | NumPy | PyTorch |
|------|-------|---------|
| 核心数据 | ndarray | Tensor |
| GPU支持 | ❌ | ✅ |
| 自动微分 | ❌ | ✅ |
| 深度学习 | 手动实现 | 内置支持 |
| 生态系统 | 科学计算 | 深度学习 |

### 学习建议

1. **从NumPy过渡到PyTorch**：
   - 两者API高度相似
   - 重点理解自动微分机制
   - 掌握GPU加速的使用

2. **实践建议**：
   - 多动手实现简单模型
   - 理解计算图的构建过程
   - 学会调试梯度问题

3. **后续学习方向**：
   - 深度学习模型（CNN、RNN、Transformer）
   - 高级优化器和学习率调度
   - 模型保存与加载
   - 分布式训练

### 关键收获

✅ 理解PyTorch的核心概念和设计理念  
✅ 掌握张量的各种操作和变换  
✅ 理解自动微分机制及其应用  
✅ 能够使用PyTorch实现简单的机器学习模型  
✅ 为后续深度学习模型的学习打下基础  

---

**最后更新**：2026年2月1日
