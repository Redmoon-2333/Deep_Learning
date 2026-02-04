# 第5章 PyTorch核心概念详解

## 本章导读

在第5章的基础上，本章将深入探讨PyTorch的核心机制和高级特性。我们将从张量的基础操作开始，逐步深入到自动微分、计算图机制、梯度管理等核心概念，帮助读者建立起对PyTorch工作原理的深刻理解。

**学习目标**：
- 掌握PyTorch张量的创建、转换和操作方法
- 理解PyTorch的自动微分机制和计算图原理
- 学会使用梯度追踪、detach等高级特性
- 能够熟练运用PyTorch实现机器学习模型

**学习路线**：
```
张量基础 → 张量操作 → 数值计算 → 统计运算 → 索引操作 → 形状变换 → 自动微分 → 梯度管理 → 实战应用
(数据核心)  (变换操作)  (数学运算)  (数据分析)  (数据访问)  (维度调整)  (核心机制)  (梯度控制)  (线性回归)
```

**核心概念**：
- 张量（Tensor）：PyTorch的基本数据单元
- 计算图：动态图机制的实现原理
- 自动微分：梯度自动计算的核心技术
- 梯度管理：requires_grad、detach等控制机制

---

## 5.1 张量创建详解

### 5.1.1 基本张量创建方法

PyTorch提供了多种创建张量的方式，每种方法都有其特定的使用场景和特点。理解这些创建方法的区别对于高效使用PyTorch至关重要。

#### 张量创建的核心概念

在深度学习中，张量是数据的基本载体。PyTorch中的张量不仅可以存储数据，还能记录计算历史以支持自动微分。创建张量时需要考虑以下因素：

1. **数据类型**：整数类型（int8, int16, int32, int64）vs 浮点类型（float16, float32, float64）
2. **存储位置**：CPU内存 vs GPU显存
3. **梯度追踪**：是否需要计算梯度（requires_grad参数）
4. **内存布局**：连续内存 vs 非连续内存

**从Python数据结构创建张量**：

`torch.tensor()`是最基础的创建方法，它会根据输入数据自动推断数据类型。需要注意的是，`torch.tensor()`会复制数据，这意味着对创建的张量的修改不会影响原始数据。

```python
import torch
import numpy as np

# 创建标量（0维张量）
# torch.tensor()根据输入数据自动推断数据类型
# 输入整数→torch.int64，输入浮点数→torch.float32
tensor1 = torch.tensor(10)
print("张量值:", tensor1)
print("张量形状:", tensor1.size())  # torch.Size([])表示0维标量
print("数据类型:", tensor1.dtype)   # torch.int64

# 从NumPy数组创建张量
# torch.tensor()会复制数据，修改tensor不影响原ndarray
# 对比：torch.from_numpy()共享内存，修改会互相影响
# 应用：将NumPy数据转换为PyTorch张量用于神经网络训练
ndarray1 = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3的NumPy数组
tensor2 = torch.tensor(ndarray1)
print("张量值:\n", tensor2)
print("张量形状:", tensor2.size())  # torch.Size([2, 3])
print("数据类型:", tensor2.dtype)   # torch.int64（继承NumPy的int类型）
```

**创建指定形状的张量**：

```python
# torch.Tensor(维度参数)：创建指定形状的未初始化张量
# 注意：默认类型为float32，值为内存中的随机数（不是真随机，是未初始化内存值）
# 应用：占位符张量，后续会被赋值（如预先分配输出空间）
# 示例：torch.Tensor(3,2,4)创建3x2x4的三维张量，可理解为3个2x4矩阵
tensor1 = torch.Tensor(3, 2, 4)
print("张量值（未初始化，显示内存随机值）:\n", tensor1)
print("张量形状:", tensor1.size())  # torch.Size([3, 2, 4])
print("数据类型:", tensor1.dtype)   # torch.float32（Tensor默认类型）

# torch.Tensor(列表/数组)：从数据创建张量并转换为float32
# 区别：torch.tensor()保持原数据类型，torch.Tensor()强制转为float32
# 示例：输入整数列表[[1,2],[3,4]]，Tensor()输出[[1.0,2.0],[3.0,4.0]]
tensor2 = torch.Tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
print("张量值（整数被转为浮点）:\n", tensor2)
print("张量形状:", tensor2.size())  # torch.Size([2, 4])
print("数据类型:", tensor2.dtype)   # torch.float32（被强制转换）

# torch.Tensor(单个整数)：创建指定长度的一维未初始化张量
# 注意：与torch.tensor(10)不同！torch.tensor(10)创建标量，torch.Tensor(10)创建长度10的向量
# 对比：torch.tensor(10)→shape=[]，torch.Tensor(10)→shape=[10]
tensor3 = torch.Tensor(10)  # 创建长度为10的一维张量
print("张量值（未初始化）:", tensor3)
print("张量形状:", tensor3.size())  # torch.Size([10])
```

### 5.1.2 指定数据类型的张量创建

PyTorch支持多种数据类型，可以根据不同的计算需求选择合适的数据类型。选择合适的数据类型对于模型性能和内存效率都有重要影响。

#### 数据类型选择指南

| 数据类型 | 位数 | 范围/精度 | 适用场景 | 内存占用 |
|---------|------|----------|---------|---------|
| int8 | 8位 | -128~127 | 量化模型、图像像素 | 1字节 |
| int16 | 16位 | -32768~32767 | 小范围整数 | 2字节 |
| int32 | 32位 | ±21亿 | 一般整数计算 | 4字节 |
| int64 | 64位 | ±922亿亿 | 大索引、标签 | 8字节 |
| float16 | 16位 | 3位精度 | 混合精度训练 | 2字节 |
| float32 | 32位 | 7位精度 | 默认浮点类型 | 4字节 |
| float64 | 64位 | 15位精度 | 高精度科学计算 | 8字节 |

**选择建议**：
- 神经网络权重和激活值通常使用float32（默认）
- 大规模模型训练可考虑float16以节省显存
- 标签和索引通常使用int64（PyTorch默认整数类型）
- 图像数据常用uint8（0-255像素值）

```python
# 创建整数类型张量的三种方式
# torch.IntTensor：32位整数（int32），范围约±21亿
# torch.LongTensor：64位整数（int64），范围约±922亿亿，PyTorch默认整数类型
# 应用：标签索引、数据索引通常用int64（可表示更大数据集）
tensor1 = torch.IntTensor([1, 2, 3])        # 方式1：类型构造器→int32
tensor2 = torch.tensor([1, 2, 3], dtype=torch.int64)  # 方式2：dtype参数→int64
tensor3 = torch.LongTensor([1, 2, 3])      # 方式3：Long=int64别名
print("IntTensor类型:", tensor1.dtype)     # torch.int32
print("dtype=int64类型:", tensor2.dtype)   # torch.int64
print("LongTensor类型:", tensor3.dtype)    # torch.int64

# 创建短整数类型张量（int16）
# torch.short/ShortTensor：16位整数，范围-32768到32767
# 应用：节省内存的小范围整数存储（如图像像素索引、类别数<32767的标签）
# 对比：int16（2字节）< int32（4字节）< int64（8字节），但表示范围递减
tensor1 = torch.ShortTensor([1, 2, 3])
print("ShortTensor类型:", tensor1.dtype)   # torch.int16
tensor2 = torch.tensor([1, 2, 3], dtype=torch.short)
print("dtype=short类型:", tensor2.dtype)   # torch.int16

# 创建字节类型张量
# torch.ByteTensor：8位无符号整数（uint8），范围0-255
# torch.int8：8位有符号整数，范围-128到127
# 应用：图像数据（像素值0-255）、掩码（0/1二值）、极小范围整数存储
tensor1 = torch.ByteTensor([1, 2, 3])      # 无符号uint8
print("ByteTensor类型:", tensor1.dtype)    # torch.uint8
tensor2 = torch.tensor([1, 2, 3], dtype=torch.int8)  # 有符号int8
print("dtype=int8类型:", tensor2.dtype)    # torch.int8

# 创建单精度浮点数张量（float32）
# torch.float32/FloatTensor：PyTorch默认浮点类型，32位单精度，精度约7位小数
# 应用：神经网络权重、激活值、损失值等绝大多数深度学习计算（速度与精度平衡）
# 示例：0.1234567→float32存储为0.1234567，float16可能变为0.1235
tensor1 = torch.FloatTensor([1, 2, 3])
print("FloatTensor类型:", tensor1.dtype)   # torch.float32
tensor2 = torch.tensor([1, 2, 3], dtype=torch.float32)
print("dtype=float32类型:", tensor2.dtype) # torch.float32

# 创建双精度浮点数张量（float64）
# torch.float64/DoubleTensor：64位双精度，精度约15位小数，内存是float32的2倍
# 应用：科学计算、高精度数值模拟（深度学习中较少用，因速度慢且显存占用大）
# 对比：float32精度够用且快，float64精度高但慢，float16快但精度低
tensor1 = torch.DoubleTensor(2, 3)  # 未初始化的2x3双精度张量
tensor2 = torch.tensor([1, 2, 3], dtype=torch.float64)
print("DoubleTensor类型:", tensor1.dtype)  # torch.float64
print("dtype=float64类型:", tensor2.dtype) # torch.float64
print("tensor1值（未初始化）:", tensor1)

# 创建半精度浮点数张量（float16）
# torch.float16/HalfTensor：16位半精度，精度约3位小数，内存是float32的1/2
# 应用：混合精度训练（AMP）、GPU显存受限时、推理加速（精度损失可接受场景）
# 优势：显存占用少、GPU计算快（Tensor Core加速） | 劣势：数值范围小易溢出
tensor1 = torch.tensor([1, 2, 3], dtype=torch.float16)
print("dtype=float16类型:", tensor1.dtype) # torch.float16
tensor2 = torch.tensor([1, 2, 3], dtype=torch.half)  # half是float16别名
print("dtype=half类型:", tensor2.dtype)    # torch.float16

# 创建布尔类型张量
# torch.bool：存储True/False，仅占1位（内存高效）
# 应用：掩码（mask）、条件筛选、注意力机制的padding标记
# 示例：mask=[True,False,True]筛选data=[1,2,3]→结果[1,3]
tensor1 = torch.BoolTensor([True, False, True])
print("BoolTensor类型:", tensor1.dtype)    # torch.bool
tensor2 = torch.tensor([True, False, True], dtype=torch.bool)
print("dtype=bool类型:", tensor2.dtype)    # torch.bool
```

### 5.1.3 指定区间的张量创建

在实际应用中，我们经常需要创建特定区间内的数值序列：

```python
# torch.arange(start, end, step)：生成等差数列
# 参数：start（起始，包含），end（终止，不包含），step（步长）
# 类似Python的range()，但返回张量而非列表
# 示例：arange(10,30,2)→[10,12,14,...,28]（不含30）
tensor1 = torch.arange(10, 30, 2)  # 从10到30（不含），步长2
print("等差数列:", tensor1)  # tensor([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

# torch.arange(n)：生成0到n-1的整数序列
# 等价于arange(0, n, 1)，常用于生成索引
# 应用：批次索引、数据采样索引、位置编码
tensor1 = torch.arange(6)  # 生成[0,1,2,3,4,5]
print("索引序列:", tensor1)

# torch.linspace(start, end, steps)：在区间均匀生成指定数量的点
# 参数：start（起始，包含），end（终止，包含），steps（生成点数）
# 区别arange：linspace指定点数，arange指定步长
# 示例：linspace(10,30,5)在[10,30]间均匀取5个点→[10,15,20,25,30]
# 应用：生成学习率衰减序列、可视化采样点
tensor1 = torch.linspace(10, 30, 5)  # 在[10,30]均匀取5个点
print("均匀采样点:", tensor1)  # tensor([10., 15., 20., 25., 30.])

# torch.logspace(start, end, steps, base)：在对数空间均匀生成点
# 参数：生成base^start到base^end的steps个点
# 公式：[base^start, base^(start+Δ), ..., base^end]，其中Δ=(end-start)/(steps-1)
# 示例：logspace(1,3,3,base=2)→[2^1, 2^2, 2^3]=[2,4,8]
# 应用：学习率指数衰减、对数刻度的超参数搜索
tensor1 = torch.logspace(1, 3, 3, 2)  # base=2，从2^1到2^3取3个点
print("对数空间点:", tensor1)  # tensor([2., 4., 8.])
```

### 5.1.4 按数值填充的张量创建

在初始化神经网络参数时，我们经常需要创建特定值填充的张量：

```python
# torch.zeros(*shape)：创建全0张量
# 应用：偏置初始化为0、梯度清零、占位符张量
# 示例：zeros(2,3)→[[0,0,0],[0,0,0]]（2行3列全0矩阵）
tensor1 = torch.zeros(2, 3)
print("全0张量:\n", tensor1)

# torch.ones(*shape)：创建全1张量
# 应用：权重初始化、掩码初始化
# 示例：ones(2,3)→[[1,1,1],[1,1,1]]（2行3列全1矩阵）
tensor1 = torch.ones(2, 3)
print("全1张量:\n", tensor1)

# torch.full(shape, value)：创建全部填充指定值的张量
# 应用：初始化LSTM遗忘门偏置为1、常量掩码、特定值填充
# 示例：full((3,2), 6)→[[6,6],[6,6],[6,6]]（3x2全6矩阵）
tensor1 = torch.full((3, 2), 6)  # 注意：shape用元组
print("全6张量:\n", tensor1)

# torch.eye(n)：创建nxn单位矩阵（对角线为1，其余为0）
# 应用：矩阵初等变换、残差连接初始化、协方差矩阵初始化
# 示例：eye(3)→[[1,0,0],[0,1,0],[0,0,1]]（3x3单位矩阵）
tensor1 = torch.eye(3)
print("单位矩阵:\n", tensor1)

# torch.zeros_like(input)：创建与input相同形状的全0张量
# 应用：梯度初始化、创建相同形状的辅助张量
tensor2 = torch.zeros_like(tensor1)  # 与tensor1同形状（3x3）
print("zeros_like张量:\n", tensor2)

# torch.empty_like(input)：创建与input相同形状和数据类型的未初始化张量
# 区别zeros_like：empty_like不初始化（内存随机值），创建速度更快
# 应用：预分配输出空间（后续会完全覆盖，不需要初始化为0）
# 注意：值不可预测，使用前必须赋值，否则结果错误
tensor3 = torch.empty_like(tensor1)  # 与tensor1同形状（3x3）
print("未初始化张量（随机值）:\n", tensor3)
```

### 5.1.5 随机张量创建

随机数在神经网络初始化、数据增强等方面发挥重要作用：

```python
# (1) rand：生成[0, 1)区间均匀分布的随机数
# 用途：权重初始化、数据增强、dropout等需要均匀随机值的场景
# 示例：torch.rand(2, 3)生成2x3张量，元素如[[0.234, 0.891, 0.456], [0.123, 0.678, 0.901]]
tensor1 = torch.rand(2, 3)
print("均匀分布随机数:\n", tensor1)
print("数据类型:", tensor1.dtype)  # torch.float32

# (2) randn：生成标准正态分布（均值0，方差1）的随机数
# 用途：神经网络权重初始化（Xavier/He初始化的基础）、噪声生成
# 示例：torch.randn(2, 3)生成2x3张量，元素如[[0.34, -1.23, 0.56], [-0.89, 1.45, -0.12]]
tensor2 = torch.randn(3, 4)
print("正态分布随机数:\n", tensor2)
print("数据类型:", tensor2.dtype)  # torch.float32

# (3) randint：生成指定范围[low, high)的整数随机数
# 用途：生成标签索引、随机采样、数据打乱等需要整数随机值的场景
# 示例：torch.randint(0, 10, (2, 3))生成[0,10)的整数，如[[3, 7, 1], [9, 0, 5]]
tensor3 = torch.randint(0, 10, (3, 3))
print("整数随机数:\n", tensor3)
print("数据类型:", tensor3.dtype)  # torch.int64

# (4) randperm：生成0到n-1的随机排列
# 用途：数据打乱顺序、随机采样索引、交叉验证数据分割
# 示例：torch.randperm(10)生成[0,1,2,...,9]的随机排列，如[3, 7, 1, 9, 0, 5, 2, 8, 4, 6]
tensor4 = torch.randperm(10)
print("随机排列:", tensor4)
print("数据类型:", tensor4.dtype)  # torch.int64

# (5) normal：生成指定均值和标准差的正态分布随机数
# 用途：自定义权重初始化（如LSTM的遗忘门偏置初始化为均值1）、生成特定分布的噪声
# 示例：torch.normal(mean=5.0, std=2.0, size=(2,3))生成均值5，标准差2的张量
#       结果如[[5.67, 3.21, 6.89], [4.12, 7.45, 3.78]]
tensor5 = torch.normal(mean=0.0, std=1.0, size=(2, 4))
print("自定义正态分布:\n", tensor5)
print("数据类型:", tensor5.dtype)  # torch.float32

# (6) rand_like/randn_like：生成与给定张量相同形状的随机张量
# 用途：保持张量形状一致的随机初始化、生成相同形状的噪声
# 示例：若x.shape=(3,4)，torch.rand_like(x)生成3x4的[0,1)均匀分布张量
x = torch.zeros(2, 3)
tensor6 = torch.rand_like(x)  # 生成与x相同形状的[0,1)均匀分布张量
print("rand_like结果:\n", tensor6)
tensor7 = torch.randn_like(x)  # 生成与x相同形状的标准正态分布张量
print("randn_like结果:\n", tensor7)

# (7) 设置随机种子：保证随机数可复现
# 用途：实验结果可重复、调试代码、对比不同模型在相同随机初始化下的表现
# 示例：设置seed=42后，每次运行torch.randn(2,2)都会得到相同的结果
torch.manual_seed(42)
tensor8 = torch.randn(2, 2)
print("第一次生成:\n", tensor8)

torch.manual_seed(42)  # 重新设置相同种子
tensor9 = torch.randn(2, 2)
print("第二次生成（相同种子）:\n", tensor9)
print("两次结果是否相等:", torch.equal(tensor8, tensor9))
```

---

## 5.2 张量转换详解

### 5.2.1 张量元素类型转换

在深度学习实践中，经常需要在不同类型之间转换张量：

```python
# (1) 使用.type()转换数据类型
# 功能：将张量转换为指定类型，返回新张量（不修改原张量）
# 示例：int64→float32，用于模型输入数据类型统一
tensor1 = torch.tensor([1, 2, 3])  # 默认int64
print("原始类型:", tensor1.dtype)
tensor2 = tensor1.type(torch.float32)  # 转换为float32
print("转换后类型:", tensor2.dtype)
print("转换后值:", tensor2)  # tensor([1., 2., 3.])

# (2) 使用快捷方法转换类型
# .int()→int32，.long()→int64，.float()→float32，.double()→float64，.half()→float16
# 应用：标签索引用.long()，权重用.float()，混合精度训练用.half()
tensor1 = torch.tensor([1.5, 2.7, 3.9])
print("原始float:", tensor1)
print("转int32:", tensor1.int())      # tensor([1, 2, 3])
print("转int64:", tensor1.long())     # tensor([1, 2, 3])
print("转float16:", tensor1.half())   # tensor([1.5000, 2.6992, 3.9004], dtype=float16)

# (3) 使用.to()转换类型和设备（推荐方法）
# 优势：同时支持类型转换和设备迁移（CPU↔GPU），代码更统一
# 示例：.to(torch.float32)转类型，.to('cuda')转GPU，.to('cuda', dtype=torch.float16)同时转换
tensor1 = torch.tensor([1, 2, 3])
tensor2 = tensor1.to(torch.float32)  # 转换为float32
print("使用.to()转换:", tensor2, tensor2.dtype)
# tensor3 = tensor1.to('cuda')  # 转移到GPU（需要GPU环境）
# 转化为复数
tensor1 = torch.tensor([1, 2, 3])
tensor2 = tensor1.to(torch.complex64)
print("转为复数:", tensor2)
```

### 5.2.2 张量与其他数据结构转换

PyTorch与NumPy、Python原生数据结构的互操作性是其重要特性：

```python
# (1) 张量 ↔ NumPy数组
# torch.from_numpy(ndarray)：共享内存，修改一个会影响另一个（仅CPU张量）
# tensor.numpy()：转为NumPy数组，也共享内存
# 应用：与NumPy生态交互，如使用matplotlib绘图、sklearn预处理

# NumPy → Tensor（共享内存）
np_array = np.array([1, 2, 3])
tensor1 = torch.from_numpy(np_array)
print("从NumPy创建:", tensor1)
np_array[0] = 999  # 修改NumPy数组
print("修改NumPy后tensor:", tensor1)  # tensor也变了！tensor([999, 2, 3])

# Tensor → NumPy（共享内存）
tensor2 = torch.tensor([4, 5, 6])
np_array2 = tensor2.numpy()
print("转为NumPy:", np_array2)
tensor2[0] = 888  # 修改张量
print("修改tensor后NumPy:", np_array2)  # NumPy也变了！array([888, 5, 6])

# (2) 张量 ↔ Python列表
# tensor.tolist()：转为Python原生列表，不共享内存
# torch.tensor(list)：从列表创建张量，复制数据
# 应用：数据持久化（JSON存储）、小规模数据调试查看

# Tensor → List
tensor1 = torch.tensor([[1, 2], [3, 4]])
py_list = tensor1.tolist()
print("转为列表:", py_list)  # [[1, 2], [3, 4]]
print("列表类型:", type(py_list))  # <class 'list'>

# List → Tensor
tensor2 = torch.tensor(py_list)
print("从列表创建:", tensor2)

# (3) 获取张量的标量值
# .item()：仅适用于单元素张量，返回Python标量
# 应用：提取损失值、准确率等单个数值用于日志记录
# 注意：多元素张量调用.item()会报错
tensor1 = torch.tensor(3.14)  # 0维标量张量
value = tensor1.item()
print("标量值:", value)  # 3.14
print("值类型:", type(value))  # <class 'float'>

# 实际应用示例
loss = torch.tensor(0.523)  # 假设这是损失值
print(f"Epoch Loss: {loss.item():.4f}")  # 格式化输出
```

### 5.2.3 张量形状变换

形状变换是深度学习中最重要的操作之一，特别是在网络层之间传递数据时：

```python
# (1) .view()重塑张量形状
# 功能：返回新视图（共享数据），不复制内存，要求张量在内存中连续
# 参数：-1表示自动推断该维度大小
# 应用：全连接层输入展平（batch,C,H,W）→（batch,C*H*W）
# 示例：6个元素可重塑为(2,3)、(3,2)、(6,)、(1,6)等
tensor1 = torch.arange(12)  # tensor([0,1,2,...,11])
print("原始形状:", tensor1.shape)  # torch.Size([12])

tensor2 = tensor1.view(3, 4)  # 重塑为3x4
print("重塑为3x4:\n", tensor2)

tensor3 = tensor1.view(2, -1)  # 2行，-1自动计算为6列
print("重塑为2x?:\n", tensor3, "形状:", tensor3.shape)

# (2) .reshape()重塑张量（推荐）
# 区别view：reshape在张量不连续时会自动复制，更安全但可能慢
# 应用：与view相同，但适用于转置等操作后的张量
# 选择建议：能用view就用view（快），不确定连续性用reshape（安全）
tensor1 = torch.arange(12)
tensor2 = tensor1.reshape(3, 4)
print("reshape结果:\n", tensor2)

# reshape可处理转置后的张量
tensor3 = tensor2.t()  # 转置，在内存中不连续
tensor4 = tensor3.reshape(2, 6)  # reshape成功
print("转置后reshape:\n", tensor4)

# (3) .flatten()展平张量
# 功能：将多维张量展平为一维，默认从第0维开始
# 参数：start_dim, end_dim指定展平的维度范围
# 应用：CNN输出→全连接层输入（保留batch维度）
# 示例：(batch=2, C=3, H=4, W=4)→flatten(1)→(2, 48)
tensor1 = torch.randn(2, 3, 4)  # 形状(2,3,4)
print("原始形状:", tensor1.shape)

tensor2 = tensor1.flatten()  # 全部展平
print("全部展平:", tensor2.shape)  # torch.Size([24])

tensor3 = tensor1.flatten(start_dim=1)  # 保留第0维（batch），展平后续维度
print("保留batch展平:", tensor3.shape)  # torch.Size([2, 12])
```

---

## 5.3 张量数值计算详解

### 5.3.1 基本算术运算

PyTorch支持丰富的数学运算，语法与NumPy类似：

```python
import torch

# 四则运算与广播机制
# PyTorch支持逐元素运算，当形状不同时自动广播
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 基本运算
print(a + b)        # 逐元素相加
print(a * 10)       # 广播：标量与张量运算

# 原地操作（节省内存）
print(a.add_(10))   # 带_后缀的方法原地修改
print(a.sub_(10))   # 恢复原始值

# 幂运算与开方
print(a.pow_(2))    # 平方
print(a.sqrt_())    # 开方

# 指数运算
tensor1 = torch.tensor([1.0, 2.0, 3.0])
print(tensor1.exp())  # e^x
```

### 5.3.2 哈达玛积（元素级乘法）

哈达玛积是指两个相同形状张量的对应元素相乘：

```python
# 哈达玛积（Hadamard Product）：逐元素相乘
# 区别矩阵乘法：对应位置元素相乘，不是线性代数的矩阵乘法
# 示例：[[1,2],[3,4]] ⊙ [[5,6],[7,8]] = [[1*5,2*6],[3*7,4*8]] = [[5,12],[21,32]]
# 应用：注意力机制的mask、dropout、特征融合
tensor1 = torch.tensor([[1,2],[3,4]])
tensor2 = torch.tensor([[5,6],[7,8]])
print(tensor1 * tensor2)  # 运算符方式
print(torch.mul(tensor1,tensor2))  # 函数方式（等价）
```

### 5.3.3 矩阵乘法运算

矩阵乘法是深度学习中最核心的运算之一：

```python
# 矩阵乘法（Matrix Multiplication）：线性代数中的矩阵乘法
# 规则：(m,n)@(n,p)→(m,p)，第一个的列数必须等于第二个的行数
# 计算：结果[i,j] = Σ(A[i,k] * B[k,j])
# 示例：[[1,2],[3,4]] @ [[5,6],[7,8]]
#       →[[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
# 应用：全连接层、注意力分数计算、线性变换
tensor1 = torch.tensor([[1,2],[3,4]])
tensor2 = torch.tensor([[5,6],[7,8]])
print(tensor1 @ tensor2)  # @运算符（推荐）
print(torch.matmul(tensor1,tensor2))  # matmul函数（等价）

# 批量矩阵乘法（Batched Matrix Multiplication）
# 三维张量：(batch, m, n) @ (batch, n, p) → (batch, m, p)
# 每个batch独立进行矩阵乘法
# 应用：Transformer中的批量注意力计算
tensor1 = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])  # shape:(2,2,2)
tensor2 = torch.tensor([[[9,10],[11,12]],[[13,14],[15,16]]])  # shape:(2,2,2)
print(tensor1 @ tensor2)  # batch0:[[1,2],[3,4]]@[[9,10],[11,12]]
print(torch.matmul(tensor1,tensor2))  # batch1:[[5,6],[7,8]]@[[13,14],[15,16]]
```

### 5.3.4 内存管理与原地操作

在大规模深度学习训练中，内存管理至关重要：

```python
import torch

# 内存分配验证：非原地操作会创建新对象
X = torch.randint(1, 10, (3, 2, 4))
print(f"原始地址: {id(X)}")
X = X + 10  # 创建新张量
print(f"新地址: {id(X)}")  # 地址改变

# 原地操作节省内存
X = torch.randint(1, 10, (3, 2, 4))
Y = torch.randint(1, 10, (3, 4, 1))
original_id = id(X)

# 使用切片赋值实现原地操作
X[:] = X @ Y  # 保持内存地址不变
print(f"原地操作后地址: {id(X)}")  # 与original_id相同
```

---

## 5.4 张量统计运算详解

### 5.4.1 基本统计函数

统计运算是数据分析和模型评估的重要工具：

```python
import torch

# 创建测试张量
tensor1 = torch.randint(1, 10, (3, 2, 4)).float()

# 求和运算
print(tensor1.sum())                 # 全局求和
print(tensor1.sum(dim=0))            # 沿第0维求和
print(tensor1.sum(dim=(0, 2)))       # 沿多维度求和

# 均值与标准差
print(tensor1.mean())                # 全局均值
print(tensor1.mean(dim=0))           # 沿指定维度均值
print(tensor1.std())                 # 标准差

# 最大最小值
print(tensor1.max())                 # 全局最大值
print(tensor1.max(dim=0))            # 返回(values, indices)
print(tensor1.argmin())              # 最小值索引

# 其他统计函数
print(torch.unique(tensor1))         # 去重
print(tensor1.sort())                # 排序
```

---

## 5.5 张量索引操作详解

### 5.5.1 基础索引

张量索引是访问和操作张量元素的基本方法：

```python
import torch

# 创建测试张量
tensor1 = torch.randint(1, 10, (3, 5, 4))

# 基础索引
print(tensor1[2, 1, 3])     # 提取单个元素
print(tensor1[:, 2, 3])     # 提取一列
print(tensor1[:, 3])        # 提取一行
```

### 5.5.2 范围索引（切片）

切片索引允许我们提取张量的子区域：

```python
# 范围索引（切片）
print(tensor1[1:])          # 从第1个矩阵开始
print(tensor1[-1:, 1:4])    # 最后一个矩阵，取1-3行
```

### 5.5.3 列表索引（花式索引）

列表索引提供了更灵活的数据提取方式：

```python
# 列表索引（花式索引）
print(tensor1[[1, 2, 0], [0, 1, 2]])      # 一维列表索引
print(tensor1[[[0], [1]], [0, 1, 2]])     # 二维列表索引（广播）
```

### 5.5.4 布尔索引（条件筛选）

布尔索引是基于条件的数据筛选方法：

```python
# 布尔索引（条件筛选）
mask = tensor1[:, :, 0] > 5       # 创建条件掩码
print(tensor1[mask])               # 筛选符合条件的行

mask = tensor1[:, 1, 2] > 5       # 筛选矩阵
print(tensor1[mask])

print(tensor1[tensor1 > 5])        # 直接筛选所有大于5的元素
```

---

## 5.6 张量形状操作详解

### 5.6.1 维度交换操作

维度交换是调整张量结构的重要操作：

```python
# 创建一个三维张量用于演示
# shape=[2,3,6]可理解为：2个样本，每个样本3行6列的矩阵
# 应用场景：批次数据（batch_size=2, height=3, width=6）
tensor1 = torch.randint(1, 10, [2, 3, 6])
print("张量形状:", tensor1.shape)      # torch.Size([2, 3, 6])
print("张量维度数:", tensor1.ndim)     # 3（三维张量）
print("张量元素总数:", tensor1.numel()) # 2*3*6=36
print("张量内容:\n", tensor1)

# transpose(dim0, dim1)：交换两个指定维度
# 应用：矩阵转置、调整通道顺序（如NCHW↔NHWC）
# 示例：shape=[2,3,6]交换维度1和2→shape=[2,6,3]
# 注意：transpose只能交换两个维度，多维交换需用permute
tensor2 = tensor1.transpose(1, 2)  # 交换第1维(3)和第2维(6)
print("原始形状:", tensor1.shape)    # torch.Size([2, 3, 6])
print("交换后形状:", tensor2.shape)  # torch.Size([2, 6, 3])
print("交换后内容:\n", tensor2)

# .T：二维张量的转置快捷方式（等价于transpose(0,1)）
# 应用：矩阵转置、线性代数运算
# 注意：.T只适用于二维张量，高维张量会交换前两维
matrix = torch.randint(1, 10, [3, 4])  # 3x4矩阵
print("原始矩阵形状:", matrix.shape)    # torch.Size([3, 4])
print("转置后形状:", matrix.T.shape)    # torch.Size([4, 3])
print("原始矩阵:\n", matrix)
print("转置矩阵:\n", matrix.T)

# permute(*dims)：重新排列所有维度
# 参数：新维度的排列顺序（必须包含所有维度）
# 应用：图像格式转换（CHW→HWC）、调整批次维度顺序
# 示例：shape=[2,3,6]，permute(2,0,1)→维度顺序变为[第2维,第0维,第1维]→shape=[6,2,3]
tensor3 = tensor1.permute(2, 0, 1)  # 将维度重排为[6,2,3]
print("原始形状:", tensor1.shape)    # torch.Size([2, 3, 6])
print("permute后形状:", tensor3.shape) # torch.Size([6, 2, 3])
print("permute后内容:\n", tensor3)
```

### 5.6.2 张量重塑操作

重塑操作改变张量的形状而不改变数据：

```python
# reshape(*shape)：改变张量形状（不改变元素顺序和总数）
# 规则：新形状的元素总数必须等于原形状元素总数
# 应用：全连接层输入展平、调整批次大小
# 示例：[2,3,6]共36个元素→可reshape为[2,18]、[6,6]、[36]等
# 技巧：可用-1让PyTorch自动推断该维度大小
tensor4 = tensor1.reshape(2, 18)  # 2x3x6=36元素→reshape为2x18
print("原始形状:", tensor1.shape)   # torch.Size([2, 3, 6])
print("reshape后形状:", tensor4.shape) # torch.Size([2, 18])
print("reshape后内容:\n", tensor4)

# reshape使用-1自动推断维度
# -1表示：根据其他维度和总元素数自动计算该维度大小
# 示例：[2,3,6]共36元素，reshape(3,-1)→PyTorch计算-1位置为36/3=12
tensor5 = tensor1.reshape(3, -1)  # 自动计算为[3,12]
print("reshape(3,-1)形状:", tensor5.shape)  # torch.Size([3, 12])
tensor6 = tensor1.reshape(-1)     # 展平为一维向量
print("reshape(-1)形状:", tensor6.shape)    # torch.Size([36])

# view(*shape)：类似reshape，但要求张量内存连续
# 区别：view要求内存连续(contiguous)，reshape会在必要时复制数据
# 性能：view更快（不复制数据），但有内存连续性要求
# 检查：用is_contiguous()检查内存是否连续
tensor5 = tensor1  # 直接赋值，内存连续
print("tensor5是否内存连续:", tensor5.is_contiguous())  # True
print("view操作成功:", tensor5.view(2, 18).shape)      # torch.Size([2, 18])

# 处理非连续内存的张量
# transpose等操作会导致内存不连续，此时view会报错
# 解决：先调用contiguous()使内存连续，再使用view
tensor6 = tensor1.T  # 转置后内存不连续
print("tensor6是否内存连续:", tensor6.is_contiguous())  # False
# tensor6.view(6, 6)  # 这行会报错：RuntimeError

# 使用contiguous()转为连续内存
tensor6 = tensor6.contiguous()  # 复制数据使内存连续
print("contiguous后是否连续:", tensor6.is_contiguous())  # True
print("view操作成功:", tensor6.view(-1, 18).shape)      # 现在可以用view了

# flatten(start_dim, end_dim)：展平指定维度范围
# 参数：start_dim（起始维度），end_dim（结束维度，默认-1表示最后一维）
# 应用：卷积层输出展平后输入全连接层
# 示例：[2,3,6]从dim=1开始展平→[2, 3*6]=[2,18]
tensor7 = tensor1.flatten(start_dim=1)  # 保持dim0，展平后续维度
print("flatten(1)后形状:", tensor7.shape)  # torch.Size([2, 18])
tensor8 = tensor1.flatten()  # 全部展平为一维
print("flatten()后形状:", tensor8.shape)   # torch.Size([36])
```

### 5.6.3 维度增减操作

维度增减操作用于调整张量的维度数：

```python
# unsqueeze(dim)：在指定位置增加一个大小为1的维度
# 参数：dim（插入位置，可为负数）
# 应用：增加batch维度、扩展维度以匹配广播规则
# 示例：[3]→unsqueeze(0)→[1,3]，unsqueeze(1)→[3,1]
tensor1 = torch.tensor([1, 2, 3])
print("原始形状:", tensor1.shape)  # torch.Size([3])

tensor2 = tensor1.unsqueeze(0)  # 在第0维插入
print("unsqueeze(0)形状:", tensor2.shape)  # torch.Size([1, 3]) - 行向量
print("unsqueeze(0)内容:", tensor2)

tensor3 = tensor1.unsqueeze(1)  # 在第1维插入
print("unsqueeze(1)形状:", tensor3.shape)  # torch.Size([3, 1]) - 列向量
print("unsqueeze(1)内容:\n", tensor3)

# unsqueeze_()：原地操作版本（会修改原张量）
# 区别：unsqueeze()返回新张量，unsqueeze_()直接修改原张量
# 注意：原地操作更节省内存，但会改变原张量
tensor4 = torch.tensor([1, 2, 3])
print("原地操作前:", tensor4.shape)      # torch.Size([3])
tensor4.unsqueeze_(dim=0)  # 直接修改tensor4
print("原地操作后:", tensor4.shape)      # torch.Size([1, 3])
print("tensor4现在是:", tensor4)

# squeeze(dim)：移除大小为1的维度
# 参数：dim（可选，指定移除哪个维度；不指定则移除所有大小为1的维度）
# 应用：移除冗余维度、降维操作
# 示例：[1,3,1]→squeeze()→[3]，squeeze(0)→[3,1]
tensor5 = torch.tensor([[[1, 2, 3]]])  # shape=[1,1,3]
print("原始形状:", tensor5.shape)  # torch.Size([1, 1, 3])

tensor6 = tensor5.squeeze()  # 移除所有大小为1的维度
print("squeeze()形状:", tensor6.shape)  # torch.Size([3])

tensor7 = tensor5.squeeze(0)  # 只移除第0维
print("squeeze(0)形状:", tensor7.shape)  # torch.Size([1, 3])
```

### 5.6.4 张量拼接操作

拼接操作用于合并多个张量：

```python
# torch.cat(tensors, dim)：沿指定维度拼接张量
# 参数：tensors（张量列表/元组），dim（拼接维度）
# 要求：除拼接维度外，其他维度大小必须相同
# 应用：合并批次数据、特征拼接
# 示例：两个[2,3]沿dim=0拼接→[4,3]，沿dim=1拼接→[2,6]
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])    # shape=[2,3]
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]]) # shape=[2,3]

# 沿第0维（行）拼接 - 增加行数
tensor3 = torch.cat((tensor1, tensor2), dim=0)
print("沿dim=0拼接:\n", tensor3)
print("拼接后形状:", tensor3.shape)  # torch.Size([4, 3])

# 沿第1维（列）拼接 - 增加列数
tensor4 = torch.cat((tensor1, tensor2), dim=1)
print("沿dim=1拼接:\n", tensor4)
print("拼接后形状:", tensor4.shape)  # torch.Size([2, 6])

# torch.stack(tensors, dim)：沿新维度堆叠张量
# 区别cat：cat在现有维度拼接，stack创建新维度
# 要求：所有张量形状必须完全相同
# 应用：批次数据组合、时序数据堆叠
# 示例：两个[4,3,5]沿dim=0堆叠→[2,4,3,5]，沿dim=1堆叠→[4,2,3,5]
tensor1 = torch.randint(1, 10, [4, 3, 5])  # shape=[4,3,5]
tensor2 = torch.randint(1, 10, [4, 3, 5])  # shape=[4,3,5]

# 在第0维堆叠 - 创建新的批次维度
tensor3 = torch.stack([tensor1, tensor2], dim=0)
print("沿dim=0堆叠形状:", tensor3.shape)  # torch.Size([2, 4, 3, 5])

# 在第1维堆叠 - 在第1维位置插入新维度
tensor4 = torch.stack([tensor1, tensor2], dim=1)
print("沿dim=1堆叠形状:", tensor4.shape)  # torch.Size([4, 2, 3, 5])

# cat vs stack 对比示例
# cat：在现有维度上拼接，不增加维度数
# stack：创建新维度进行堆叠，维度数+1
a = torch.tensor([1, 2, 3])  # shape=[3]
b = torch.tensor([4, 5, 6])  # shape=[3]

cat_result = torch.cat([a, b], dim=0)
print("cat结果:", cat_result)           # tensor([1,2,3,4,5,6])
print("cat结果形状:", cat_result.shape) # torch.Size([6])

stack_result = torch.stack([a, b], dim=0)
print("stack结果:\n", stack_result)     # tensor([[1,2,3],[4,5,6]])
print("stack结果形状:", stack_result.shape) # torch.Size([2, 3])
```

### 5.6.5 张量分割操作

分割操作用于将张量拆分为多个部分：

```python
# torch.chunk(tensor, chunks, dim)：将张量分割为指定数量的块
# 参数：tensor（待分割张量），chunks（分割块数），dim（分割维度）
# 规则：如果不能均分，最后一块会较小
# 应用：多GPU训练数据分配、大批次数据分块处理
tensor1 = torch.arange(10)  # tensor([0,1,2,3,4,5,6,7,8,9])
chunks = torch.chunk(tensor1, 3, dim=0)  # 分成3块
print("分割成3块:")
for i, chunk in enumerate(chunks):
    print(f"  块{i}: {chunk}")  # [0,1,2,3], [4,5,6,7], [8,9]

# torch.split(tensor, split_size_or_sections, dim)：按指定大小分割
# 参数：split_size_or_sections（整数或列表）
#       - 整数：每块的大小
#       - 列表：每块的具体大小
# 应用：按需分割数据、多任务学习特征分离
tensor2 = torch.arange(10)

# 按固定大小分割
splits1 = torch.split(tensor2, 3, dim=0)  # 每块大小3
print("固定大小分割:")
for i, s in enumerate(splits1):
    print(f"  块{i}: {s}")  # [0,1,2], [3,4,5], [6,7,8], [9]

# 按列表指定每块大小
splits2 = torch.split(tensor2, [2, 3, 5], dim=0)  # 分别为2,3,5大小
print("\n自定义大小分割:")
for i, s in enumerate(splits2):
    print(f"  块{i}: {s}")  # [0,1], [2,3,4], [5,6,7,8,9]
```

### 5.6.6 张量重复操作

重复操作用于扩展张量的尺寸：

```python
# repeat(*sizes)：沿各维度重复张量
# 参数：各维度的重复次数（长度必须≥张量维度数）
# 应用：数据增强、广播操作的显式实现
# 示例：[2,3]repeat(2,3)→沿dim0重复2次，dim1重复3次→[4,9]
tensor1 = torch.tensor([[1, 2], [3, 4]])  # shape=[2,2]
tensor2 = tensor1.repeat(2, 3)  # 沿dim0重复2次，dim1重复3次
print("原始张量:\n", tensor1)
print("repeat(2,3)形状:", tensor2.shape)  # torch.Size([4, 6])
print("repeat后张量:\n", tensor2)

# repeat_interleave(repeats, dim)：在指定维度上交错重复元素
# 区别repeat：repeat整体重复，repeat_interleave逐元素重复
# 应用：上采样、标签扩展
# 示例：[1,2,3]repeat_interleave(2)→[1,1,2,2,3,3]
tensor3 = torch.tensor([1, 2, 3])
tensor4 = tensor3.repeat_interleave(2)  # 每个元素重复2次
print("原始张量:", tensor3)  # tensor([1, 2, 3])
print("repeat_interleave(2):", tensor4)  # tensor([1, 1, 2, 2, 3, 3])

# 二维张量的repeat_interleave
tensor5 = torch.tensor([[1, 2], [3, 4]])
tensor6 = tensor5.repeat_interleave(2, dim=0)  # 沿行重复
print("\n原始矩阵:\n", tensor5)
print("沿dim=0 repeat_interleave(2):\n", tensor6)
# 结果：[[1,2], [1,2], [3,4], [3,4]]
```

---

## 5.7 自动微分机制详解

### 5.7.1 计算图与梯度基础

PyTorch的自动微分（Autograd）机制是其最核心的特性之一，它能够自动计算张量的梯度，为神经网络的训练提供基础支持。

#### 自动微分的核心原理

自动微分的关键在于记录节点的数据与运算。数据记录在张量的 `data` 属性中，计算过程记录在 `grad_fn` 属性中。

计算图根据搭建方式可分为**静态图**和**动态图**：
- **静态图**（如TensorFlow 1.x）：先定义计算图，再执行计算
- **动态图**（PyTorch）：在计算过程中逐步搭建计算图，更灵活直观

PyTorch采用动态图机制，在计算的过程中逐步搭建计算图，同时对每个Tensor都存储 `grad_fn` 供自动微分使用。

#### 梯度计算的基本流程

若设置张量参数 `requires_grad=True`，则PyTorch会追踪所有基于该张量的操作，并在反向传播时计算其梯度。依赖于叶子节点的节点，`requires_grad` 默认为True。当计算到根节点后，在根节点调用 `backward()` 方法即可反向传播计算图中所有节点的梯度。

**重要概念**：
- **叶子节点**（Leaf Node）：用户直接创建的张量，是计算图的起点
- **非叶子节点**：通过运算得到的中间结果张量
- **梯度累积**：非叶子节点的梯度在反向传播之后会被释放（除非设置 `retain_grad=True`），而叶子节点的梯度会保留（累积）

通常需要使用 `optimizer.zero_grad()` 清零参数的梯度，防止梯度累积影响训练。

有时我们希望将某些计算移动到计算图之外，可以使用 `Tensor.detach()` 返回一个新的变量，该变量与原变量具有相同的值，但丢失计算图中如何计算原变量的信息。换句话说，梯度不会在该变量处继续向下传播。

```python
# 定义数据
# x：标量输入值，不参与梯度计算
# y：目标值，形状为(1,1)的二维张量
x=torch.tensor(10.0)
y=torch.tensor([[3.0]])
print('输入值 x:', x)
print('目标值 y:', y)

# 初始化参数
# w：权重参数，形状(1,1)，requires_grad=True表示需要计算梯度
# b：偏置参数，形状(1,1)，同样需要梯度计算
# rand()生成[0,1)区间均匀分布的随机数
w=torch.rand(1,1,requires_grad=True)
b=torch.rand(1,1,requires_grad=True)

# 前向传播，得到输出
# z = w*x + b：线性变换，构建计算图
# 由于w和b设置了requires_grad=True，z会记录grad_fn（梯度函数）
z=w*x+b
print('前向传播输出 z:', z)

# 检查张量的叶子节点属性
# is_leaf：True表示叶子节点（用户创建的张量），False表示中间计算结果
# 只有叶子节点才能积累梯度（grad属性）
print('x.is_leaf (输入):', x.is_leaf)
print('w.is_leaf (权重):', w.is_leaf)
print('b.is_leaf (偏置):', b.is_leaf)
print('z.is_leaf (计算结果):', z.is_leaf)
print('y.is_leaf (目标值):', y.is_leaf)

# 设置损失函数
# MSELoss：均方误差损失函数，计算预测值与真实值的平方差平均
# loss_value：损失值，is_leaf=False因为它是计算图的终点
loss=torch.nn.MSELoss()
loss_value=loss(z,y)
print('损失值:', loss_value, '是否为叶子节点:', loss_value.is_leaf)

# 反向传播
# backward()：从loss_value开始反向传播，自动计算所有requires_grad=True张量的梯度
# 梯度存储在对应张量的.grad属性中
loss_value.backward()

# 查看梯度
# w.grad：权重w相对于loss的梯度∂L/∂w
# b.grad：偏置b相对于loss的梯度∂L/∂b
# 梯度值用于指导参数更新方向
print('权重w的梯度:', w.grad)
print('偏置b的梯度:', b.grad)
```

### 5.7.2 requires_grad属性详解

requires_grad属性控制张量是否参与梯度计算：

```python
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

### 5.7.3 梯度计算示例

让我们通过具体例子来理解梯度计算过程：

```python
# 标量梯度
# 定义输入
x = torch.tensor(2.0, requires_grad=True)

# 定义函数 y = 3x^2 + 2x + 1
y = 3 * x ** 2 + 2 * x + 1

# 反向传播
y.backward()

# dy/dx = 6x + 2 = 14
print(x.grad)  # tensor(14.)

# 向量梯度
# 向量输入
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 定义函数 y = sum(x^2)
y = torch.sum(x ** 2)

# 反向传播
y.backward()

# dy/dx = 2x
print(x.grad)  # tensor([2., 4., 6.])
```

---

## 5.8 梯度管理详解

### 5.8.1 detach()方法详解

`detach()` 方法用于创建一个新的张量，该张量与原始张量共享数据，但不再参与梯度计算。这在某些场景下非常有用，例如：

1. **冻结部分网络参数**：在迁移学习中，冻结预训练模型的参数
2. **生成对抗网络（GAN）**：分离生成器和判别器的梯度计算
3. **特征提取**：只需要特征而不需要梯度时
4. **数值稳定性**：避免某些计算参与梯度传播

#### detach()的核心特性

- **共享数据**：`detach()` 返回的张量与原始张量共享底层数据存储
- **切断计算图**：新张量的 `requires_grad=False`，且没有 `grad_fn`
- **内存高效**：不复制数据，只是创建一个新的张量视图

#### 使用场景示例

在训练过程中，有时需要计算一些指标（如准确率）但不需要对这些计算进行反向传播。此时可以使用 `detach()` 将张量从计算图中分离出来，既能获取数值又能避免不必要的梯度计算。

```python
# detach()方法演示
# x：带梯度计算的张量，requires_grad=True
# y：通过detach()分离得到的新张量，不参与梯度计算
x=torch.tensor(2.0,requires_grad=True)
y=x.detach()
print('原始张量 x:', x)
print('分离张量 y:', y)

# 检查梯度计算属性
# requires_grad：True表示参与自动微分，False表示不参与
# detach()会创建新的张量，且requires_grad=False
print('x.requires_grad:', x.requires_grad)
print('y.requires_grad:', y.requires_grad)

print(id(x))
print(id(y))

print(x.untyped_storage().data_ptr())
print(y.untyped_storage().data_ptr())

# 分别对x和y进行后续计算
z1=x**2
z2=y**2
print(z1)
print(z2)

z1.sum().backward()
# z2.sum().backward()  # 这会报错，因为y不参与梯度计算

# 更复杂的detach示例
x=torch.ones(2,2,requires_grad=True)
y=x*x
print(x)
print(y)

# 分离y
u = y.detach()
print(u)

# 让u参与新张量计算
z=u*x

# 反向传播，计算所有梯度
z.sum().backward()
print(x.grad==u)
```

### 5.8.2 .data与.detach()的区别

在早期版本的PyTorch中，使用 `.data` 属性来获取不参与梯度计算的张量，但现在推荐使用 `.detach()` 方法。理解两者的区别对于编写健壮的PyTorch代码非常重要。

#### 主要区别对比

| 特性 | `.data` | `.detach()` |
|------|---------|-------------|
| 是否推荐 | ❌ 已弃用 | ✅ 推荐使用 |
| 返回类型 | 张量 | 张量 |
| 是否共享内存 | 是 | 是 |
| 安全性 | 低（可能修改原计算图） | 高（安全切断计算图） |
| 异常检测 | 不会检测 | 会检测in-place操作 |

#### 为什么推荐使用detach()？

1. **安全性**：`.data` 可能意外地修改原计算图，导致梯度计算错误
2. **异常检测**：`.detach()` 会更好地检测和处理in-place操作
3. **语义清晰**：明确表达了"切断梯度"的意图
4. **未来兼容**：`.data` 可能在未来的PyTorch版本中被移除

#### 最佳实践

- **总是使用 `.detach()`** 来创建不参与梯度计算的张量
- **避免使用 `.data`**，除非你明确知道自己在做什么
- **注意内存共享**：无论是 `.data` 还是 `.detach()`，返回的张量都与原张量共享数据，修改会影响原张量

```python
# 创建两个相同的张量用于对比实验
# x1, x2：都是需要梯度计算的张量，用于演示detach()和data的区别
x1=torch.tensor([1.0,2,3],requires_grad=True)
x2=torch.tensor([1.0,2,3],requires_grad=True)

# 对两个张量应用sigmoid激活函数
# sigmoid(x) = 1/(1+exp(-x))：S型激活函数，输出范围(0,1)
# y1, y2：都会记录计算图信息(grad_fn存在)
y1=x1.sigmoid()
y2=x2.sigmoid()
print('y1 (来自x1):', y1)
print('y2 (来自x2):', y2)

# detach() vs data 属性对比
# z1：使用.data属性获取数据，不推荐使用（旧API）
# z2：使用.detach()方法分离张量，推荐方式
# 两者都会返回不参与梯度计算的新张量
z1=y1.data
z2=y2.detach()
print('z1 (.data方式):', z1)
print('z2 (.detach()方式):', z2)

# 验证两种方式都不参与梯度计算
# .data和.detach()得到的张量requires_grad都为False
print('z1.requires_grad (.data):', z1.requires_grad)
print('z2.requires_grad (.detach()):', z2.requires_grad)

# 关键区别：.data可能修改原计算图，.detach()不会
# zero_()：将张量所有元素置零
# 由于z1通过.data获取，修改z1会影响y1的值
# z2通过.detach()获取，修改z2也会影响y2的值（共享数据）
z1.zero_()
z2.zero_()
print('修改后y1:', y1)
print('修改后y2:', y2)

# 反向传播验证梯度计算
# 由于y1的值被.zero_()修改为0，其梯度也会受到影响
# x1.grad应该为0，因为y1全为0
y1.sum().backward()
print('x1的梯度:', x1.grad)
```

---

## 5.9 机器学习实战：线性回归

本节将使用PyTorch实现一个完整的线性回归模型，综合运用本章所学的知识。

### 5.9.1 问题定义

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

### 5.9.2 数据准备

```python
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
noise=torch.randn(100,1)*0.5  # 生成噪声并缩放（标准差为0.5）
y=X*w+b+noise  # 根据线性模型 y = wx + b + noise 生成目标值

# 构建数据集和数据加载器
# TensorDataset将特征和标签打包成数据集
# 每个样本包含一对(X[i], y[i])
dataset=TensorDataset(X,y)
# DataLoader提供批量加载和数据打乱功能
# batch_size=10：每次训练使用10个样本
# shuffle=True：每个epoch随机打乱数据顺序，提高训练效果
dataloader=DataLoader(dataset,batch_size=10,shuffle=True)
```

### 5.9.3 模型定义

```python
# 2. 构建模型
# 使用PyTorch内置的线性层nn.Linear
# in_features=1：输入特征维度为1
# out_features=1：输出维度为1
# 模型形式：y = wx + b，其中w和b是待学习的参数
model=nn.Linear(in_features=1,out_features=1)
```

### 5.9.4 训练配置

```python
# 3. 定义损失函数和优化器
# 损失函数：均方误差（Mean Squared Error）
# 衡量预测值与真实值之间的平方差
loss=nn.MSELoss()
# 优化器：随机梯度下降（SGD）
# model.parameters()：获取模型中所有可学习参数（w和b）
# lr=0.001：学习率，控制参数更新的步长
optimizer=optim.SGD(model.parameters(),lr=0.001)
```

### 5.9.5 训练过程

```python
# 4. 训练模型
epoch_num=1000  # 训练轮次：遍历完整数据集的次数
loss_list=[]  # 记录每个epoch的平均损失值，用于绘制损失曲线

# 开始训练循环
for epoch in range(epoch_num):
    # 一个训练轮次（epoch）的迭代过程
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
```

### 5.9.6 结果分析

```python
# 5. 打印训练后的模型参数
# model.weight：线性层的权重参数（形状为[1,1]）
# model.bias：线性层的偏置参数（形状为[1]）
# .item()将单元素张量转换为Python标量
print(f'训练得到的参数 - 权重w: {model.weight.item():.4f}, 偏置b: {model.bias.item():.4f}')
print(f'真实参数 - 权重w: 2.5000, 偏置b: 5.2000')
```

### 5.9.7 可视化结果

```python
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
```

---

## 本章小结

### 核心概念回顾

1. **张量基础**：
   - PyTorch的核心数据结构是张量（Tensor）
   - 支持多种数据类型（int、float、bool等）
   - 可以创建标量、向量、矩阵和高维张量

2. **张量操作**：
   - **创建**：从数据、随机数、特定值创建
   - **转换**：类型转换、与NumPy互转、形状变换
   - **计算**：四则运算、矩阵乘法、哈达玛积
   - **索引**：基础索引、切片、列表索引、布尔索引
   - **形状**：维度交换、重塑、增减维度、拼接分割

3. **自动微分机制**：
   - 通过`requires_grad`属性控制梯度追踪
   - 使用计算图记录运算过程
   - 通过`.backward()`方法进行反向传播
   - 梯度存储在张量的`.grad`属性中

4. **梯度管理**：
   - `detach()`方法切断计算图连接
   - `.data`属性（已弃用，推荐使用`detach()`）
   - 梯度清零的重要性

5. **实战应用**：
   - 使用`nn.Module`定义模型
   - 使用`DataLoader`处理数据
   - 完整的训练循环流程

### PyTorch vs NumPy对比

| 特性 | NumPy | PyTorch |
|------|-------|---------|
| 核心数据 | ndarray | Tensor |
| GPU支持 | ❌ | ✅ |
| 自动微分 | ❌ | ✅ |
| 深度学习 | 手动实现 | 内置支持 |
| 生态系统 | 科学计算 | 深度学习 |
| 内存管理 | CPU only | CPU/GPU统一接口 |

### 学习建议

1. **循序渐进**：先掌握张量基础操作，再学习自动微分
2. **动手实践**：多写代码，通过实际例子加深理解
3. **关注内存**：理解原地操作和内存共享的概念
4. **调试技巧**：学会使用`.grad`、`.requires_grad`等属性进行调试
5. **性能优化**：了解view vs reshape、in-place操作等性能差异

### 下一步学习

掌握了本章内容后，建议继续学习：
- 神经网络层的使用（nn.Module的高级特性）
- 优化器的选择和使用
- 数据加载和预处理
- GPU加速和分布式训练
- 模型保存和加载

通过系统学习这些内容，你将能够使用PyTorch构建和训练各种深度学习模型。