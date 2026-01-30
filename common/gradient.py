
import numpy as np

# 数值微分求导
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h )) / (2 * h)

# 数值微分求梯度
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    # 遍历x的特征
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp
    return grad

# 数值梯度计算（支持矩阵输入）
# 用于计算损失函数对参数的梯度，是梯度下降算法的基础
# 参数:
#   f: 损失函数，接受参数并返回损失值
#   X: 输入参数，可以是一维数组（单样本）或二维矩阵（多样本，每行一个样本）
# 返回:
#   与X同形状的梯度矩阵
# 示例:
#   f = lambda x: x[0]**2 + x[1]**2  # 损失函数
#   X = np.array([[3.0, 4.0], [1.0, 2.0]])  # 2个样本
#   grad = numerical_gradient(f, X)  # 返回 [[6.0, 8.0], [2.0, 4.0]]
def numerical_gradient(f, X):
    # 判断输入维度，分别处理单样本和多样本情况
    if X.ndim == 1:
        # 一维数组（单样本）：直接调用底层梯度计算函数
        return _numerical_gradient(f, X)
    else:
        # 二维矩阵（多样本）：创建同形状的零矩阵存储梯度
        grad = np.zeros_like(X)
        
        # 遍历矩阵的每一行（每个样本）
        # enumerate返回 (i, x)，i是行索引，x是该行数据
        for i, x in enumerate(X):
            grad[i] = _numerical_gradient(f, x)
        return grad
