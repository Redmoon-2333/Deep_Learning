# 阶跃函数
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0

import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax0(x):
    return np.exp(x) / np.sum(np.exp(x))
# 考虑输入为矩阵
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x,axis=0)
        y = np.exp(x) / np.sum(np.exp(x),axis=0)
        return y.T
    # 溢出处理策略
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def identity(x):
    return x

def leaky_relu(x):
    return np.maximum(0.01*x, x)

def swish(x):
    return x * sigmoid(x)

def softplus(x):
    return np.log(1 + np.exp(x))

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4,5,-1, -2,-3,-4,-5])
    print(step_function(x))
    print(sigmoid(x))
    print(relu(x))
    print(softmax0(x))
    X=np.array([[0,1,2],[3,4,5],[6,7,8]])
    # 保留4位小数
    print(np.round(softmax0(X),4))
