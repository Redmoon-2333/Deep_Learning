import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['b2'] = np.zeros((1, output_size))

    # 前向传播
    # x @ W1 等价于 np.dot(x, W1)，是Python 3.5+的矩阵乘法运算符
    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = x @ W1 + b1       # (batch, 784) @ (784, 50) -> (batch, 50)
        z1 = sigmoid(a1)
        a2 = z1 @ W2 + b2      # (batch, 50) @ (50, 10) -> (batch, 10)
        y = softmax(a2) 
        return y

    # 计算损失
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy(y, t)
        return loss

    # 计算准确度
    def accuracy(self, x, t):
        y = self.forward(x)
        # 根据最大概率的到分类号
        y = np.argmax(y, axis=1)
        # 与正确的标签对比，获得准确率
        accuracy = np.sum(y==t) / x.shape[0]
        return accuracy

    # 计算梯度
    # 注意：lambda 参数名用 _ 而非 x，避免覆盖外部输入数据 x
    # numerical_gradient 会直接修改 self.params 中的值来计算数值梯度
    def numerical_gradient(self, x, t):
        loss_f = lambda _: self.loss(x, t)  # _ 是占位符，实际用的是被扰动的 self.params
        grads = {}
        grads['W1'] = numerical_gradient(loss_f, self.params['W1'])
        grads['W2'] = numerical_gradient(loss_f, self.params['W2'])
        grads['b1'] = numerical_gradient(loss_f, self.params['b1'])
        grads['b2'] = numerical_gradient(loss_f, self.params['b2'])
        return grads
    
