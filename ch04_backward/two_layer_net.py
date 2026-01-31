import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict # 有序字典，用来保存层结构

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['b2'] = np.zeros((1, output_size))
        # 定义层结构
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    # 前向传播
    def forward(self, x):
        # 对于每一层，依次调用forward方法
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 计算损失
    def loss(self, x, t):
        y = self.forward(x)
        loss = self.lastLayer.forward(y, t)
        return loss

    # 计算准确度
    def accuracy(self, x, t):
        y = self.forward(x) # 预测分类数值
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
    # 计算梯度：反向传播
    def gradient(self, x, t):
        # 前向传播计算损失
        self.loss(x, t)
        # 反向传播计算梯度
        dy = 1
        dy = self.lastLayer.backward(dy)
        # 将所有层反向处理
        layers = list(self.layers.values())
        for layer in reversed(layers):
            dy = layer.backward(dy)
        # 提取各层的梯度
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['W2'] = self.layers['Affine2'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['b2'] = self.layers['Affine2'].db
        return grads
