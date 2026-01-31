import numpy as np
from common.functions import *
import matplotlib.pyplot as plt
from common.gradient import numerical_gradient

# 定义梯度下降法的函数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x, np.array(x_history)

# 定义目标函数
def f(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    # 定义初始值
    init_x = np.array([-3.0, 4.0])
    # 定义超参数
    lr = 0.9
    num_iter = 2000
    # 梯度下降法
    x, x_history = gradient_descent(f, init_x, lr=lr, step_num=num_iter)
    print("最小值点", x)
    # 画图
    plt.plot([-5,5],[0,0],'--b')
    plt.plot([0,0],[-5,5],'--b')
    plt.plot(x_history[:, 0], x_history[:, 1], "o")
    plt.xlim([-3.5,3.5])
    plt.ylim([-4.5,4.5])
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()

