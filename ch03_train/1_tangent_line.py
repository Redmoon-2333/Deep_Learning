import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_diff


# 原函数 y=0.01x^2+0.1x
def f(x):
    return 0.01*x**2 + 0.1*x
# 切线方程函数，返回切线函数
def tangent_line(f, x):
    #计算x处切线的斜率（利用数值微分）
    y = f(x)
    a = numerical_diff(f, x)
    print("斜率为", a)
    # 计算截距
    b = y - a*x
    return lambda x: a*x + b

# 定义画图范围
x_range = np.arange(0.0, 20.0, 0.1)
y_range = f(x_range)

# 计算x=5的切线方程
f_line = tangent_line(f, x=5)
y_line = f_line(x_range)

plt.plot(x_range, y_range)
plt.plot(x_range, y_line)
plt.show()

