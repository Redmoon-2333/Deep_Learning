import numpy as np
from common.functions import *
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
def get_data():
    # 从文件加载数据集
    data = pd.read_csv("../data/train.csv")
    # 2 . 划分数据集
    X= data.drop(["label"],axis=1)
    y= data["label"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # 3. 特征转换
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test,y_test

# 初始化神经网络
def init_network():
    # 直接从文件中加载字典对象
    network=joblib.load("../data/nn_sample")
    return network

# 前向传播
def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = identity(a3)
    return y


# 主流程
x,y=get_data()

network=init_network()

# 定义变量
batch_size = 100
accuracy_cnt = 0
n=x.shape[0]

# 循环迭代，分批次测试，前向传播，累计预测准确个数
for i in range(0, n, batch_size):
    x_batch = x[i:i+batch_size]
    y_pred = forward(network, x_batch)
    p = np.argmax(y_pred, axis=1)
    accuracy_cnt += np.sum(p == y[i:i+batch_size])

# 计算分类准确率
print("Accuracy: ", accuracy_cnt/n)

