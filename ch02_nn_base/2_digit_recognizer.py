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
# print(network['W1'].shape)
# print(network['W2'].shape)
# print(network['W3'].shape)
# print(network['b1'].shape)
# print(network['b2'].shape)
# print(network['b3'].shape)

y_pred=forward(network,x)

# 分类概率转换为预测标签
y_pred=np.argmax(y_pred,axis=1)
print (y_pred)

# 计算分类准确率
accuracy=np.mean(y_pred==y)
print("Accuracy: ", accuracy)

