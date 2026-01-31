import numpy as np
import pandas as pd
from common.load_data import get_data
from two_layer_net import TwoLayerNet

# 1. 加载数据
x_train, y_train, x_test, y_test = get_data()
# 2. 初始化模型
model = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# 3. 设置超参数
learning_rate = 0.1
num_epochs = 20
batch_size = 100
train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size
iter_num = num_epochs * iter_per_epoch

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 4. 循环迭代，用梯度下降法训练模型
for i in range(iter_num):
    # 4.1 随机选取批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = y_train[batch_mask]
    # 4.2 计算梯度（反向传播）
    grad = model.gradient(x_batch, t_batch)
    # 4.3 更新参数
    for key in ('W1', 'W2', 'b1', 'b2'):
        model.params[key] -= learning_rate * grad[key]
    # 4.4 保存当前训练损失
    train_loss_list.append(model.loss(x_batch, t_batch))
    # 4.5 每个epoch结束时计算准确度
    if i % iter_per_epoch == 0:
        train_acc = model.accuracy(x_train, y_train)
        test_acc = model.accuracy(x_test, y_test)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

# 5. 画图
import matplotlib.pyplot as plt
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc')
plt.legend(loc='best')
plt.show()



