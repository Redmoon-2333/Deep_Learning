import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 1. 加载数据
fashion_data = pd.read_csv('../data/fashion-mnist_train.csv').fillna(0)  # 填充NaN为0
fashion_test = pd.read_csv('../data/fashion-mnist_test.csv').fillna(0)  # 填充NaN为0
# 提取X和y，转换为张量
x_train = fashion_data.iloc[:, 1:].values
x_train = torch.tensor(x_train, dtype=torch.float).reshape(-1, 1,28, 28)
y_train = fashion_data.iloc[:, 0].values
y_train = torch.tensor(y_train, dtype=torch.int64)
x_test = fashion_test.iloc[:, 1:].values
x_test = torch.tensor(x_test, dtype=torch.float).reshape(-1,1, 28, 28)
y_test = fashion_test.iloc[:, 0].values
y_test = torch.tensor(y_test, dtype=torch.int64)
# # 显示图片和标签信息
# plt.imshow(x_train[12345,0,:,:],cmap='gray')
# plt.show()
# print(y_train[12345])

# 构建数据集
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
# 2. 创建模型
model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=400, out_features=120),
    nn.Sigmoid(),
    nn.Linear(in_features=120, out_features=84),
    nn.Sigmoid(),
    nn.Linear(in_features=84, out_features=10)
)
x=torch.rand(size=(1,1,28,28),dtype=torch.float)
for layer in model:
    x=layer(x)
    print(f"{layer.__class__.__name__}: {x.shape}")

# 3. 训练模型
def train_test(model, train_dataset, test_dataset,lr,epoch_num,batch_size,device):
    # 参数初始化函数
    def init_params(layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
    # 3.1 初始化相关操作
    model.apply(init_params)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3.2 训练过程
    for epoch in range(epoch_num):
        model.train()
        # 定义DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss = 0
        train_correct_num = 0
        # 循环迭代
        for batch_idx, (X,y) in enumerate(train_loader):
            X,y = X.to(device), y.to(device)
            # 3.2.1 前向传播
            output = model(X)
            # 3.2.2 计算损失
            loss_value = loss(output, y)
            # 3.2.3 反向传播
            loss_value.backward()
            # 3.2.4 更新参数
            optimizer.step()
            # 3.2.5 梯度清零
            optimizer.zero_grad()
            # 累计损失
            train_loss += loss_value.item()*X.shape[0]
            # 累加预测正确的数量
            pred=output.argmax(dim=1)
            train_correct_num += (pred==y).sum()

            # 打印进度条
            print(f"\rEpoch:{epoch+1:0>2}[{'='* int((batch_idx+1)/len(train_loader)*50)}]",end="")
        # 一轮结束，计算平均误差和准确率
        this_loss = train_loss/len(train_dataset)
        accuracy = train_correct_num/len(train_dataset)

        # 3.3 测试验证
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_correct_num = 0
        # 迭代预测，进行准确数量的累加
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                pred = output.argmax(dim=1)
                test_correct_num += (pred == y).sum()
        # 计算准确率
        this_test_accuracy = test_correct_num/len(test_dataset)

        print(f"train_loss: {this_loss:.4f}, train_acc: {accuracy:.4f}, test_acc: {this_test_accuracy:.4f}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 超参数
lr=0.01
epoch_num=20
batch_size=256
train_test(model, train_dataset, test_dataset,lr,epoch_num,batch_size,device)

# 选取一个数据，进行测试对比
plt.imshow(x_test[666,0,:,:],cmap='gray')
plt.show()
print(y_test[666]) # 查看真是标签
# 传入模型，前向传播，进行预测
output = model(x_test[666].unsqueeze(0).to(device))
y_pred = output.argmax(dim=1)
print(y_pred)



