import torch
import torch.nn as nn

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 只定义一个全连接层
        self.linear = nn.Linear(in_features=5, out_features=3)
        # 权重初始化
        self.linear.weight.data=torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.2, 0.3, 0.2, 0.1, 0.5]
        ])
        self.linear.bias.data=torch.tensor([0.1, 0.2, 0.3])
    # 前向传播
    def forward(self, x):
        return self.linear(x)

# 主流程
# 1. 定义数据2*5
x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],dtype=torch.float)
# 目标值2*3
y = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],dtype=torch.float)
# 2. 模型定义
model = Model()
# 3. 损失函数定义
criterion = nn.MSELoss()
# 4. 损失计算
loss = criterion(model(x), y)
# 5. 反向传播，计算梯度
loss.backward()
# 6. 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 7. 参数更新
optimizer.step()
optimizer.zero_grad()
# 8. 输出参数
for param in model.state_dict():
    print(param)
    print(model.state_dict()[param])









