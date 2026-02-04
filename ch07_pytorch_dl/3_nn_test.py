import torch
import torch.nn as nn

# 自定义神经网络类
class Model(nn.Module):
    # 初始化方法
    def __init__(self, device):
       super().__init__()
       # 定义三个线性层
       self.linear1 = nn.Linear(3, 4, device=device)
       nn.init.xavier_normal_(self.linear1.weight)
       self.linear2 = nn.Linear(4, 4, device=device)
       nn.init.kaiming_normal_(self.linear2.weight)
       self.out= nn.Linear(4, 2, device='cuda')
    # 前向传播
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.out(x)
        x = torch.softmax(x,dim=1)
        return x
# 全局变量device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 测试
# 1. 定义输入数据
x = torch.randn(10, 3, device=device)

# 2. 创建模型
model = Model(device=device)

# 3. 前向传播
output=model(x)
print("神经网络输出为:", output)

# 调用 parameters() 方法
# for param in model.parameters():
#     print(param)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param)

# 调用state_dict() 方法，得到模型的参数
print()
print(model.state_dict())

from torchsummary import summary
summary(model, (3, ),batch_size=10, device='cuda')

