import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

# 创建数据集
def create_dataset():
    # 1. 从文件读取数据
    data = pd.read_csv('../data/house_prices.csv')
    # 2. 数据预处理,去除无关列
    data = data.drop(['Id'], axis=1)
    # 3. 划分特征和目标
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    # 4. 划分训练集和测试机
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 5. 特征转换
    # 5.1 按照特征数据类型划分为数值型和类别行
    # 转换为列表以避免 Pandas Index 与 scikit-learn 的兼容性问题
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
    # 5.2 创建特征转换器
    # 5.2.1 数值型特征：平均值填充，然后标准化
    numeric_transformer = Pipeline(
        steps=[
        ('fillna', SimpleImputer(strategy='mean')),
        ('std', StandardScaler())
        ]
    )
    # 5.2.2 类别特征：用默认值填充，然后独热编码
    categorical_transformer = Pipeline(steps=[
        ('fillna', SimpleImputer(strategy='constant', fill_value='NaN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    # 5.2.3 组合列转换器
    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    # 5.3 进行特征转换
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    X_train = pd.DataFrame(X_train.toarray(), columns=transformer.get_feature_names_out())
    X_test = pd.DataFrame(X_test.toarray(), columns=transformer.get_feature_names_out())
    # 6. 构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))
    # 7. 返回数据集以及特征数量
    return train_dataset, test_dataset, X_train.shape[1]

# 测试
train_dataset, test_dataset, feature_num = create_dataset()
# print(feature_num)
# 创建模型
model = nn.Sequential(
    nn.Linear(feature_num, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1)
)
# 自定义损失函数
def loss_rmse(y_pred, y_true):
    y_pred = torch.clamp(y_pred,1,float("inf"))
    mse = nn.MSELoss()
    return torch.sqrt(mse(torch.log(y_pred+1e-10), torch.log(y_true+1e-10)))
# 模型训练和测试
def train_test(model, train_dataset, test_dataset,lr,epoch_num,batch_size,device):
    # 1. 初始化相关操作
    def init_params(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
    # 1.1 参数初始化
    model.apply(init_params)
    # 1.2 将模型加载到设备
    model = model.to(device)
    # 1.3 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 1.4，定义训练误差和测试误差变化列表
    train_loss_list = []
    test_loss_list = []
    # 2. 模型训练
    for epoch in range(epoch_num):
        model.train()
        # 2.1 创建数据加载器
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss_total = 0
        # 2.2 迭代训练
        for batch_idx,(X,y) in enumerate(dataloader):
            # 加载数据到设备
            X = X.to(device)
            y = y.to(device)
            #2.3.1 前向传播
            y_pred = model(X)
            # 2.3.2 计算损失
            loss = loss_rmse(y_pred.squeeze(), y)
            # 2.3.3 反向传播
            loss.backward()
            # 2.3.4 参数更新
            optimizer.step()
            optimizer.zero_grad()
            # 累加损失
            train_loss_total+=loss.item()*X.shape[0]
        train_loss_list.append(train_loss_total/len(train_dataset))
        # 3. 测试
        test_train_loss=0
        model.eval()
        # 3.1 创建数据加载器
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # 3.2 计算测试误差
        test_loss_total = 0
        with torch.no_grad(): # 禁用梯度计算
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss_value = loss_rmse(y_pred.squeeze(), y)
                test_loss_total += loss_value.item() * X.shape[0]
        this_test_loss = test_loss_total / len(test_dataset)
        test_loss_list.append(this_test_loss)

        print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {train_loss_total/len(train_dataset):.4f}, Test Loss: {this_test_loss:.4f}')
    return train_loss_list, test_loss_list

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
lr = 0.1
epoch_num = 200
batch_size = 64
train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device)

# 画图
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



