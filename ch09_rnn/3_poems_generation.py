"""
================================================================================
古诗生成系统 - 基于RNN的文本生成模型
================================================================================

【整体设计思路】

本系统实现了一个基于循环神经网络(RNN)的中文古诗自动生成模型。核心设计思想是：
通过训练RNN学习古诗的字符级序列模式，使模型能够根据给定的起始字符，逐字生成
符合古诗格律和语义连贯性的诗句。

【算法选型依据】

1. **RNN架构选择**：选用基础RNN而非LSTM/GRU，原因如下：
   - 古诗字符序列相对较短（通常4-8句，每句5-7字）
   - 基础RNN参数量少，训练速度快
   - 对于短序列，基础RNN已能捕捉足够的时序依赖

2. **字符级建模**：采用字符级(Character-level)而非词级(Word-level)建模：
   - 中文古诗以单字为最小语义单位
   - 字符级词表小，训练效率高
   - 能够生成更灵活的字符组合

3. **自回归生成**：采用自回归(Autoregressive)生成策略：
   - 每个新生成的字符作为下一个预测的输入
   - 保证生成序列的连贯性
   - 支持变长序列生成

【架构设计理念】

系统采用经典的三层架构设计：

┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据预处理层                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  文本读取    │ -> │  清洗分字    │ -> │  构建词表    │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据加载层                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PoetryData (Dataset)                                               │   │
│  │  - 序列切分：将长序列切分为固定长度的输入-目标对                      │   │
│  │  - 滑动窗口：使用滑动窗口生成训练样本                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           模型层                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │  Embedding  │ -> │    RNN      │ -> │   Linear    │                     │
│  │  (字符嵌入)  │    │ (序列建模)   │    │ (输出预测)   │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           训练与生成层                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  模型训练    │ -> │  损失计算    │ -> │  古诗生成    │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘

【核心功能模块划分】

1. **数据预处理模块 (preprocess_poems)**
   - 功能：读取古诗文本，构建字符词表
   - 输入：古诗文本文件路径
   - 输出：ID序列列表、id2word映射、word2id映射

2. **数据集模块 (PoetryData)**
   - 功能：将长序列切分为训练样本
   - 核心：滑动窗口机制生成(input, target)对
   - 继承：torch.utils.data.Dataset

3. **模型模块 (PoetryRNN)**
   - 功能：定义RNN网络结构
   - 组成：Embedding层 + RNN层 + 全连接层
   - 继承：torch.nn.Module

4. **训练模块 (train)**
   - 功能：模型训练循环
   - 包含：前向传播、损失计算、反向传播、参数更新

5. **生成模块 (generate_poem)**
   - 功能：使用训练好的模型生成古诗
   - 策略：基于概率分布的随机采样

【各模块间的交互逻辑与数据流向】

1. 训练阶段数据流：
   poems.txt -> preprocess_poems() -> id_seqs
   id_seqs -> PoetryData -> DataLoader -> model.forward() -> loss.backward()

2. 生成阶段数据流：
   start_token -> word2id -> model.forward() -> softmax -> multinomial
   -> next_id -> id2word -> 生成字符 -> 循环直到完成

================================================================================
"""

import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import re


# ================================================================================
# 模块一：数据预处理
# ================================================================================

def preprocess_poems(file_path):
    """
    古诗数据预处理函数

    【功能描述】
    读取原始古诗文本文件，进行数据清洗、字符分割、词表构建和序列化转换。
    将文本形式的数据转换为模型可处理的数值ID序列。

    【处理流程】
    1. 文件读取：逐行读取UTF-8编码的古诗文本
    2. 数据清洗：使用正则表达式去除中文标点符号
    3. 字符提取：将每句诗拆分为单个字符列表
    4. 词表构建：基于字符集合构建id2word和word2id双向映射
    5. 序列化：将字符序列转换为ID序列

    【参数说明】
    :param file_path: str类型，古诗文本文件的相对或绝对路径
                      文件格式要求：每行一首诗，UTF-8编码

    【返回值】
    :return: tuple类型，包含三个元素：
             - id_seqs: list[list[int]]，所有诗的ID序列列表
             - id2word: list[str]，索引到字符的映射列表
             - word2id: dict[str, int]，字符到索引的映射字典

    【关键技术点】
    1. 正则表达式 re.sub(r"[，。？！、：]", "", line)
        - 匹配中文常用标点：逗号、句号、问号、叹号、顿号、冒号
        - 替换为空字符串，实现标点去除

    2. 集合(set)数据结构
        - char_set 使用集合存储字符，自动去重
        - 时间复杂度O(1)的查找效率

    3. 词表构建策略
        - 在字符集合基础上添加'<UNK>'特殊标记
        - '<UNK>'用于处理未登录词(Out-Of-Vocabulary)
    """
    char_set = set()
    poems = []

    # 1.1 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 数据清洗，去掉标点以及两侧空白
            # re.sub(pattern, repl, string): 将string中匹配pattern的部分替换为repl
            line = re.sub(r"[，。？！、：]", "", line).strip()
            # 按字分割，去重，保存到set
            char_set.update(list(line))
            poems.append(list(line))

    # 1.2 构建词表
    # id2word: 列表结构，索引即字符ID，值即字符本身
    # 添加'<UNK>'标记用于处理训练时未出现的字符
    id2word = list(char_set) + ['<UNK>']
    # word2id: 字典推导式构建反向映射，字符为键，索引为值
    word2id = {word: id for id, word in enumerate(id2word)}

    # 1.3 将诗句id化
    # 遍历每首诗，将字符序列转换为ID序列
    id_seqs = []
    for poem in poems:
        # word2id.get(word): 获取字符对应的ID
        id_seq = [word2id.get(word) for word in poem]
        id_seqs.append(id_seq)

    return id_seqs, id2word, word2id


# 执行数据预处理，加载古诗数据
# 假设数据文件位于上级目录的data文件夹中
id_seqs, id2word, word2id = preprocess_poems('../data/poems.txt')


# ================================================================================
# 模块二：自定义数据集
# ================================================================================

class PoetryData(Dataset):
    """
    古诗数据集类

    【功能描述】
    继承PyTorch的Dataset类，用于加载和预处理古诗训练数据。
    采用滑动窗口机制将长序列切分为固定长度的训练样本对(input, target)。

    【核心机制 - 滑动窗口】
    对于序列 [c1, c2, c3, c4, c5, c6] 和 seq_len=3：
    - 样本1: X=[c1,c2,c3], y=[c2,c3,c4]  (预测下一个字符)
    - 样本2: X=[c2,c3,c4], y=[c3,c4,c5]
    - 样本3: X=[c3,c4,c5], y=[c4,c5,c6]

    这种设计使得模型学习"给定前文，预测下一个字符"的任务。

    【属性说明】
    :attr id_seqs: list[list[int]]，所有古诗的ID序列
    :attr seq_len: int，序列长度，即每次输入的字符数
    :attr data: list[tuple[list[int], list[int]]]，训练样本列表

    【方法说明】
    :method __init__: 构造函数，初始化数据集
    :method __len__: 返回数据集大小，供DataLoader使用
    :method __getitem__: 根据索引获取单个样本，供DataLoader使用
    """

    def __init__(self, id_seqs, seq_len):
        """
        构造函数

        【参数说明】
        :param id_seqs: list[list[int]]，所有诗的ID序列列表
        :param seq_len: int，序列长度L，即输入序列的固定长度

        【初始化逻辑】
        1. 保存原始ID序列和序列长度参数
        2. 使用滑动窗口遍历所有诗，生成(input, target)训练对
        3. 将生成的样本存储在self.data列表中

        【滑动窗口算法详解】
        对于每首诗的ID序列，从索引0开始，以步长1滑动：
        - input: id_seq[i : i+seq_len]
        - target: id_seq[i+1 : i+1+seq_len] (input整体右移一位)
        这样target就是input的"下一个字符"序列，构成监督学习的标签。
        """
        self.id_seqs = id_seqs
        self.seq_len = seq_len
        self.data = []  # 保存元组（X,y）的列表

        # 遍历所有诗
        for id_seq in id_seqs:
            # 遍历所有可能的起始位置
            # range(0, len(id_seq) - self.seq_len) 确保有足够长度截取
            for i in range(0, len(id_seq) - self.seq_len):
                # 以当前字为起点，截取长度为seq_len的序列
                # X: 输入序列，用于预测
                # y: 目标序列，作为预测目标
                self.data.append((
                    id_seq[i:i + self.seq_len],           # 输入序列
                    id_seq[i + 1:i + 1 + self.seq_len]    # 目标序列(右移一位)
                ))

    def __len__(self):
        """
        返回数据集大小

        【功能】
        返回self.data列表的长度，即训练样本总数。
        该方法被PyTorch的DataLoader调用，用于确定数据迭代次数。

        【返回值】
        :return: int，训练样本总数
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        通过索引获取数据样本

        【功能】
        根据给定的索引idx，返回对应的训练样本(input, target)。
        将Python列表转换为PyTorch的LongTensor类型，便于GPU计算。

        【参数说明】
        :param idx: int，样本索引，范围在[0, len(self.data))之间

        【返回值】
        :return: tuple[torch.LongTensor, torch.LongTensor]
                 - x: 输入序列张量，形状为(seq_len,)
                 - y: 目标序列张量，形状为(seq_len,)

        【数据类型说明】
        torch.LongTensor: 64位整数张量，用于存储类别索引
                          在Embedding层中，输入必须是Long类型
        """
        x = torch.LongTensor(self.data[idx][0])
        y = torch.LongTensor(self.data[idx][1])
        return x, y


# 创建数据集实例，序列长度设置为24
# 这个长度可以覆盖大多数古诗的一句或两句
dataset = PoetryData(id_seqs, seq_len=24)


# ================================================================================
# 模块三：模型定义
# ================================================================================

class PoetryRNN(nn.Module):
    """
    古诗生成RNN模型

    【功能描述】
    定义基于RNN的古诗生成神经网络架构。模型由三个主要部分组成：
    1. Embedding层：将字符ID映射为稠密向量
    2. RNN层：学习序列的时序依赖关系
    3. 全连接层：将RNN输出映射为词表大小的概率分布

    【架构详解】

    输入(字符ID) -> Embedding -> RNN -> Linear -> 输出(词表概率分布)
         |            |         |        |
         |            ↓         ↓        |
         |       [embedding_dim] [hidden_size] [vocab_size]
         |            |         |        |
         ↓            ↓         ↓        ↓
    [batch, seq] [batch, seq, embed] [batch, seq, hidden] [batch, seq, vocab]

    【数学原理】

    1. Embedding层：
       e_t = W_embed[x_t]  # 查表操作，x_t是字符ID

    2. RNN层：
       h_t = tanh(W_ih * e_t + W_hh * h_{t-1} + b)
       其中：
       - W_ih: 输入到隐状态的权重矩阵
       - W_hh: 隐状态到隐状态的权重矩阵（循环连接）
       - h_t: 第t时间步的隐状态

    3. 全连接层：
       o_t = W_out * h_t + b_out
       输出维度为vocab_size，每个值代表对应字符的logit分数

    【属性说明】
    :attr embed: nn.Embedding，字符嵌入层
    :attr rnn: nn.RNN，循环神经网络层
    :attr linear: nn.Linear，输出投影层
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        """
        构造函数

        【参数说明】
        :param vocab_size: int，词汇表大小，即不同字符的总数
        :param embedding_dim: int，嵌入维度，每个字符用多少维向量表示
        :param hidden_size: int，RNN隐藏状态维度，决定模型的记忆容量
        :param num_layers: int，RNN层数，默认1层，多层可增强表达能力

        【层初始化详解】

        1. nn.Embedding(vocab_size, embedding_dim):
           - 创建一个可学习的查找表，形状为(vocab_size, embedding_dim)
           - 每行对应一个词表中的字符的向量表示
           - 初始值为随机初始化，训练过程中不断更新

        2. nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True):
           - input_size=embedding_dim: 输入特征维度，即Embedding的输出维度
           - hidden_size: 隐藏状态维度，决定RNN的表达能力
           - num_layers: 堆叠的RNN层数
           - batch_first=True: 输入输出张量的第一维是batch_size

        3. nn.Linear(hidden_size, vocab_size):
           - 将RNN的输出从hidden_size维度映射到vocab_size维度
           - 输出可视为每个字符的"分数"，经softmax后成为概率分布
        """
        super(PoetryRNN, self).__init__()
        # 定义嵌入层、RNN层和全连接层
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hx=None):
        """
        前向传播函数

        【功能描述】
        定义数据从输入到输出的正向计算流程。

        【参数说明】
        :param input: torch.LongTensor，输入字符ID序列
                      形状: (batch_size, seq_len)
                      例如: [[1, 5, 8, ...], [2, 3, 7, ...], ...]
        :param hx: torch.Tensor或None，RNN的初始隐藏状态
                   形状: (num_layers, batch_size, hidden_size)
                   若为None，RNN会自动初始化为全0

        【返回值】
        :return: tuple[torch.Tensor, torch.Tensor]
                 - output: RNN每个时间步的输出，经Linear层投影后
                           形状: (batch_size, seq_len, vocab_size)
                           含义: 每个位置对每个字符的预测分数
                 - hn: 最后一个时间步的隐藏状态
                       形状: (num_layers, batch_size, hidden_size)
                       用途: 可用于序列生成时传递状态

        【计算流程】
        1. embed(input): (batch, seq) -> (batch, seq, embedding_dim)
        2. rnn(embed, hx): (batch, seq, embed_dim) -> (batch, seq, hidden_size)
        3. linear(output): (batch, seq, hidden_size) -> (batch, seq, vocab_size)
        """
        embed = self.embed(input)
        output, hn = self.rnn(embed, hx)
        output = self.linear(output)
        return output, hn


# 创建模型实例
# 词汇表大小为所有不同字符数
# 嵌入维度256，隐藏层维度512，2层RNN
model = PoetryRNN(
    vocab_size=len(id2word),
    embedding_dim=256,
    hidden_size=512,
    num_layers=2
)


# ================================================================================
# 模块四：模型训练
# ================================================================================

def train(model, dataset, lr, epoch_num, batch_size, device):
    """
    模型训练函数

    【功能描述】
    执行RNN模型的训练循环，包括数据加载、前向传播、损失计算、
    反向传播和参数更新。同时提供训练进度可视化。

    【训练流程详解】

    1. 初始化阶段：
       - 将模型移动到指定设备(CPU/GPU)
       - 设置模型为训练模式(model.train())
       - 定义损失函数(CrossEntropyLoss)和优化器(Adam)

    2. 迭代训练阶段(Epoch循环)：
       - 每个epoch遍历整个数据集
       - 使用DataLoader进行批处理

    3. 批次训练阶段(Batch循环)：
       - 将数据移动到设备
       - 前向传播获取预测输出
       - 计算损失值
       - 反向传播计算梯度
       - 优化器更新参数
       - 梯度清零

    【参数说明】
    :param model: PoetryRNN，待训练的模型实例
    :param dataset: PoetryData，训练数据集
    :param lr: float，学习率(learning rate)，控制参数更新步长
    :param epoch_num: int，训练轮数，整个数据集遍历的次数
    :param batch_size: int，批次大小，每次处理的样本数
    :param device: torch.device，计算设备(CPU或CUDA)

    【关键技术点】

    1. CrossEntropyLoss:
       - 交叉熵损失，分类任务的标准损失函数
       - 输入: (N, C, ...) 的预测分数，(N, ...) 的目标类别
       - 内部自动应用Softmax，因此模型输出不需要Softmax

    2. output.transpose(1, 2):
       - CrossEntropyLoss要求预测张量形状为(N, C, ...)
       - 原始output形状: (batch, seq_len, vocab_size)
       - 转置后形状: (batch, vocab_size, seq_len)

    3. 训练进度条:
       - 使用\r实现动态进度显示
       - '='字符数量表示完成百分比
    """
    # 4.1 初始化
    model.to(device)
    model.train()

    # 定义损失函数和优化器
    # CrossEntropyLoss: 适用于多分类任务，内部包含Softmax
    loss_fn = nn.CrossEntropyLoss()
    # Adam优化器: 自适应学习率，通常比SGD收敛更快
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4.2 迭代训练
    for epoch in range(epoch_num):
        train_loss = 0
        # 定义数据加载器
        # shuffle=True: 每个epoch打乱数据顺序，增加随机性
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, (x, y) in enumerate(data_loader):
            # 将数据移动到指定设备(CPU/GPU)
            x, y = x.to(device), y.to(device)

            # 4.2.1 前向传播
            # 注意：模型返回(output, hn)元组，需要同时接收两个值
            output, hn = model(x)

            # 4.2.2 计算损失
            # output.transpose(1, 2): 调整维度以匹配CrossEntropyLoss要求
            # 原始: (batch, seq_len, vocab_size) -> 目标: (batch, vocab_size, seq_len)
            loss_value = loss_fn(output.transpose(1, 2), y)

            # 4.2.3 反向传播
            # 计算损失函数对模型参数的梯度
            loss_value.backward()

            # 4.2.4 更新参数
            # 根据梯度更新模型参数
            optimizer.step()

            # 4.2.5 梯度清零
            # 清除本轮计算的梯度，为下一轮做准备
            # 注意：必须在step()之后调用，否则梯度会累积
            optimizer.zero_grad()

            # 累加损失值，用于计算平均损失
            # loss_value.item()获取标量值，x.shape[0]是当前batch大小
            train_loss += loss_value.item() * x.shape[0]

            # 打印训练进度条
            # \r: 回车符，使光标回到行首，实现动态更新
            # {:0>2}: 格式化为2位数字，不足补0
            # '=' * int(...): 根据进度生成等号字符
            progress = int((batch_idx + 1) / len(data_loader) * 50)
            print(f"\rEpoch:{epoch + 1:0>2}[{'=' * progress}]{' ' * (50 - progress)}]", end="")

        # 本轮训练结束，计算并打印平均损失
        this_loss = train_loss / len(dataset)
        print(f" train_loss:{this_loss:.4f}")


# 设备配置：优先使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
lr = 1e-3           # 学习率: 0.001，Adam常用的初始学习率
epoch_num = 20      # 训练轮数: 遍历整个数据集20次
batch_size = 32     # 批次大小: 每次处理32个样本

# 执行训练
train(model, dataset, lr, epoch_num, batch_size, device)


# ================================================================================
# 模块五：古诗生成
# ================================================================================

def generate_poem(model, id2word, word2id, start_token, line_num=4, line_length=7):
    """
    古诗生成函数

    【功能描述】
    使用训练好的RNN模型生成一首古诗。采用自回归生成策略，
    从给定的起始字符开始，逐字预测下一个字符，直到生成完整的诗句。

    【生成策略详解】

    1. **自回归生成(Autoregressive Generation)**:
       - 每个新生成的字符作为下一个预测的输入
       - 形成链式生成过程，保证序列连贯性

    2. **随机采样策略**:
       - 不使用贪婪搜索(取概率最大)，而是按概率分布随机采样
       - 增加生成结果的多样性
       - torch.multinomial(prob, num_samples=1)实现按概率采样

    3. **格式控制**:
       - 支持指定行数(line_num)和每行字数(line_length)
       - 自动添加中文标点：逗号(，)和句号(。)

    【参数说明】
    :param model: PoetryRNN，训练好的模型实例
    :param id2word: list[str]，索引到字符的映射列表
    :param word2id: dict[str, int]，字符到索引的映射字典
    :param start_token: str，起始字符，生成从这里开始
    :param line_num: int，生成诗的行数，默认4行（绝句）
    :param line_length: int，每行的字数，默认7字（七言）

    【返回值】
    :return: str，生成的完整古诗字符串，包含标点和换行

    【生成流程】

    1. 起始处理：
       - 将start_token转换为ID
       - 如果token在词表中，加入生成结果
       - 初始化输入张量

    2. 逐行生成(外层循环 line_num次)：
       - 每行生成两个半句（用逗号分隔）
       - 每半句生成line_length个字

    3. 逐字生成(内层循环 line_length次)：
       - 模型前向传播得到输出
       - Softmax转换为概率分布
       - 多项式采样得到下一个字符ID
       - ID转换为字符加入结果
       - 更新输入为新生成的ID

    4. 标点添加：
       - 半句结束添加逗号(，)
       - 整句结束添加句号和换行(。\n)

    【关键技术点】

    1. model.eval():
       - 设置模型为评估模式
       - 关闭Dropout和BatchNorm的训练行为
       - 确保生成结果可复现

    2. torch.no_grad():
       - 禁用梯度计算
       - 减少内存消耗，加速推理
       - 生成阶段不需要反向传播

    3. torch.softmax(output[0, 0], dim=-1):
       - 将模型输出的logits转换为概率分布
       - output[0, 0]: 取batch第0个样本的最后时间步输出
       - dim=-1: 在最后一个维度(vocab_size)上计算softmax

    4. torch.multinomial(prob, num_samples=1):
       - 按照给定的概率分布prob进行采样
       - 返回采样得到的索引
       - 比argmax更灵活，能生成多样化结果
    """
    model.eval()  # 设置评估模式
    poem = []  # 记录生成的诗
    current_rest_line = line_length  # 记录当前行剩余的字数

    # 5.1 token id 化
    # 获取起始字符的ID，若不存在则使用<UNK>
    start_id = word2id.get(start_token, word2id['<UNK>'])

    # 如果起始字符有效，加入生成结果
    if start_id != word2id['<UNK>']:
        poem.append(start_token)
        current_rest_line -= 1

    # 5.2 定义输入数据
    # 形状: (1, 1)，batch_size=1，seq_len=1
    input_tensor = torch.LongTensor([[start_id]]).to(device)

    # 5.3 迭代生成诗句
    with torch.no_grad():  # 禁用梯度计算
        # 按行生成诗句
        for i in range(line_num):
            # 生成两句诗（每行由两个半句组成，用逗号和句号分隔）
            for interpunction in ["，", "。\n"]:
                # 逐字生成诗句
                while current_rest_line > 0:
                    # 前向传播
                    output, hn = model(input_tensor)

                    # 得到每个词的分类概率
                    # output[0, 0]: 取第一个batch的最后一个时间步输出
                    # 形状从(1, 1, vocab_size)变为(vocab_size,)
                    prob = torch.softmax(output[0, 0], dim=-1)

                    # 基于概率分布，得到下一个随机的id
                    # multinomial按照prob的概率分布进行采样
                    next_id = torch.multinomial(prob, num_samples=1)

                    # 根据id得到word
                    # .item() 将张量转为Python整数
                    poem.append(id2word[next_id.item()])

                    # 更新input，长度减1
                    current_rest_line -= 1

                    # unsqueeze(0): 在维度0处增加一维，形状从()变为(1,)
                    # 再unsqueeze(0)变为(1, 1)，符合模型输入要求
                    input_tensor = next_id.unsqueeze(0)

                # 本句生成完成，添加标点
                poem.append(interpunction)
                current_rest_line = line_length  # 重置剩余字数

    # 将字符列表拼接为字符串返回
    return "".join(poem)


# 生成10首古诗，起始字符为"一"
# 每首4行，每行7字（七言绝句格式）
for i in range(10):
    print(generate_poem(
        model, id2word, word2id,
        start_token="一",
        line_length=7,
        line_num=4
    ))
