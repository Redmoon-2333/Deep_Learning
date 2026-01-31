import numpy as np
from common.functions import sigmoid
from common.functions import softmax
from common.functions import cross_entropy

# ReLu
class Relu:
    # 初始化
    def __init__(self):
        # mask: 用于保存输入数据中小于等于0的元素的索引
        self.mask = None

    # 前向传播
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        # 将输入数据中小于等于0的元素设为0
        out[self.mask] = 0

        return out

    # 反向传播
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

# Sigmoid
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

# Affine 仿射层（全连接层）
# ============================================================================
# 功能说明：
#   Affine 层执行仿射变换 y = x·W + b，是神经网络中最基本的线性变换层
#   - 前向传播：将输入 x 与权重 W 做矩阵乘法，再加上偏置 b
#   - 反向传播：根据链式法则计算各参数的梯度
#
# 计算流程示例（以 2 个样本、3 输入特征、2 输出特征为例）：
# ----------------------------------------------------------------------------
# 假设输入数据：
#   x = [[1, 2, 3],      # 样本1：3个特征
#        [4, 5, 6]]      # 样本2：3个特征
#   x.shape = (2, 3)     # (batch_size=2, input_features=3)
#
#   W = [[0.1, 0.2],     # 权重矩阵
#        [0.3, 0.4],
#        [0.5, 0.6]]
#   W.shape = (3, 2)     # (input_features=3, output_features=2)
#
#   b = [0.1, 0.2]       # 偏置向量
#   b.shape = (2,)       # (output_features=2)
#
# 前向传播计算：
#   out = x·W + b
#       = [[1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6],   # 样本1
#          [4*0.1+5*0.3+6*0.5, 4*0.2+5*0.4+6*0.6]]   # 样本2
#         + [0.1, 0.2]
#       = [[2.2, 2.8],
#          [4.9, 6.4]]
#         + [0.1, 0.2]
#       = [[2.3, 3.0],
#          [5.0, 6.6]]
#   out.shape = (2, 2)   # (batch_size=2, output_features=2)
#
# 反向传播计算（假设上游梯度 dout）：
#   dout = [[1, 1],      # 上游传来的梯度
#           [1, 1]]
#   dout.shape = (2, 2)
#
#   1. 计算输入梯度 dx（用于继续向前层传播）：
#      dx = dout·W^T
#         = [[1, 1],    ·  [[0.1, 0.3, 0.5],
#            [1, 1]]        [0.2, 0.4, 0.6]]
#         = [[0.3, 0.7, 1.1],
#            [0.3, 0.7, 1.1]]
#      dx.shape = (2, 3)  # 与输入 x 形状相同
#
#   2. 计算权重梯度 dW（用于更新权重）：
#      dW = x^T·dout
#         = [[1, 4],     ·  [[1, 1],
#            [2, 5],         [1, 1]]
#            [3, 6]]
#         = [[5, 5],
#            [7, 7],
#            [9, 9]]
#      dW.shape = (3, 2)  # 与权重 W 形状相同
#
#   3. 计算偏置梯度 db（用于更新偏置）：
#      db = sum(dout, axis=0)  # 沿 batch 维度求和
#         = [1+1, 1+1] = [2, 2]
#      db.shape = (2,)   # 与偏置 b 形状相同
# ============================================================================
class Affine:
    def __init__(self, W, b):
        self.W = W  # 权重矩阵，shape: (input_features, output_features)
        self.b = b  # 偏置向量，shape: (output_features,)
        # 用于保存输入数据，方便反向传播计算梯度
        self.x = None
        self.original_x_shape = None  # 保存原始输入形状，用于反向传播时恢复
        # 用于保存梯度，方便梯度下降法更新参数
        self.dW = None  # 权重梯度
        self.db = None  # 偏置梯度

    def forward(self, x):
        """
        前向传播：计算 out = x·W + b
        参数：
            x: 输入数据，shape 为 (batch_size, ..., input_features)
               可以是多维张量，会自动展平为二维矩阵
        返回：
            out: 输出数据，shape 为 (batch_size, output_features)
        """
        # 保存原始形状，用于反向传播时恢复 dx 的形状
        self.original_x_shape = x.shape
        # 将输入展平为二维矩阵：(batch_size, total_features)
        # 例如：(2, 3, 4, 5) → (2, 60)，其中 -1 表示自动计算剩余维度乘积
        self.x = x.reshape(x.shape[0], -1)
        # 执行仿射变换：矩阵乘法 + 偏置
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        """
        反向传播：根据链式法则计算各参数的梯度
        参数：
            dout: 上游（后一层）传来的梯度，shape 为 (batch_size, output_features)
        返回：
            dx: 输入的梯度，shape 与原始输入 x 相同
        """
        # 计算输入梯度：dx = dout·W^T
        # 推导：∂L/∂x = ∂L/∂out · ∂out/∂x = dout · W^T
        dx = np.dot(dout, self.W.T)
        # 恢复为原始输入形状（如果输入是多维张量）
        dx = dx.reshape(*self.original_x_shape)

        # 计算权重梯度：dW = x^T·dout
        # 推导：∂L/∂W = ∂L/∂out · ∂out/∂W = x^T · dout
        self.dW = np.dot(self.x.T, dout)

        # 计算偏置梯度：db = sum(dout, axis=0)
        # 推导：∂L/∂b = ∂L/∂out · ∂out/∂b = sum(dout)，因为 b 对每个样本都加了一次
        self.db = np.sum(dout, axis=0)

        return dx

# SoftmaxWithLoss 输出层（Softmax + 交叉熵损失）
# ============================================================================
# 功能说明：
#   将 Softmax 激活函数与交叉熵损失函数组合成一层
#   - 前向传播：先对输入 x 做 Softmax 得到概率分布 y，再计算与标签 t 的交叉熵损失
#   - 反向传播：利用组合求导的简洁结果 dx = (y - t) / batch_size
#
# 为什么要组合？
#   Softmax 和交叉熵单独求导很复杂，但组合后的梯度公式极其简洁：dx = y - t
#   这是数学上的巧合，也是深度学习中常用的技巧
#
# 计算流程示例（以 2 个样本、3 分类问题为例）：
# ----------------------------------------------------------------------------
# 假设输入数据（最后一层全连接层的输出，即 logits）：
#   x = [[2.0, 1.0, 0.1],    # 样本1 的 logits
#        [0.5, 2.5, 0.3]]    # 样本2 的 logits
#   x.shape = (2, 3)         # (batch_size=2, num_classes=3)
#
# 【前向传播】
# Step 1: Softmax 转换为概率分布
#   y = softmax(x)
#   对于样本1: exp([2.0, 1.0, 0.1]) = [7.39, 2.72, 1.11]
#              sum = 11.22
#              y[0] = [7.39/11.22, 2.72/11.22, 1.11/11.22]
#                   = [0.659, 0.242, 0.099]
#   对于样本2: exp([0.5, 2.5, 0.3]) = [1.65, 12.18, 1.35]
#              sum = 15.18
#              y[1] = [0.109, 0.802, 0.089]
#   最终 y = [[0.659, 0.242, 0.099],
#             [0.109, 0.802, 0.089]]
#
# Step 2: 计算交叉熵损失
#   假设真实标签（独热编码）：
#   t = [[1, 0, 0],    # 样本1 属于类别0
#        [0, 1, 0]]    # 样本2 属于类别1
#
#   loss = -mean(t * log(y))
#        = -mean([1*log(0.659) + 0 + 0,
#                 0 + 1*log(0.802) + 0])
#        = -mean([-0.417, -0.220])
#        = 0.319
#
# 【反向传播】
# 梯度计算（关键公式推导见下方）：
#   dx = (y - t) / batch_size
#      = ([[0.659, 0.242, 0.099],   -  [[1, 0, 0],
#          [0.109, 0.802, 0.089]]      [0, 1, 0]]) / 2
#      = [[-0.341, 0.242, 0.099],
#         [0.109, -0.198, 0.089]] / 2
#      = [[-0.171, 0.121, 0.050],
#         [0.055, -0.099, 0.045]]
#
# 梯度含义解释：
#   - 负值（如 -0.171）：该位置是正确类别，需要增大对应的 logit
#   - 正值（如 0.121）：该位置是错误类别，需要减小对应的 logit
#   - 除以 batch_size：取平均，使梯度大小与批量大小无关
#
# 两种标签格式的处理：
#   格式1 - 独热编码：t = [[1,0,0], [0,1,0]]，直接用 dx = (y - t)
#   格式2 - 整数标签：t = [0, 1]，需要索引操作 y[i, t[i]] -= 1
#
# ----------------------------------------------------------------------------
# 【补充示例】整数标签格式的反向传播计算过程：
# ----------------------------------------------------------------------------
# 沿用上面的例子，Softmax 输出：
#   y = [[0.659, 0.242, 0.099],
#        [0.109, 0.802, 0.089]]
#
# 假设真实标签为整数格式：
#   t = [0, 1]    # 样本1 属于类别0，样本2 属于类别1
#   t.shape = (2,)
#
# Step 1: 复制 y 到 dx
#   dx = y.copy()
#      = [[0.659, 0.242, 0.099],
#         [0.109, 0.802, 0.089]]
#
# Step 2: 生成样本索引
#   np.arange(batch_size) = np.arange(2) = [0, 1]
#
# Step 3: 对正确类别位置减 1
#   dx[np.arange(2), t] -= 1
#   即：dx[[0, 1], [0, 1]] -= 1
#   这会访问 dx[0, 0] 和 dx[1, 1]，分别减 1：
#     dx[0, 0] = 0.659 - 1 = -0.341
#     dx[1, 1] = 0.802 - 1 = -0.198
#   结果：
#   dx = [[-0.341, 0.242, 0.099],
#         [0.109, -0.198, 0.089]]
#
# Step 4: 除以 batch_size
#   dx /= 2
#      = [[-0.171, 0.121, 0.050],
#         [0.055, -0.099, 0.045]]
#
# 注意：结果与独热编码方式完全一致！
# 这证明两种方式是等价的，只是实现方式不同
# ============================================================================
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 保存损失值
        self.y = None     # 保存 Softmax 输出（概率分布）
        self.t = None     # 保存真实标签

    def forward(self, x, t):
        """
        前向传播：计算 Softmax 概率和交叉熵损失
        参数：
            x: 输入数据（logits），shape 为 (batch_size, num_classes)
            t: 真实标签，可以是独热编码 (batch_size, num_classes) 或整数标签 (batch_size,)
        返回：
            loss: 交叉熵损失值（标量）
        """
        self.t = t
        # Step 1: Softmax 将 logits 转换为概率分布
        self.y = softmax(x)
        # Step 2: 计算与真实标签的交叉熵损失
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        """
        反向传播：计算输入的梯度
        参数：
            dout: 上游梯度，对于损失函数通常为 1
        返回：
            dx: 输入 x 的梯度，shape 与 x 相同

        数学推导（为什么 dx = y - t）：
            设 L = -log(y_k)，其中 y = softmax(x)，k 是正确类别
            对于正确类别 k：∂L/∂x_k = y_k - 1
            对于其他类别 j：∂L/∂x_j = y_j
            合并表示：∂L/∂x = y - t（t 是独热向量）
        """
        batch_size = self.t.shape[0]

        # 根据标签格式选择计算方式
        if self.t.size == self.y.size:
            # 独热编码格式：t.shape == y.shape，例如 t = [[1,0,0], [0,1,0]]
            # 直接向量化计算：dx = (y - t) / batch_size
            dx = (self.y - self.t) / batch_size
        else:
            # 整数标签格式：t 是一维数组，例如 t = [0, 1]
            # 需要通过索引操作实现 y - t 的效果
            dx = self.y.copy()
            # np.arange(batch_size) = [0, 1, ..., batch_size-1] 生成样本索引
            # self.t = [0, 1] 是每个样本的正确类别索引
            # dx[np.arange(batch_size), self.t] 访问每个样本正确类别位置的值
            # 例如：dx[0, 0] 和 dx[1, 1]，对这些位置减 1，等效于 y - t
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size

        return dx
