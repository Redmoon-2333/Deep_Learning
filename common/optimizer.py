import numpy as np

# ================== 随机梯度下降(SGD) ==================
# 【核心思想】
#   最基础的优化算法，每次沿着当前梯度的负方向更新参数
#   类比：在山坡上，每一步都朝着最陡峭的下坡方向走
#
# 【更新公式】
#   W = W - lr * grad
#   参数 = 参数 - 学习率 × 梯度
#
# 【优缺点】
#   优点：简单直观，计算开销小
#   缺点：
#     1. 收敛慢：所有参数使用相同的学习率
#     2. 易震荡：在峡谷地形（不同方向曲率差异大）中来回震荡
#     3. 易陷入局部最优/鞍点
#
# 【数值计算示例】
#   假设 params = {'W': 2.0}, grads = {'W': 0.5}, lr=0.1
#
#   第1次更新: W = 2.0 - 0.1 * 0.5 = 1.95
#   第2次更新: W = 1.95 - 0.1 * 0.5 = 1.90
#   → 每次固定步长下降，无加速效果
#
# 【典型使用】
#   optimizer = SFGD(lr=0.01)
#   for epoch in range(1000):
#       grads = network.gradient(x, t)
#       optimizer.update(network.params, grads)
# ===========================================================
class SGD:
    """
    随机梯度下降优化器 (Stochastic Gradient Descent)
    
    Parameters
    ----------
    lr : float
        学习率，控制每次更新的步长（默认0.01）
        - 过大：可能跳过最优点，损失震荡甚至发散
        - 过小：收敛太慢，训练时间长
        - 常用值：0.1, 0.01, 0.001
    """
    def __init__(self, lr=0.01):
        self.lr = lr  # 学习率
    
    def update(self, params, grads):
        """
        参数更新
        
        Parameters
        ----------
        params : dict
            参数字典，如 {'W1': ndarray, 'b1': ndarray, ...}
        grads : dict
            梯度字典，键与params对应
        """
        # 遍历所有参数，按公式 W = W - lr * grad 更新
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    """
    动量优化器
    
    Parameters
    ----------
    lr : float
        学习率，控制每次更新的步长（默认0.01）
        - 过大：可能跳过最优点，甚至发散
        - 过小：收敛太慢
    momentum : float
        动量系数μ，控制历史速度的保留比例（默认0.9）
        - 0.9：保留90%的历史速度，常用值
        - 0.99：更强的惯性，适合稳定的梯度方向
        - 0：退化为普通SGD
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr              # 学习率
        self.momentum = momentum  # 动量系数
        self.v = None             # 速度字典，首次update时初始化
    
    def update(self, params, grads):
        """
        参数更新
        
        Parameters
        ----------
        params : dict
            参数字典，如 {'W1': ndarray, 'b1': ndarray, ...}
        grads : dict
            梯度字典，键与params对应
        """
        # 首次调用时，为每个参数初始化速度为0
        if self.v is None:
            self.v = {}
            for key in params.keys():
                # 速度shape与参数shape一致
                self.v[key] = np.zeros_like(params[key])
        
        # 遍历所有参数进行更新
        for key in params.keys():
            # 核心公式: v = μ*v - lr*grad
            # 速度 = 动量衰减后的旧速度 + 当前梯度方向的加速
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            # 用速度更新参数: W = W + v
            params[key] += self.v[key]


# ================== AdaGrad ==================
# 【核心思想】
#   自适应学习率：为每个参数维护独立的学习率
#   更新频繁的参数 → 学习率逐渐变小（已经学得差不多了）
#   更新稀疏的参数 → 学习率相对较大（还需要多学习）
#
# 【更新公式】
#   h = h + grad²              （累积历史梯度平方）
#   W = W - lr * grad / √h     （学习率被√h缩放）
#
# 【与SGD对比】
#   SGD:     W = W - lr * grad           （固定学习率）
#   AdaGrad: W = W - lr * grad / √h      （自适应学习率）
#
# 【优缺点】
#   优点：自动调整学习率，适合稀疏数据（如NLP词向量）
#   缺点：h单调递增 → 学习率持续衰减 → 后期几乎停止学习
#
# 【数值计算示例】
#   假设 params = {'W': 2.0}, grads = {'W': 0.5}, lr=0.1
#
#   第1次更新（h初始为0）:
#     h['W'] = 0 + 0.5² = 0.25
#     W = 2.0 - 0.1 * 0.5 / √0.25 = 2.0 - 0.1 = 1.90
#
#   第2次更新（假设新grad仍为0.5）:
#     h['W'] = 0.25 + 0.25 = 0.5
#     W = 1.90 - 0.1 * 0.5 / √0.5 = 1.90 - 0.071 ≈ 1.829
#     → 学习率从0.1降到0.071，步长在减小！
#
# 【典型使用】
#   optimizer = AdaGrad(lr=0.01)
#   for epoch in range(1000):
#       grads = network.gradient(x, t)
#       optimizer.update(network.params, grads)
# ===========================================================
class AdaGrad:
    """
    AdaGrad优化器（Adaptive Gradient）
    
    Parameters
    ----------
    lr : float
        初始学习率（默认0.01）
        - 实际学习率会随训练自动衰减
        - 可设置较大初始值，如0.01~0.1
    """
    def __init__(self, lr=0.01):
        self.lr = lr   # 初始学习率
        self.h = None  # 梯度平方累积量，首次update时初始化

    def update(self, params, grads):
        """
        参数更新
        
        Parameters
        ----------
        params : dict
            参数字典，如 {'W1': ndarray, 'b1': ndarray, ...}
        grads : dict
            梯度字典，键与params对应
        """
        # 首次调用时，为每个参数初始化累积量h为0
        if self.h is None:
            self.h = {}
            for key in params.keys():
                self.h[key] = np.zeros_like(params[key])

        for key in params.keys():
            # 累积历史梯度的平方: h = h + grad²
            self.h[key] += grads[key] * grads[key]
            # 自适应更新: W = W - lr * grad / √h
            # 1e-7 防止除零错误
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# ================== RMSProp ==================
# 【核心思想】
#   AdaGrad的改进版：用指数移动平均代替简单累加
#   解决AdaGrad学习率单调递减、后期停滞的问题
#   只关注"近期"的梯度变化，遗忘久远的历史
#
# 【更新公式】
#   h = decay * h + (1-decay) * grad²   （指数移动平均）
#   W = W - lr * grad / √h
#
# 【与AdaGrad对比】
#   AdaGrad: h = h + grad²              （无限累加，只增不减）
#   RMSProp: h = 0.9*h + 0.1*grad²      （加权平均，可增可减）
#
# 【优势】
#   1. 学习率不会单调衰减至0
#   2. 能适应非平稳目标（如RNN中的时序变化）
#   3. 在深度网络中表现稳定
#
# 【数值计算示例】
#   假设 params = {'W': 2.0}, grads = {'W': 0.5}, lr=0.1, decay=0.9
#
#   第1次更新（h初始为0）:
#     h['W'] = 0.9 * 0 + 0.1 * 0.5² = 0.025
#     W = 2.0 - 0.1 * 0.5 / √0.025 = 2.0 - 0.316 ≈ 1.684
#
#   第2次更新（假设新grad仍为0.5）:
#     h['W'] = 0.9 * 0.025 + 0.1 * 0.25 = 0.0475
#     W = 1.684 - 0.1 * 0.5 / √0.0475 ≈ 1.684 - 0.229 ≈ 1.455
#     → h增长变缓，学习率衰减更平滑
#
# 【典型使用】
#   optimizer = RMSProp(lr=0.01, decay=0.9)
#   for epoch in range(1000):
#       grads = network.gradient(x, t)
#       optimizer.update(network.params, grads)
# ===========================================================
class RMSProp:
    """
    RMSProp优化器（Root Mean Square Propagation）
    
    Parameters
    ----------
    lr : float
        学习率（默认0.01）
    decay : float
        衰减率，控制历史信息的保留比例（默认0.9）
        - 0.9：保留90%历史 + 10%当前，常用值
        - 越大：历史影响越大，学习率变化越平滑
        - 越小：更关注近期梯度，响应更快
    """
    def __init__(self, lr=0.01, decay=0.9):
        self.lr = lr       # 学习率
        self.decay = decay # 衰减率
        self.h = None      # 梯度平方的指数移动平均

    def update(self, params, grads):
        """
        参数更新
        
        Parameters
        ----------
        params : dict
            参数字典，如 {'W1': ndarray, 'b1': ndarray, ...}
        grads : dict
            梯度字典，键与params对应
        """
        # 首次调用时初始化h
        if self.h is None:
            self.h = {}
            for key in params.keys():
                self.h[key] = np.zeros_like(params[key])

        for key in params.keys():
            # 指数移动平均: h = decay*h + (1-decay)*grad²
            self.h[key] *= self.decay
            self.h[key] += (1 - self.decay) * grads[key] * grads[key]
            # 自适应更新: W = W - lr * grad / √h
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# ================== Adam ==================
# 【核心思想】
#   Momentum + RMSProp 的结合体，目前最流行的优化器
#   同时利用：
#     - 一阶矩（梯度的指数移动平均）→ Momentum的加速效果
#     - 二阶矩（梯度平方的指数移动平均）→ RMSProp的自适应学习率
#   并加入偏差修正，解决初期估计偏差问题
#
# 【更新公式】
#   v = α1*v + (1-α1)*grad        （一阶矩：梯度的移动平均）
#   h = α2*h + (1-α2)*grad²       （二阶矩：梯度平方的移动平均）
#   lr_t = lr * √(1-α2^t) / (1-α1^t)  （偏差修正后的学习率）
#   W = W - lr_t * v / √h
#
# 【为什么需要偏差修正？】
#   初始时v=h=0，前几步的移动平均会严重偏向0
#   修正因子 1/(1-α^t) 在初期较大，补偿这种偏差
#   随着t增大，修正因子趋近于1，影响消失
#
# 【优势】
#   1. 结合Momentum和RMSProp的优点
#   2. 对超参数不敏感，默认值通常就能work
#   3. 适用于大多数深度学习任务
#
# 【数值计算示例】
#   假设 params={'W':2.0}, grads={'W':0.5}, lr=0.1, α1=0.9, α2=0.999
#
#   第1次更新（t=1, v=h=0）:
#     v['W'] = 0.9*0 + 0.1*0.5 = 0.05
#     h['W'] = 0.999*0 + 0.001*0.25 = 0.00025
#     lr_t = 0.1 * √(1-0.999) / (1-0.9) = 0.1 * 0.0316 / 0.1 = 0.0316
#     W = 2.0 - 0.0316 * 0.05 / √0.00025 ≈ 2.0 - 0.1 = 1.90
#
#   第2次更新（t=2）:
#     v['W'] = 0.9*0.05 + 0.1*0.5 = 0.095
#     h['W'] = 0.999*0.00025 + 0.001*0.25 ≈ 0.0005
#     lr_t = 0.1 * √(1-0.999²) / (1-0.9²) ≈ 0.024
#     W ≈ 1.90 - 0.024 * 0.095 / √0.0005 ≈ 1.80
#
# 【典型使用】
#   optimizer = Adam(lr=0.001)  # Adam常用较小学习率
#   for epoch in range(1000):
#       grads = network.gradient(x, t)
#       optimizer.update(network.params, grads)
# ===========================================================
class Adam:
    """
    Adam优化器（Adaptive Moment Estimation）
    
    Parameters
    ----------
    lr : float
        学习率（默认0.01，推荐0.001）
    alpha1 : float
        一阶矩衰减率β1（默认0.9）
        - 控制梯度移动平均的衰减，类似Momentum的动量系数
    alpha2 : float
        二阶矩衰减率β2（默认0.999）
        - 控制梯度平方移动平均的衰减，类似RMSProp的decay
        - 通常取接近1的值，使二阶矩估计更稳定
    """
    def __init__(self, lr=0.01, alpha1=0.9, alpha2=0.999):
        self.lr = lr         # 学习率
        self.alpha1 = alpha1 # 一阶矩衰减率(β1)
        self.alpha2 = alpha2 # 二阶矩衰减率(β2)
        self.v = None        # 一阶矩：梯度的指数移动平均
        self.h = None        # 二阶矩：梯度平方的指数移动平均
        self.t = 0           # 时间步，用于偏差修正

    def update(self, params, grads):
        """
        参数更新
        
        Parameters
        ----------
        params : dict
            参数字典，如 {'W1': ndarray, 'b1': ndarray, ...}
        grads : dict
            梯度字典，键与params对应
        """
        # 首次调用时初始化一阶矩v
        if self.v is None:
            self.v = {}
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])

        # 首次调用时初始化二阶矩h
        if self.h is None:
            self.h = {}
            for key in params.keys():
                self.h[key] = np.zeros_like(params[key])

        # 时间步+1，用于偏差修正
        self.t += 1
        # 偏差修正后的学习率: lr_t = lr * √(1-β2^t) / (1-β1^t)
        lr_t = self.lr * np.sqrt(1 - self.alpha2**self.t) / (1 - self.alpha1**self.t)
        
        for key in params.keys():
            # 更新一阶矩: v = β1*v + (1-β1)*grad
            self.v[key] = self.alpha1 * self.v[key] + (1 - self.alpha1) * grads[key]
            # 更新二阶矩: h = β2*h + (1-β2)*grad²
            self.h[key] = self.alpha2 * self.h[key] + (1 - self.alpha2) * grads[key] * grads[key]
            # 参数更新: W = W - lr_t * v / √h
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.h[key]) + 1e-7)
