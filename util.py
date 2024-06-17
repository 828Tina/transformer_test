import torch
import math

def attention(Q, K, V, mask):
    # b句话,每句话40个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 40, 8]
    # Q,K,V = [b, head_num, 40_words, 8]

    # [b, 4, 40, 8] * [b, 4, 8, 40] -> [b, 4, 40, 40]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    # QK.T
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头维数的平方根,做数值缩放
    # [b, 4, 40, 40]
    # QK.T/根号dk
    score /= (8 ** 0.5)

    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask = [b, 1, 40, 40]
    score = score.masked_fill_(mask, -float('inf'))
    # softmax过程
    score = torch.softmax(score, dim=-1)

    # 以注意力分数乘以V,得到最终的注意力结果
    # softmax((QK.T)/根号dk)*V
    # [b, 4, 40, 40] * [b, 4, 40, 8] -> [b, 4, 40, 8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 4, 40, 8] -> [b, 40, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 40, 32)

    return score


class MultiHead(torch.nn.Module):

    def __init__(self):
        # 这些是Q,K,V对应的线性矩阵，把输入x通过线性矩阵变成对应的Q,K,V
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)

        # 输出的全连接层
        self.out_fc = torch.nn.Linear(32, 32)

        # 规范化LN
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

        # 防止过拟合
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # 多头注意力层

        # b句话,每句话40个词,每个词编码成32维向量
        # Q,K,V = [b, 40, 32]
        b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()

        # 规范化，放在所有过程之前是因为经过验证，放在之前可以让模型收敛的更快
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算,维度不变
        # [b, 40, 32] * [b, 32, 32]-> [b, 40, 32]
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)

        # 拆分成多个头
        # b句话,每句话4=个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 40, 32] -> [b, 4, 40, 8]
        Q = Q.reshape(b, 40, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 40, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 40, 4, 8).permute(0, 2, 1, 3)

        # 这下这些Q,K,V的形状和attention要求的输入形状一样了
        # 计算注意力
        # [b, 4, 40, 8] -> [b, 40, 32]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 40, 32] -> [b, 40, 32]
        score = self.dropout(self.out_fc(score))

        # 短接，transformer里的残差链接吧
        score = clone_Q + score
        return score


class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # pos是第几个词,i是第几个维度,d_model是维度总数
        def get_pe(pos, i, d_model):
            # 分母
            fenmu = 1e4 ** (i / d_model)
            # sin里的
            pe = pos / fenmu

            if i % 2 == 0:
                # 偶数
                return math.sin(pe)
            # 奇数
            return math.cos(pe)

        # 初始化位置编码矩阵
        pe = torch.empty(40, 32)
        for i in range(40):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe.unsqueeze(0)

        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层
        self.embed = torch.nn.Embedding(39, 32)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8, 40] -> [8, 40, 32]
        # 表示8句话，每句话40个字，每个字32个维度
        embed = self.embed(x)

        # 词编码和位置编码相加
        # [8, 40, 32] + [1, 40, 32] -> [8, 40, 32]
        embed = embed + self.pe
        return embed


class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 40, 32] -> [b, 40, 32]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out
