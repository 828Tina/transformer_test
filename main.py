import torch
from util import MultiHead, FullyConnectedOutput, PositionEmbedding
from mask import mask_pad, mask_tril


class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 多头注意力层
        self.mh = MultiHead()
        # 全连接层
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 计算自注意力,维度不变
        # [b, 40, 32] -> [b, 40, 32]
        score = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 40, 32] -> [b, 40, 32]
        out = self.fc(score)

        return out


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # mask的那部分
        self.mh1 = MultiHead()
        # 从encoder那边来的部分
        self.mh2 = MultiHead()
        # 全连接层
        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # 先计算y的自注意力,维度不变
        # [b, 40, 32] -> [b, 40, 32]
        # mask_tril_y为了预测下一个词
        y = self.mh1(y, y, y, mask_tril_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 40, 32],[b, 40, 32] -> [b, 40, 32]
        y = self.mh2(y, x, x, mask_pad_x)

        # 全连接输出,维度不变
        # [b, 40, 32] -> [b, 40, 32]
        y = self.fc(y)

        return y


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x = PositionEmbedding()
        self.embed_y = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = torch.nn.Linear(32, 39)

    def forward(self, x, y):
        # [b, 1, 40, 40]
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)

        # 编码,添加位置信息
        # x = [b, 40] -> [b, 40, 32]
        # y = [b, 40] -> [b, 40, 32]
        x, y = self.embed_x(x), self.embed_y(y)

        # 编码层计算
        # [b, 40, 32] -> [b, 40, 32]
        x = self.encoder(x, mask_pad_x)

        # 解码层计算
        # [b, 40, 32],[b, 40, 32] -> [b, 40, 32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,维度不变
        # [b, 40, 32] -> [b, 40, 39]
        y = self.fc_out(y)

        return y
