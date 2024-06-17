import torch
from data import dataset_x, dataset_y, dataset_xr, dataset_yr


def mask_pad(data):
    # 用于遮挡<PAD>

    # b句话,每句话40个词,这里是还没embed的
    # data = [b, 40]
    # 判断每个词是不是<PAD>
    # 这里data表示的是字典里键值对对应的数字
    mask = data == dataset_x['<PAD>']

    # [b, 40] -> [b, 1, 1, 40]
    mask = mask.reshape(-1, 1, 1, 40)

    # 在计算注意力时,是计算40个词和40个词相互之间的注意力,所以是个40*40的矩阵
    # 是pad的列是true,意味着任何词对pad的注意力都是0
    # 但是pad本身对其他词的注意力并不是0
    # 所以是pad的行不是true

    # 复制n次
    # [b, 1, 1, 40] -> [b, 1, 40, 40]
    mask = mask.expand(-1, 1, 40, 40)

    return mask


def mask_tril(data):
    # 用于遮挡未来词
    # 比如一句话是"a,b,c"，那么考虑a的时候，不考虑a和b以及a的c的注意力

    # b句话,每句话40个词,这里是还没embed的
    # data = [b, 40]

    # 上三角矩阵,不包括对角线,意味着,对每个词而言,他只能看到他自己,和他之前的词,而看不到之后的词
    # [1, 40, 40]

    """
    [[0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]"""

    tril = 1 - torch.tril(torch.ones(1, 40, 40, dtype=torch.long))

    # 判断y当中每个词是不是pad,如果是pad则不可见
    # [b, 40]
    mask = data == dataset_y['<PAD>']

    # 变形+转型,为了之后的计算
    # [b, 1, 40]
    mask = mask.unsqueeze(1).long().to(data.device)
    tril = tril.to(data.device)
    # mask和tril求并集
    # [b, 1, 40] + [1, 40, 40] -> [b, 40, 40]
    mask = mask + tril

    # 转布尔型
    mask = mask > 0

    # 转布尔型,增加一个维度,便于后续的计算
    mask = (mask == 1).unsqueeze(dim=1)

    return mask
