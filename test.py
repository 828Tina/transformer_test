import torch
from data import dataset_y, loader, dataset_xr, dataset_x, dataset_yr
from mask import mask_pad, mask_tril
import torch.nn as nn
def predict(x):
    # 加载模型，并将模型移动到GPU上
    model = torch.load("./result_model/transformer.model")
    # x = [1, 40]:一句话，40个词
    model.eval()

    # [1, 1, 40, 40]
    mask_pad_x = mask_pad(x)
    mask_pad_x = mask_pad_x.to('cuda')
    # 初始化输出,这个是固定值
    # [1, 40]
    # [[0,2,2,2...]]
    target = [dataset_y['<BEGIN>']] + [dataset_y['<PAD>']] * 39
    target = torch.LongTensor(target).unsqueeze(0)

    # x编码,添加位置信息
    # [1, 40] -> [1, 40, 32]
    x = x.to('cuda')
    x = model.module.embed_x(x)

    # 编码层计算,维度不变
    # [1, 40, 32] -> [1, 40, 32]
    x = model.module.encoder(x, mask_pad_x)

    # 遍历生成第1个词到第39个词
    # 因为是预测，所以最后一个词是预测出来的
    for i in range(39):
        # [1, 40]
        # [[0,2,2,2...]]
        y = target
        y = y.to('cuda')
        # [1, 1, 40, 40]
        mask_tril_y = mask_tril(y)
        mask_tril_y = mask_tril_y.to('cuda')
        # y编码,添加位置信息
        # [1, 40] -> [1, 40, 32]
        y = model.module.embed_y(y)

        # 解码层计算,维度不变
        # [1, 40, 32],[1, 40, 32] -> [1, 40, 32]
        y = model.module.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,39分类
        # [1, 40, 32] -> [1, 40, 39]
        out = model.module.fc_out(y)

        # 取出当前词的输出
        # [1, 40, 39] -> [1, 40]
        out = out[:, i, :]

        # 取出分类结果
        # [1, 39] -> [1]
        out = out.argmax(dim=1).detach()

        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    return target


if __name__ == "__main__":
    # 测试
    for i, (x, y) in enumerate(loader):
        break
    for i in range(5):
        print(i)
        print('x输入：', ''.join([dataset_xr[i] for i in x[i].tolist()]).replace("<BEGIN>", "").split("<END>")[0])
        print('y答案：', ''.join([dataset_yr[i] for i in y[i].tolist()]).replace("<BEGIN>", "").split("<END>")[0])
        print('y预测：',
              ''.join([dataset_yr[i] for i in predict(x[i].unsqueeze(0))[0].tolist()]).replace("<BEGIN>", "").split(
                  "<END>")[0])
