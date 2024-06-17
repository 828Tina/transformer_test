import random
import numpy as np
import torch

# 输入
dataset_x = '<BEGIN>,<END>,<PAD>,a,b,c,d,e,f,g,1,2,3,4,5,6,7,8,9'
# 输入表示为字典形式
dataset_x = {word: i for i, word in enumerate(dataset_x.split(","))}
dataset_xr = [k for k, v in dataset_x.items()]
dataset_y = {k.upper(): v for k, v in dataset_x.items()}
dataset_yr = [k for k, v in dataset_y.items()]


def get_data():
    # 定义词的集合，总共像上面的x
    words = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ]

    # 定义被选中的概率
    all_words_len = len(words)
    p = np.arange(1, all_words_len + 1)

    p = p / p.sum()

    # 随机选n个词
    n = random.randint(20, 30)
    x = np.random.choice(words, size=n, replace=True, p=p)

    # 采样的结果就是x，变成List
    x = x.tolist()

    # 怎么变换
    def convert(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 10 - int(i)
        return str(i)

    y = [convert(i) for i in x]
    # 因为输出和输入不一定是一样的大小，所以假设y多了1位
    # 这1位是最后一个词放前面
    y = [y[-1]] + y

    # 加上首尾符号
    x = ['<BEGIN>'] + x + ['<END>']
    y = ['<BEGIN>'] + y + ['<END>']

    # 补PAD，既然上面的是随机20到30，这里补长度到40
    x = x + ['<PAD>'] * 40
    y = y + ['<PAD>'] * 41

    x = x[:40]
    y = y[:41]

    # 编码成数据
    # 编码成数据
    x = [dataset_x[i] for i in x]
    y = [dataset_y[i] for i in y]

    # 转tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    return x, y


def get_data():
    # 定义词的集合，总共像上面的x
    words = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ]

    # 定义被选中的概率
    all_words_len = len(words)
    p = np.arange(1, all_words_len + 1)

    p = p / p.sum()

    # 随机选n个词
    n = random.randint(20, 30)
    x = np.random.choice(words, size=n, replace=True, p=p)

    # 采样的结果就是x，变成List
    x = x.tolist()

    # 怎么变换
    def convert(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 10 - int(i)
        return str(i)

    y = [convert(i) for i in x]
    # 因为输出和输入不一定是一样的大小，所以假设y多了1位
    # 这1位是最后一个词放前面
    y = [y[-1]] + y

    # 加上首尾符号
    x = ['<BEGIN>'] + x + ['<END>']
    y = ['<BEGIN>'] + y + ['<END>']

    # 补PAD，既然上面的是随机20到30，这里补长度到40
    x = x + ['<PAD>'] * 40
    y = y + ['<PAD>'] * 41

    x = x[:40]
    y = y[:41]

    # 编码成数据
    # 编码成数据
    x = [dataset_x[i] for i in x]
    y = [dataset_y[i] for i in y]

    # 转tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    return x, y


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, i):
        return get_data()


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=Dataset(),
                                     batch_size=8,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)
