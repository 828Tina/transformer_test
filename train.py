import math
import torch
from main import Transformer
from data import loader, dataset_y
from tqdm import tqdm
import swanlab
import torch.nn as nn
import os
# 声明GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实例化模型
model = Transformer()
# 如果要多卡训练，要先把 model 移动到 GPU 上，然后再训练
if torch.cuda.device_count() > 2:  # 检查电脑是否有多块GPU
    # 设置要使用的前四块GPU
    device_ids = [0, 1]  # 这里选择的是前2块GPU，可以根据实际情况修改
    print(f"Let's use {len(device_ids)} GPUs!")
    # 将模型移到指定的GPU上
    model = nn.DataParallel(model, device_ids=device_ids)

model.to(device)  # 把并行的模型移动到GPU上

# 计算loss
loss_func = torch.nn.CrossEntropyLoss().to(device)
# 优化器选择adam
optim = torch.optim.Adam(model.parameters(), lr=2e-3)
# 学习率是0.002
# 学习率调度器，每3个步长epochs后，学习率*0.5
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)

# 打开文件以写入结果
# 创建进度条
progress_bar = tqdm(total=len(loader), desc="Training Progress")
# 初始化swanlab
lxy = swanlab.init(
    project="DaMoXing_test",
    experiment_name="Transformer_test_epoch_1",
    workspace=None,
    description="Transformer的一次简单的全过程训练",
    config={'epochs': 2, 'learning_rate': 2e-3},  # 通过config参数保存输入或超参数
    logdir="./logs",  # 指定日志文件的保存路径
)

with open('transformer_lxy.txt', 'a') as f:
    for epoch in range(lxy.config.epochs):
        for i, (x, y) in enumerate(loader):
            # x = [8, 40]
            # y = [8, 41]

            # 将输入数据移到GPU上
            x, y = x.to(device), y.to(device)

            # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
            # [8, 40]
            # [8, 40] -> [8, 40, 39]
            pred = model(x, y[:, :-1])

            # [8, 40, 39] -> [320, 39]
            pred = pred.reshape(-1, 39)

            # [8, 41] -> [320]
            # 去掉'<BEGIN>'
            y = y[:, 1:].reshape(-1)

            # 忽略pad
            select = y != dataset_y['<PAD>']
            pred = pred[select]
            y = y[select]

            loss = loss_func(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # 更新进度条
            progress_bar.update(1)

            if i % 250 == 0:
                # [select, 39] -> [select]
                pred = pred.argmax(1)
                # 表示沿着第二个维度（即每一行）找到最大值的索引，也就是找到每个样本预测概率最高的类别对应的索引。
                correct = (pred == y).sum().item()
                accuracy = correct / len(pred)
                lr = optim.param_groups[0]['lr']

                # 将结果写入文件
                f.write("Epoch: {} Step: {} LR: {:.6f} Accuracy: {:.4f} Loss: {}\n".format(epoch, i, lr, accuracy,
                                                                                           loss.item()))

                swanlab.log({"loss": loss, "accuracy": accuracy, "learning_rate": lr})
        sched.step()
if not os.path.exists('result_model'):
    os.makedirs('result_model')
torch.save(model, "./result_model/transformer.model")
# 关闭进度条
progress_bar.close()
