# Transformer在Swanlab上的可视化训练

## 环境安装

需要安装以下内容：

```
torch
swanlab
```

## 使用swanlab可视化结果

```python
# 可视化部署
swanlab.init(
    project="TransformerTrain",
    experiment_name="Transformer_test_epoch_2",
    workspace=None,
    description="Transformer的一次简单的全过程训练",
    config={'epochs': 2, 'learning_rate': 2e-3},  # 通过config参数保存输入或超参数
    logdir="./logs",  # 指定日志文件的保存路径
)
```

想了解更多关于SwanLab的知识，请看[SwanLab官方文档](https://docs.swanlab.cn/zh/guide_cloud/general/what-is-swanlab.html)。

## 训练

在首次使用SwanLab时，需要去[官网](https://swanlab.cn)注册一下账号，然后在[用户设置](https://swanlab.cn/settings)复制一下你的API Key。

然后在终端输入`swanlab login`:

```bash
swanlab login
```

把API Key粘贴进去即可完成登录，之后就不需要再次登录了。
