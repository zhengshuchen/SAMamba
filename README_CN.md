
# SAMamba
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
SAMamba 是一个用于红外小目标分割的框架，我们将在后续公布完整代码
## 数据集结构
如果你想要在自己的数据集上训练，你需要按照下列的结构准备数据:
```
|-SIRST
    |-trainval
        |-images
            |-xxx.png
        |-masks
            |-xxx.png
    |-test
        |-images
            |-xxx.png
        |-masks
            |-xxx.png
```

## 训练

如果您想训练自己的模型，请从[官方存储库](https://github.com/facebookresearch/sam2)
下载预训练模型，并使用下面的命令进行训练:

```train
python train.py --opt ./options/train.yaml
```
## 预测


使用下面的命令进行预测:

```eval
python test.py --opt ./options/test.yaml
```
