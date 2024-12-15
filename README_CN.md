
# SAMamba
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
SAMamba 是一个用于红外小目标分割的框架
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

使用下面的命令进行训练:

```train
python train.py --opt ./options/train.yaml
```
## 预测


使用下面的命令进行预测:

```eval
python test.py --opt ./options/test.yaml
```
## 模型权重

- [SAMamba for SIRST](https://drive.google.com/drive/folders/1_Ef2rpJXUkGti1qxPSvrnez-3m1OegLa?usp=drive_link)

| Model name | IoU   | nIoU  |
|------------|-------|-------|
| SAMamba    | 80.40 | 78.57 |
- [SAMamba for IRSTD](https://drive.google.com/drive/folders/1_Ef2rpJXUkGti1qxPSvrnez-3m1OegLa?usp=sharing)

| Model name | IoU   | nIoU  |
|------------|-------|-------|
| SAMamba    | 72.27 | 66.03 |
