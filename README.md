
# SAMamba
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
SAMamba is a framework for infrared small target segmentation, we will publish the full code and model weights later.
## Dataset Structe
If you want to train on custom datasets you should paper dataset as following structure:
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
## Training
If you want to train your own model, please download the pre-trained segment anything 2 from [the official repository](https://github.com/facebookresearch/sam2), and run this command:
```train
python train.py --opt ./options/train.yaml
```
## Evaluation


To evaluate pretrained model, run:

```eval
python test.py --opt ./options/test.yaml
```
## Model weight
Once the code is fully exposed, you can download the model weights:
- SAMamba

| Dateset | IoU   | nIoU  | F1  |
|------------|-------|-------|-------|
|  NUAA-SIRST  | 81.08 | 79.17 | 98.55 |
|  IRSTD-1K  | 73.53 | 68.99 | 84.75 |
|  NUDT-SIRST  | 93.13 | 93.15 | 96.44 |