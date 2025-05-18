
# SAMamba
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
SAMamba is a framework for infrared small target segmentation and has been successfully accepted by Information Fusion. Meanwhile, our complete code and model weights have been published!
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
You can download the model weights:
- [SAMamba](https://drive.google.com/drive/folders/1UrIj44_NIq5C6ldRuH1DMfM6m0kiiz5d?usp=drive_link)

| Dateset | IoU   | nIoU  | F1  |
|------------|-------|-------|-------|
|  NUAA-SIRST  | 81.08 | 79.17 | 89.55 |
|  IRSTD-1K  | 73.53 | 68.99 | 84.75 |
|  NUDT-SIRST  | 93.13 | 93.15 | 96.44 |