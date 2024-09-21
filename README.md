# Evanyseg
## Dataset
The EvanySeg model was trained based on 107,055 2D images, accompanied by 206,596 object-level ground truth masks. Segmentation predictions for training the EvanySeg model were generated using SAM, MedSAM, and SAM-Med2D. In total, 619,044 object-level image-mask pairs were utilized, with each image and corresponding mask resized to 244 × 244 pixels for training the segmentation evaluation model.

The filesystem hierarchy of the dataset is as follows:

```
├─QAAS_Med
│  ├─checkpoints
│  ├─results
│  ├─datasets
│  │  ├─preprocess
│  │  │  └─train_bbox_Polyp
│  │  │      ├─crop_image
│  │  │      │      0_SAM_Polyp_train_175.png
│  │  │      │      0_SAM_Med2D_Polyp_train_175.png
│  │  │      │      0_MedSAM_bbox_Polyp_train_175.png
│  │  │      │      . . .
│  │  │      │      
│  │  │      ├─crop_mask
│  │  │      │      0_SAM_Polyp_train_175.png
│  │  │      │      0_SAM_Med2D_Polyp_train_175.png
│  │  │      │      0_MedSAM_bbox_Polyp_train_175.png
│  │  │      │      . . .
│  │  │      │      
│  │  │      └─crop_predict
│  │  │      │      0_SAM_Polyp_train_175.png
│  │  │      │      0_SAM_Med2D_Polyp_train_175.png
│  │  │      │      0_MedSAM_bbox_Polyp_train_175.png
│  │  │             . . .
│  │  │              
│  │  └─raw          
│  │      └─Polyp
│  │          └─train
│  │              ├─images
│  │              │      175.png
│  │              │      . . .
│  │              │      
│  │              └─masks
│  │                     175.png
│  │                     . . .
│  │         

```

The data set needs to be sorted into Poly form below datasets / raw. After preprocessing.py processing, it will become the form data below datasets / preprocess directory. The processed data naming rules are as follows:

```
├─crop_images
       {i}_{model_name}_{directory}_{part}_{sample_name}
```

Note: "i" represents the index of the connected domain being processed in the current iteration, "model_name" indicates the model SAM, MedicalSAM and SAM-Med2D, "directory" represents the directory name of the dataset such as Polyp, "part" indicates the subdirectory, sample_name, "sample_name" indicates the original name of the image

## Framework

## Results



Pearson correlation：

|      Model      | Resolution | Bbox | 1point | 5point |
| :-------------: | :--------: | :--: | :----: | :----: |
| EvanySeg(ViT-b) |  224×224   |      |        |        |
| EvanySeg(ViT-l) |  224×224   |      |        |        |

checkpoint：

|      Model      | Bbox | 1point | 5point |
| :-------------: | :--: | :----: | :----: |
| EvanySeg(ViT-b) |      |        |        |
| EvanySeg(ViT-l) |      |        |        |



```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```



## Quick Start

Download the datasets and pre-trained models to the corresponding folders, and configure the environment.

```python
pip install -r requirements.txt
```

#### Train

```python
python preprocessing.py
python train.py
```

#### test

```python
python preprocessing.py
python test.py
```



## Reference

```
@article{*****,
title={Towards Ground-truth-free Evaluation of Any Segmentation in Medical Images*},
author={Ahjol Senbi, Tianyu Huang, Fei Lyu, Qing Li, Yuhui Tao, Wei Shao, Qiang Chen,
Chengyan Wang, Shuo Wang, Tao Zhou, Yizhe Zhang†},
journal={ },
year={2024}
}
```

