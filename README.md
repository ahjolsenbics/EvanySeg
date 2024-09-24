<p align="center">
    <h1 align="center">Towards Ground-truth-free Evaluation of Any Segmentation in Medical Images*</h1>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/ahjolsenbics/EvanySeg/blob/main/README.md#Framework">Framework</a> |
        <a href="#-Citing Us">Citing Us</a> |
        <a href="#-Installation">Installation</a> |
        <a href="#-Getting Started">Getting Started</a> |
        <a href="#-Demo">Demo</a> |
        <a href="https://github.com/ahjolsenbics/EvanySeg">Page Main</a>
    <p>
</h4>


<p align="center">
    <a href="https://github.com/confident-ai/deepeval/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/confident-ai/deepeval.svg?color=violet">
    </a>
    <a href="https://drive.google.com/drive/folders/1Ngme9APByRTAOOsLGtwzVYzS2Il4jc1n?usp=drive_link">
        <img alt="Try Download EvanySeg checkpoints" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/confident-ai/deepeval/blob/master/LICENSE.md">
        <img alt="License" src="https://img.shields.io/github/license/confident-ai/deepeval.svg?color=yellow">
    </a>
</p>

## Framework

###### workflow
![workflow](utils\readme_img\workflow.png)
EvanySeg is a companion model to SAM and its variants, designed to enhance reliability and trustworthiness in the deployment of SAM (and its variants) on medical images.

## Citing Us
If you're interested in learning more about EvanySeg, we would appreciate your references to [our paper](https://arxiv.org/pdf/2409.14874).

## Dataset
The EvanySeg model was trained based on 2D images, accompanied by object-level ground truth masks. Segmentation predictions for training the EvanySeg model were generated using SAM, MedSAM, and SAM-Med2D.    
  
The filesystem hierarchy of the dataset is as follows:

```
📁 EvanySeg
├── 📁 checkpoints
├── 📁 results
├── 📁 datasets
│   ├── 📁 preprocess
│   │   └── 📁 train_bbox_Polyp
│   │       ├── 📁 crop_image
│   │       │       0_SAM_Polyp_train_175.png
│   │       ├── 📁 crop_mask
│   │       │       0_SAM_Polyp_train_175.png
│   │       └── 📁 crop_predict
│   │       │       0_SAM_Polyp_train_175.png
│   └── 📁 raw          
│       └── 📁 Polyp
│           └── 📁 train
│               ├── 📁 images
│               │       175.png
│               └── 📁 masks
│                       175.png

```
 The processed data naming rules are as follows:
```
├─crop_images
       {i}_{model_name}_{directory}_{part}_{sample_name}
```

Note: "i" represents the index of the connected domain being processed in the current iteration, "model_name" indicates the model SAM, MedicalSAM and SAM-Med2D, "directory" represents the directory name of the dataset such as Polyp, "part" indicates the subdirectory, sample_name, "sample_name" indicates the original name of the image


## Getting Started
### Download
Please download the EvanySeg cheeckpoints from [ResNet101 checkpoint](https://drive.google.com/file/d/1Hj7LwH8zIJUaiQmDOkHM6JUgxkoTyGpu/view?usp=drive_link) and  [Vit-b checkpoint](https://drive.google.com/file/d/1S_s8zUgv8V2F8LP_h_4HM96j1LWHzjBB/view?usp=drive_link). 

The example datasets are  provided [train.zip](https://drive.google.com/file/d/1zXRUoL2BJzuUDszQb0M3SOyC3-O2STn1/view?usp=drive_link) and [test.zip](https://drive.google.com/file/d/1jfd-5et6kgPqr4stIEA_uYX62P3f8GRI/view?usp=drive_link)


### Installation
Download the datasets and pre-trained models to the corresponding folders, and configure the environment.

```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Test

```python
python test.py
```

### Train

```python
python train.py
```

## Demo
 [Demo](https://modelscope.cn/studios/ahjolsenbi/EvanySeg)
