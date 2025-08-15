# CSFNet-pytorch

This is the official pytorch implementation of "Cosine Similarity Fusion Network for Real-Time RGB-X Semantic Segmentation of Driving Scenes".

![Network.png](https://github.com/Danial-Qashqai/CSFNet/blob/main/figures/Network_main.png)
*Overview of our proposed Cosine Similarity Fusion Network (CSFNet) for real-time RGB-X semantic segmentation.*

## Results
We offer the pre-trained weights on different RGB-X (RGB-D/T/P) datasets:

### Validation on Cityscapes (19 categories) 
| Architecture | Backbone | Params(M) | FPS | mIoU.half | Weight | Training Log |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.31 | 106.1 | 74.73 | [Google Drive](https://drive.google.com/file/d/1yK1Fg7NX1zryVDQTbzIDVGnn8prxLsjY/view?usp=sharing) | log |
| CSFNet-2 | STDC2 | 19.37 | 72.3 | 76.36 | [Google Drive](https://drive.google.com/file/d/1yQGGVAOUcSeWYz-vjoIIViIU_uV6uBpy/view?usp=sharing) | log |


![Fig4.png](https://github.com/Danial-Qashqai/CSFNet/blob/main/figures/Figure_4.png)
*Visual results of CSFNet on Cityscapes val set (half resolution). From left to right: RGB input, depth input, prediction, and ground truth.*

### Validation on MFNet (9 categories)
| Architecture | Backbone | Params(M) | FPS | mIoU | Weight | Training Log |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.30 | 106.3 | 56.05 | [Google Drive](https://drive.google.com/file/d/1y_YTWsq-W3PQYiq1XFDygnX3SG9ZuvUk/view?usp=sharing) | log |
| CSFNet-2 | STDC2 | 19.36 | 72.7 | 59.98 | [Google Drive](https://drive.google.com/file/d/1yfAk7pFSeb6QBedaK_M_n2OUg53jLYqJ/view?usp=sharing) | log |

### Validation on ZJU (8 categories)
| Architecture | Backbone | Params(M) | FPS | mIoU | Weight | Training Log |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.30 | 108.5 | 90.85 | [Google Drive](https://drive.google.com/file/d/1yT1yAtqfDvQDjEO3ypvhmr8V3b-Tgh-u/view?usp=sharing) | log |
| CSFNet-2 | STDC2 | 19.36 | 75 | 91.40 | [Google Drive](https://drive.google.com/file/d/1ycSKi80HhilbX2U7dQUF-a8vdrR_vFEV/view?usp=sharing) | log |

### Validation on FMB (14 categories)
| Architecture | Backbone | Params(M) | FPS | mIoU | Weight | Training Log |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.30 | 106.2 | 60.61 | [Google Drive](https://drive.google.com/file/d/1QzrOoYmB4C9pAbFuKhJ-YsRx77s0drtx/view?usp=sharing) | log |
| CSFNet-2 | STDC2 | 19.36 | 72.5 | 62.73 | [Google Drive](https://drive.google.com/file/d/1M4TQb19LmH7A0pM68RluQH7gf5_4aFoL/view?usp=sharing) | log |
 
We measured the FPS using a single NVIDIA RTX 3090 GPU.

## Usage
### Installation
Clone repository:
Please navigate to the cloned directory.
```
git clone https://github.com/Danial-Qashqai/CSFNet
cd /path/to/this/repository
```
Requirements:
we are using Python 3.10.14, Torch 2.3.1, torchvision 0.18.1 and CUDA 12.1.

Install pytorch, cuda and cudnn, then install other dependencies via:
```shell
pip install -r requirements.txt
```
### Datasets
For easy access, we have prepared the Cityscapes, MFNet, and ZJU datasets at the following links:
- Cityscapes_val [Google Drive](https://drive.google.com/file/d/11oBaU3lXQHzVk3Gp2WIa14n4yk9mlhXz/view?usp=sharing) / [kaggle](https://www.kaggle.com/datasets/danialqashqai/cityscapes-rgbd-val).
- Cityscapes_train [kaggle](https://www.kaggle.com/datasets/danialqashqai/cityscapes-rgbd-train).
- MFNet  [Google Drive](https://drive.google.com/file/d/1ytbhoiFpkRk_iMbL0qGGa_Q6feWlCibS/view?usp=sharing).
- ZJU   [Google Drive](https://drive.google.com/file/d/1TugQ16fcxbmPBJD0EPMHHmjdK9IE4SAO/view?usp=sharing).
- FMB   [Google Drive](https://drive.google.com/drive/folders/1T_jVi80tjgyHTQDpn-TjfySyW4CK1LlF?usp=sharing).

You may also refer to their official websites for data preparation and further details:
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)
- [ZJU](https://huggingface.co/datasets/Zhonghua/ZJU_RGB_P/tree/main)
- [FMB](https://github.com/JinyuanLiu-CV/SegMiF?tab=readme-ov-file)

Orgnize the Cityscapes dataset in the following structure:

```shell
<Dataset>
|-- <cityscapes>
    |-- <train>
        |-- <rgb>
        |-- <depth_raw>
        |-- <labels_19>
    |-- <valid>
        |-- <rgb>
        |-- <depth_raw>
        |-- <labels_19>
```

### Pre-trained ImageNet Backbones
The pre-trained weights for the [STDC1](https://github.com/MichaelFan01/STDC-Seg) and [STDC2](https://github.com/MichaelFan01/STDC-Seg) backbones are available at the following links:
- [STDC1](https://drive.google.com/file/d/1xR7Hg0CQcGyCFRgoF6vuhFNClE4ACpF_/view?usp=sharing)
- [STDC2](https://drive.google.com/file/d/1xecVDI_8WvExrybZIzweC6urcllkFPQq/view?usp=sharing)


## Training

* Train our CSFNet-2 on Cityscapes:
```
python train.py \
    --batch_size 16 \
    --batch_size_valid 8 \
    --num_gpus 1 \
    --network "CSFNet-2" \
    --dataset "Cityscapes" \
    --num_classes 19 \
    --epochs 300 \
    --lr 0.02  \
    --pretrained True \
    --backbone_path "Backbone/STDCNet_2.tar"  \
    --crop_H 512 \
    --crop_W 1024 \
    --img_train_dir "./cityscapes/train" \
    --img_test_dir "./cityscapes/valid" 
```

* Train our CSFNet-2 on MFNet:
```
python train.py \
    --batch_size 8 \
    --batch_size_valid 8 \
    --num_gpus 1 \
    --network "CSFNet-2" \
    --dataset "MFNet" \
    --num_classes 9 \
    --epochs 600 \
    --lr 0.01  \
    --pretrained True \
    --backbone_path "Backbone/STDCNet_2.tar"  \
    --crop_H 480 \
    --crop_W 640 \
    --img_train_dir "./MFNet/ir_seg_dataset" \
    --img_test_dir "./MFNet/ir_seg_dataset" 
```

* Train our CSFNet-2 on ZJU:
```
python train.py \
    --batch_size 8 \
    --batch_size_valid 8 \
    --num_gpus 1 \
    --network "CSFNet-2" \
    --dataset "ZJU" \
    --num_classes 8 \
    --epochs 600 \
    --lr 0.01  \
    --pretrained True \
    --backbone_path "Backbone/STDCNet_2.tar"  \
    --crop_H 512 \
    --crop_W 612 \
    --img_train_dir "./ZJU/train" \
    --img_test_dir "./ZJU/val" 
```

* Train our CSFNet-2 on FMB:
```
python train.py \
    --batch_size 8 \
    --batch_size_valid 8 \
    --num_gpus 1 \
    --network "CSFNet-2" \
    --dataset "FMB" \
    --num_classes 14 \
    --epochs 600 \
    --lr 0.01  \
    --pretrained True \
    --backbone_path "Backbone/STDCNet_2.tar"  \
    --crop_H 600 \
    --crop_W 800 \
    --img_train_dir "./FMB/train" \
    --img_test_dir "./FMB/test" 
```

## Evaluation
* Evaluate our CSFNet-2 on Cityscapes:
```
python eval.py \
    --batch_size_valid 8 \
    --num_gpus 1 \
    --network "CSFNet-2" \
    --dataset "Cityscapes" \
    --num_classes 19 \
    --img_test_dir "./cityscapes/valid"  \
    --weight_path "./best_CSFNet_2_city.pth"
```

* Evaluate our CSFNet-2 on MFNet:
```
python eval.py \
    --batch_size_valid 8 \
    --num_gpus 1 \
    --network "CSFNet-2" \
    --dataset "MFNet" \
    --num_classes 9 \
    --img_test_dir "./MFNet/ir_seg_dataset"  \
    --weight_path "./best_CSFNet_2_MFNet.pth"
```

* Evaluate our CSFNet-2 on ZJU:
```
python eval.py \
    --batch_size_valid 8 \
    --num_gpus 1 \
    --network "CSFNet-2" \
    --dataset "ZJU" \
    --num_classes 8 \
    --img_test_dir "/ZJU/val"  \
    --weight_path "./best_CSFNet_2_ZJU.pth"
```

* Evaluate our CSFNet-2 on FMB:
```
python eval.py \
    --batch_size_valid 8 \
    --num_gpus 1 \
    --network "CSFNet-2" \
    --dataset "FMB" \
    --num_classes 14 \
    --img_test_dir "/FMB/test"  \
    --weight_path "./best_CSFNet_2_FMB.pth"
```


## Contact

Danial Qashqai：danialqashqai99@gmail.com
