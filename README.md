# CSFNet-pytorch

This is the official pytorch implementation of "CSFNet: A Cosine Similarity Fusion Network for Real-Time RGB-X Semantic Segmentation of Driving Scenes".

![Network.png](https://github.com/Danial-Qashqai/CSFNet/blob/main/figures/Network.png)


## Result
We offer the pre-trained weights on different RGB-X (RGB-D/T/P) datasets:

### Validation on Cityscapes (19 categories) 
| Architecture | Backbone | Params(M) | FPS | mIoU.half | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.31 | 106.1 | 74.73 | [Google Drive](https://drive.google.com/file/d/1yK1Fg7NX1zryVDQTbzIDVGnn8prxLsjY/view?usp=sharing) |
| CSFNet-2 | STDC2 | 19.37 | 72.3 | 76.36 | [Google Drive](https://drive.google.com/file/d/1yQGGVAOUcSeWYz-vjoIIViIU_uV6uBpy/view?usp=sharing) |

### Validation on MFNet (9 categories)
| Architecture | Backbone | Params(M) | FPS | mIoU | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.30 | 106.3 | 56.05 | [Google Drive](https://drive.google.com/file/d/1y_YTWsq-W3PQYiq1XFDygnX3SG9ZuvUk/view?usp=sharing) |
| CSFNet-2 | STDC2 | 19.36 | 72.7 | 59.98 | [Google Drive](https://drive.google.com/file/d/1yfAk7pFSeb6QBedaK_M_n2OUg53jLYqJ/view?usp=sharing) |

### Validation on ZJU (8 categories)
| Architecture | Backbone | Params(M) | FPS | mIoU | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.30 | 108.5 | 90.85 | [Google Drive](https://drive.google.com/file/d/1yT1yAtqfDvQDjEO3ypvhmr8V3b-Tgh-u/view?usp=sharing) |
| CSFNet-2 | STDC2 | 19.36 | 75 | 91.40 | [Google Drive](https://drive.google.com/file/d/1ycSKi80HhilbX2U7dQUF-a8vdrR_vFEV/view?usp=sharing) |

### Validation on FMB (14 categories)
| Architecture | Backbone | Params(M) | FPS | mIoU | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.30 | 106.2 | -- | [Google Drive]() |
| CSFNet-2 | STDC2 | 19.36 | 72.5 | 62.73 | [Google Drive](https://drive.google.com/file/d/1M4TQb19LmH7A0pM68RluQH7gf5_4aFoL/view?usp=sharing) |
 
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

* Train our CSFNet-2 on MFNet:

* Train our CSFNet-2 on ZJU:

* Train our CSFNet-2 on FMB:

## Evaluation
* Evaluate our CSFNet-2 on Cityscapes:

* Evaluate our CSFNet-2 on MFNet:

* Evaluate our CSFNet-2 on ZJU:

* Evaluate our CSFNet-2 on FMB:
  
## Notes

Our code will be released soon.


## Contact

Danial Qashqai：danialqashqai99@gmail.com
