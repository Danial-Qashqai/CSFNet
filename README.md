# CSFNet-pytorch

This is the official pytorch implementation of "CSFNet: A Cosine Similarity Fusion Network for Real-Time RGB-X Semantic Segmentation of Driving Scenes".

![Network.png](https://github.com/Danial-Qashqai/CSFNet/blob/main/figures/Network.png)


## Result
We offer the pre-trained weights on different RGBX datasets:

### Cityscapes(19 categories) 
| Architecture | Backbone | Params(M) | FPS | mIoU.half | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.31 | 106.1 | 74.73 | [Google Drive](https://drive.google.com/file/d/1yK1Fg7NX1zryVDQTbzIDVGnn8prxLsjY/view?usp=sharing) |
| CSFNet-2 | STDC2 | 19.37 | 72.3 | 76.36 | [Google Drive](https://drive.google.com/file/d/1yQGGVAOUcSeWYz-vjoIIViIU_uV6uBpy/view?usp=sharing) |


### MFNet(9 categories)
| Architecture | Backbone | Params(M) | FPS | mIoU.half | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.30 | 106.3 | 56.05 | [Google Drive](https://drive.google.com/file/d/1y_YTWsq-W3PQYiq1XFDygnX3SG9ZuvUk/view?usp=sharing) |
| CSFNet-2 | STDC2 | 19.36 | 72.7 | 59.98 | [Google Drive](https://drive.google.com/file/d/1yfAk7pFSeb6QBedaK_M_n2OUg53jLYqJ/view?usp=sharing) |

### ZJU(8 categories)
| Architecture | Backbone | Params(M) | FPS | mIoU.half | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | - | 108.5 | 90.85 | [Google Drive](https://drive.google.com/file/d/1yT1yAtqfDvQDjEO3ypvhmr8V3b-Tgh-u/view?usp=sharing) |
| CSFNet-2 | STDC2 | - | 75 | 91.40 | [Google Drive](https://drive.google.com/file/d/1ycSKi80HhilbX2U7dQUF-a8vdrR_vFEV/view?usp=sharing) |

## Notes

Our code will be released soon.


## Contact

Danial Qashqai：danialqashqai99@gmail.com
