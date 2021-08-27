# KPAC: Kernel-Sharing Parallel Atrous Convolutional block
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-blue.svg?style=plastic)

<p align="center">
   <img src="./assets/KPAC.jpg" />
</p>

This repository contains the official Tensorflow implementation of the following paper:

> **[Single Image Defocus Deblurring Using Kernel-Sharing Parallel Atrous Convolutions](https://arxiv.org/abs/2108.09108)**<br>
> Hyeongseok Son, Junyong Lee, Sunghyun Cho, Seungyong Lee, ICCV 2021


## Getting Started
### Prerequisites
*Tested environment*

![Ubuntu16.04](https://img.shields.io/badge/Ubuntu-16.0.4-blue.svg?style=plastic)
![Python 2.7.12](https://img.shields.io/badge/Python-2.7.12-green.svg?style=plastic)
![Tensorflow 1.10.0](https://img.shields.io/badge/Tensorflow-1.10.0-green.svg?style=plastic)
![CUDA 9.0](https://img.shields.io/badge/CUDA-9.0-green.svg?style=plastic)

1. **Pre-trained models**
    * Download and unzip [pretrained weights](https://www.dropbox.com/sh/frpegu68s0yx8n9/AACrptFFhxejSyKJBvLdk9IJa?dl=1) under `./ckpt/`:

        ```
        ├── ./pretrained
        │   ├── single_2level.npz
        │   ├── single_3level.npz
        │   ├── dual.npz
        ```

## Testing models of ICCV2021

```shell
# Our 2-level model 
CUDA_VISIBLE_DEVICES=0 python main_eval_2level.py

# Our 3-level model 
CUDA_VISIBLE_DEVICES=0 python main_eval_3level.py

# Our dual pixel-based model
CUDA_VISIBLE_DEVICES=0 python main_eval_dual.py
```
