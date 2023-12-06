# ⚡LightGAN: GAN with PyTorch Lightning

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11-255074?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![poetry](https://img.shields.io/badge/-Poetry_1.6+-1e293b?logo=poetry&logoColor=white)](https://python-poetry.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_10.7_%7C_10.8_%7C_12.1-91c733?logo=cuda&logoColor=white)](https://pytorch.org/get-started/previous-versions/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Hydra_1.3+-89b8cf)](https://hydra.cc/)
[![conda](https://anaconda.org/conda-forge/mlconjug/badges/version.svg)](https://hydra.cc/)
[![contributors](https://img.shields.io/github/contributors/unerue/lightning-boilerplate.svg)](https://github.com/unerue/lightning-boilerplate/graphs/contributors)

</div>

- [ ] GAN (2014)
- [ ] DCGAN (2015)
- [ ] WGAN
- [X] CycleGAN (2017) 
- [ ] GcGAN (2019): https://github.com/hufu6371/GcGAN
- [X] CUT (2020)
- [X] FastCUT (2020)
- [ ] SINCUT (2020)
- [ ] NOT (2023): https://github.com/iamalexkorotin/NeuralOptimalTransport
- [X] SB (2023)
- [ ] Image Style Transfer using CNN (2016): https://github.com/ali-gtw/ImageStyleTransfer-CNN

## Lightning hooks

* https://github.com/Lightning-AI/lightning/src/lightning/pytorch/core/hooks.py
* https://lightning.ai/docs/pytorch/latest/common/trainer.html#trainer-class-api


## TODO or Check?

* loss naming
* 

### 2021
* 1st: https://github.com/lyndonzheng/F-LSeSim 
* NAVER: https://github.com/clovaai/tunit

* https://github.com/utkarshojha/few-shot-gan-adaptation
* https://github.com/microsoft/CoCosNet-v2
* https://github.com/fnzhan/UNITE
* https://github.com/jiupinjia/stylized-neural-painting
* https://github.com/DinoMan/DINO

## Code references
* AWESOME list: https://github.com/weihaox/awesome-image-translation
* https://github.com/eriklindernoren/PyTorch-GAN
* https://github.com/nocotan/pytorch-lightning-gans
* https://github.com/TengdaHan/Image-Colorization
* https://github.com/sagiebenaim/DistanceGAN
* https://github.com/hufu6371/GcGAN
* https://github.com/jamesloyys/PyTorch-Lightning-GAN
* https://github.com/deepakhr1999/cyclegans


* https://docs.python.org/3/library/typing.html

imagedata/
    a/b/c/d/
        a1.jpg
    b/c/d/e/
        b1.jpg
        ...
    c/
    a.json
    b.json
    c.json

imagedata/
    images/
        b/
          a1
        c/
    annotations.json
    images
     file_name: images/b/a.jpg