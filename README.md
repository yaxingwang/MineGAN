# MineGAN: effective knowledge transfer from  GANs to target domains with few images 
# Abstract: 
One of the attractive characteristics of deep neural networks is their ability to transfer knowledge obtained in one domain to other related domains. As a result, high-quality networks can be trained in domains with relatively little training data. This property has been extensively studied for discriminative networks but has received significantly less attention for generative models.  Given the often enormous effort required to train GANs, both computationally as well as in the dataset collection, the re-use of pretrained GANs is a desirable objective.  We propose a novel knowledge transfer method for generative models based on mining the knowledge that is most beneficial to a specific target domain, either from a single or multiple pretrained GANs.  This is done using a miner network that identifies which part of the generative distribution of each pretrained GAN outputs samples closest to the target domain.  Mining effectively steers GAN sampling towards suitable regions of the latent space, which facilitates the posterior finetuning and avoids pathologies of other methods such as mode collapse and lack of flexibility.  We perform experiments on several complex datasets using various GAN architectures (BigGAN, Progressive GAN) and show that the proposed method, called MineGAN, effectively transfers knowledge to domains with few target images, outperforming existing methods.  In addition, MineGAN can successfully transfer knowledge from multiple pretrained GANs. 

# Updating 
Training for MineGAN on [StyleGANv2](https://github.com/yaxingwang/MineGAN/tree/master/styleGANv2)

------------------------------------------------------------

Training for MineGAN on [StyleGAN](https://github.com/yaxingwang/MineGAN/tree/master/styleGAN)

------------------------------------------------------------

Training for MineGAN on [mnist](https://github.com/yaxingwang/MineGAN/tree/master/MNISTtf)

------------------------------------------------------------

# Overview 
- [Dependences](#dependences)
- [Installation](#installtion)
- [Instructions](#instructions)
- [Results](#results)
- [References](#references)
- [Contact](#contact)
# Dependences 
- Python2.7, NumPy, SciPy, NVIDIA GPU
- **Tensorflow/Pytorch:** the version of tensorflow should be more 1.0(https://www.tensorflow.org/), pytorch is more 0.4
- **Dataset:** MNIST, CelebA, HHFQ, Imagenet, Places365 or your dataset 

# Installation 
- Install tensorflow/pytorch
# Instructions

- `git clone git@github.com:yaxingwang/MineGAN.git` to get `MineGA`

- Pretrained model: downloading the pretrained model from [Biggan](https://github.com/ajbrock/BigGAN-PyTorch), and put it into `data/your_data/weights`. Note using `G_ema.pth` to replace `G.pth`, since we dones't use `ema`. The pretrained model is moved into `MineGA/weights/biggan` 

- Downloading [inception model](https://drive.google.com/file/d/1A5C1jYieAcu_CDml0mhrLGqCCBF4uujG/view?usp=sharing) and moving it into `MineGA` 

- Preparing data: leveraging  `sh scripts/utils/prepare_data.py`, and put it into `data/your_data/data`. Please check [Biggan](https://github.com/ajbrock/BigGAN-PyTorch) to learn how to generate the data 

- Traing: ```sh scripts/launch_BigGAN_bs256x8.sh```


 


# References 
- \[1\] [BigGAN](https://arxiv.org/abs/1809.11096) 
- \[2\] [PGAN](https://arxiv.org/abs/1710.10196) 
# Contact


If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: yaxing@cvc.uab.es
