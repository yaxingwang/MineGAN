# MineGAN: effective knowledge transfer from  GANs to target domains with few images 
# Abstract: 
One of the attractive characteristics of deep neural networks is their ability to transfer knowledge obtained in one domain to other related domains. As a result, high-quality networks can be trained in domains with relatively little training data. This property has been extensively studied for discriminative networks but has received significantly less attention for generative models.  Given the often enormous effort required to train GANs, both computationally as well as in the dataset collection, the re-use of pretrained GANs is a desirable objective.  We propose a novel knowledge transfer method for generative models based on mining the knowledge that is most beneficial to a specific target domain, either from a single or multiple pretrained GANs.  This is done using a miner network that identifies which part of the generative distribution of each pretrained GAN outputs samples closest to the target domain.  Mining effectively steers GAN sampling towards suitable regions of the latent space, which facilitates the posterior finetuning and avoids pathologies of other methods such as mode collapse and lack of flexibility.  We perform experiments on several complex datasets using various GAN architectures (BigGAN, Progressive GAN) and show that the proposed method, called MineGAN, effectively transfers knowledge to domains with few target images, outperforming existing methods.  In addition, MineGAN can successfully transfer knowledge from multiple pretrained GANs. 
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

Coming soon


# References 
- \[1\] 'Large Scale GAN Training for High Fidelity Natural Image Synthesis' by Andrew Brock et. al, (https://arxiv.org/abs/1809.11096)[paper], (https://github.com/ajbrock/BigGAN-PyTorch)[code] 
- \[2\] 'Progressive Growing of GANs for Improved Quality, Stability, and Variation' by Martin Heusel  et. al, (https://arxiv.org/abs/1710.10196)[paper], (https://github.com/tkarras/progressive_growing_of_gans)[code] 
# Contact


If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: yaxing@cvc.uab.es
