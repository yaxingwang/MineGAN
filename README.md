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
- **Tensorflow:** the version should be more 1.0(https://www.tensorflow.org/)
- **Dataset:** lsun-bedroom(http://lsun.cs.princeton.edu/2017/) or your dataset 

# Installation 
- Install tensorflow
- Opencv 
# Instructions
- Using 'git clone https://github.com/yaxingwang/Transferring-GANs'

    You will get new folder whose name is 'Transferring-GANs' in your current path, then  use 'cd Transferring-GANs' to enter the downloaded new folder
    
- Download pretrain models[Google driver](https://drive.google.com/file/d/1e7Pw-m-DgAiB_aQnNUUwBRVFc2izRiRw/view?usp=sharing); [Tencent qcloud](https://share.weiyun.com/5mBsISh)

    Uncompressing downloaded folder to current folder, then you have new folder 'transfer_model'  which contains two folders: 'conditional', 'unconditional', each of which has four folders: 'imagenet', 'places', 'celebA', 'bedroom'

- Download dataset or use your dataset.

    I have shown one example and you could make it with same same form.

- Run 'python transfer_gan.py'

   Runing code with default setting. The pretrained model can be seleted by changing the parameter 'TARGET_DOMAIN'
 
- Conditional GAN 
  If you are interested in using conditional model, just setting parameter 'ACGAN = True'
# Results 
Using pretrained models not only get high performance, but fastly attach convergence. In following figure, we show conditional and unconditional settings.
<br>
<p align="center"><img width="100%" height='60%'src="results/FID.png" /></p>



# References 
- \[1\] 'Improved Training of Wasserstein GANs' by Ishaan Gulrajani et. al, https://arxiv.org/abs/1704.00028, (https://github.com/igul222/improved_wgan_training)[code] 
- \[2\] 'GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium' by Martin Heusel  et. al, https://arxiv.org/abs/1704.00028
# Contact

If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: yaxing@cvc.uab.es
