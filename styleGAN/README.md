# MineGAN for StyleGAN

### Dependences 
- Python3.7, NumPy, SciPy, NVIDIA GPU
- **Pytorch:**  Pytorch is more 1.2 (pytorch14 doesn't work)

### Download datasets
```
Animal Face: https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/AnimalFace.zip
Anime Face: http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip
```

### Preprocess datasets
```
python prepare_data.py --out dataset/DATASET_lmdb --n_worker 8 dataset/DATASET
```

### Download pre-traind GAN models
```
# Download from https://drive.google.com/file/d/1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ/view
# Save model in ./checkpoint directory
```

### Pre-compute FID activations
```
python precompute_acts.py --dataset DATASET
```

### Run experiments
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --name DATASET_finetune --mixing --miner --loss r1 --sched --dataset DATASET --save_path result/DATASET  
```

If you use the provided data and code, please cite the following papers:
 
```
@inproceedings{
    mo2020freeze,
    title={Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs},
    author={Mo, Sangwoo and Cho, Minsu and Shin, Jinwoo},
    booktitle = {CVPR AI for Content Creation Workshop},
    year={2020},
}

@InProceedings{Wang_2020_CVPR,
author = {Wang, Yaxing and Gonzalez-Garcia, Abel and Berga, David and Herranz, Luis and Khan, Fahad Shahbaz and Weijer, Joost van de},
title = {MineGAN: Effective Knowledge Transfer From GANs to Target Domains With Few Images},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
} 

```
