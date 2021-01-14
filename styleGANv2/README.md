# MineGAN for StyleGANv2

### Dependences 
- Python3.7, NumPy, SciPy, NVIDIA GPU(CUDA10.2)
- **Pytorch:**  Pytorch is  1.5 (I just test 1.5)

### Download datasets
```
Animal Face: https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/AnimalFace.zip
Anime Face: http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip
```

### Preprocess datasets
```
python prepare_data.py  data/CatHead --out data/CatHead_lmdb --size 256
```

### Download pre-traind GAN models
This pretrained model is  [unoffical StyleGANv2 one](https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/view), please cite this [repository](https://github.com/rosinality/stylegan2-pytorch) if you use the pretrained model. Given the downloaded pretrained model, we can creat new folder(e.g. 'model'), and move the downloaded model into this folder. 


### Run experiments
```
python -m torch.distributed.launch --nproc_per_node=1  train.py   data/CatHead_lmdb  --test_number 160  --batch 4 --size 256  --channel_multiplier  2 --ckpt model/550000.pt  --output_dir results/MineGAN
```

If you use the provided data and code, please cite the following papers:
 
```
@InProceedings{Wang_2020_CVPR,
author = {Wang, Yaxing and Gonzalez-Garcia, Abel and Berga, David and Herranz, Luis and Khan, Fahad Shahbaz and Weijer, Joost van de},
title = {MineGAN: Effective Knowledge Transfer From GANs to Target Domains With Few Images},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
} 

```
