import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from metric.inception import InceptionV3
from metric.metric import get_fake_images_and_acts, compute_fid

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator, Miner, Miner_semantic_conv
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment

def evaluate(args,real_acts=None):
    miner.eval()
    miner_semantic.eval()
    fake_images, fake_acts = get_fake_images_and_acts(inception, g_ema, miner, miner_semantic, code_size=args.latent, sample_num=args.sample_num, batch_size=8,device=device)
    # Real:
    if real_acts is  None:
        acts=[]
        pbar = tqdm(total=args.test_number)
        for  real_image in loader_test:
             real_image=real_image.cuda()
             with torch.no_grad():
                 out = inception(real_image)
                 out = out[0].squeeze(-1).squeeze(-1)
             acts.append(out.cpu().numpy())  # numpy
             pbar.update(len(real_image))  # numpy
        real_acts = np.concatenate(acts, axis=0)  # N x d    
    fid = compute_fid(real_acts, fake_acts)
	
    miner.train()
    miner_semantic.train()
    metrics = {'fid': fid}
    return fid,real_acts  
	
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
def sample_data_test(dataset, batch_size, image_size=4, drop_last=True):

    dataset.resolution = image_size

    loader = data.DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=1, drop_last=drop_last)
    return loader

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device, miner=None):
    if n_noise == 1:
        return  miner(torch.randn(batch, latent_dim, device=device))[0] # the output of miner is list  

    noises =  miner(torch.randn(n_noise, batch, latent_dim, device=device).unbind(0))

    return noises


def mixing_noise(batch, latent_dim, prob, device, miner=None):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device, miner)

    else:
        return [make_noise(batch, latent_dim, 1, device, miner)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, miner, miner_semantic):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        miner_module = miner.module
        miner_semantic_module = miner_semantic.module

    else:
        g_module = generator
        d_module = discriminator
        miner_module = miner
        miner_semantic_module = miner_semantic

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    step_dis = 1000
    best_fid,real_acts = evaluate(args)
    print('--------fid:%f----------'%best_fid)
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(miner, False)
        requires_grad(miner_semantic, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device, miner=miner)
        fake_img, _ = generator(noise, miner_semantic=miner_semantic)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss
        if i > (args.start_iter +  step_dis):
            requires_grad(generator, True)
        else:
            requires_grad(generator, False)
        requires_grad(miner, True)
        requires_grad(miner_semantic, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device, miner=miner)
        fake_img, _ = generator(noise, miner_semantic=miner_semantic)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:#I do not regularize the miner
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device, miner=miner)
            fake_img, latents = generator(noise, miner_semantic=miner_semantic, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

                        #f"sample/{str(i).zfill(6)}.png",
            if i % 100 == 0:
                fid, _ = evaluate(args, real_acts=real_acts)
                print('------fid:%f-------'%fid)
                if fid<best_fid: 
                    torch.save(
                        {
                            "miner": miner.state_dict(),
                            "miner_semantic": miner_semantic.state_dict(),
                            "d": d_module.state_dict(),
                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "args": args,
                            "ada_aug_p": ada_aug_p,
                        },
                        '%s/best_model.pt'%(os.path.join(args.output_dir, 'checkpoint')),
                    )
                    best_fid=fid.copy() 
                with torch.no_grad():
                    g_ema.eval()
                    miner.eval()
                    miner_semantic.eval()
                    sample, _ = g_ema(miner(sample_z), miner_semantic=miner_semantic)
                    utils.save_image(
                        g_ema([sample_z])[0],
                        '%s/%s_w_o_miner.png'%(os.path.join(args.output_dir, 'samples'), str(i).zfill(6)),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        sample,
                        '%s/%s.png'%(os.path.join(args.output_dir, 'samples'), str(i).zfill(6)),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    miner.train()
                    miner_semantic.train()

                    #f"checkpoint/{str(i).zfill(6)}.pt",
            if i % 100000 == 0:
                torch.save(
                    {
                        "miner": miner_module.state_dict(),
                        "miner_semantic": miner_semantic_module.state_dict(),
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    '%s/%s.pt'%(os.path.join(args.output_dir, 'checkpoint'), str(i).zfill(6)),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("path_test", type=str, help="path to the lmdb dataset")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--sample_num", type=int, default=5000, help="the number of samples computing FID"
    )
    parser.add_argument(
        "--test_number", type=int, default=100, help="the number of test samples"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="path to save the generatd image and  model",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )


    args = parser.parse_args()

    print(args)
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    if not os.path.exists(os.path.join(args.output_dir, 'checkpoint')):
        os.makedirs(os.path.join(args.output_dir, 'checkpoint'))

    if not os.path.exists(os.path.join(args.output_dir, 'samples')):
        os.makedirs(os.path.join(args.output_dir, 'samples'))

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    miner = Miner(args.latent).to(device)  
    miner_semantic = Miner_semantic_conv(code_dim=8, style_dim=args.latent).to(device)  # using conv 

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    g_optim.add_param_group({'params': miner.parameters(), 'lr': args.lr * g_reg_ratio, 'betas': (0 ** g_reg_ratio, 0.99 ** g_reg_ratio)})
    g_optim.add_param_group({'params': miner_semantic.parameters(), 'lr': args.lr * g_reg_ratio, 'betas': (0 ** g_reg_ratio, 0.99 ** g_reg_ratio)})
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)#  I add strict=False, since the provided model is little different. And we add two miner networks
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)#  I add strict=False

        #g_optim.load_state_dict(ckpt["g_optim"]) 
        #d_optim.load_state_dict(ckpt["d_optim"]) 

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        miner = nn.parallel.DistributedDataParallel(
            miner,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        miner_semantic = nn.parallel.DistributedDataParallel(
            miner_semantic,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    dataset_test = MultiResolutionDataset(args.path, transform_test, args.size)#here we can give the test data path 
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    loader_test = sample_data_test(

        dataset_test, batch_size=args.batch, image_size=args.size, drop_last= True

    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")
    inception = nn.DataParallel(InceptionV3()).cuda()
    inception.eval()
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, miner, miner_semantic)
