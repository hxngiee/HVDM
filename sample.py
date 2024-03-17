import sys; sys.path.extend(['.'])
import os
import argparse
import torch
from omegaconf import OmegaConf
from exps.diffusion import diffusion
from exps.first_stage import first_stage
from utils import set_random_seed
import os
import json

from tools.trainer import latentDDPM
from tools.dataloader import get_loaders
from tools.scheduler import LambdaLinearScheduler
from models.autoencoder.autoencoder_vit import ViTAutoencoder
from models.ddpm.unet import UNetModel, DiffusionWrapper
from losses.ddpm import DDPM

import copy
from utils import file_name, Logger, download

from models.ema import LitEma

from einops import rearrange

import numpy as np
import torch.nn.functional as F
from torchvision import transforms

import torchvision
import PIL

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained

from utils import DWT_3D, IDWT_3D

def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print (f'Saving Video with {T} frames, img shape {H}, {W}')

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    return img

import time
import random
import numpy as np
import torch
start = time.time()

seed = 2023

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='experiment name to run')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='main', help='experiment identifier')

""" Args about Data """
parser.add_argument('--data', type=str, default='SKY')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--ds', type=int, default=4)

""" Args about Model """
parser.add_argument('--pretrain_config', type=str, default='configs/autoencoder/base.yaml')
parser.add_argument('--diffusion_config', type=str, default='configs/latent-diffusion/base.yaml')

# for diffusion model path specification
parser.add_argument('--first_model', type=str, required=True, help='the path of pretrained autoencoder model')
parser.add_argument('--second_model', type=str, required=True, help='the path of pretrained diffusion model')

parser.add_argument('--mode', type=str, choices=['short', 'long'], required=True, help='the sampling mode of pretrained diffusion model')

args = parser.parse_args()
""" FIX THE RANDOMNESS """
set_random_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.n_gpus = torch.cuda.device_count()

device = torch.device('cuda')

# init and save configs

""" RUN THE EXP """
config = OmegaConf.load(args.diffusion_config)
first_stage_config = OmegaConf.load(args.pretrain_config)
args.unetconfig = config.model.params.unet_config
args.lr = config.model.base_learning_rate
args.scheduler = config.model.params.scheduler_config
args.res = first_stage_config.model.params.ddconfig.resolution
args.timesteps = first_stage_config.model.params.ddconfig.timesteps
args.skip = first_stage_config.model.params.ddconfig.skip
args.ddconfig = first_stage_config.model.params.ddconfig
args.embed_dim = first_stage_config.model.params.embed_dim
args.ddpmconfig = config.model.params
args.cond_model = config.model.cond_model

rank = 0

first_stage_model = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
first_stage_model_ckpt = torch.load(args.first_model)
first_stage_model.load_state_dict(first_stage_model_ckpt)

unet = UNetModel(**args.unetconfig)

model = DiffusionWrapper(unet).to(device)

ema_model = copy.deepcopy(model)
dir = args.second_model
ema_model_ckpt = torch.load(dir)
ema_model.load_state_dict(ema_model_ckpt)
ema_model.eval()

first_stage_model.eval()
model.eval()

cond_model = ema_model.diffusion_model.cond_model # True
diffusion_model = DDPM(ema_model,
                       channels=ema_model.diffusion_model.in_channels,
                       image_size=ema_model.diffusion_model.image_size,
                       sampling_timesteps=100,
                       w=0.).to(device)

def main():

    epoch = dir.split('_')[-1].split('.')[0]

    if args.mode == 'short':
        print('Sampling Short Video Batches')

        os.makedirs(f'./results/test_{epoch}/short', exist_ok=True)

        with torch.no_grad():
            k = args.batch_size
            for i in range(2048 // k):
                print(f'gen : {i}')

                z = diffusion_model.sample(batch_size=k)
                fake  = first_stage_model.decode_from_sample(z)
                fake  = fake.clamp(-1, 1).cpu()

                fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=k)) * 127.5
                fake = fake.type(torch.uint8)

                fakes = []
                fakes.append(rearrange(fake[:k], 'b t h w c -> b c t h w'))
                fakes = torch.cat(fakes)

                print(fakes.shape)

                for j in range(k):
                    save_image_grid(fakes[j:j+1,:,:,:,:].cpu().numpy(), os.path.join(f'results/test_{epoch}/short', f'generated_{k*i + j}.gif'), drange=[0, 255], grid_size=(1, 1))

    elif args.mode == 'long':

        dwt = DWT_3D("haar")

        print('Sampling Long Video Batches')

        os.makedirs(f'./results/test_{epoch}/long', exist_ok=True)

        with torch.no_grad():
            k = args.batch_size
            for i in range(2048 // k):
                fakes = []

                print(f'gen : {i}')

                z = diffusion_model.sample(batch_size=k)
                fake = first_stage_model.decode_from_sample(z)
                fake = fake.clamp(-1, 1)

                res = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=k)) * 127.5
                _res = rearrange(res / 127.5 - 1, 'b t h w c -> b c t h w')
                res_dwt = dwt(_res)
                prev = first_stage_model.extract(_res, res_dwt).detach()

                fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=k)) * 127.5
                fake = fake.type(torch.uint8)

                _first = fake.cpu()
                _first = rearrange(_first[:k], 'b t h w c -> b c t h w')
                fakes.append(_first[:, :, :, :, :])

                for j in range(7):
                    z = diffusion_model.sample(batch_size=k, cond=prev)
                    fake = first_stage_model.decode_from_sample(z)
                    fake = fake.clamp(-1, 1)

                    res = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=k)) * 127.5
                    _res = rearrange(res / 127.5 - 1, 'b t h w c -> b c t h w')
                    res_dwt = dwt(_res)
                    prev = first_stage_model.extract(_res, res_dwt).detach()

                    res = res.type(torch.uint8).cpu()
                    intermediates = rearrange(res[:k], 'b t h w c -> b c t h w')
                    fakes.append(intermediates[:, :, :, :, :])

                fakes = torch.cat(fakes, dim=2)

                print(fakes.shape)

                for j in range(k):
                    save_image_grid(fakes[j:j + 1, :, :, :, :].cpu().numpy(), os.path.join(f'results/test_{epoch}/long', f'generated_{k * i + j}.gif'), drange=[0, 255], grid_size=(1, 1))

    print("time :", time.time() - start)


if __name__ == '__main__':
    main()