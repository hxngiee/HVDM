## [ECCV 2024] HVDM: Hybrid Video Diffusion Models with 2D Triplane and 3D Wavelet Representation

Official PyTorch implementation of **["Hybrid Video Diffusion Models with 2D Triplane and 3D Wavelet Representation"](https://arxiv.org/abs/2402.13729)**.   

### 1. Environment setup
```bash
conda create -n hvdm python=3.8 -y
source activate hvdm
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install natsort tqdm gdown omegaconf einops lpips pyspng tensorboard imageio av moviepy PyWavelets
```

### 2. Dataset 

#### Dataset download
We conduct experiments on three datasets: [SkyTimelapse](https://github.com/weixiong-ur/mdgan), [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php), [TaiChi](https://github.com/AliaksandrSiarohin/first-order-model/tree/master/data/taichi-loading). Please refer to the directories structure below and locate it in the `/data` folder. You can modify the data directory path where data is stored by changing the `data_location` variable in `tools/dataloader.py`.

#### Directories structure
The dataset and checkpoints should be placed in the following structures below
```
HVDM
├── configs
├── data
    └── SKY
        ├── 001.png
        └── ...
    └── TaiChi
        ├── 001.png
        └── ...
    └── UCF-101
        ├── folder
            ├── 001.avi    
            └── ...    
├── ...
├── results
    ├── ddpm_final_[DATASET]_42
        ├── model_[EPOCH].pth
        └── ...
    └── first_stage_ae_final_[DATASET]_42
        ├── model_[EPOCH].pth
        └── ...
├── tools
└── main.py
```

### 3. Training
For settings related to the experiment name, please refer to the [PVDM](https://github.com/sihyun-yu/PVDM) which is the repository our code is based on. Here, `[EXP_NAME]` is an experiment name you want to specifiy, `[DATASET]` is either `SKY` or `UCF101` or `TaiChi`, and `[DIRECTOTY]` denotes a directory of the autoencoder to be used.

#### Autoencoder

```bash
 python main.py 
 --exp first_stage \
 --id [EXP_NAME] \
 --pretrain_config configs/autoencoder/base.yaml \
 --data [DATASET_NAME] \
 --batch_size [BATCH_SIZE]
```
This script will automatically save logs and checkpoints in `./results` folder.

#### Diffusion model

```bash
 python main.py \
 --exp ddpm \
 --id [EXP_NAME] \
 --pretrain_config configs/autoencoder/base.yaml \
 --data [DATASET] \
 --first_model [AUTOENCODER DIRECTORY] 
 --diffusion_config configs/latent-diffusion/base.yaml \
 --batch_size [BATCH_SIZE]
```

### 4. Inference 
We are currently working on incorporating code for Image2Video and Video Dynamics Control. Also the model checkpoints will be released soon.

#### Short Video Generation
```bash
python sample.py 
--exp ddpm \
--first_model './results/model_[EPOCH].pth' \
--second_model 'results/ddpm_main_UCF101_42/ema_model_[EPOCH].pth' \
--mode short
```

#### Long Video Generation
```bash
python sample.py 
--exp ddpm \
--first_model '.results/model_[EPOCH].pth' \ 
--second_model 'results/ddpm_main_[DATASET]_42/ema_model_[EPOCH].pth' \
--mode long
```

### Citation
```bibtex
@article{kim2024hybrid,
  title={Hybrid Video Diffusion Models with 2D Triplane and 3D Wavelet Representation},
  author={Kim, Kihong and Lee, Haneol and Park, Jihye and Kim, Seyeon and Lee, Kwanghee and Kim, Seungryong and Yoo, Jaejun},
  journal={arXiv preprint arXiv:2402.13729},
  year={2024}
}
```

### Reference
HVDM draws significant inspiration from the following projects: [pvdm](https://github.com/sihyun-yu/PVDM), [wavediff](https://github.com/VinAIResearch/WaveDiff), [latent-diffusion](https://github.com/CompVis/latent-diffusionn), and [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repositories. We thank to all contributors for making their work openly accessible.
