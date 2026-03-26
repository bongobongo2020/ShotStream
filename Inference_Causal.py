import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import MultiShots_FrameConcat_Dataset
import sys
from pipeline import (
    CausalInferenceArPipeline,
)
from utils.misc import set_seed

import logging

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default=None)
parser.add_argument("--resume_ckpt", type=str, default=None)
parser.add_argument("--resume_lora_ckpt", type=str, default=None)
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--output_folder", type=str, default=None)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--multi_caption", type=bool, default=True)
parser.add_argument("--use_wo_rope_cache", type=bool, default=False)

args = parser.parse_args()

# Config setting
def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

# Initialize distributed inference
rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
local_rank = int(os.getenv("LOCAL_RANK", 0))
device = local_rank
print(f"Device is {device}")
_init_logging(rank)

if world_size > 1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size)

logging.info(f"Generation job args: {args}")

if dist.is_initialized():
    base_seed = [args.seed] if rank == 0 else [None]
    dist.broadcast_object_list(base_seed, src=0)
    args.seed = base_seed[0]

current_seed = args.seed + rank 
set_seed(current_seed)
logging.info(f"Rank {rank} set seed to {current_seed}")

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("ckpts/default_config.yaml")
config = OmegaConf.merge(default_config, config)

config.multi_caption = args.multi_caption
config.use_wo_rope_cache = args.use_wo_rope_cache

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    pipeline = CausalInferenceArPipeline(config, device=device)

# Load Ckpt
if args.resume_ckpt is not None:
    config.resume_ckpt = args.resume_ckpt

if config.resume_ckpt:
    state_dict = torch.load(config.resume_ckpt, map_location="cpu")
    print(f"resume generator's ckpt from {config.resume_ckpt}")
    pipeline.generator.load_state_dict(state_dict['generator'])

pipeline = pipeline.to(dtype=torch.bfloat16)

# Dataset
if args.data_path is not None:
    config.data_path = args.data_path
print(f"Dataset Path is {config.data_path}")

dataset = MultiShots_FrameConcat_Dataset(csv_path=config.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Output Dir
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    video = pipeline.inference(
        batch=batch,
        use_wo_rope_cache=config.use_wo_rope_cache,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)

    video = 255.0 * torch.cat(all_video, dim=1)

    pipeline.vae.model.clear_cache()

    caption=batch['shots_captions'][0][-1][0][0]
    output_path = f"{args.output_folder}/{i:03d}_{caption[:50]}.mp4"
    write_video(output_path, video[0], fps=16)
