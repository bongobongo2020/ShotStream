# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import imageio
import warnings

warnings.filterwarnings('ignore')

from wan.utils.utils import str2bool

import gc
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.scheduler import FlowMatchScheduler

from einops import rearrange
from utils.dataset import MultiShots_FrameConcat_Dataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Wan 2.1 T2V FrameConcat"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-1.3B",
        help="The task to run.")
    
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="ckpts/bidirectional_teacher.pt",
        help="The path to the checkpoint directory.")
    
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="demo/data/sample.csv",
        help="The path to the checkpoint directory.")

    parser.add_argument(
        "--max_context_frames",
        type=int,
        default=6,
        help="Whether use dynamic sample frames")

    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=True,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="demo/data/ode_sample",
        help="The file to save the generated image or video to.")
    
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='flow_match',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    
    parser.add_argument(
        "--sample_steps", type=int, default=50, help="The sampling steps.")
    
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=8,
        help="Sampling shift factor for flow matching schedulers.")
    
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=6.0,
        help="Classifier free guidance scale.")

    parser.add_argument(
        "--multi_caption",
        type=bool,
        default=True)

    args = parser.parse_args()

    return args

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.param_dtype = torch.bfloat16

        self.text_encoder = WanTextEncoder().to(self.device)
        self.vae = WanVAEWrapper().to(self.device)

        self.model = WanDiffusionWrapper(model_name='Wan2.1-T2V-1.3B', is_causal=False).to(self.device)
        self.model.model.eval().requires_grad_(False)

        ckpt = torch.load(checkpoint_dir, map_location='cpu', mmap=True)['generator']
        self.model.load_state_dict(ckpt)
        print(f"resume ckpt from {checkpoint_dir} done")

        self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        self.model.to(self.device)

        self.dynamic_sample_frames=True  # Hard Code 
        self.change_rope = True
        self.max_context_frames = config.max_context_frames
        self.restrict_max_length = 81
        self.multi_caption = getattr(config, "multi_caption", False)  # for multi caption


    def generate(self,
                 batch,
                 guide_scale=5.0,
                 n_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                 offload_model=True,
                 save_dir=""):
        r"""
        """
        data_path = batch['data_path'][0]
        print(f"GENERATE data path is {data_path}")
        data_basename = os.path.basename(data_path)[:-4] 
        video_data = torch.tensor(batch['data'][0])  # [f h w c]
        global_captions = batch['global_captions']
        shots_captions = batch['shots_captions']
        shot_flags = torch.tensor([batch['shot_flag']]).to(torch.int32) 

        shot_flags_unique = torch.unique(shot_flags)
        if self.dynamic_sample_frames:
            shot_number = shot_flags_unique.shape[0] - 1  # except the noise latent 
            base_count = self.max_context_frames // shot_number
            remainder = self.max_context_frames % shot_number
            counts = [base_count] * shot_number
            for i in range(1, remainder + 1):
                counts[-i] += 1
            
        condition_indices = []
        shot_flags_for_rope = []
        for shot_index, shot_flag in enumerate(shot_flags_unique):
            indices = torch.where(shot_flags[0]==shot_flag)
            if shot_flag != max(shot_flags_unique):
                if self.dynamic_sample_frames:  # for dynamic sample frames
                    start_idx = min(indices[0]).item()
                    end_idx = max(indices[0]).item()
                    sampled_steps = torch.linspace(start_idx, end_idx, steps=counts[shot_index])
                    sampled_indices = sampled_steps.round().long().tolist()
                    condition_indices += sampled_indices
                    shot_flags_for_rope += [shot_index] * len(sampled_indices)
                else:
                    condition_indices.append(min(indices[0]).item())
                    condition_indices.append(max(indices[0]).item())
            else:
                latent_indices = indices
        condition_indices = torch.tensor(condition_indices, dtype=torch.int32, device=video_data.device)
        condition_frames = video_data[condition_indices]  # f h w c 
        if self.dynamic_sample_frames:
            assert condition_frames.shape[0] == self.max_context_frames

        video_data = video_data[latent_indices[0]]  # f h w c 
        if self.restrict_max_length is not None:
            if video_data.shape[0] >= self.restrict_max_length:
                stride = video_data.shape[0] // self.restrict_max_length  # downsample for larger motion 
                video_data = video_data[ ::stride]
                video_data = video_data[ :self.restrict_max_length]

        if video_data.shape[0] % 4 != 1:
            video_data = video_data[ :(video_data.shape[0]-1) // 4 * 4+1]

        device, dtype = self.vae.model.encoder.conv1.weight.device, self.vae.model.encoder.conv1.weight.dtype 
        # VAE shape: Input [b, c, f, h, w] -> [b, f, c, h, w]
        condition_frames = rearrange(condition_frames, 'f h w c -> f c 1 h w').to(device).to(dtype)  # encode each frame
        video_data = rearrange(video_data, 'f h w c -> 1 c f h w').to(device).to(dtype)
        with torch.no_grad():
            condition_latents = self.vae.encode_to_latent(condition_frames)  # [f, 1, c, h, w]
            condition_latents = rearrange(condition_latents, 'f 1 c h w -> 1 f c h w')
            video_latents = self.vae.encode_to_latent(video_data)  # [1, f, c, h, w]

        if self.multi_caption:
            caption_s = []
            for i in range(len(shots_captions[0])):
                caption = global_captions[0][0] + shots_captions[0][i][0][0]
                caption_s.append(caption)
        else:
            caption = global_captions[0][0] + shots_captions[0][-1][0][0]

        shot_flags_for_rope += [shot_flags_for_rope[-1]+1] * video_latents.shape[1]

        self.text_encoder.text_encoder.to(self.device)
        with torch.no_grad():
            prompts = caption_s if self.multi_caption else [caption]
            context = self.text_encoder(prompts)
            print(f"prompts is {prompts}")
        # context = self.text_encoder(caption)
        context_null = self.text_encoder(n_prompt)
        if offload_model:
            self.text_encoder.cpu()

        noise = torch.randn_like(video_latents)  # [1, f, c, h, w]
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        print(f"shot_flags_for_rope is {shot_flags_for_rope }")
        shot_flags_for_rope = torch.tensor(shot_flags_for_rope).to(torch.int32).to(device)

        # save_index list
        save_index_list = [0, 13, 25, 37]
        save_dict = {}
        save_dict['data_path'] = data_basename
        save_dict['condition_latents'] = condition_latents.cpu().detach()
        save_dict['noise'] = noise.cpu().detach()
        save_dict['caption'] = prompts
        save_dict['prompt_embeddings'] = context
        save_dict['shot_flags_for_rope'] = shot_flags_for_rope.cpu()

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            sample_scheduler = FlowMatchScheduler(shift=8, sigma_min=0.0, extra_one_step=True)
            sample_scheduler.set_timesteps(50, denoising_strength=1.0, shift=8.0)
            timesteps = sample_scheduler.timesteps
           
            latents = noise
           
            for idx, t in enumerate(tqdm(timesteps)):
                # for save
                if idx in save_index_list:
                    print(f"idx is {idx}, timestep is {t}, saved")
                    save_dict[f"{idx}_input"] = latents.cpu().detach()

                latent_model_input = torch.concat([condition_latents, latents], dim =1)

                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, context, timestep, shot_flags_for_rope=shot_flags_for_rope, frameconcat_infer=True)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, context_null, timestep,shot_flags_for_rope=shot_flags_for_rope, frameconcat_infer=True)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latents = sample_scheduler.step(
                    noise_pred[::, condition_latents.shape[1]:],
                    timestep,
                    latents)

            x0 = latents

            save_dict[f"pred_x0"] = x0.cpu().detach()

            torch.save(save_dict, f"{save_dir}/{data_basename}.pt")
            print(f"{save_dir}/{data_basename}.pt is saved")
            if offload_model:
                self.model.cpu()
            device, dtype = self.vae.model.encoder.conv1.weight.device, self.vae.model.encoder.conv1.weight.dtype 
            videos = self.vae.decode_to_pixel(x0.to(device).to(dtype))  # [b f c h w]

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0], video_data[0], save_dict

def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
    args = _parse_args()

    # Rank
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
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
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Dataset
    dataset = MultiShots_FrameConcat_Dataset(csv_path=args.data_csv_path)
    num_prompts = len(dataset)
    print(f"Number of prompts: {num_prompts}")
    
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    # Model
    logging.info("Creating WanT2V pipeline.")
    wan_t2v = WanT2V(
        config=args,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,                                                                                                                                                                                                                              
        rank=rank,
        t5_cpu=False,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        data_path = batch['data_path'][0]
        data_basename = os.path.basename(data_path)[:-4] 
        
        print(f"debug data path is {data_path}")

        video, video_data, save_dict = wan_t2v.generate(
            batch=batch,
            guide_scale=args.sample_guide_scale,
            offload_model=args.offload_model,
            save_dir=args.save_dir)

        print(f"Rank {rank}: Saved index {i} done")
