# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
# import logging
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

# from .distributed.fsdp import shard_model
# from .modules.model import WanModel
# from .modules.t5 import T5EncoderModel
# from .modules.vae import WanVAE
# from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
#                                get_sampling_sigmas, retrieve_timesteps)
# from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
# from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.scheduler import FlowMatchScheduler

from einops import rearrange
from utils.dataset import MultiShots_FrameConcat_Dataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        # shard_fn = partial(shard_model, device_id=device_id)
        # self.text_encoder = T5EncoderModel(
        #     text_len=config.text_len,
        #     dtype=config.t5_dtype,
        #     device=torch.device('cpu'),
        #     checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
        #     tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        #     shard_fn=shard_fn if t5_fsdp else None)

        # self.vae_stride = config.vae_stride
        # self.patch_size = config.patch_size
        # self.vae = WanVAE(
        #     vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        #     device=self.device)
        self.text_encoder = WanTextEncoder().to(self.device)
        self.vae = WanVAEWrapper().to(self.device)

        # logging.info(f"Creating WanModel from {checkpoint_dir}")
        # self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = WanDiffusionWrapper(model_name='Wan2.1-T2V-1.3B', is_causal=False).to(self.device)
        self.model.model.eval().requires_grad_(False)

        ckpt_path = "/m2v_intern/luoyawen/Coding/Interleaved_Multishots/LongLive/logs/wan_train_frame_concat_debug_4_node_data_la_e_81/WanT2V_Training_FrameConcat-2025-11-19_11-09-48/checkpoint_model_001300/model.pt"
        ckpt = torch.load(ckpt_path, map_location='cpu')['generator']
        self.model.load_state_dict(ckpt)
        print(f"resume ckpt from {ckpt_path} done")

        dataset = MultiShots_FrameConcat_Dataset(csv_path="/ytech_m2v_hdd/shixiaoyu/dataset/yawen_multishots_internal/csv/008_processed_la_e81.csv")
        num_prompts = len(dataset)
        print(f"Number of prompts: {num_prompts}")
        
        if dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
        else:
            sampler = SequentialSampler(dataset)
        self.dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

        # if use_usp:
        #     from xfuser.core.distributed import \
        #         get_sequence_parallel_world_size

        #     from .distributed.xdit_context_parallel import (usp_attn_forward,
        #                                                     usp_dit_forward)
        #     for block in self.model.blocks:
        #         block.self_attn.forward = types.MethodType(
        #             usp_attn_forward, block.self_attn)
        #     self.model.forward = types.MethodType(usp_dit_forward, self.model)
        #     self.sp_size = get_sequence_parallel_world_size()
        # else:
        self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        # if dit_fsdp:
        #     self.model = shard_fn(self.model)
        # else:
        self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                #  sample_solver='unipc',
                 sample_solver='flow_match',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        # F = frame_num
        # target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
        #                 size[1] // self.vae_stride[1],
        #                 size[0] // self.vae_stride[2])

        # seq_len = math.ceil((target_shape[2] * target_shape[3]) /
        #                     (self.patch_size[1] * self.patch_size[2]) *
        #                     target_shape[1] / self.sp_size) * self.sp_size


        batch = next(iter(self.dataloader))
        data_path = batch['data_path']
        video_data = torch.tensor(batch['data'][0])  # [f h w c]
        global_captions = batch['global_captions']
        shots_captions = batch['shots_captions']
        shot_flags = torch.tensor([batch['shot_flag']]).to(torch.int32) 

        shot_flags_unique = torch.unique(shot_flags)
        condition_indices = []
        for shot_flag in shot_flags_unique:
            indices = torch.where(shot_flags[0]==shot_flag)
            if shot_flag != max(shot_flags_unique):
                condition_indices.append(min(indices[0]).item())
                condition_indices.append(max(indices[0]).item())
            else:
                latent_indices = indices
        condition_indices = torch.tensor(condition_indices, dtype=torch.int32, device=video_data.device)
        condition_frames = video_data[condition_indices]  # f h w c 
        video_data = video_data[latent_indices[0]]

        if video_data.shape[0] % 4 != 1:
            video_data = video_data[ :(video_data.shape[0]-1) // 4 * 4+1]
        if video_data.shape[0] > 81:
            video_data = video_data[ :81]
        device, dtype = self.vae.model.encoder.conv1.weight.device, self.vae.model.encoder.conv1.weight.dtype 
        # VAE shape: Input [b, c, f, h, w] -> [b, f, c, h, w]
        condition_frames = rearrange(condition_frames, 'f h w c -> f c 1 h w').to(device).to(dtype)  # encode each frame
        video_data = rearrange(video_data, 'f h w c -> 1 c f h w').to(device).to(dtype)
        with torch.no_grad():
            condition_latents = self.vae.encode_to_latent(condition_frames)  # [f, 1, c, h, w]
            condition_latents = rearrange(condition_latents, 'f 1 c h w -> 1 f c h w')
            video_latents = self.vae.encode_to_latent(video_data)  # [1, f, c, h, w]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        caption = global_captions[0][0] + shots_captions[0][0][-1][0]
        # seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        # seed_g = torch.Generator(device=self.device)
        # seed_g.manual_seed(seed)

        # if not self.t5_cpu:
        #     self.text_encoder.model.to(self.device)
        #     context = self.text_encoder([input_prompt], self.device)
        #     context_null = self.text_encoder([n_prompt], self.device)
        #     if offload_model:
        #         self.text_encoder.model.cpu()
        # else:
        #     context = self.text_encoder([input_prompt], torch.device('cpu'))
        #     context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        #     context = [t.to(self.device) for t in context]
        #     context_null = [t.to(self.device) for t in context_null]

        # if not self.t5_cpu:
        self.text_encoder.text_encoder.to(self.device)
        # context = self.text_encoder(input_prompt)
        context = self.text_encoder(caption)
        context_null = self.text_encoder(n_prompt)
        if offload_model:
            self.text_encoder.cpu()
        # else:
        #     context = self.text_encoder.text_encoder(input_prompt, torch.device('cpu'))
        #     context_null = self.text_encoder.text_encoder(n_prompt, torch.device('cpu'))
        #     context = [t.to(self.device) for t in context]
        #     context_null = [t.to(self.device) for t in context_null]


        # noise = [
        #     torch.randn(
        #         target_shape[0],
        #         target_shape[1],
        #         target_shape[2],
        #         target_shape[3],
        #         dtype=torch.float32,
        #         device=self.device,
        #         # generator=seed_g
        #         )
        # ]
        noise = torch.randn_like(video_latents)  # [1, f, c, h, w]
        print(f"condition latent shape is {condition_latents.shape}, noise shape is {noise.shape}")
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            # if sample_solver == 'unipc':
            #     sample_scheduler = FlowUniPCMultistepScheduler(
            #         num_train_timesteps=self.num_train_timesteps,
            #         shift=1,
            #         use_dynamic_shifting=False)
            #     sample_scheduler.set_timesteps(
            #         sampling_steps, device=self.device, shift=shift)
            #     timesteps = sample_scheduler.timesteps
            # elif sample_solver == 'dpm++':
            #     sample_scheduler = FlowDPMSolverMultistepScheduler(
            #         num_train_timesteps=self.num_train_timesteps,
            #         shift=1,
            #         use_dynamic_shifting=False)
            #     sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            #     timesteps, _ = retrieve_timesteps(
            #         sample_scheduler,
            #         device=self.device,
            #         sigmas=sampling_sigmas)
            # # elif 
            # sample_solver == "flow_match":
            sample_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
            sample_scheduler.set_timesteps(50, denoising_strength=1.0, shift=5.0)
            timesteps = sample_scheduler.timesteps

            # else:
            #     raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            print(f"DEBUG Ori T2V Timestep is {timesteps}")
            # arg_c = {'context': context, 'seq_len': seq_len}
            # arg_null = {'context': context_null, 'seq_len': seq_len}

            # add for debug
            # arg_c = {"prompt_embeds": context[0].unsqueeze(0), 'seq_len': seq_len}
            # arg_null = {"prompt_embeds": context_null[0].unsqueeze(0), 'seq_len': seq_len}


            # latent_model_input = latents[0].unsqueeze(0)
            # latent_model_input = rearrange(latent_model_input, '1 c f h w -> 1 f c h w')
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = torch.concat([condition_latents, latents], dim =1)

                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, context, timestep)[0]
                    # latent_model_input, arg_c, timestep)[0]
                    # latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, context_null, timestep)[0]
                    # latent_model_input, arg_null, timestep)[0]
                    # latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                print(f"[DEBUG] Before step, noise_pred shape is {noise_pred.shape}, latents shape is {latents[0].shape}")
                # temp_x0 = sample_scheduler.step(
                latents = sample_scheduler.step(
                    # noise_pred.unsqueeze(0),
                    noise_pred[::, condition_latents.shape[1]:],
                    timestep,
                    latents)
                    # return_dict=False,
                    # generator=seed_g)[0]
                # latent_model_input = latents
                # latents = [temp_x0.squeeze(0)]

            x0 = latents
            # x0 = rearrange(x0, 'b f c h w -> b c f h w')
            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                device, dtype = self.vae.model.encoder.conv1.weight.device, self.vae.model.encoder.conv1.weight.dtype 
                videos = self.vae.decode_to_pixel(x0.to(device).to(dtype))

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
