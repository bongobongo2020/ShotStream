# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import gc
import logging

from model.flow_matching import Flow_Matching

from utils.dataset import cycle, MultiShots_FrameConcat_Dataset
from utils.distributed import fsdp_wrap, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist

import torch
from torch.utils.tensorboard import SummaryWriter # MODIFIED: Added TensorBoard
import time
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)

from utils.memory import log_gpu_memory
from utils.debug_option import LOG_GPU_MEMORY
import time

from einops import rearrange

import datetime

class Trainer:
    
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_tensorboard = config.disable_tensorboard 

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
            print(f"set the random_seed as {random_seed}")

        set_seed(config.seed + global_rank)
        print(f"set the seed as {config.seed} + global_rank")

        self.use_one_logger = getattr(config, "use_one_logger", False)
        self.max_context_frames = getattr(config, "max_context_frames", 6)  # for dynamic sample frames
        self.dynamic_sample_frames = getattr(config, "dynamic_sample_frames", False)
        self.change_rope = getattr(config, "change_rope", False)  # for change rope
        self.multi_caption = getattr(config, "multi_caption", False)  # for multi caption
        self.only_sample_first_frame = getattr(config, "only_sample_first_frame", False)  # for multi caption ablation
        
        self.output_path = config.logdir
        if self.is_main_process:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.output_path = os.path.join(self.output_path, f"{config.log_name}-{current_time}")
            os.makedirs(self.output_path, exist_ok=True)

        # MODIFIED: Replaced wandb setup with TensorBoard SummaryWriter
        self.writer = None
        if self.is_main_process and not self.disable_tensorboard:
            print(f"Initializing TensorBoard SummaryWriter in {self.output_path}")
            self.writer = SummaryWriter(log_dir=self.output_path)
            
        # Step 2: Initialize the model
        self.model = Flow_Matching(config, device=self.device)

        if self.config.resume_ckpt is not None:
            print(f"resume ckpt from {self.config.resume_ckpt}")
            ckpt = torch.load(self.config.resume_ckpt, map_location='cpu', mmap=True)['generator']
            self.model.generator.load_state_dict(ckpt)
            print(f"resume ckpt from {self.config.resume_ckpt} done")

        self.model.generator = fsdp_wrap(  
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
        
        # Step 3: Set Trainable Parameters
        print("Freezing all model parameters...")
        self.model.requires_grad_(False)
        print("Unfreezing generator parameters...")
        # self.model.generator.requires_grad_(True)
        if config.only_train3d is True:
            for name, module in self.model.generator.named_modules():
                if any(keyword in name for keyword in ["self_attn"]):
                    print(f"Trainable: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            self.model.generator.requires_grad_(True)
        
        print("\nVerifying parameter grad status:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  - [TRAINABLE] {name}")
        

        # Step 4: Set Optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 5: Set Dataset
        dataset = MultiShots_FrameConcat_Dataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=dataset.custom_collate_fn,
            num_workers=0)
        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: Set default param for gradient clip
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        with FSDP.state_dict_type(
            self.model.generator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),          # newly added
        ):
            generator_state_dict  = self.model.generator.state_dict()
            generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                            self.generator_optimizer)

        state_dict = {
            "generator": generator_state_dict,
            "generator_optimizer": generator_opim_state_dict,
            "step": self.step,
        }

        if self.is_main_process:  # Only save in main process
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print("Model saved to", checkpoint_file)
            
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
        
    def fwdbwd_one_step(self, batch, train_generator):
        if self.step % 50 == 0:
            torch.cuda.empty_cache()
        # Step 1: Prepare data
        data_path = batch['data_path']
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
        if self.only_sample_first_frame:
            counts = [1] * (shot_number - 1) + [self.max_context_frames - shot_number + 1]
            
        condition_indices = []
        shot_flags_for_rope = []
        for shot_index, shot_flag in enumerate(shot_flags_unique):
            indices = torch.where(shot_flags[0][0]==shot_flag)
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
        if self.config.restrict_max_length is not None:
            if video_data.shape[0] >= self.config.restrict_max_length:
                stride = video_data.shape[0] // self.config.restrict_max_length  # downsample for larger motion 
                video_data = video_data[ ::stride]
                video_data = video_data[ :self.config.restrict_max_length]

        if video_data.shape[0] % 4 != 1:
            video_data = video_data[ :(video_data.shape[0]-1) // 4 * 4+1]

        device, dtype = self.model.vae.model.encoder.conv1.weight.device, self.model.vae.model.encoder.conv1.weight.dtype 

        # VAE shape: Input [b, c, f, h, w] -> [b, f, c, h, w]
        condition_frames = rearrange(condition_frames, 'f h w c -> f c 1 h w').to(device).to(dtype)  # encode each frame
        video_data = rearrange(video_data, 'f h w c -> 1 c f h w').to(device).to(dtype)

        with torch.no_grad():
            condition_latents = self.model.vae.encode_to_latent(condition_frames)  # [f, 1, c, h, w]
            import random
            if random.random() < 0.1:
                condition_latents = torch.zeros_like(condition_latents)
                shot_flags_for_rope = [0] * len(shot_flags_for_rope)  # [Note] Also set flag as zero
            condition_latents = rearrange(condition_latents, 'f 1 c h w -> 1 f c h w')
            video_latents = self.model.vae.encode_to_latent(video_data)  # [1, f, c, h, w]

        if self.multi_caption:
            caption_s = []
            for i in range(len(shots_captions[0][0])):
                caption = global_captions[0][0] + shots_captions[0][0][i][0]
                caption_s.append(caption)
        else:
            caption = global_captions[0][0] + shots_captions[0][0][-1][0]

        shot_flags_for_rope += [shot_flags_for_rope[-1]+1] * video_latents.shape[1]
        # print(f"[DEBUG] shot flags for rope is {shot_flags_for_rope}")
        
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            prompts = caption_s if self.multi_caption else [caption]
            conditional_dict = self.model.text_encoder(
                text_prompts=prompts)
            import random
            if random.random() < 0.1:
                conditional_dict["prompt_embeds"] = torch.zeros_like(conditional_dict["prompt_embeds"])  # 对于prompt这里目前处理比较简单 就是全部置0

        # Step 3: Store gradients for the generator (if training the generator)
        shot_flags_for_rope = torch.tensor(shot_flags_for_rope).to(torch.int32).to(device)
        generator_loss = self.model.flow_matching_loss(
            latent = video_latents,
            condition_latent = condition_latents,
            conditional_dict = conditional_dict,
            unconditional_dict=None,
            clean_latent=None,
            initial_latent=None,
            shot_flags_for_rope = shot_flags_for_rope if self.change_rope else None,
        )

        # Scale loss for gradient accumulation and backward
        scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
        scaled_generator_loss.backward()
        if LOG_GPU_MEMORY:
            log_gpu_memory("After train_generator backward pass", device=self.device, rank=dist.get_rank())
        generator_log_dict = {}
        generator_log_dict.update({"generator_loss": generator_loss,
                                    "generator_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation
        return generator_log_dict

    def train(self):
        start_step = self.step
        skipped_steps_counter = 0
        # try:
        while True:
            error_detected_this_step = torch.tensor(0.0, device=self.device)
            self.generator_optimizer.zero_grad(set_to_none=True)
            
            accumulated_generator_logs = []
            
            for accumulation_step in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                extra_gen = self.fwdbwd_one_step(batch, True)
                accumulated_generator_logs.append(extra_gen)

            # Compute grad norm and update parameters
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
            
            dist.all_reduce(error_detected_this_step, op=dist.ReduceOp.SUM)

            if error_detected_this_step.item() > 0:
                skipped_steps_counter += 1
                if self.is_main_process:
                    print(
                        f"At least one rank failed at step {self.step}. "
                        f"Synchronously skipping optimizer step. "
                        f"Total skipped steps so far: {skipped_steps_counter}."
                    )
                continue

            generator_log_dict = merge_dict_list(accumulated_generator_logs)
            generator_log_dict["generator_grad_norm"] = generator_grad_norm
            
            self.generator_optimizer.step()
            self.step += 1

            # Save the model
            if self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process and self.writer:
                if generator_log_dict:
                    self.writer.add_scalar("Loss/generator", generator_log_dict["generator_loss"].mean().item(), self.step)
                    self.writer.add_scalar("GradNorm/generator", generator_log_dict["generator_grad_norm"].mean().item(), self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                # MODIFIED: Log iteration time to TensorBoard
                if self.writer:
                    self.writer.add_scalar("Time/per_iteration", iteration_time, self.step)
                self.previous_time = current_time
                current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                print(f"Time [{current_time}], Step {self.step}, Loss is {generator_log_dict['generator_loss'].mean().item():.3f}, Grad Norm is {generator_log_dict['generator_grad_norm'].mean().item():.3f}, Time is {iteration_time:.3f}")   
                
            if self.step > self.config.max_iters:
                break

