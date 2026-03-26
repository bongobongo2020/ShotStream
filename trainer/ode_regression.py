# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import gc
import logging
# from model.flow_matching import Flow_Matching
from model.ode_regression import Ode_Regression

from utils.dataset import cycle, ODE_Sample_Dataset
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


import datetime
from collections import defaultdict

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
        self.model = Ode_Regression(config, device=self.device)

        if self.config.resume_ckpt is not None:
            print(f"resume ckpt from {self.config.resume_ckpt}")
            ckpt = torch.load(self.config.resume_ckpt, map_location='cpu', mmap=True)['generator']
            self.model.generator.load_state_dict(ckpt, strict=False)
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
        dataset = ODE_Sample_Dataset(config.data_path)
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
        # add for window and frist frame sink
        self.local_attn_size = getattr(config, "local_attn_size", None)
        self.sink_size = getattr(config, "sink_size", None)
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
        device, dtype = self.model.vae.model.encoder.conv1.weight.device, self.model.vae.model.encoder.conv1.weight.dtype 
        shot_flags_for_rope  = batch['shot_flags_for_rope'][0] if 'shot_flags_for_rope' in batch.keys() else None
        condition_latents = batch['condition_latents'][0].to(device).to(dtype)  # [b, f, c, h, w]
        latents_all = batch['latent_all'][0].to(device).to(dtype)  # [b, num_sample, f, c, h, w]
        
        if condition_latents.shape[1] + latents_all.shape[2] > 28:  # restrict for training
            latents_all = latents_all[::, ::, :28-condition_latents.shape[1]]

        if latents_all.shape[2] % self.model.num_frame_per_block != 0:
            latents_all = latents_all[::, ::, :(latents_all.shape[2]//self.model.num_frame_per_block)*self.model.num_frame_per_block]
        shot_flags_for_rope = shot_flags_for_rope[:condition_latents.shape[1] + latents_all.shape[2]]
        caption = batch['caption'][0]
       
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=caption)
            import random
            if random.random() < 0.1:
                conditional_dict["prompt_embeds"] = torch.zeros_like(conditional_dict["prompt_embeds"])

        # Step 3: Store gradients for the generator (if training the generator)
        generator_loss, log_dict = self.model.ode_regression_loss(
            latent = latents_all,
            condition_latent = condition_latents,
            conditional_dict=conditional_dict,
            unconditional_dict=None,
            clean_latent=None,
            initial_latent=None,
            shot_flags_for_rope = shot_flags_for_rope,
            
        )

        # Scale loss for gradient accumulation and backward
        scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
        scaled_generator_loss.backward()
        if LOG_GPU_MEMORY:
            log_gpu_memory("After train_generator backward pass", device=self.device, rank=dist.get_rank())
        generator_log_dict = {}
        generator_log_dict.update({"generator_loss": generator_loss,
                                    "generator_grad_norm": torch.tensor(0.0, device=self.device),
                                    "unnormalized_loss": log_dict['unnormalized_loss'],
                                    "timestep": log_dict["timestep"]})  # Will be computed after accumulation
        return generator_log_dict

    def train(self):
        start_step = self.step
        skipped_steps_counter = 0
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
            # print(f"Step {self.step} done.")
            # add for save loss in different timestep
            timestep = generator_log_dict["timestep"]
            loss_breakdown = defaultdict(list)
            buckets = [260, 500, 740, 1000] 
            stats = {}
            import bisect 
            unnormalized_loss = generator_log_dict["unnormalized_loss"]
            for index, t in enumerate(timestep):
                t_val = int(t.item())
                bucket_index = bisect.bisect_left(buckets, t_val)
                bucket_index = min(bucket_index, len(buckets) - 1)
                bucket_label = str(buckets[bucket_index])
                loss_breakdown[bucket_label].append(unnormalized_loss[index].item())

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

                    for bucket_label, loss_list in loss_breakdown.items():
                        if len(loss_list) > 0: # 确保这个桶里有数据，避免除以0
                            avg_loss = sum(loss_list) / len(loss_list)
                            self.writer.add_scalar(f"Loss_Timestep/{bucket_label}", avg_loss, self.step)

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

