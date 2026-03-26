# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Tuple
from einops import repeat
from model.base import FrameConcatModel

class Flow_Matching(FrameConcatModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
    
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

    def flow_matching_loss(
        self,
        latent: torch.Tensor,
        condition_latent: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        shot_flags_for_rope: torch.Tensor = None  # add for rope when shot is changed
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        batch_size, num_frame = latent.shape[0], latent.shape[1]

        # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
        # timestep = self._get_timestep(
        #     batch_size = batch_size,
        #     num_frame = num_frame,
        #     device = latent.device,
        # )
        # if self.timestep_shift > 1:
        #     timestep = self.timestep_shift * \
        #         (timestep / 1000) / \
        #         (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=conditional_dict['prompt_embeds'].dtype, device=conditional_dict['prompt_embeds'].device)

        noise = torch.randn_like(latent)
        noisy_latent = self.scheduler.add_noise(
            latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep
        ).unflatten(0, (batch_size, num_frame))

        condition_frame = condition_latent.shape[1]
        latent_input = torch.concat([condition_latent, noisy_latent], dim=1).to(conditional_dict['prompt_embeds'].dtype)
        timestep = repeat(timestep, 'b -> b f', f=latent_input.shape[1])

        condition_frame_number = condition_latent.shape[1]

        noise_pred, _ = self.generator(
            noisy_image_or_video=latent_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            # uniform_timestep = False,
            causal_use_condition_mask = self.is_causal,
            condition_frame_number = condition_frame_number,
            shot_flags_for_rope = shot_flags_for_rope,
        )

        training_target = noise - latent
        loss = torch.nn.functional.mse_loss(noise_pred[:, condition_frame:].float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep[:,0])
        
        return loss