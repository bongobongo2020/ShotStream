# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple
import torch

from model.base import FrameConcatCausalModel

from einops import rearrange, repeat
import torch.nn.functional as F


class Ode_Regression(FrameConcatCausalModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
        self.device = device
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.local_attn_size = getattr(args, "local_attn_size", None)
        self.sink_size = getattr(args, "sink_size", None)

    def _process_timestep(self, timestep, task="causal_video"):
        """
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.

        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if task == "image":
            assert timestep.shape[1] == 1
            return timestep
        elif task == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif task == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        index = torch.randint(0, len(self.denoising_step_list), [
            batch_size, num_frames], device=self.device, dtype=torch.long)

        index = self._process_timestep(index).to(ode_latent.device)

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width)
        ).squeeze(1)  # [b, f, c, h, w] torch.Size([1, 21, 16, 60, 104])

        timestep = self.denoising_step_list[index].to(ode_latent.device)

        # if self.extra_noise_step > 0:
        #     random_timestep = torch.randint(0, self.extra_noise_step, [
        #                                     batch_size, num_frames], device=self.device, dtype=torch.long)
        #     perturbed_noisy_input = self.scheduler.add_noise(
        #         noisy_input.flatten(0, 1),
        #         torch.randn_like(noisy_input.flatten(0, 1)),
        #         random_timestep.flatten(0, 1)
        #     ).detach().unflatten(0, (batch_size, num_frames)).type_as(noisy_input)

        #     noisy_input[timestep == 0] = perturbed_noisy_input[timestep == 0]
        # print(f"timestep is {timestep}")
        return noisy_input, timestep

    def ode_regression_loss(
        self,
        latent: torch.Tensor,  # [b, num_sample, f, c, h, w]
        condition_latent: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        shot_flags_for_rope:  torch.Tensor = None,
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
        target_latent = latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=latent)

        batch_size, num_frame = latent.shape[0], latent.shape[1]

        latent_input = torch.concat([condition_latent, noisy_input], dim=1)
        condition_frame_number = condition_latent.shape[1]
        condition_timestep = torch.zeros([timestep.shape[0], condition_frame_number]).to(timestep.device).to(timestep.dtype)
        timestep = torch.concat([condition_timestep, timestep], dim=-1)

        noise_pred, pred_image_or_video = self.generator(
            noisy_image_or_video=latent_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            causal_use_condition_mask = True,
            condition_frame_number = condition_frame_number,
            shot_flags_for_rope = shot_flags_for_rope,
            local_attn_size = self.local_attn_size,
            sink_size = self.sink_size,
        )

        # Step 2: Compute the regression loss
        mask = timestep > 0
        loss = F.mse_loss(
            pred_image_or_video[mask], target_latent[mask[::, condition_frame_number:]], reduction="mean")
        
        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video[::, condition_frame_number:], target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach()
        }

        return loss, log_dict