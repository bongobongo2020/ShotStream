"""
Merge LoRA checkpoint with base model checkpoint into a single checkpoint.

Usage:
    python merge_lora_checkpoint.py \
        --config_path <path_to_config.yaml> \
        --base_ckpt <path_to_base_checkpoint.pth> \
        --lora_ckpt <path_to_lora_checkpoint.pth> \
        --output_ckpt <path_to_output_merged_checkpoint.pth> \
        [--use_ema]
"""

import argparse
import torch
from omegaconf import OmegaConf
import peft
from pipeline import CausalInferencePipeline
import os


def merge_lora_to_base(config_path, base_ckpt_path, lora_ckpt_path, output_path, use_ema=False):
    """
    Merge LoRA weights into base model and save as a single checkpoint.
    
    Args:
        config_path: Path to config YAML file
        base_ckpt_path: Path to base model checkpoint
        lora_ckpt_path: Path to LoRA checkpoint
        output_path: Path to save merged checkpoint
        use_ema: Whether to use EMA weights from base checkpoint
    """
    print(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    
    # Initialize pipeline on CPU to save memory
    print("Initializing pipeline...")
    device = "cpu"
    pipeline = CausalInferencePipeline(config, device=device)
    
    # Load base model checkpoint
    print(f"Loading base checkpoint from: {base_ckpt_path}")
    base_state_dict = torch.load(base_ckpt_path, map_location="cpu", mmap=True)
    model_key = 'generator_ema' if use_ema else 'generator'
    pipeline.generator.load_state_dict(base_state_dict[model_key])
    print(f"Loaded base model (using {model_key})")
    
    # Configure LoRA for the model
    print("Configuring LoRA...")
    pipeline.generator.model = pipeline._configure_lora_for_model(
        pipeline.generator.model, 
        "generator"
    )
    
    # Load LoRA checkpoint
    print(f"Loading LoRA checkpoint from: {lora_ckpt_path}")
    lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu", mmap=True)
    peft.set_peft_model_state_dict(
        pipeline.generator.model, 
        lora_checkpoint["generator_lora"]
    )
    print("LoRA weights loaded successfully")
    
    # Merge LoRA weights into base model
    print("Merging LoRA weights into base model...")
    pipeline.generator.model = pipeline.generator.model.merge_and_unload()
    print("Merge completed!")
    
    # Get the merged state dict
    merged_state_dict = pipeline.generator.state_dict()
    
    # Create output checkpoint with same structure as base
    print(f"Saving merged checkpoint to: {output_path}")
    output_checkpoint = {
        'generator': merged_state_dict,
    }
    
    # If base had EMA, save merged weights as both generator and generator_ema
    if use_ema or 'generator_ema' in base_state_dict:
        output_checkpoint['generator_ema'] = merged_state_dict
        print("Saved as both 'generator' and 'generator_ema'")
    
    # Copy other keys from base checkpoint if they exist
    for key in base_state_dict.keys():
        if key not in ['generator', 'generator_ema']:
            output_checkpoint[key] = base_state_dict[key]
            print(f"Copied '{key}' from base checkpoint")
    
    # Save merged checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_checkpoint, output_path)
    print(f"✓ Merged checkpoint saved successfully to: {output_path}")
    print(f"✓ Checkpoint size: {os.path.getsize(output_path) / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA checkpoint with base model checkpoint"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--base_ckpt", 
        type=str, 
        required=True,
        help="Path to base model checkpoint (.pth or .pt)"
    )
    parser.add_argument(
        "--lora_ckpt", 
        type=str, 
        required=True,
        help="Path to LoRA checkpoint (.pth or .pt)"
    )
    parser.add_argument(
        "--output_ckpt", 
        type=str, 
        required=True,
        help="Path to save merged checkpoint"
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true",
        help="Use EMA weights from base checkpoint"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    if not os.path.exists(args.base_ckpt):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_ckpt}")
    if not os.path.exists(args.lora_ckpt):
        raise FileNotFoundError(f"LoRA checkpoint not found: {args.lora_ckpt}")
    
    # Perform merge
    merge_lora_to_base(
        config_path=args.config_path,
        base_ckpt_path=args.base_ckpt,
        lora_ckpt_path=args.lora_ckpt,
        output_path=args.output_ckpt,
        use_ema=args.use_ema
    )


if __name__ == "__main__":
    main()
