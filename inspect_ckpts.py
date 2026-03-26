#!/usr/bin/env python3
"""Inspect checkpoint files and print their keys."""

import torch
from pathlib import Path

CKPTS_DIR = Path("ckpts")

def inspect_checkpoint(ckpt_path):
    """Load checkpoint and print keys."""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Size: {ckpt_path.stat().st_size / (1024**3):.2f} GB")
    print(f"{'='*60}")
    
    try:
        # Load checkpoint (map to CPU to avoid GPU memory issues)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if isinstance(ckpt, dict):
            print(f"\nType: dict")
            print(f"Number of keys: {len(ckpt)}")
            print(f"\nTop-level keys:")
            for key in ckpt.keys():
                value = ckpt[key]
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: Tensor {tuple(value.shape)}, dtype={value.dtype}")
                elif isinstance(value, dict):
                    print(f"  - {key}: dict with {len(value)} keys")
                elif isinstance(value, list):
                    print(f"  - {key}: list with {len(value)} items")
                else:
                    print(f"  - {key}: {type(value).__name__}")
            
            # Print nested keys for dict values
            for key, value in ckpt.items():
                if isinstance(value, dict):
                    print(f"\n  Keys in '{key}':")
                    for nested_key in list(value.keys())[:10]:  # Show first 10
                        nested_value = value[nested_key]
                        if isinstance(nested_value, torch.Tensor):
                            print(f"    - {nested_key}: Tensor {tuple(nested_value.shape)}")
                        else:
                            print(f"    - {nested_key}: {type(nested_value).__name__}")
                    if len(value) > 10:
                        print(f"    ... and {len(value) - 10} more keys")
        else:
            print(f"\nType: {type(ckpt).__name__}")
            if isinstance(ckpt, torch.Tensor):
                print(f"Shape: {tuple(ckpt.shape)}")
                print(f"Dtype: {ckpt.dtype}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

def main():
    # Find all .pt files
    pt_files = sorted(CKPTS_DIR.glob("*.pt"))
    
    print(f"Found {len(pt_files)} checkpoint files:")
    for f in pt_files:
        print(f"  - {f.name}")
    
    # Inspect each checkpoint
    for pt_file in pt_files:
        inspect_checkpoint(pt_file)

if __name__ == "__main__":
    main()
