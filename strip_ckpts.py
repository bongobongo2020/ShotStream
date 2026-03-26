#!/usr/bin/env python3
"""Strip checkpoint files to only keep the generator weights."""

import torch
from pathlib import Path
import shutil

CKPTS_DIR = Path("ckpts")

def strip_checkpoint(ckpt_path, backup=True):
    """Strip checkpoint to only keep generator weights."""
    print(f"\n{'='*60}")
    print(f"Processing: {ckpt_path.name}")
    print(f"Original size: {ckpt_path.stat().st_size / (1024**3):.2f} GB")
    
    # Create backup if requested
    if backup:
        backup_path = ckpt_path.with_suffix('.pt.bak')
        if not backup_path.exists():
            print(f"Creating backup: {backup_path.name}")
            shutil.copy2(ckpt_path, backup_path)
        else:
            print(f"Backup already exists: {backup_path.name}")
    
    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if isinstance(ckpt, dict):
        print(f"Original keys: {list(ckpt.keys())}")
        
        # Keep only generator
        if 'generator' in ckpt:
            stripped_ckpt = {'generator': ckpt['generator']}
            
            # Save stripped checkpoint
            print("Saving stripped checkpoint...")
            torch.save(stripped_ckpt, ckpt_path)
            
            new_size = ckpt_path.stat().st_size / (1024**3)
            print(f"New size: {new_size:.2f} GB")
            print(f"Size reduction: {((ckpt_path.stat().st_size / (1024**3)) - new_size):.2f} GB")
            print("✓ Successfully stripped checkpoint")
        else:
            print("⚠ No 'generator' key found in checkpoint")
    else:
        print(f"⚠ Checkpoint is not a dict, it's {type(ckpt).__name__}")
    
    print(f"{'='*60}")

def main():
    # Find all .pt files
    pt_files = sorted(CKPTS_DIR.glob("*.pt"))
    
    print(f"Found {len(pt_files)} checkpoint files to process:")
    for f in pt_files:
        print(f"  - {f.name}")
    
    # Strip each checkpoint
    for pt_file in pt_files:
        strip_checkpoint(pt_file, backup=True)
    
    print(f"\n{'='*60}")
    print("✓ All checkpoints have been stripped!")
    print("Original files are backed up with .pt.bak extension")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
