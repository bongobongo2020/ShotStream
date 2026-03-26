# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from omegaconf import OmegaConf
import wandb
from trainer import ScoreDistillationTrainer, FrameConcatTrainer, ODERegressionTrainer, ScoreDistillationFrameConcatTrainer, StreamingScoreDistillationFrameConcatTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/wan_train_frame_concat.yaml")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--no-one-logger", action="store_true", help="Disable One Logger (enabled by default)")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("tools/train/config/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.dirname(args.config_path).split("/")[-1]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_tensorboard = args.disable_tensorboard
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume  # Default to True unless --no-auto-resume is specified
    config.use_one_logger = not args.no_one_logger

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "wan_frame_concat":
        trainer = FrameConcatTrainer(config)
    elif config.trainer == "ode_regression":
        trainer = ODERegressionTrainer(config)
    elif config.trainer == "score_distillation_frameconcat":
        trainer = ScoreDistillationFrameConcatTrainer(config)
    elif config.trainer == "score_distillation_frameconcat_stream":
        trainer = StreamingScoreDistillationFrameConcatTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
