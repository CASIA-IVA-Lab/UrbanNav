import os
import glob
import yaml
import json
import logging
import argparse
import numpy as np
from datetime import datetime
from statistics import mean
import wandb
import clip

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.urbannav_dataset import custom_collate
from utils.distributed import init_distributed
from utils.urbannav_trainer import UrbanNavTrainer
from utils.setup_utils import setup_optimizer, setup_dataset, setup_model


WANDB_API_KEY = "<YOUR_WANDB_KEY_API>"
WANDB_ENTITY = "<YOUR_WANDB_ENTITY>"


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, stream=False):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        if stream:
            handlers = [logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        else:
            handlers = [logging.FileHandler(f"{logging_dir}/log.txt")]
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def main(args):
    """
    Training UrbanNav from scratch.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup DDP:
    _, rank, gpu, _ = init_distributed()
    device = f'cuda:{gpu}'
    seed = config.get('seed', 0) * dist.get_world_size() + rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Set up result folder
    if rank == 0:
        if config["mode"] == 'train':
            time_str = datetime.now().strftime("%m-%d-%H-%M")
            result_folder = os.path.join('logs', config["project_name"], config["run_name"]+'-'+time_str)
            os.makedirs(result_folder, exist_ok=True)
            os.makedirs(os.path.join(result_folder, 'checkpoints'), exist_ok=True)
            with open(os.path.join(result_folder, config["mode"]+'_config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logger = create_logger(result_folder, stream=True)
        elif config["mode"] == 'finetune':
            result_folder = config["project_folder"]
            with open(os.path.join(result_folder, config["mode"]+'_config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
    else:
        logger = create_logger(None)

    # Load dataset
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    train_dataset, val_dataset, _, _ = setup_dataset(config)
    if rank == 0:
        logger.info(f"Train dataset samples: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=dist.get_world_size(), 
        rank=rank,
        shuffle=True,
        seed=config.get("seed", 0)
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        sampler=train_sampler,
        collate_fn=custom_collate
    )
    
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=dist.get_world_size(), 
        rank=rank,
        shuffle=True,
        seed=config.get("seed", 0)
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        sampler=val_sampler,
        collate_fn=custom_collate
    )

    # Set up model
    model = setup_model(
        model_type=config["model"]["feature_fusion"],
        checkpoint=None,
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        visual_feat_size=config["model"]["visual_feat_size"],
        num_freqs=config["model"]["num_freqs"],
        attn_dim=config["model"]["attn_dim"],
        num_attn_layers=config["model"]["num_attn_layers"],
        num_attn_heads=config["model"]["num_attn_heads"],
        ff_dim_factor=config["model"]["ff_dim_factor"],
        dropout=config["model"]["dropout"],
        film_feature_size=config["model"]['film_feat_size'],
        clip_type=config["model"]['clip_type']
    ).to(device)
    vision_encoder = torch.hub.load(
        'facebookresearch/dinov2', 
        config['model']['visual_encoder'],
        ).to(torch.float32).to(device)
    text_encoder, _ = clip.load(config['model']['clip_type'])
    text_encoder = text_encoder.to(torch.float32).to(device)
    

    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer(
        model=model,
        optimizer_type=config["train"]["optimizer"],
        scheduler_type=config["train"]["scheduler"],
        learning_rate=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
        step_size=int(config["train"]["step_size"]),
        gamma=float(config["train"]["gamma"]),
        epochs=int(config["epochs"])
        )
    
    # Set up trainer
    trainer = UrbanNavTrainer(
        model=model,
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        batch_size=batch_size,
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        direction_loss_weight=config["train"]["direction_loss_weight"],
        feature_loss_weight=config["train"]["feature_loss_weight"],
        arrived_loss_weight=config["train"]["arrived_loss_weight"],
        log_freq=config.get("log_freq", 100),
        use_wandb=config.get("use_wandb", False)
    )

    # Prepare model for DDP training
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    model.train()
    vision_encoder.eval()
    text_encoder.eval()

    if rank == 0 and config.get("use_wandb", False):
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=config["project_name"],
            entity=WANDB_ENTITY,
            name=config["run_name"]+ '-' + time_str,
            config=config
        )
        logger.info(f"Starting training for {config['epochs']} epochs...")

    # Start training
    epochs = config["epochs"]
    for epoch in range(epochs):
        train_epoch_losses = trainer.train_epoch(epoch)
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} Training Loss: " +
                        f"Total: {train_epoch_losses['loss']:.4f}, " +
                        f"Waypoint: {train_epoch_losses['waypoint_loss']:.4f}, " +
                        f"Direction: {train_epoch_losses['direction_loss']:.4f}, " +
                        f"Feature: {train_epoch_losses['feature_loss']:.4f}, " +
                        f"Arrived: {train_epoch_losses['arrived_loss']:.4f}"
            )

        # Evaluation
        if (epoch) % config["eval_freq"] == 0:
            val_epoch_losses = trainer.val_epoch()
            if rank == 0:
                logger.info(f"Epoch {epoch}/{epochs} Validation Loss: " +
                            f"Total: {val_epoch_losses['loss']:.4f}, " +
                            f"Waypoint: {val_epoch_losses['waypoint_loss']:.4f}, " +
                            f"Direction: {val_epoch_losses['direction_loss']:.4f}, " +
                            f"Feature: {val_epoch_losses['feature_loss']:.4f}, " +
                            f"Arrived: {val_epoch_losses['arrived_loss']:.4f}"
                            )

        # Save model 
        if rank == 0:
            last_path = os.path.join(result_folder, 'checkpoints', f'last.pth')
            torch.save(model.module.state_dict(), last_path)
            if epoch % config["save_model_freq"] == 0 and epoch != 0:
                model_path = os.path.join(result_folder, 'checkpoints', f'{str(epoch).zfill(4)}.pth')
                torch.save(model.module.state_dict(), model_path)

    model.eval()
    cleanup()


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="UrbanNav")
    parser.add_argument("--config", "-c", type=str, help="Path to the config file in configs folder",)
    args = parser.parse_args()

    main(args)
    
    