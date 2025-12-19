import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
import torch.distributed as dist

import os
import random
import pickle as pkl
from utils.urbannav_dataset import UrbanNavDataset
from model.urban_mlp import UrbanNavMLP
from model.urban_ca import UrbanNavCrossAttention
from model.urban_film import UrbanNavFiLM

UrbanNav = {
    "mlp": UrbanNavMLP,
    "cross_attention": UrbanNavCrossAttention,
    "film": UrbanNavFiLM
}


def setup_DDP(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def setup_model(
    model_type: int,
    checkpoint: str,
    context_size: int,
    len_traj_pred: int,
    visual_feat_size: int,
    num_freqs: int,
    attn_dim: int,
    num_attn_layers: int,
    num_attn_heads: int,
    ff_dim_factor: int,
    dropout: float,
    film_feature_size: int = None,
    clip_type: str = None,
):
    """
    Set up and initialize the UrbanNav model.

    Args:
        model_type (int): The type of model to use (must be a key in UrbanNav).
        checkpoint (str): Path to the checkpoint file to load model weights from.
        context_size (int): Number of context frames.
        len_traj_pred (int): Length of trajectory to predict.
        visual_feat_size (int): Size of the visual feature vector.
        num_freqs (int): Number of frequencies for positional encoding.
        attn_dim (int): Attention dimension.
        num_attn_layers (int): Number of attention layers.
        num_attn_heads (int): Number of attention heads.
        ff_dim_factor (int): Feedforward dimension factor.
        dropout (float): Dropout rate.
        film_feature_size (int, optional): Feature size for FiLM model. Default is None.
        clip_type (str, optional): Type of CLIP model to use. Default is None.

    Returns:
        nn.Module: The initialized model, optionally loaded with checkpoint weights.

    Raises:
        ValueError: If the model_type is not recognized.
    """
    if model_type in UrbanNav:
        model: nn.Module = UrbanNav[model_type](
            context_size=context_size,
            len_traj_pred=len_traj_pred,
            visual_feat_size=visual_feat_size,
            num_freqs=num_freqs,
            attn_dim=attn_dim,
            num_attn_layers=num_attn_layers,
            num_attn_heads=num_attn_heads,
            ff_dim_factor=ff_dim_factor,
            dropout=dropout,
            film_feature_size=film_feature_size,
            clip_type=clip_type
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Loaded checkpoint from {checkpoint}")
    
    return model


def setup_optimizer(
        model: nn.Module,
        optimizer_type: str = "adamw",
        scheduler_type: str = 'cosine',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        step_size: int = 10,
        gamma: float = 0.1,
        epochs: int = 100
        ):

    if optimizer_type == "adam":
        optimizer = Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
            )
    elif optimizer_type == "adamw":
        optimizer = AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
            )
    elif optimizer_type == "sgd":
        optimizer = SGD(
            model.parameters(), 
            lr=learning_rate
            )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    if scheduler_type == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
            )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs
            )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown optimizer: {scheduler_type}")

    return optimizer, scheduler


# Set up dataset
def setup_dataset(config):

    train_dataset = None
    val_dataset = None
    test_seen_dataset = None
    test_unseen_dataset = None

    if config["mode"] in ['train', 'finetune']:
        with open(config["data"]["train_split"], 'rb') as split_file:
            data_list = pkl.load(split_file)

        random.shuffle(data_list)
        split_idx = int(len(data_list) * config["data"]["split"])
        train_data_list = data_list[:split_idx]
        val_data_list = data_list[split_idx:]
        
        train_dataset = UrbanNavDataset(
            mode="train", 
            data_dir=config["data"]["data_dir"],
            feat_file=config["data"]["feat_file"],
            data_list=train_data_list,
            crop=config["data"]["crop"],
            resize=config["data"]["resize"],
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            video_freq=config["data"]["video_freq"],
            pose_freq=config["data"]["pose_freq"],
            target_freq=config["data"]["target_freq"],
            arrived_prob=config["data"]["arrived_prob"],
            input_noise=config["data"]["input_noise"],
            search_range=[config["data"]["search_lower_bound"], config["data"]["search_upper_bound"]]
            )
        val_dataset = UrbanNavDataset(
            mode="val",
            data_dir=config["data"]["data_dir"],
            feat_file=config["data"]["feat_file"],
            data_list=val_data_list,
            crop=config["data"]["crop"],
            resize=config["data"]["resize"],
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            video_freq=config["data"]["video_freq"],
            pose_freq=config["data"]["pose_freq"],
            target_freq=config["data"]["target_freq"],
            arrived_prob=config["data"]["arrived_prob"],
            input_noise=config["data"]["input_noise"],
            search_range=[config["data"]["search_lower_bound"], config["data"]["search_upper_bound"]]
            )

    elif config["mode"] == 'test':
        with open(config["data"]["test_seen_split"], 'rb') as seen_split_file:
            seen_data_list = pkl.load(seen_split_file)
        test_seen_dataset = UrbanNavDataset(
            mode="test",
            data_dir=config["data"]["data_dir"],
            feat_file=config["data"]["feat_file"],
            data_list=seen_data_list,
            crop=config["data"]["crop"],
            resize=config["data"]["resize"],
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            video_freq=config["data"]["video_freq"],
            pose_freq=config["data"]["pose_freq"],
            target_freq=config["data"]["target_freq"],
            arrived_prob=config["data"]["arrived_prob"],
            input_noise=config["data"]["input_noise"],
            search_range=[config["data"]["search_lower_bound"], config["data"]["search_upper_bound"]]
        )

        with open(config["data"]["test_unseen_split"], 'rb') as unseen_split_file:
            unseen_data_list = pkl.load(unseen_split_file)
        test_unseen_dataset = UrbanNavDataset(
            mode="test",
            data_dir=config["data"]["data_dir"],
            feat_file=config["data"]["feat_file"],
            data_list=unseen_data_list,
            crop=config["data"]["crop"],
            resize=config["data"]["resize"],
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            video_freq=config["data"]["video_freq"],
            pose_freq=config["data"]["pose_freq"],
            target_freq=config["data"]["target_freq"],
            arrived_prob=config["data"]["arrived_prob"],
            input_noise=config["data"]["input_noise"],
            search_range=[config["data"]["search_lower_bound"], config["data"]["search_upper_bound"]]
        )

    return train_dataset, val_dataset, test_seen_dataset, test_unseen_dataset

