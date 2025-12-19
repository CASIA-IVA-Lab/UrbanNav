import os
import yaml
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from statistics import mean
import clip

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from utils.urbannav_dataset import custom_collate
from utils.urbannav_trainer import test_epoch
from utils.setup_utils import setup_dataset, setup_model


def test(args):

    # Load config
    ckpt_file = args.checkpoint
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"Checkpoint file {ckpt_file} does not exist.")
    run_dir = str(Path(ckpt_file).parent.parent)
    with open(os.path.join(run_dir, 'train_config.json'), "r") as f:
        config = json.load(f)

    config["mode"] = "test"
    config["batch_size"] = args.batch_size
    device = f'cuda:{args.gpu}'
    
    # Set random seed
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    # Set train and test datasets
    _, _, seen_dataset, unseen_dataset = setup_dataset(config)

    # Build dataloader
    batch_size = config["batch_size"]
    seen_loader = DataLoader(
        seen_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
        collate_fn=custom_collate
    )
    unseen_loader = DataLoader(
        unseen_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
        collate_fn=custom_collate
    )
        
    # Load model
    model = setup_model(
        model_type=config["model"]["feature_fusion"],
        checkpoint=ckpt_file,
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

    # Test Seen
    seen_metric = test_epoch(
        mode="seen",
        config=config,
        model=model,
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        dataloader=seen_loader,
        device=device
    )

    # Test Unseen
    unseen_metric = test_epoch(
        mode="unseen",
        config=config,
        model=model,
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        dataloader=unseen_loader,
        device=device
    )

    metric = {}
    metric["test_seen"] = seen_metric
    metric["test_unseen"] = unseen_metric

    res_path = os.path.join(run_dir, "result.json")
    with open(res_path, "w", encoding="utf-8") as json_file:
        json.dump(metric, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {res_path}")


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="UrbanNav")
    parser.add_argument(
        "--checkpoint", "-c", 
        type=str, 
        required=True, 
        help="Path to the checkpoint file for loading the trained model."
    )
    parser.add_argument(
        "--gpu", "-g", 
        type=int, 
        help="GPU device id to use for testing (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--batch_size", "-b", 
        type=int, 
        default=64, 
        help="Batch size to use during testing."
    )
    args = parser.parse_args()
        
    # Test
    test(args)

    