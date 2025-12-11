import os
import glob
import yaml
import json
import argparse
import numpy as np
from datetime import datetime
from statistics import mean
import wandb

import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.logger import TensorboardLogger
from utils.train_utils import train_epoch, val_epoch
from utils.setup_utils import setup_DDP, setup_optimizer, setup_dataset, setup_model


def train(
        rank: int,
        world_size: int,
        config: dict,
        train_dataset: Dataset,
        val_dataset: Dataset,
        result_folder: str
):
    # Set up DDP
    setup_DDP(rank, world_size, config["master_name"])

    # Init logger
    if rank == 0:
        logger = TensorboardLogger(config["mode"], result_folder, world_size, config["iter_log_freq"])
    else:
        logger = None

    # Set random seed
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    # Build dataloader
    batch_size = config["batch_size"]
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        drop_last=True,
        persistent_workers=True,
        sampler=train_sampler
    )
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        drop_last=True,
        persistent_workers=True,
        sampler=val_sampler
    )

    # Set up model
    model = setup_model(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer(config=config, model=ddp_model)

    # Start training
    epochs = config["epochs"]
    best_val_loss = 10000
    for epoch in range(epochs):
        train_epoch_losses = train_epoch(
            config=config,
            ddp_model=ddp_model,
            optimizer=optimizer,
            dataloader=train_loader,
            device=rank,
            epoch=epoch
        )
        if (epoch + 1) % config["eval_freq"] == 0:
            val_epoch_losses = val_epoch(
                config=config,
                model=model,
                dataloader=val_loader,
                device=rank,
                epoch=epoch,
            )

        scheduler.step()
        
        if rank == 0:
            # Logging
            logger.update("train", epoch, train_epoch_losses)
            logger.update("val", epoch, val_epoch_losses)

            # Save model
            val_loss = mean(val_epoch_losses["loss"])
            last_path = os.path.join(result_folder, 'checkpoints', f'last.pth')
            torch.save(model.state_dict(), last_path)

            if val_loss <= best_val_loss:
                
                for file in glob.glob(os.path.join(result_folder, 'checkpoints', f'{config["mode"]}_best*')):
                    os.remove(file)
                best_path = os.path.join(result_folder, 'checkpoints', f'{config["mode"]}_best_{epoch+1}.pth')
                torch.save(model.state_dict(), best_path)
                best_val_loss = val_loss
                
            if (epoch+1) % config["save_model_freq"] == 0:
                model_path = os.path.join(result_folder, 'checkpoints', f'{config["mode"]}_{epoch+1}.pth')
                torch.save(model.state_dict(), model_path)

    dist.destroy_process_group()


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="UrbanNav")
    parser.add_argument("--config", "-c", type=str, help="Path to the config file in configs folder",)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set random seed
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    # Set up result folder
    if config["mode"] == 'train':
        time_str = datetime.now().strftime("%m-%d-%H-%M")
        result_folder = os.path.join('results', config["project_name"], config["run_name"]+'-'+time_str)
        os.makedirs(result_folder, exist_ok=False)
        os.makedirs(os.path.join(result_folder, 'checkpoints'), exist_ok=False)
        with open(os.path.join(result_folder, config["mode"]+'_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    elif config["mode"] == 'finetune':
        result_folder = config["project_folder"]
        with open(os.path.join(result_folder, config["mode"]+'_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    # Set cuda
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config["gpu_ids"]])
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    # Set train and test datasets
    train_dataset, val_dataset, _, _ = setup_dataset(config)

    # Train
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config, train_dataset, val_dataset, result_folder), nprocs=world_size, join=True)
    
    