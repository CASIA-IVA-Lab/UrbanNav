# -----------------------------------------------------------------------------
# Description: 
#   This script extracts image features using a specified deep learning model 
#   (default: DINOv2 ViT-L/14) and stores them in an LMDB database for efficient 
#   retrieval. It supports distributed feature extraction across multiple GPUs 
#   using PyTorch's distributed and multiprocessing modules.
#
# Usage:
#   python extract_features_cache.py --data-dir <image_dir> --output-dir <lmdb_dir> \
#       --model-name <model_name> --gpus <gpu_ids>
#
# Author: UrbanNav Project Contributors
# -----------------------------------------------------------------------------

import os
import lmdb
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


def build_data_list(data_dir):
    jpg_files = []
    dir_path = Path(data_dir).resolve()

    for file in dir_path.rglob('*.jpg'):
        if file.is_file():
            relative_path = file.relative_to(dir_path)
            jpg_files.append(str(relative_path))

    return jpg_files


class ImagePathDataset(Dataset):
    """
    PyTorch Dataset for loading images from relative paths and applying preprocessing transforms.
    Returns image names and processed image tensors.
    """
    def __init__(self, image_rel_paths, data_dir):

        self.image_rel_paths = image_rel_paths
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize([360, 640]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"All images num: {len(self.image_rel_paths)}")

    def __len__(self):
        return len(self.image_rel_paths)

    def __getitem__(self, idx):

        img_rel_path = self.image_rel_paths[idx]
        img_path = os.path.join(self.data_dir, img_rel_path)
        img_path_parts = img_path.split('/')
        img_name = img_path_parts[-2] + '_' + img_path_parts[-1].split('.')[0]

        image = self.transform(Image.open(img_path).convert("RGB"))
        return img_name, image


@torch.no_grad()
def run(
        rank: int,
        world_size: int,
        dataset: Dataset, 
        output_lmdb: str, 
        batch_size: int = 64, 
        num_workers: int = 12
    ):

    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12368)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        sampler=sampler
        )

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(rank)
    model.eval()

    if rank == 0:
        process_bar = tqdm(dataloader, desc=f"Extract features", ncols=120)
        # Initialize LMDB environment
        env = lmdb.open(output_lmdb, map_size=2**40)  # 1TB map size
        txn = env.begin(write=True)
    else:
        process_bar = dataloader

    batch_idx = 0
    for batch_names, batch_images in process_bar:

        batch_images = TF.center_crop(batch_images, [350, 630])
        batch_images = TF.resize(batch_images, [350, 630])
        batch_images = batch_images.to(rank)

        features = model(batch_images)
        features = features.cpu().numpy()

        # Gather all keys and features to rank 0
        gathered_keys = [None] * world_size
        gathered_features = [None] * world_size

        dist.gather_object(list(batch_names), gathered_keys if rank == 0 else None, dst=0)
        dist.gather_object(features, gathered_features if rank == 0 else None, dst=0)

        if rank == 0:
            # Flatten the gathered data
            all_keys = [key for sublist in gathered_keys for key in sublist]
            all_features = np.concatenate(gathered_features)

            # Write the current batch's features to the LMDB file
            for key, feature in zip(all_keys, all_features):
                txn.put(key.encode('ascii'), pickle.dumps(feature))  # Serialize feature with pickle

            # Commit every 1000 writes to avoid excessive memory usage
            if batch_idx % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)

            # Clear memory
            del gathered_keys, gathered_features, all_keys, all_features

        batch_idx += 1

    if rank == 0:
        # Final commit and close LMDB environment
        txn.commit()
        env.close()
        
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract Features to LMDB")
    parser.add_argument("--data-dir", "-d", type=str, help="Path to the image data directory")
    parser.add_argument("--cache-dir", "-o", type=str, help="Path to the output LMDB directory")
    parser.add_argument("--model-name", "-m", type=str, default='dinov2_vitl14', help="Model name to use")
    parser.add_argument("--gpus", "-g", type=int, nargs='+', default=[0], help="List of GPU ids to use")
    args = parser.parse_args()

    gpu_ids = args.gpus
    world_size = len(gpu_ids)

    os.makedirs(args.cache_dir, exist_ok=True)
    output_file = os.path.join(args.cache_dir, f"urbannav_{args.model_name}_feat.lmdb")
    if os.path.exists(output_file):
        raise FileExistsError(f"'{output_file}' already exists.")
    
    # Set cuda
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if type(gpu_ids) == int:
            gpu_ids = [gpu_ids]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_ids])
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    # Setup dataset
    image_rel_paths = build_data_list(args.data_dir)
    print(f"All images: {len(image_rel_paths)}")
    dataset = ImagePathDataset(image_rel_paths, args.data_dir)

    mp.spawn(run, args=(world_size, dataset, output_file, 64, 12), nprocs=world_size, join=True)
    print(f"Features extracted and saved to {output_file}")

