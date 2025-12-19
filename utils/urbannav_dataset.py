import os
import random
import numpy as np
import json
import lmdb
import pickle as pkl
from PIL import Image
from tqdm import tqdm
from typing import List

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as TF


def process_frames(frames, target_size):

        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0  # Corrected normalization
    
        desired_height = target_size[0]
        desired_width = target_size[1]
        _, _, height, width = frames.shape

        pad_height = desired_height - height
        pad_width = desired_width - width
        
        # Only pad if necessary
        if pad_height > 0 or pad_width > 0:
            # Calculate padding for each side (left, right, top, bottom)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            # Apply padding
            frames = TF.pad(
                frames, 
                (pad_left, pad_top, pad_right, pad_bottom),
            )
        elif pad_height < 0  or pad_width < 0:
            frames = TF.center_crop(frames, (desired_height, desired_width))
        
        return frames


def pose_to_matrix(pose):

    position = pose[:3]
    rotation = Rotation.from_quat(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position

    return matrix


def poses_to_matrices(poses):

    positions = poses[:, :3]
    quats = poses[:, 3:]
    rotations = Rotation.from_quat(quats)
    matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    matrices[:, :3, :3] = rotations.as_matrix()
    matrices[:, :3, 3] = positions

    return matrices


def latlon_to_local(lat, lon, lat0, lon0):
    R_earth = 6378137  # Earth's radius in meters
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    x = dlon * np.cos((lat_rad + lat0_rad) / 2) * R_earth
    y = dlat * R_earth
    return x, y


class UrbanNavDataset(Dataset):
    """
    UrbanNav main PyTorch Dataset class for loading and processing trajectory data.
    """
    def __init__(self, 
                 mode: str, 
                 data_dir: str,
                 feat_file: str,
                 data_list: List[str],
                 context_size: int,
                 len_traj_pred: int,
                 crop: List[int] = [350, 630],
                 resize: List[int] = [350, 630],
                 video_freq: int = 1,
                 pose_freq: int = 5,
                 target_freq: int = 1,
                 arrived_prob: float = 0.2,
                 input_noise: float = 0.1,
                 search_range: List[int] = [10, 60]
                 ):
        super().__init__()

        self.mode = mode

        self.data_dir = data_dir
        self.feat_file = feat_file
        self.video_freq = video_freq
        self.pose_freq = pose_freq
        self.target_freq = target_freq
        self.crop = crop
        self.resize = resize
        self.arrived_prob = arrived_prob

        self.frame_multiplier = self.video_freq // self.target_freq
        self.pose_multiplier = self.pose_freq // self.target_freq

        self.context_size = context_size
        self.len_traj_pred = len_traj_pred

        self.input_noise = input_noise
        self.search_lower_bound = search_range[0]
        self.search_upper_bound = search_range[1]

        traj_list = data_list
        self.transform = transforms.Compose([
            transforms.CenterCrop(self.crop),
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Build look up table
        self.poses = {}
        self.labels = {}
        self.lut = []
        if int(os.environ.get("RANK", "0")) == 0:
            traj_list = tqdm(traj_list, desc=f'Building {self.mode} dataset', ncols=120)
        for traj_id in traj_list:
            pose_data = {}
            pose_path = os.path.join(self.data_dir, traj_id, 'traj_data.txt')
            label_path = os.path.join(self.data_dir, traj_id, 'label.json')
            if not os.path.exists(pose_path) or not os.path.exists(label_path):
                continue
            
            # Pose filter
            pose = np.loadtxt(pose_path, delimiter=" ")[::max(1, self.pose_multiplier), 1:]
            pose_nan = np.isnan(pose).any(axis=1)
            if np.any(pose_nan):
                first_nan_idx = np.argmin(pose_nan)
                pose = pose[:first_nan_idx]
            usable = pose.shape[0] - self.context_size - max(self.search_lower_bound, self.len_traj_pred)
            if usable < 0:
                continue

            pose_data["count"] = usable
            pose_data["step_scale"] = np.linalg.norm(np.diff(pose[:, [0, 2]], axis=0), axis=1).mean()
            pose_data["pose"] = pose
            self.poses[traj_id] = pose_data
            
            with open(label_path, 'r', encoding='utf-8') as file:
                label_data = json.load(file)
            self.labels[traj_id] = label_data
            
            for label_idx in label_data.keys():
                label_idx = int(label_idx)
                if label_idx > self.context_size + self.search_lower_bound and label_idx < pose.shape[0]:
                    self.lut.append((traj_id, label_idx))
        
        # Load feat
        self._load_visual_feat()

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Waypoints number: {len(self.lut)}")


    def transform_poses(self, poses, current_pose_array):

        current_pose_matrix = pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)

        pose_matrices = poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]

        return positions


    def transform_target_pose(self, target_pose, current_pose_array):

        current_pose_matrix = pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)

        target_pose_matrix = pose_to_matrix(target_pose)
        transformed_target_matrix = np.matmul(current_pose_inv, target_pose_matrix)
        target_position = transformed_target_matrix[:3, 3]

        return target_position
    

    def _load_images(self, traj_id, indices):
        images = []
        for frame_idx in indices:
            frame_path = os.path.join(self.data_dir, traj_id, str(frame_idx).zfill(4) + '.jpg')
            image = self.transform(Image.open(frame_path).convert("RGB"))
            images.append(image)
        images = torch.stack(images, dim=0)
        return images
    

    def _load_visual_feat(self):
        if self.feat_file:
            self.visual_feat = lmdb.open(self.feat_file, readonly=True, lock=False)
        else:
            self.visual_feat = None


    def __getstate__(self):
        state = self.__dict__.copy()
        state["visual_feat"] = None
        return state
    

    def __setstate__(self, state):
        self.__dict__ = state
        if self.config["data"]["feat_file"] is not None:
            self._load_visual_feat()
    

    def __len__(self):
        return len(self.lut)
    

    def __getitem__(self, index):

        traj_id, target_idx = self.lut[index]

        # Get label data
        label = self.labels[traj_id]
        instr_data = random.choice(label[str(target_idx).zfill(4)])
        instr = instr_data["prompt"]
        bbox = instr_data["bbox"]
        distance = instr_data["distance"]
        if distance < 5:
            arrived = (random.random() < self.arrived_prob)
        else:
            arrived = False

        # Sample start indices
        if arrived:
            pose_start_idx = target_idx - self.context_size - self.len_traj_pred
        else:
            
            lower_bound = max(target_idx - self.search_upper_bound, self.context_size)
            upper_bound = target_idx - self.search_lower_bound
            pose_start_idx = random.randint(lower_bound, upper_bound) - self.context_size

        # Get frame indices
        frame_start_idx = pose_start_idx * self.frame_multiplier
        frame_indices = frame_start_idx + np.arange(self.context_size + self.len_traj_pred) * self.frame_multiplier
        

        # Load frames and features
        if self.visual_feat is not None:
            features = []
            with self.visual_feat.begin() as txn:
                for frame_idx in frame_indices:
                    while True:
                        key = traj_id + '_' + str(frame_idx).zfill(4)
                        value = txn.get(key.encode('ascii'))
                        if value:
                            feature = pkl.loads(value)
                            features.append(feature)
                            break
                        else:
                            frame_idx -= self.frame_multiplier

            obs_goal_feat = np.stack(features, axis=0)
            obs_goal_frames = None
        else:
            obs_goal_feat = None
            obs_goal_frames = self._load_images(traj_id, frame_indices)

        curr_idx = frame_indices[self.context_size - 1]
        curr_frame_path = os.path.join(self.data_dir, traj_id, str(curr_idx).zfill(4) + '.jpg')
        curr_frame = self.transform(Image.open(curr_frame_path).convert("RGB"))

        # Get pose data
        current_pose_idx = pose_start_idx + self.context_size
        pose = self.poses[traj_id]["pose"]
        input_poses = pose[pose_start_idx: current_pose_idx]
        assert int(target_idx) > current_pose_idx
        
        # Get target pose and waypoint poses
        target_pose = pose[int(target_idx)]
        waypoint_poses = pose[current_pose_idx: current_pose_idx + self.len_traj_pred]

        # Transform poses
        current_pose = input_poses[-1]
        transformed_input_poses = self.transform_poses(input_poses, current_pose)[:, [0, 2]]
        target_pose = self.transform_target_pose(target_pose, current_pose)[[0, 2]]
        transformed_waypoint_poses = self.transform_poses(waypoint_poses, current_pose)

        # Convert data to tensors
        step_scale = torch.tensor(self.poses[traj_id]["step_scale"], dtype=torch.float32)
        step_scale = torch.clamp(step_scale, min=1e-2)

        input_poses = torch.tensor(transformed_input_poses, dtype=torch.float32) / step_scale + torch.randn(self.context_size, 2) * self.input_noise
        target_pose = torch.tensor(target_pose, dtype=torch.float32)
        waypoint_poses = torch.tensor(transformed_waypoint_poses[:, [0, 2]], dtype=torch.float32) / step_scale

        arrived = torch.tensor(arrived, dtype=torch.float32)

        sample = {
            'instr': instr,
            'bbox': bbox,
            'obs_goal_frames': obs_goal_frames,
            'obs_goal_feat': obs_goal_feat,
            'curr_frame': curr_frame,
            'input_poses': input_poses,
            'target_pose': target_pose,
            'waypoint_poses': waypoint_poses,
            'step_scale': step_scale,
            'arrived': arrived
        }

        return sample
    

def custom_collate(batch):
    """
    Custom collate function for DataLoader.
    Handles cases where some fields in the batch may be None.
    - If all values for a key are None, returns None for that key.
    - Otherwise, uses the default_collate for non-None values.
    - If default_collate fails, returns the raw list of values.
    Args:
        batch (list): List of samples (dicts) from the dataset.
    Returns:
        dict: Batched data with appropriate handling of None values.
    """
    from torch.utils.data._utils.collate import default_collate
    elem = batch[0]
    collated = {}
    for key in elem:
        values = [d[key] for d in batch]
        # If all values are None, return None for this key
        if all(v is None for v in values):
            collated[key] = None
        else:
            # Try to collate non-None values using default_collate
            try:
                collated[key] = default_collate([v for v in values if v is not None])
            except Exception:
                # If default_collate fails, return the raw list
                collated[key] = values
    return collated

