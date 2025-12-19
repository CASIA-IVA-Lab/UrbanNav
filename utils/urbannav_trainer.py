import os
import clip
import numpy as np
from tqdm import tqdm
import wandb
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF

    
def mean_reduce(local_tensor: torch.tensor) -> torch.Tensor:
    """
    Average a tensor across all distributed processes.
    Args: local_tensor (torch.Tensor): Tensor to be averaged.
    Returns: torch.Tensor: Mean value across all processes.
    """
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    return local_tensor / dist.get_world_size()


class UrbanNavTrainer:
    """
    Trainer class for UrbanNav navigation models.
    """
    def __init__(self,
                 model,
                 vision_encoder,
                 text_encoder,
                 device,
                 optimizer,
                 scheduler,
                 train_dataloader,
                 val_dataloader,
                 batch_size,
                 context_size: int = 4,
                 len_traj_pred: int = 4,
                 direction_loss_weight: float = 5.0,
                 feature_loss_weight: float = 0.1,
                 arrived_loss_weight: float = 1.0,
                 log_freq: int = 100,
                 use_wandb: bool = True
                 ):
        
        self.device = device
        self.model = model
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.batch_size = batch_size
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred

        self.direction_loss_weight = direction_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.arrived_loss_weight = arrived_loss_weight

        self.log_freq = log_freq
        self.use_wandb = use_wandb
        self.train_step = 0


    @torch.no_grad()
    def vision_encode(self, frames):

        B, T = frames.shape[:2]
        frames = frames.flatten(0, 1)
        features = self.vision_encoder(frames)
        features = features.view(B, T, -1)

        return features
    
    @staticmethod
    def compute_loss(waypoint_pred, arrived_pred, feature_pred, waypoint_gt, arrived_gt, feature_gt):
        """
        Compute multiple loss components for UrbanNav model training.
        Args:
            waypoint_pred (Tensor): Predicted waypoints, shape (..., 2).
            arrived_pred (Tensor): Predicted arrival logits.
            feature_pred (Tensor or None): Predicted features, or None if not used.
            waypoint_gt (Tensor): Ground truth waypoints, shape (..., 2).
            arrived_gt (Tensor): Ground truth arrival labels.
            feature_gt (Tensor or None): Ground truth features, or None if not used.
        Returns:
            tuple: (waypoint_loss, arrived_loss, direction_loss, feature_loss)
                - waypoint_loss (Tensor): L1 loss between predicted and ground truth waypoints.
                - arrived_loss (Tensor): Binary cross-entropy loss for arrival prediction.
                - direction_loss (Tensor): 1 - mean cosine similarity between predicted and ground truth directions.
                - feature_loss (Tensor or float): MSE loss for features, or 0.0 if not used.
        """
        # Compute feature loss
        if feature_pred is not None and feature_gt is not None:
            feature_loss = F.mse_loss(feature_pred, feature_gt)
        else:
            feature_loss = 0.0

        # Compute Waypoiint loss
        waypoint_loss = F.l1_loss(waypoint_pred, waypoint_gt)

        # Compute arrived loss
        arrived_loss = F.binary_cross_entropy_with_logits(arrived_pred.flatten(), arrived_gt)

        # Compute direction loss
        waypoint_pred = waypoint_pred.view(-1, 2)
        waypoint_gt = waypoint_gt.view(-1, 2)

        dot_product = (waypoint_pred * waypoint_gt).sum(dim=1)
        waypoint_pred_norm = waypoint_pred.norm(dim=1)
        waypoint_gt_norm = waypoint_gt.norm(dim=1)
        cos_sim = dot_product / (waypoint_pred_norm * waypoint_gt_norm + 1e-8)

        direction_loss = 1 - cos_sim.mean()

        return waypoint_loss, arrived_loss, direction_loss, feature_loss


    def train_epoch(self, epoch: int):
        """
        Run one training epoch for the UrbanNav model.
        """
        self.model.train()
        
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Start training for epoch {epoch}")
            training_bar = tqdm(self.train_dataloader, desc=f"Training for epoch {epoch}", leave=True, ncols=120)
        else:
            training_bar = self.train_dataloader

        epoch_losses = dict()
        epoch_losses["loss"] = []
        epoch_losses["waypoint_loss"] = []
        epoch_losses["direction_loss"] = []
        epoch_losses["feature_loss"] = []
        epoch_losses["arrived_loss"] = []

        for _, data in enumerate(training_bar):

            instr = data["instr"]
            curr_frame = data["curr_frame"].to(self.device)
            input_poses = data["input_poses"].to(self.device)
            waypoint_gt = data["waypoint_poses"].to(self.device)
            arrived_gt = data["arrived"].to(self.device)

            if data['obs_goal_feat'] is not None:
                obs_feat = data['obs_goal_feat'][:, :self.context_size, ...].to(self.device)
                feat_gt = data['obs_goal_feat'][:, self.context_size:, ...].to(self.device)
            else:
                with torch.no_grad():
                   obs_frames = data['obs_goal_frames'][:, :self.context_size, ...].to(self.device)
                   goal_frames = data['obs_goal_frames'][:, self.context_size:, ...].to(self.device)
                   obs_feat = self.vision_encode(obs_frames)
                   feat_gt = self.vision_encode(goal_frames)
            curr_frame = TF.resize(curr_frame, (224, 224))
                  
            # Extract text features
            with torch.no_grad():
                text_inputs = clip.tokenize(instr, truncate=True).to(self.device)
                text_feat = self.text_encoder.encode_text(text_inputs)

            # Predict
            waypoint_pred, arrived_pred, feature_pred = self.model(
                text_feat, obs_feat,
                curr_obs_img=curr_frame,
                coord=input_poses
                )

            # Compute Loss
            waypoint_loss, arrived_loss, direction_loss, feature_loss = self.compute_loss(
                waypoint_pred=waypoint_pred, 
                arrived_pred=arrived_pred, 
                feature_pred=feature_pred, 
                waypoint_gt=waypoint_gt, 
                arrived_gt=arrived_gt, 
                feature_gt=feat_gt
                )
            loss = waypoint_loss + self.arrived_loss_weight * arrived_loss \
                  + self.direction_loss_weight * direction_loss + feature_loss * self.feature_loss_weight
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            # logging
            self.train_step += 1
            loss_cpu = mean_reduce(loss).item()
            
            epoch_losses["loss"].append(loss_cpu)
            epoch_losses["waypoint_loss"].append(mean_reduce(waypoint_loss).item())
            epoch_losses["direction_loss"].append(mean_reduce(direction_loss).item())
            epoch_losses["feature_loss"].append(mean_reduce(feature_loss).item())
            epoch_losses["arrived_loss"].append(mean_reduce(arrived_loss).item())

            if int(os.environ.get("RANK", "0")) == 0:
                training_bar.set_postfix(loss=loss_cpu)
                
                if self.use_wandb and self.train_step % self.log_freq == 0:
                    running_loss = statistics.mean(epoch_losses["loss"][-self.log_freq:])
                    running_waypoint_loss = statistics.mean(epoch_losses["waypoint_loss"][-self.log_freq:])
                    running_direction_loss = statistics.mean(epoch_losses["direction_loss"][-self.log_freq:])
                    running_feature_loss = statistics.mean(epoch_losses["feature_loss"][-self.log_freq:])
                    running_arrived_loss = statistics.mean(epoch_losses["arrived_loss"][-self.log_freq:])
                    wandb.log({
                        'loss': running_loss,
                        'waypoint_loss': running_waypoint_loss,
                        'direction_loss': running_direction_loss,
                        'feature_loss': running_feature_loss,
                        'arrived_loss': running_arrived_loss
                    }, step=self.train_step)
        
        for k in epoch_losses.keys():
            epoch_losses[k] = statistics.mean(epoch_losses[k])

        return epoch_losses


    @torch.no_grad()
    def val_epoch(self):
        """
        Run one evaluating epoch for the UrbanNav model.
        """
        self.model.eval()

        if int(os.environ.get("RANK", "0")) == 0:
            val_bar = tqdm(self.val_dataloader, desc=f"Evaluating:", leave=True, ncols=120)
        else:
            val_bar = self.val_dataloader

        epoch_losses = {}
        epoch_losses["loss"] = []
        epoch_losses["waypoint_loss"] = []
        epoch_losses["direction_loss"] = []
        epoch_losses["feature_loss"] = []
        epoch_losses["arrived_loss"] = []

        for _, data in enumerate(val_bar):

            instr = data["instr"]
            curr_frame = data["curr_frame"].to(self.device)
            input_poses = data["input_poses"].to(self.device)
            waypoint_gt = data["waypoint_poses"].to(self.device)
            arrived_gt = data["arrived"].to(self.device)

            if data['obs_goal_feat'] is not None:
                obs_feat = data['obs_goal_feat'][:, :self.context_size, ...].to(self.device)
                feat_gt = data['obs_goal_feat'][:, self.context_size:, ...].to(self.device)
            else:
                with torch.no_grad():
                   obs_frames = data['obs_goal_frames'][:, :self.context_size, ...].to(self.device)
                   goal_frames = data['obs_goal_frames'][:, self.context_size:, ...].to(self.device)
                   obs_feat = self.vision_encode(obs_frames)
                   feat_gt = self.vision_encode(goal_frames)
            curr_frame = TF.resize(curr_frame, (224, 224))

            # Extract text features
            with torch.no_grad():
                text_inputs = clip.tokenize(instr, truncate=True).to(self.device)
                text_feat = self.text_encoder.encode_text(text_inputs)
            
            # Predict
            waypoint_pred, arrived_pred, feature_pred = self.model(
                text_feat, obs_feat,
                curr_obs_img=curr_frame,
                coord=input_poses
                )

            # Compute Loss
            waypoint_loss, arrived_loss, direction_loss, feature_loss = self.compute_loss(
                waypoint_pred, arrived_pred, feature_pred, waypoint_gt, arrived_gt, feat_gt
                )
            loss = waypoint_loss + self.arrived_loss_weight * arrived_loss + \
                  self.direction_loss_weight * direction_loss + feature_loss * self.feature_loss_weight

            # Logging
            loss_cpu = mean_reduce(loss).item()
            if int(os.environ.get("RANK", "0")) == 0:
                val_bar.set_postfix(loss=loss_cpu)

            epoch_losses["loss"].append(loss_cpu)
            epoch_losses["waypoint_loss"].append(mean_reduce(waypoint_loss).item())
            epoch_losses["direction_loss"].append(mean_reduce(direction_loss).item())
            epoch_losses["feature_loss"].append(mean_reduce(feature_loss).item())
            epoch_losses["arrived_loss"].append(mean_reduce(arrived_loss).item())

        for k in epoch_losses.keys():
            epoch_losses[k] = statistics.mean(epoch_losses[k])
            
        return epoch_losses


@torch.no_grad()
def test_epoch(
        mode: str,
        config: dict,
        model: nn.Module,
        vision_encoder,
        text_encoder,
        dataloader: DataLoader,
        device: torch.device,
        ):
    """
    Run one testing epoch for the UrbanNav model.
    """
    
    context_size = config["context_size"]
    len_traj_pred = config["len_traj_pred"]
    
    model.eval()
    vision_encoder.eval()
    text_encoder.eval()

    direction_loss_weight = config["train"]["direction_loss_weight"]
    feature_loss_weight = config["train"]["feature_loss_weight"]

    test_bar = tqdm(dataloader, desc=f"Testing {mode}: ", leave=True, ncols=120)

    metric = {}
    metric["loss"] = []
    metric["waypoint_loss"] = []
    metric["direction_loss"] = []
    metric["feature_loss"] = []
    metric["arrived_loss"] = []
    metric["arrived_acc"] = []
    for i in range(len_traj_pred):
        metric[f"angle_{i+1}"] = []
        metric[f"distance_{i+1}"] = []
    metric["AOE"] = []
    metric["MAOE"] = []
    metric["ADE"] = []
    metric["MADE"] = []
    metric["weighted_fitness"] = []

    for i, data in enumerate(test_bar):

        instr = data["instr"]
        curr_frame = data["curr_frame"].to(device)
        input_poses = data["input_poses"].to(device)
        waypoint_gt = data["waypoint_poses"].to(device)
        arrived_gt = data["arrived"].to(device)

        if data['obs_goal_feat'] is not None:
            obs_feat = data['obs_goal_feat'][:, :context_size, ...].to(device)
            feat_gt = data['obs_goal_feat'][:, context_size:, ...].to(device)
        else:
            with torch.no_grad():
                obs_frames = data['obs_goal_frames'][:, :context_size, ...].to(device)
                goal_frames = data['obs_goal_frames'][:, context_size:, ...].to(device)
                obs_feat = vision_encode(vision_encoder, obs_frames)
                feat_gt = vision_encode(vision_encoder, goal_frames)
        curr_frame = TF.resize(curr_frame, (224, 224))

        # Extract text features
        with torch.no_grad():
            text_inputs = clip.tokenize(instr, truncate=True).to(device)
            text_feat = text_encoder.encode_text(text_inputs)
        
        # Predict
        with torch.no_grad():
            waypoint_pred, arrived_pred, feature_pred = model(
                text_feat, obs_feat,
                curr_obs_img=curr_frame,
                coord=input_poses
                )

        # Compute loss and metric
        waypoint_loss, arrived_loss, direction_loss, feature_loss = UrbanNavTrainer.compute_loss(
            waypoint_pred, arrived_pred, feature_pred, waypoint_gt, arrived_gt, feat_gt
            )
        loss = waypoint_loss + direction_loss_weight * direction_loss + feature_loss * feature_loss_weight
        waypoint_pred_view = waypoint_pred.view(-1, 2)
        waypoint_gt_view = waypoint_gt.view(-1, 2)

        # Compute arrived accuracy
        arrived_probs = torch.sigmoid(arrived_pred)
        arrived_pred_binary = (arrived_probs >= 0.5).float().squeeze(-1)
        arrived_correct = (arrived_pred_binary == arrived_gt).float()
        arrived_acc = arrived_correct.mean()

        # Compute AOE and MAOE
        cos_sim = F.cosine_similarity(waypoint_pred_view, waypoint_gt_view, dim=1)
        angles = torch.acos(cos_sim) * 180 / torch.pi
        angles = angles.view(config["batch_size"], config["len_traj_pred"])
        mean_angle_step = angles.mean(dim=0)
        mean_angle_step_np = mean_angle_step.cpu().numpy()
        max_angles, _ = torch.max(angles, dim=1)
        mean_angle_max = max_angles.mean(dim=0)
        mean_angle = torch.mean(angles)

        # Compute distance
        distances = torch.norm(waypoint_pred_view - waypoint_gt_view, p=2, dim=1)
        distances = distances.view(config["batch_size"], config["len_traj_pred"])
        mean_distance_step = distances.mean(dim=0)
        mean_distance_step_np = mean_distance_step.cpu().numpy()
        max_distances, _ = torch.max(distances, dim=1)
        mean_distance_max = max_distances.mean(dim=0)
        mean_distance = torch.mean(distances)

        # WF
        metric_factor = torch.linspace(1 / config['len_traj_pred'], 1, steps=config['len_traj_pred']).to(device)
        weighted_fitness = 0.1 * torch.dot(mean_angle_step, metric_factor.flip(0)) + torch.dot(mean_distance_step, metric_factor)

        metric["loss"].append(loss.item())
        metric["waypoint_loss"].append(waypoint_loss.item())
        metric["direction_loss"].append(direction_loss.item())
        metric["feature_loss"].append(feature_loss.item())
        metric["arrived_loss"].append(arrived_loss.item())
        
        for i in range(len_traj_pred):
            metric[f'angle_{i+1}'].append(float(mean_angle_step_np[i]))
            metric[f'distance_{i+1}'].append(float(mean_distance_step_np[i]))

        metric["AOE"].append(mean_angle.item())
        metric["MAOE"].append(mean_angle_max.item())
        metric["ADE"].append(mean_distance.item())
        metric["MADE"].append(mean_distance_max.item())
        metric["weighted_fitness"].append(weighted_fitness.item())
        metric["arrived_acc"].append(arrived_acc.item())

    for k, v in metric.items():
        metric[k] = np.nanmean(v)

    return metric


def vision_encode(vision_encoder, frames):

    B, T = frames.shape[:2]
    frames = frames.flatten(0, 1)
    features = vision_encoder(frames)
    features = features.view(B, T, -1)

    return features

