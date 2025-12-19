import torch
import torch.nn as nn

import math
from typing import Optional, Callable
from efficientnet_pytorch import EfficientNet


# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


class PositionalEncoding(nn.Module):
    """
    ref: https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py
    """
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x


class PolarEmbedding(nn.Module):
    def __init__(self, num_freqs):
        super(PolarEmbedding, self).__init__()

        self.num_freqs = num_freqs
        freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        self.out_dim = 2 + 4 * self.num_freqs
        
    def forward(self, coords):
        """
        Args:
            coords: Tensor of shape (B, N, 2)
        Returns:
            Tensor of shape (B, N, D) where D = 2 (if include_input) + 4 * num_freqs
        """
        x, y = coords[..., 0], coords[..., 1]  # Shape: (B, N)
        r = torch.sqrt(x**2 + y**2).unsqueeze(-1)  # Shape: (B, N)
        theta = torch.atan2(y, x).unsqueeze(-1)  # Shape: (B, N)
        enc = [r, theta]

        # Expand freq_bands to (1, 1, num_freqs) for broadcasting
        freq_bands = self.freq_bands.reshape(1, 1, -1)  # Shape: (1, 1, num_freqs)

        # Compute sin and cos for theta and r with frequency bands
        enc.append(torch.sin(theta * freq_bands))  # Shape: (B, N, num_freqs)
        enc.append(torch.cos(theta * freq_bands))  # Shape: (B, N, num_freqs)
        enc.append(torch.sin(r * freq_bands))      # Shape: (B, N, num_freqs)
        enc.append(torch.cos(r * freq_bands))      # Shape: (B, N, num_freqs)

        # Concatenate all encodings along the last dimension
        enc = torch.cat(enc, dim=-1)  # Shape: (B, N, D)

        return enc


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x


class InitialFeatureExtractor(nn.Module):
    def __init__(self):
        super(InitialFeatureExtractor, self).__init__()

        self.layers = nn.Sequential(
            self._conv_layer(3, 128, 5, 2, 2),
            self._conv_layer(128, 128, 3, 2, 1),
            self._conv_layer(128, 128, 3, 2, 1)
        )

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.layers(x)
    

class IntermediateFeatureExtractor(nn.Module):
    def __init__(self):
        super(IntermediateFeatureExtractor, self).__init__()

        self.layers = nn.Sequential(
            self._conv_layer(128, 256, 3, 2, 1),
            self._conv_layer(256, 512, 3, 2, 1),
            self._conv_layer(512, 1024, 3, 2, 1),
            self._conv_layer(1024, 1024, 3, 2, 1)
        )

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.layers(x)
    

class FiLMTransform(nn.Module):
    def __init__(self):
        super(FiLMTransform, self).__init__()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):

        beta = beta.reshape(x.size(0), x.size(1), 1, 1)
        gamma = gamma.reshape(x.size(0), x.size(1), 1, 1)

        return gamma * x + beta
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.film_transform = FiLMTransform()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film_transform(x, beta, gamma)
        x = self.relu2(x)

        return x + identity


class FinalClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FinalClassifier, self).__init__()

        self.conv = nn.Conv2d(in_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor):

        x = self.conv(x)
        feature_map = x

        x = self.global_pool(x)
        x.reshape(x.size(0), x.size(1))
        x = self.fc_layer(x)

        return x, feature_map


class FiLMNetwork(nn.Module):
    def __init__(self, num_resblock, num_channel, feat_text_dim):
        super(FiLMNetwork, self).__init__()

        self.num_resblock = num_resblock
        self.num_channel = num_channel

        self.film_param_generator = nn.Linear(feat_text_dim, 2*num_resblock*num_channel)
        self.initial_feaure_extrator = InitialFeatureExtractor()
        self.residual_blocks = nn.ModuleList()
        self.intermediate_feature_extractor = IntermediateFeatureExtractor()

        for _ in range(num_resblock):
            self.residual_blocks.append(ResidualBlock(num_channel+2, num_channel))


    def forward(self, x: torch.Tensor, feat_text: torch.Tensor):

        batch_size = x.size(0)
        device = x.device

        x = self.initial_feaure_extrator(x)
        film_param = self.film_param_generator(feat_text).reshape(batch_size, self.num_resblock, 2, self.num_channel)

        # Represent spatial information
        d = x.size(2)
        coords = torch.arange(-1, 1+0.00001, 2/(d-1)).to(device)
        coord_x = coords.expand(batch_size, 1, d, d)
        coord_y = coords.reshape(d, 1).expand(batch_size, 1, d, d)

        for i, resblock in enumerate(self.residual_blocks):
            beta = film_param[:, i, 0, :]
            gamma = film_param[:, i, 1, :]

            x = torch.cat([x, coord_x, coord_y], 1)
            x = resblock(x, beta, gamma)

        features = self.intermediate_feature_extractor(x)

        return features


class UrbanNavFiLM(nn.Module):
    def __init__(self,
                 context_size: int,
                 len_traj_pred: int,
                 visual_feat_size: int,
                 num_freqs: int,
                 attn_dim: int,
                 num_attn_layers: int,
                 num_attn_heads: int,
                 ff_dim_factor: int,
                 dropout: float,
                 **kwargs,
                 ):
        super().__init__()

        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.visual_feat_size = visual_feat_size
        self.film_feature_size = kwargs['film_feature_size']

        self.attn_dim = attn_dim
        self.num_attn_layers = num_attn_layers
        self.num_attn_heads = num_attn_heads
        self.ff_dim_factor = ff_dim_factor
        self.dropout = dropout

        if kwargs['clip_type'] == "ViT-B/32":
            self.obsgoal_encoder = FiLMNetwork(8, 128, 512)
        elif kwargs['clip_type'] == "ViT-L/14@336px":
            self.obsgoal_encoder = FiLMNetwork(8, 128, 768)
        elif kwargs['clip_type'] == "RN50x64":
            self.obsgoal_encoder = FiLMNetwork(8, 128, 1024)
        self.obsgoal_encoder = replace_bn_with_gn(self.obsgoal_encoder)

        self.obsgoal_compress = nn.Linear(self.film_feature_size, self.visual_feat_size)
        self.vision_compress = nn.Linear(self.visual_feat_size, self.attn_dim)
        self.text_compress = nn.Sequential(
            nn.Linear(512, self.attn_dim),
            nn.BatchNorm1d(self.attn_dim)
        )

        # Feature fusion MLP
        self.linear1 = nn.Linear(self.attn_dim, self.ff_dim_factor * self.attn_dim)
        self.bn1 = nn.BatchNorm1d(self.ff_dim_factor * self.attn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.ff_dim_factor * self.attn_dim, self.attn_dim)
        self.bn2 = nn.BatchNorm1d(self.attn_dim)

        self.coord_embedding = PolarEmbedding(num_freqs)
        self.coord_embedding_size = self.coord_embedding.out_dim * self.context_size
        self.coord_compress = nn.Linear(self.coord_embedding_size, self.attn_dim)

        # Transformer encoder
        self.positional_encoding = PositionalEncoding(self.attn_dim, self.context_size + 1)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.attn_dim,
            nhead=self.num_attn_heads,
            dim_feedforward=self.attn_dim * self.ff_dim_factor,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=self.num_attn_layers)

        # Set up decoder
        self.mlp_decoder = nn.Sequential(
            nn.Linear((self.context_size + 1) * self.visual_feat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Set up predictor
        self.wp_predictor = nn.Linear(32, self.len_traj_pred * 2)
        self.arrived_predictor = nn.Linear(32, 1)


    def forward(self, text_feat, obs_feat, **kwargs):
        B = text_feat.size(0)
        curr_obs_img = kwargs["curr_obs_img"]

        obsgoal_enc = self.obsgoal_encoder(curr_obs_img, text_feat)
        obsgoal_enc = self.obsgoal_compress(obsgoal_enc.flatten(start_dim=1)).unsqueeze(1)

        text_feat = self.text_compress(text_feat).unsqueeze(1)
        obs_feat = obs_feat[:, :-1, :]

        obsgoal_enc = torch.cat((obsgoal_enc, obs_feat, text_feat), dim=1)
        obsgoal_enc = self.linear1(obsgoal_enc).permute(0, 2, 1)
        obsgoal_enc = self.relu(self.bn1(obsgoal_enc).permute(0, 2, 1))
        obsgoal_enc = self.linear2(obsgoal_enc).permute(0, 2, 1)
        obsgoal_enc = self.bn2(obsgoal_enc).permute(0, 2, 1)

        # Encode
        input_tokens = obsgoal_enc
        input_tokens = self.positional_encoding(input_tokens)
        feature_pred = self.sa_encoder(input_tokens)

        # Decode
        decode_out = self.mlp_decoder(feature_pred.reshape(B, -1))

        # Predict
        wp_pred = self.wp_predictor(decode_out).reshape(B, self.len_traj_pred, 2)
        arrived_pred = self.arrived_predictor(decode_out).reshape(B, 1)
        return wp_pred, arrived_pred, feature_pred[:, 1:]

