import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable


class PolarEmbedding(nn.Module):
    def __init__(self, cfg):
        super(PolarEmbedding, self).__init__()
        self.num_freqs = cfg.model.cord_embedding.num_freqs
        self.include_input = cfg.model.cord_embedding.include_input
        # Register freq_bands as a buffer to ensure it's moved to the correct device
        freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        self.out_dim = 2 + 4 * self.num_freqs if self.include_input else 4 * self.num_freqs

    def forward(self, coords):
        """
        Args:
            coords: Tensor of shape (B, N, 2)
        
        Returns:
            Tensor of shape (B, N, D) where D = 2 (if include_input) + 4 * num_freqs
        """
        # Ensure coords has the correct shape
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError(f"Expected coords of shape (B, N, 2), but got {coords.shape}")
        
        x, y = coords[..., 0], coords[..., 1]  # Shape: (B, N)
        r = torch.sqrt(x**2 + y**2).unsqueeze(-1)            # Shape: (B, N)
        theta = torch.atan2(y, x).unsqueeze(-1)              # Shape: (B, N)
        
        enc = [r, theta] if self.include_input else []
        
        # Expand freq_bands to (1, 1, num_freqs) for broadcasting
        freq_bands = self.freq_bands.view(1, 1, -1)  # Shape: (1, 1, num_freqs)
        
        # Compute sin and cos for theta and r with frequency bands
        enc.append(torch.sin(theta * freq_bands))  # Shape: (B, N, num_freqs)
        enc.append(torch.cos(theta * freq_bands))  # Shape: (B, N, num_freqs)
        enc.append(torch.sin(r * freq_bands))      # Shape: (B, N, num_freqs)
        enc.append(torch.cos(r * freq_bands))      # Shape: (B, N, num_freqs)
        
        # Concatenate all encodings along the last dimension
        enc = torch.cat(enc, dim=-1)  # Shape: (B, N, D)
        
        return enc

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
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class MultiLayerDecoder(nn.Module):
    """
    ref: https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py
    """
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
    
class FeatPredictor(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, nhead=8, num_layers=8, ff_dim_factor=4):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
    
    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        return x
    

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with the output of func.
    args:
        predicate: Return true if the module is to be replaced.
        func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)
    
    bn_list = [k.split('.') for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]

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
    bn_list = [k.split('.') for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]

    assert len(bn_list) == 0
    return root_module
    

def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int = 16
) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module
    
    