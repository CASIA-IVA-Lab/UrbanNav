import torch
import torch.nn as nn
import math


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

    def forward(self, query, key, value, mask=None):
        # query: (seq_len, batch_size, embed_dim)
        # key: (seq_len, batch_size, embed_dim)
        # value: (seq_len, batch_size, embed_dim)
        # mask: padding mask

        attn_output, attn_weights = self.attn(query, key, value, key_padding_mask=mask)
        return attn_output, attn_weights


class CrossAttentionEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super(CrossAttentionEncoder, self).__init__()

        # Cross Attention
        self.cross_attention_layers = nn.ModuleList([
            MultiHeadCrossAttention(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, image_features, text_features, mask=None):
        
        for attn_layer in self.cross_attention_layers:
            # Text features for Query，Image features for Key and Value
            x, _ = attn_layer(image_features, text_features, text_features, mask)

        return x


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
        freq_bands = self.freq_bands.view(1, 1, -1)  # Shape: (1, 1, num_freqs)

        # Compute sin and cos for theta and r with frequency bands
        enc.append(torch.sin(theta * freq_bands))  # Shape: (B, N, num_freqs)
        enc.append(torch.cos(theta * freq_bands))  # Shape: (B, N, num_freqs)
        enc.append(torch.sin(r * freq_bands))      # Shape: (B, N, num_freqs)
        enc.append(torch.cos(r * freq_bands))      # Shape: (B, N, num_freqs)

        # Concatenate all encodings along the last dimension
        enc = torch.cat(enc, dim=-1)  # Shape: (B, N, D)

        return enc


class UrbanNavCrossAttention(nn.Module):
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
                 **kwargs
                 ):
        super().__init__()

        self.context_size = context_size

        self.len_traj_pred = len_traj_pred
        self.visual_feat_size = visual_feat_size

        self.attn_dim = attn_dim
        self.num_attn_layers = num_attn_layers
        self.num_attn_heads = num_attn_heads
        self.ff_dim_factor = ff_dim_factor
        self.dropout = dropout
        self.coord_embedding = PolarEmbedding(num_freqs)
        self.coord_embedding_size = self.coord_embedding.out_dim * (self.context_size)
        self.coord_compress = nn.Linear(self.coord_embedding_size, self.attn_dim)

        # Feature project layer
        self.vision_compress = nn.Sequential(
            nn.Linear(self.visual_feat_size, self.attn_dim),
            # nn.BatchNorm1d(self.attn_dim)
        )
        self.text_compress = nn.Sequential(
            nn.Linear(512, self.attn_dim),
            nn.BatchNorm1d(self.attn_dim)
        )

        # Multi-head cross-attention
        self.ca_encoder = CrossAttentionEncoder(self.attn_dim, self.num_attn_heads, self.num_attn_layers, self.dropout)

        # Set up transformer encoder
        self.positional_encoding = PositionalEncoding(self.attn_dim, self.context_size + 2)
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
        coord = kwargs["coord"]

        # Feature project
        text_feat = self.text_compress(text_feat).unsqueeze(1)
        text_feat_attn = text_feat.transpose(0, 1)
        text_feat_attn = text_feat_attn.expand(self.context_size, -1, -1)

        obs_feat = self.vision_compress(obs_feat).transpose(0, 1)

        coord_enc = self.coord_embedding(coord).view(B, -1)
        coord_enc = self.coord_compress(coord_enc).view(B, 1, -1)

        # Feature Fusion
        obsgoal_encoding = self.ca_encoder(text_feat_attn, obs_feat)  # query: text, key: obs, value: obs
        obsgoal_encoding = obsgoal_encoding.transpose(0, 1)

        # Encode
        input_tokens = torch.cat([obsgoal_encoding, coord_enc], dim=1)
        input_tokens = self.positional_encoding(input_tokens)
        feature_pred = self.sa_encoder(input_tokens)

        # Decode
        decode_out = self.mlp_decoder(feature_pred.view(B, -1))

        # Predict
        wp_pred = self.wp_predictor(decode_out).view(B, self.len_traj_pred, 2)
        arrived_pred = self.arrived_predictor(decode_out).view(B, 1)
        return wp_pred, arrived_pred, feature_pred[:, 1:-1]

