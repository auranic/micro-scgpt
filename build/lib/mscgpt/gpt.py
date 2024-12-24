import torch
import torch.nn as nn

from torch.nn import functional as F

class MicroSCGPT(nn.Module):
    """TODO"""
    def __init__(
            self,
            ctx_size: int,
            bins_size: int,
            vocab_size: int,
            n_heads: int = 4,
            n_layers: int = 4,
            embed_size: int = 128,
            trans_hidden_size: int = 128,
            dropout: float = .1
        ):
        super().__init__()

        # Hyperparameters
        self.ctx_size = ctx_size
        self.bins_size = bins_size
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embed_size = embed_size
        self.dropout = dropout

        # Input module
        self.emb_gid = nn.Embedding(self.vocab_size, self.embed_size)
        self.emb_bin = nn.Embedding(self.bins_size, self.embed_size)
        self.ln_gid = nn.LayerNorm(self.embed_size)
        self.ln_bin = nn.LayerNorm(self.embed_size)

        # Transformers module
        # TODO: Flash attention
        self.transformer_layer = nn.TransformerEncoderLayer(
            self.embed_size, 
            self.n_heads,
            trans_hidden_size,
            batch_first=True,
            dropout=self.dropout
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, self.n_layers)

        self.n_parameters = sum(p.numel() for p in self.parameters())
        print(f"> MicroSCGPT: Model initialized with {self.n_parameters} parameters.")

    def forward(self, x_gid, x_bin) -> torch.tensor:
        # -> [B, ctx_size], [B, ctx_size]
        x_gid_emb = self.ln_gid(self.emb_gid(x_gid))
        x_bin_emb = self.ln_bin(self.emb_gid(x_bin))
        x_emb = x_gid_emb + x_bin_emb
        x_emb = F.dropout(x_emb, p=self.dropout)
        return self.transformer(x_emb) # -> [B, ctx_size, embed_size]


class GeneExpressionRegressor(nn.Module):
    """TODO"""
    def __init__(
            self,
            ctx_size: int,
            embed_size: int,
            vocab_size: int,
            n_hidden_layers: int = 2,
            hidden_size: int = 64
        ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(ctx_size*embed_size, hidden_size))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.LayerNorm(hidden_size))
        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.LayerNorm(hidden_size))
        self.layers.append(nn.Linear(hidden_size, vocab_size))
        self.layers.append(nn.LeakyReLU())

    def forward(self, x_emb) -> torch.tensor:
        B, T, C = x_emb.shape
        x_emb = x_emb.view(B, T*C)
        for layer in self.layers:
            x_emb = layer(x_emb)
        return x_emb