"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
"""Modified baseline: RMSNorm + SwiGLU, based on classroom GPT baseline."""
"""Modified baseline: RMSNorm + SwiGLU, based on classroom GPT baseline."""
"""Exp-B student.py: ONLY RMSNorm, MLP remains original GELU (Ablation for RMSNorm)
Only change: All nn.LayerNorm replaced by RMSNorm. MLP unchanged as baseline.
Keep pos embedding, scaled dot product attention, predict_log_probs, weight-tying.
"""
"""Modified baseline: RMSNorm + SwiGLU, based on classroom GPT baseline."""
"""Modified baseline: RMSNorm + SwiGLU, based on classroom GPT baseline."""
"""Modified baseline: RMSNorm + SwiGLU + ALiBi (D version), based on classroom GPT baseline."""
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        hidden_dim = int(8.0 / 3.0 * emb_dim) 
        self.w1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class ALiBiAttention(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        slopes = torch.tensor([2 ** (-8 / self.heads * (i+1)) for i in range(self.heads)])
        self.register_buffer("slopes", slopes)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att_logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        pos = torch.arange(length, device=x.device)
        dist = pos.unsqueeze(0) - pos.unsqueeze(1) # shape [T, T]
        dist = dist.float()
        causal_mask = dist > 0
        dist = dist.masked_fill(causal_mask, float("-inf"))
        alibi_bias = dist * self.slopes.view(-1,1,1)
        att_logits = att_logits + alibi_bias      
        att_weights = F.softmax(att_logits, dim=-1)
        attended = att_weights @ v
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.proj(attended)



class Block(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attn = ALiBiAttention(width, heads)
        self.mlp = SwiGLUFFN(width)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([Block(width, config['heads']) for _ in range(config['depth'])])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)

        self.apply(self.initialize)
        self.head.weight = self.token.weight  
    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)

def build_model(config):
    """Student implementation: RMSNorm + SwiGLU + ALiBi"""
    return GPT(config)

