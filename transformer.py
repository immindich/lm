import torch
from torch import nn
import torch.nn.functional as F
import math
import einops
from config import Config

class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg.d_vocab, self.cfg.d_model))
        a = math.sqrt(1.0 / self.cfg.d_vocab)
        nn.init.uniform_(self.W_E, -a, a)

    def forward(self, tokens):
        return self.W_E[tokens]

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(cfg.d_model, cfg.d_vocab))
        a = math.sqrt(1.0 / cfg.d_model)
        nn.init.uniform_(self.W_U, -a, a)

    def forward(self, x):
        return x @ self.W_U

class RotaryEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # (d_head / 2)
        theta = 10000. ** (-torch.arange(0, cfg.d_head,2) / cfg.d_head)
        # (seq, d_head / 2)
        angles = torch.outer(torch.arange(cfg.ctx_len), theta)
        sines = angles.sin()
        cosines = angles.cos()
        r = torch.stack((cosines, -sines, sines, cosines), dim=-1)
        self.register_buffer("rotations", r.view(cfg.ctx_len, cfg.d_head // 2, 2, 2).requires_grad_(False), persistent=False)
    
    # x : (batch, head, seq, d_head)
    def forward(self, x):
        seq = x.shape[2]
        batch = x.shape[0]
        x = x.view(batch, self.cfg.n_heads, seq, self.cfg.d_head // 2, 2)
        rotated = einops.einsum(self.rotations[:seq], x, 'seq d i j, batch head seq d j -> batch head seq d i')
        return rotated.view(batch, self.cfg.n_heads, seq, self.cfg.d_head)

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        assert cfg.d_model % cfg.n_heads == 0
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_heads
        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, self.d_head))
        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, self.d_head))
        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, self.d_head))
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, self.d_head, cfg.d_model))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

        if self.cfg.rotary:
            self.rotary_embed = RotaryEmbedding(cfg)

        a = math.sqrt(1.0 / cfg.d_model)
        for m in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.uniform_(m, -a, a)

    # x : (batch, seq, d_model)
    def forward(self, x):
        queries = einops.einsum(x, self.W_Q, 'batch seq d_model, n_heads d_model d_head -> batch n_heads seq d_head')
        keys = einops.einsum(x, self.W_K, 'batch seq d_model, n_heads d_model d_head -> batch n_heads seq d_head')
        values = einops.einsum(x, self.W_V, 'batch seq d_model, n_heads d_model d_head -> batch n_heads seq d_head')

        if self.cfg.rotary:
            queries = self.rotary_embed(queries)
            keys = self.rotary_embed(keys)

        #scores = einops.einsum(queries, keys, 'batch n_heads seq_q d_head, batch n_heads seq_k d_head -> batch n_heads seq_q seq_k') / math.sqrt(self.d_head)
        #scores_masked = self.mask(scores)

        #weights = scores_masked.softmax(-1)
        #values_weighted = einops.einsum(weights, values, 'batch n_heads seq_q seq_k, batch n_heads seq_k d_head -> batch seq_q n_heads d_head')

        values_weighted = F.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=0.0, is_causal=True)

        output = einops.einsum(values_weighted, self.W_O, 'batch n_heads seq_q d_head, n_heads d_head d_model -> batch seq_q n_heads d_model')
        return output.sum(-2) + self.b_O

    def mask(self, scores):
        mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1), device=scores.device), diagonal=1).bool()
        scores.masked_fill_(mask, -1e5)
        return scores

class PositionalEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_P = nn.Parameter(torch.empty(cfg.ctx_len, cfg.d_model))
        a = math.sqrt(1.0 / cfg.d_model)
        nn.init.uniform_(self.W_P, -a, a)

    def forward(self, tokens):
        batch, seq = tokens.shape
        return self.W_P[0:seq].view(1,seq, self.cfg.d_model).repeat(batch,1,1)

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x):
        e = x.mean(2, True)
        s = (x.var(2, keepdim = True, unbiased = False) + self.cfg.layer_norm_eps).sqrt()
        return ((x - e) / s) * self.w + self.b

class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gain = nn.Parameter(torch.ones(cfg.d_model))

    def forward(self, x):
        # x : (batch, seq, d_model)
        rms_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * self.gain * rms_inv

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

    def forward(self, x):
        preact = self.W_in(x)
        activations = F.gelu(preact)
        return self.W_out(activations)

class GatedFFN(nn.Module):
    def __init__(self, cfg, gating_fn):
        super().__init__()
        self.cfg = cfg
        self.W1 = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.W2 = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)
        self.V = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.gating_fn = gating_fn

    def forward(self, x):
        return self.W2(self.gating_fn(self.W1(x)) * self.V(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = RMSNorm(cfg) if cfg.rmsnorm else LayerNorm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg) if cfg.rmsnorm else LayerNorm(cfg)
        self.mlp = GatedFFN(cfg, F.silu) if cfg.swiglu else MLP(cfg)

    def forward(self, x):
        resid_mid = x + self.attn(self.ln1(x))
        resid_post = resid_mid + self.mlp(self.ln2(resid_mid))
        return resid_post

class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        if not self.cfg.rotary:
            self.pos = PositionalEmbed(cfg)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for i in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)

    def forward(self, x):
        if self.cfg.rotary:
            resid = self.embed(x)
        else:
            resid = self.embed(x) + self.pos(x)
        for i in range(self.cfg.n_layers):
            resid = self.layers[i](resid)
        return self.unembed(resid)
