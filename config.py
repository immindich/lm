import torch
from dataclasses import dataclass

@dataclass
class Config:
    d_model : int
    d_mlp : int
    n_heads : int
    d_head : int
    d_vocab : int
    ctx_len : int
    layer_norm_eps : float
    n_layers : int
    rmsnorm : bool
    swiglu : bool

@dataclass
class TrainConfig:
    batch_size: int
    minibatch_size: int
    batches: int
    warmup_iters: int
    decay_iters: int
    lr: float
    decay_factor: float

def save_checkpoint(model, optimizer, name, train_cfg, step):
    checkpoint = {
        'model_cfg': model.cfg.__dict__,
        'train_cfg': train_cfg.__dict__,
        'step' : step,
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(checkpoint, f"{name}.pt")

def load_checkpoint(path):
    return torch.load(path)
