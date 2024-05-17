import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import tiktoken
import numpy as np
from dataclasses import dataclass
from math import cos, pi
import signal
import json
import argparse
import sys
import os

import transformer
from config import Config, TrainConfig, save_checkpoint, load_checkpoint

def get_lr(cfg, step):
    lr = cfg.lr
    if step >= cfg.batches:
        return lr * cfg.decay_factor
    if step < cfg.warmup_iters:
        lr *= step / cfg.warmup_iters
    decay = (1 - cfg.decay_factor) * 0.5 * (1 + cos(pi * step / cfg.decay_iters)) + cfg.decay_factor
    return lr * decay

device = "cuda"

train_data_path = "/storage/datasets/owt/train.bin"
val_data_path = "/storage/datasets/owt/val.bin"

# I have enough memory to fit all the data, so no worry about memory leaks
train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

print(f"Training set: {len(train_data)} tokens")

def sample_batch(split, size, seq_len):
    if split == "train":
        data = train_data
    else:
        data = val_data
    indices = torch.randint(len(data) - seq_len - 1, (size,))
    xs = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in indices])
    ys = torch.stack([torch.from_numpy(data[i+1:i+seq_len+1].astype(np.int64)) for i in indices])
    return xs.to(device), ys.to(device)

def validate(model):
    model.eval()
    loss = 0.0
    minibatches = 100
    for i in range(minibatches):
        xs, ys = sample_batch("val", 10, model.cfg.ctx_len)
        ypred = model(xs)
        loss += F.cross_entropy(ypred.view(-1, ypred.size(-1)), ys.view(-1)).item()
    return loss / minibatches

pause_training = False

def pause_handler(number, frame):
    global pause_training
    pause_training = True

def save_model(model, name):
    torch.save(model.state_dict(), f"{name}.pth")


def train(model, opt, name, train_cfg, interval=1000, step=0, log_dir=None):
    global pause_training
    if log_dir is not None:
        writer = SummaryWriter(os.path.join('runs', log_dir))
    model.train()

    print(f"Training {train_cfg.batches} batches of size {train_cfg.batch_size * train_cfg.minibatch_size} ({train_cfg.batch_size * train_cfg.minibatch_size * model.cfg.ctx_len * train_cfg.batches} tokens)")

    for i in range(step, train_cfg.batches):
        total_tokens = i * train_cfg.batch_size * train_cfg.minibatch_size * model.cfg.ctx_len

        lr = get_lr(train_cfg, i)
        for pg in opt.param_groups:
            pg['lr'] = lr

        batch_loss = 0.0
        opt.zero_grad()
        for j in range(train_cfg.batch_size):
            xs, ys = sample_batch("train", train_cfg.minibatch_size, model.cfg.ctx_len)
            ypred = model(xs)
            loss = F.cross_entropy(ypred.view(-1, ypred.size(-1)), ys.view(-1)) / train_cfg.batch_size
            batch_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)

        if i % interval == 0:
            print(f"batch {i}\tloss: {batch_loss}")

        if log_dir is not None:
            writer.add_scalar("training loss/step", batch_loss, i)
            writer.add_scalar("learning rate/step", opt.param_groups[0]['lr'], i)

        opt.step()

        if pause_training:
            if log_dir is not None:
                writer.flush()
            print("Signal received, saving checkpoint")
            save_checkpoint(model, opt, name+f"-step{i+1}", train_cfg, i+1)
            pause_training = False

    if log_dir is not None:
        writer.flush()

    save_checkpoint(model, opt, name, train_cfg, i+1)

def main():
    train_cfg = TrainConfig(
        batch_size = 30,
        minibatch_size = 30,
        batches = 30000,
        warmup_iters = 1000,
        decay_iters = 30000,
        lr = 4e-4,
        decay_factor = 0.1
    )

    model_cfg = transformer.Config(
        d_model = 768,
        d_mlp = 2048,
        d_vocab = 50304,
        d_head = 64,
        n_heads = 12,
        n_layers = 12,
        layer_norm_eps = 1e-8,
        ctx_len = 512,
        rmsnorm = True,
        swiglu = True
    )

    step = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-l', '--log')
    parser.add_argument('name')
    args = parser.parse_args()

    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint)
        model_cfg = Config(**checkpoint['model_cfg'])
        train_cfg = TrainConfig(**checkpoint['train_cfg'])
        step = checkpoint['step']

    model = transformer.TransformerModel(model_cfg).to(device="cuda", dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), train_cfg.lr)

    if args.checkpoint is not None:
       model.load_state_dict(checkpoint['model'])
       opt.load_state_dict(checkpoint['optimizer'])

    train(model, opt, args.name, train_cfg, interval=100, step=step, log_dir=args.log)

signal.signal(signal.SIGUSR1, pause_handler)
main()
