This repo contains code for training and running autoregressive language models based on the transformer architecture. I mainly wrote this code to teach myself PyTorch and learn more about how large language models work. It should not be used for anything serious. Heavily inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

Current features:

- Basic decoder-only transformer architecture with learned positional embeddings in the style of GPT-2
- RMSNorm
- [Gated feedforward layers](https://arxiv.org/abs/2002.05202)
- [Rotary Positional Embeddings](https://arxiv.org/abs/2104.09864)

Planned:

- Hybrid architectures like [Jamba](https://arxiv.org/pdf/2403.19887)
- Mixture of Experts
