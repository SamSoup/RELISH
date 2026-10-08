# RELISH

This repository contains the source code for [RELISH: LLM REgression with a Latent Iterative State Head](https://openreview.net/forum?id=IebWFoQwdz), published as a conference paper at COLM 2026.

[Poster (PDF)](assets/RELISH_COLM_2026_Poster.pdf)

- <ins>WHAT:</ins> RELISH is a lightweight text regression architecture that predicts scalar values directly from frozen LLM representations.
- <ins>HOW:</ins> RELISH iteratively refines a learned latent state via cross-attention over token-level representations, then maps the final state to a point estimate with a linear regressor.
- <ins>HOW WELL:</ins> Over 6 datasets, 4 LLMs, & 2 LLM training regimes, RELISH tops baselines from all 3 major LLM regression families while using only ∼3.4–3.7M trainable parameters.

<p align="center">
  <a href="https://www.cosmicai.org/">
    <img src="assets/cosmicai-logo.png" alt="NSF-Simons AI Institute for Cosmic Origins logo" width="420">
  </a>
</p>

## Install

Requires Python 3.9 or newer, PyTorch, and Hugging Face Transformers.

```bash
git clone https://github.com/SamSoup/RELISH.git
cd RELISH
python -m pip install -e .
```

## Quick start

```python
import torch
from relish import FrozenBackbone, RelishRegressor

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = FrozenBackbone(
    "Qwen/Qwen3-8B",
    device=device,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
prompts = [
    "Rate the similarity from 1 to 4.\nA: A dog runs.\nB: An animal runs.\nScore:",
    "Rate the similarity from 1 to 4.\nA: It is sunny.\nB: It is raining.\nScore:",
]
hidden, attention_mask = backbone.encode(
    prompts,
    already_prompted=True,
    max_length=256,
    return_attention_mask=True,
)
model = RelishRegressor(d_model=backbone.hidden_size).to(backbone.device)
predictions = model(hidden, attention_mask=attention_mask)
```

## Acknowledgments

We appreciate the valuable feedback we received on this manuscript from our
anonymous reviewers, Junyi Jessy Li, Leqi Liu, and members of the Artificial
Intelligence and Human-Centered Computing (AI&HCC) lab at UT Austin. This work
was supported by the NSF under Cooperative Agreement 2421782 and the Simons
Foundation grant MPS-AI-00010515, awarded to the
[NSF-Simons AI Institute for Cosmic Origins (CosmicAI)](https://www.cosmicai.org/).
This research was also supported by computational resources provided by the Texas
Advanced Computing Center (TACC) at The University of Texas at Austin. The
statements made herein are solely the opinions of the authors and do not reflect
the views of the sponsoring agencies.

## Citation

```bibtex
@inproceedings{su2026relish,
  title = {RELISH: LLM REgression with a Latent Iterative State Head},
  author = {Su, Yiheng and Lease, Matthew},
  booktitle = {Proceedings of the Third Conference on Language Modeling},
  year = {2026},
  url = {https://arxiv.org/abs/2604.01206}
}
```
