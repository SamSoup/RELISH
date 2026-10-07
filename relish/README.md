# RELISH

- [`backbone.py`](backbone.py) defines `FrozenBackbone`, which loads a Hugging
  Face tokenizer (`tok`) and language model (`model`), freezes the model, and
  encodes text into token representations through `encode()`.
- [`modeling.py`](modeling.py) defines `RelishCrossAttentionBlock` and
  `RelishRegressor`. The regressor creates an input projection, learned readout
  tokens, a stack of cross-attention blocks, and an output head that returns
  one scalar per input. Each block refines the readout through attention and
  a feed-forward network with residual connections and layer normalization.

`FrozenBackbone.encode()` supplies the representations to `RelishRegressor`.
The backbone stays frozen while the regression head is trained.

```python
import torch
from relish import FrozenBackbone, RelishRegressor

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = FrozenBackbone(
    "Qwen/Qwen3-8B",
    device=device,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
head = RelishRegressor(d_model=backbone.hidden_size).to(backbone.device)

hidden, mask = backbone.encode(
    ["Rate the similarity from 1 to 4.\nA: A dog runs.\nB: An animal runs.\nScore:"],
    already_prompted=True,
    return_attention_mask=True,
)
predictions = head(hidden, attention_mask=mask)
```
