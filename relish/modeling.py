from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_D_H = 256
DEFAULT_N_HEADS = 8
DEFAULT_N_LAYERS = 3
DEFAULT_NUM_READOUT_TOKENS = 1
DEFAULT_HEAD_TYPE = "linear"
DEFAULT_HIDDEN_LAYER_POOLING = "single"
READOUT_INIT_MODES = (
    "learned",
    "first_token",
    "last_token",
    "mean_pool",
    "max_pool",
)


class RelishCrossAttentionBlock(nn.Module):
    """
    Cross-attention update block for readout tokens over query token context.
    """

    def __init__(self, d_h: int, n_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_h)
        self.ff = nn.Sequential(
            nn.Linear(d_h, 4 * d_h),
            nn.GELU(),
            nn.Linear(4 * d_h, d_h),
        )
        self.ln2 = nn.LayerNorm(d_h)

    def forward(self, readout: torch.Tensor, query_ctx: torch.Tensor) -> torch.Tensor:
        update, _ = self.attn(readout, query_ctx, query_ctx, need_weights=False)
        readout = self.ln1(readout + update)
        readout = self.ln2(readout + self.ff(readout))
        return readout


class RelishRegressor(nn.Module):
    """
    RELISH readout regressor:
    query hidden states -> iterative readout token updates -> scalar prediction.

    The paper configuration uses the defaults below: one learned readout token,
    three cross-attention blocks, and a linear scalar head. Non-default readout
    counts and MLP heads are kept here for the copied ablation scripts.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_h: int = DEFAULT_D_H,
        n_heads: int = DEFAULT_N_HEADS,
        n_layers: int = DEFAULT_N_LAYERS,
        num_readout_tokens: int = DEFAULT_NUM_READOUT_TOKENS,
        head_type: str = DEFAULT_HEAD_TYPE,
        hidden_layer_pooling: str = DEFAULT_HIDDEN_LAYER_POOLING,
        num_source_layers: int = 1,
        readout_init: str = "learned",
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if num_readout_tokens < 1:
            raise ValueError(
                f"num_readout_tokens must be >= 1, got {num_readout_tokens}"
            )
        if num_source_layers < 1:
            raise ValueError(f"num_source_layers must be >= 1, got {num_source_layers}")

        self.d_model = int(d_model)
        self.d_h = int(d_h)
        self.n_layers = int(n_layers)
        self.num_readout_tokens = int(num_readout_tokens)
        self.head_type = str(head_type)
        self.hidden_layer_pooling = str(hidden_layer_pooling)
        self.num_source_layers = int(num_source_layers)
        self.readout_init = str(readout_init)

        if self.hidden_layer_pooling not in {"single", "mean", "learned"}:
            raise ValueError(
                f"Unsupported hidden_layer_pooling: {self.hidden_layer_pooling}"
            )
        if self.hidden_layer_pooling == "learned":
            self.hidden_layer_logits = nn.Parameter(torch.zeros(self.num_source_layers))
        else:
            self.register_parameter("hidden_layer_logits", None)
        if self.readout_init not in READOUT_INIT_MODES:
            raise ValueError(f"Unsupported readout_init: {self.readout_init}")

        self.query_proj = nn.Linear(self.d_model, self.d_h)
        self.readout_tokens = nn.Parameter(
            torch.randn(1, self.num_readout_tokens, self.d_h) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                RelishCrossAttentionBlock(d_h=self.d_h, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )

        if self.head_type == "linear":
            self.head = nn.Linear(self.d_h, 1)
        elif self.head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.d_h, self.d_h),
                nn.GELU(),
                nn.Linear(self.d_h, 1),
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    def _pool_readout(self, readout: torch.Tensor) -> torch.Tensor:
        if readout.size(1) == 1:
            return readout[:, 0, :]
        return readout.mean(dim=1)

    def _pool_source_layers(self, query_hidden) -> torch.Tensor:
        if isinstance(query_hidden, (list, tuple)):
            if self.hidden_layer_pooling != "learned":
                raise ValueError(
                    "Sequence query_hidden is only expected with "
                    "hidden_layer_pooling='learned'"
                )
            layer_count = len(query_hidden)
            if layer_count != self.num_source_layers:
                raise ValueError(
                    f"Expected {self.num_source_layers} source layers, got {layer_count}"
                )
            weights = torch.softmax(self.hidden_layer_logits, dim=0)
            acc = None
            for weight, layer_hidden in zip(weights, query_hidden):
                term = layer_hidden.float() * weight
                acc = term if acc is None else acc + term
            if acc is None:
                raise ValueError("query_hidden sequence is empty")
            return acc

        if query_hidden.dim() == 3:
            return query_hidden
        if query_hidden.dim() != 4:
            raise ValueError(
                "query_hidden must have shape [B, S, D] or [B, L, S, D], "
                f"got {tuple(query_hidden.shape)}"
            )
        if self.hidden_layer_pooling != "learned":
            raise ValueError(
                "4D query_hidden is only expected with hidden_layer_pooling='learned'"
            )
        layer_count = query_hidden.size(1)
        if layer_count != self.num_source_layers:
            raise ValueError(
                f"Expected {self.num_source_layers} source layers, got {layer_count}"
            )
        weights = torch.softmax(self.hidden_layer_logits, dim=0).to(query_hidden.dtype)
        return (query_hidden * weights.view(1, layer_count, 1, 1)).sum(dim=1)

    def _first_valid_index(self, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=torch.long)
        first = mask.argmax(dim=1)
        return first.clamp(min=0)

    def _last_valid_index(self, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=torch.long)
        positions = torch.arange(mask.size(1), device=mask.device).view(1, -1)
        return (mask * positions).max(dim=1).values.clamp(min=0)

    def _gather_token(self, query_ctx: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch_index = torch.arange(query_ctx.size(0), device=query_ctx.device)
        return query_ctx[batch_index, index.to(query_ctx.device), :]

    def _masked_mean(self, query_ctx: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(device=query_ctx.device, dtype=query_ctx.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (query_ctx * mask).sum(dim=1) / denom

    def _masked_max(self, query_ctx: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask.to(device=query_ctx.device, dtype=torch.bool).unsqueeze(-1)
        floor = torch.finfo(query_ctx.dtype).min
        pooled = query_ctx.masked_fill(~valid, floor).max(dim=1).values
        empty = ~valid.squeeze(-1).any(dim=1)
        if empty.any():
            pooled[empty] = query_ctx[empty].max(dim=1).values
        return pooled

    def _warm_start_vector(
        self,
        query_ctx: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.readout_init == "first_token":
            index = (
                self._first_valid_index(attention_mask)
                if attention_mask is not None
                else torch.zeros(query_ctx.size(0), device=query_ctx.device, dtype=torch.long)
            )
            return self._gather_token(query_ctx, index)
        if self.readout_init == "last_token":
            index = (
                self._last_valid_index(attention_mask)
                if attention_mask is not None
                else torch.full(
                    (query_ctx.size(0),),
                    query_ctx.size(1) - 1,
                    device=query_ctx.device,
                    dtype=torch.long,
                )
            )
            return self._gather_token(query_ctx, index)
        if self.readout_init == "mean_pool":
            return (
                self._masked_mean(query_ctx, attention_mask)
                if attention_mask is not None
                else query_ctx.mean(dim=1)
            )
        if self.readout_init == "max_pool":
            return (
                self._masked_max(query_ctx, attention_mask)
                if attention_mask is not None
                else query_ctx.max(dim=1).values
            )
        raise ValueError(f"Unsupported readout_init: {self.readout_init}")

    def _init_readout(
        self,
        query_ctx: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = query_ctx.size(0)
        if self.readout_init == "learned":
            return self.readout_tokens.expand(batch_size, -1, -1)
        warm_start = self._warm_start_vector(query_ctx, attention_mask)
        return warm_start.unsqueeze(1).expand(-1, self.num_readout_tokens, -1)

    def forward(
        self,
        query_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_hidden = self._pool_source_layers(query_hidden)
        query_ctx = self.query_proj(query_hidden.float())
        readout = self._init_readout(query_ctx, attention_mask)
        for layer in self.layers:
            readout = layer(readout, query_ctx)
        pooled = self._pool_readout(readout)
        return self.head(pooled).squeeze(-1)


# Compatibility aliases for copied ablation scripts and historical checkpoints.
ReadoutCrossBlock = RelishCrossAttentionBlock
M0ReadoutRegressor = RelishRegressor
