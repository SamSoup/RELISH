from __future__ import annotations

import os

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_PROMPT_TMPL = "Input:\n{text}\n\nPredict a score from 1 to 4:"
HIDDEN_LAYER_SOURCES = (
    "first",
    "last",
    "penultimate",
    "quarter",
    "middle",
    "three_quarter",
    "last4",
    "all",
)
HIDDEN_LAYER_POOLINGS = ("single", "mean", "learned")


def ensure_hf_cache(cache_dir: str | None = None) -> str | None:
    """
    Optionally set Hugging Face cache variables for model and tokenizer downloads.

    If ``cache_dir`` is omitted, Hugging Face uses its normal defaults.
    """
    if cache_dir is None:
        return os.environ.get("HF_HOME")
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    return os.environ["HF_HOME"]


def _batch_iter(values: list[str], batch_size: int):
    step = max(int(batch_size), 1)
    for start in range(0, len(values), step):
        yield values[start : start + step]


def audit_tokenized_text_lengths(
    tokenizer,
    texts,
    *,
    max_length: int,
    batch_size: int = 64,
    add_special_tokens: bool = True,
    context: str,
    allow_truncation: bool = False,
) -> dict[str, int]:
    """Fail fast before hidden-state extraction silently truncates prompts."""
    values = [str(text) for text in texts]
    lengths: list[int] = []
    for batch in _batch_iter(values, batch_size):
        encoded = tokenizer(
            batch,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    max_seen = max(lengths) if lengths else 0
    num_exceeding = sum(length > int(max_length) for length in lengths)
    stats = {
        "n": len(lengths),
        "max": int(max_seen),
        "max_length": int(max_length),
        "num_exceeding_max_length": int(num_exceeding),
    }
    if num_exceeding and not allow_truncation:
        raise ValueError(
            f"{context} would truncate {num_exceeding}/{len(lengths)} examples "
            f"at max_length={int(max_length)}. Increase max_length or pass "
            "allow_truncation=True for an explicit truncated run."
        )
    return stats


def resolve_hidden_layer_indices(num_hidden_layers: int, source: str) -> tuple[int, ...]:
    """Return hidden-state tuple indices, excluding the embedding state at index 0."""
    n_layers = int(num_hidden_layers)
    if n_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

    source = str(source).strip().lower()
    if source == "first":
        return (1,)
    if source == "last":
        return (n_layers,)
    if source == "penultimate":
        return (max(1, n_layers - 1),)
    if source == "quarter":
        return (max(1, round(n_layers * 0.25)),)
    if source == "middle":
        return (max(1, round(n_layers * 0.50)),)
    if source == "three_quarter":
        return (max(1, round(n_layers * 0.75)),)
    if source == "last4":
        start = max(1, n_layers - 3)
        return tuple(range(start, n_layers + 1))
    if source == "all":
        return tuple(range(1, n_layers + 1))
    raise ValueError(
        f"Unsupported hidden_layer_source={source!r}; "
        f"expected one of {', '.join(HIDDEN_LAYER_SOURCES)}"
    )


def select_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    *,
    source: str = "last",
    pooling: str = "single",
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Select or aggregate model hidden states for RELISH readout inputs."""
    num_hidden_layers = len(hidden_states) - 1
    indices = resolve_hidden_layer_indices(num_hidden_layers, source)
    selected = [hidden_states[index] for index in indices]

    pooling = str(pooling).strip().lower()
    if pooling == "single":
        if len(selected) != 1:
            raise ValueError(
                f"hidden_layer_pooling='single' requires a single source layer, "
                f"but hidden_layer_source={source!r} resolved to {len(selected)} layers"
            )
        return selected[0]
    if pooling == "mean":
        if len(selected) == 1:
            return selected[0]
        acc = selected[0].float()
        for state in selected[1:]:
            acc = acc + state.float()
        return acc / float(len(selected))
    if pooling == "learned":
        return tuple(selected)
    raise ValueError(
        f"Unsupported hidden_layer_pooling={pooling!r}; "
        f"expected one of {', '.join(HIDDEN_LAYER_POOLINGS)}"
    )


def _config_num_hidden_layers(config) -> int | None:
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is None:
            continue
        for attr in ("num_hidden_layers", "n_layers", "num_layers"):
            value = getattr(candidate, attr, None)
            if value is not None:
                return int(value)
    return None


def get_hidden_size(model) -> int:
    """Return the hidden size for common Hugging Face model configs."""
    config = model.config
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is None:
            continue
        for attr in ("hidden_size", "dim", "d_model", "model_dim"):
            value = getattr(candidate, attr, None)
            if value is not None:
                return int(value)
    if hasattr(model, "get_input_embeddings"):
        return int(model.get_input_embeddings().weight.shape[-1])
    raise AttributeError(f"Could not determine hidden size for {type(model).__name__}")


class FrozenBackbone:
    """Wrapper around a frozen Hugging Face backbone for RELISH hidden states."""

    def __init__(
        self,
        model_name: str,
        device: str | torch.device | None = None,
        *,
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | None = None,
        trust_remote_code: bool = False,
    ):
        hf_home = ensure_hf_cache(cache_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cache_kwargs = {"cache_dir": hf_home} if hf_home else {}
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            **cache_kwargs,
        )
        added_pad_token = False
        if self.tok.pad_token is None:
            if self.tok.eos_token is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                self.tok.add_special_tokens({"pad_token": "[PAD]"})
                added_pad_token = True
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            **cache_kwargs,
        )
        if added_pad_token:
            self.model.resize_token_embeddings(len(self.tok))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)

    def encode(
        self,
        texts,
        max_length: int = 256,
        already_prompted: bool = False,
        return_attention_mask: bool = False,
        allow_truncation: bool = False,
        hidden_layer_source: str = "last",
        hidden_layer_pooling: str = "single",
    ):
        """
        Tokenize and encode a list of prompts.

        If already_prompted=False, wraps raw text with the default template.
        If return_attention_mask=True, returns (hidden_states, attention_mask).
        """
        if already_prompted:
            prompts = list(texts)
        else:
            prompts = [DEFAULT_PROMPT_TMPL.format(text=t) for t in texts]
        audit_tokenized_text_lengths(
            self.tok,
            prompts,
            max_length=int(max_length),
            batch_size=len(prompts) or 1,
            add_special_tokens=True,
            context=f"FrozenBackbone.encode[{self.model_name}]",
            allow_truncation=bool(allow_truncation),
        )
        batch = self.tok(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch, output_hidden_states=True)
        hidden = select_hidden_states(
            out.hidden_states,
            source=hidden_layer_source,
            pooling=hidden_layer_pooling,
        )
        if return_attention_mask:
            return hidden, batch["attention_mask"]
        return hidden

    @property
    def hidden_size(self) -> int:
        return get_hidden_size(self.model)

    def selected_hidden_layer_count(self, source: str) -> int:
        num_hidden_layers = _config_num_hidden_layers(self.model.config)
        if num_hidden_layers is None:
            raise ValueError(
                f"Cannot infer num_hidden_layers from config for {self.model_name}"
            )
        return len(resolve_hidden_layer_indices(num_hidden_layers, source))


# Historical name used by the experiment scripts.
FrozenLlama = FrozenBackbone
