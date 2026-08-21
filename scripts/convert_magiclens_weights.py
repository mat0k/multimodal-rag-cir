"""Convert the official MagicLens JAX/Flax checkpoint into a PyTorch state_dict.

The official release (https://github.com/google-deepmind/magiclens) ships its
weights as a pickled msgpack blob of Flax params. This script decodes that blob
and re-lays every tensor out to match `src.retrievers.backbones.magiclens`.

Usage:
    python scripts/convert_magiclens_weights.py \
        --checkpoint models/magiclens/magic_lens_clip_base.pkl \
        --model-size base \
        --output models/magiclens/magic_lens_clip_base.pt

Nothing is written unless every source tensor is consumed exactly once and every
destination key is filled with a matching shape, so a silent partial conversion
cannot occur. Conversion passing does NOT by itself prove numerical
correctness -- run the parity check against the original JAX model for that.

Layout notes (the parts that are easy to get wrong):

* Flax `Dense` kernels are `(in, out)`; `torch.nn.Linear.weight` is `(out, in)`,
  so plain feed-forward kernels are transposed.
* Attention q/k/v kernels are `(D, N, H)` and flatten to `(N*H, D)` after a
  transpose, matching the `...D,DNH->...NH` einsum in the reference.
* Attention *output* projections differ between the two halves of the model and
  must not be treated uniformly:
    - Scenic CLIP stores `attn/out/kernel` as `(N, H, D)` (`...NH,NHD->...D`),
      so it flattens to `(N*H, D)` and is then transposed.
    - MagicLens's own blocks store `self_attention/post/w` as `(D, N, H)`
      (`...NH,DNH->...D`), so it reshapes straight to `(D, N*H)` with no
      transpose.
  Both land on `(D, N*H)`, but via different routes; swapping them yields a
  model that runs and silently returns wrong embeddings.
* `open_clip` fuses q/k/v into a single `in_proj_weight`, so three source
  tensors collapse into one destination tensor (and likewise for biases). This
  is why the source has 96 more leaves than the destination has keys.
* Scenic CLIP's LayerNorm is the standard formulation, so `scale` maps onto
  `weight` directly. MagicLens's own `MLLayerNorm` applies `(1 + scale)`; our
  port stores the raw `scale` and applies the `1 +` at runtime, so those also
  copy across directly.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from flax import serialization

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrievers.backbones.magiclens.modeling import MagicLens  # noqa: E402


def _to_torch(array) -> torch.Tensor:
    # copy(): the restored arrays are read-only, which torch.from_numpy warns about.
    return torch.from_numpy(np.asarray(array).copy()).float()


def _flatten(tree: Dict, prefix: str = "") -> Dict[str, np.ndarray]:
    """Flatten Flax's nested param dict into '/'-joined paths."""
    flat: Dict[str, np.ndarray] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


class _Source:
    """Flat view of the JAX params that tracks which tensors were consumed."""

    def __init__(self, flat: Dict[str, np.ndarray]):
        self._flat = flat
        self._used: set = set()

    def take(self, key: str) -> torch.Tensor:
        if key not in self._flat:
            raise KeyError(f"checkpoint is missing expected parameter: {key}")
        if key in self._used:
            raise RuntimeError(f"parameter consumed twice: {key}")
        self._used.add(key)
        return _to_torch(self._flat[key])

    def count_indexed(self, prefix: str, infix: str) -> int:
        """Count e.g. how many `resblocks_{i}` live under a prefix."""
        seen = set()
        for key in self._flat:
            if not key.startswith(prefix):
                continue
            for part in key[len(prefix):].split("/"):
                if part.startswith(infix) and part[len(infix):].isdigit():
                    seen.add(int(part[len(infix):]))
        return len(seen)

    def unused(self) -> list:
        return sorted(set(self._flat) - self._used)


def _convert_clip_resblock(src: _Source, jax_prefix: str, torch_prefix: str,
                           width: int, out: Dict[str, torch.Tensor]) -> None:
    """One Scenic-CLIP residual block -> one open_clip residual block."""
    out[f"{torch_prefix}.ln_1.weight"] = src.take(f"{jax_prefix}/ln_1/scale")
    out[f"{torch_prefix}.ln_1.bias"] = src.take(f"{jax_prefix}/ln_1/bias")

    # q/k/v are separate in Flax but fused in open_clip's nn.MultiheadAttention.
    kernels, biases = [], []
    for name in ("query", "key", "value"):
        kernel = src.take(f"{jax_prefix}/attn/{name}/kernel")  # (D, N, H)
        bias = src.take(f"{jax_prefix}/attn/{name}/bias")      # (N, H)
        kernels.append(kernel.reshape(width, -1).t())          # (N*H, D)
        biases.append(bias.reshape(-1))                        # (N*H,)
    out[f"{torch_prefix}.attn.in_proj_weight"] = torch.cat(kernels, dim=0)
    out[f"{torch_prefix}.attn.in_proj_bias"] = torch.cat(biases, dim=0)

    # Scenic stores the out projection as (N, H, D) -- flatten then transpose.
    out_kernel = src.take(f"{jax_prefix}/attn/out/kernel")
    out[f"{torch_prefix}.attn.out_proj.weight"] = out_kernel.reshape(-1, width).t()
    out[f"{torch_prefix}.attn.out_proj.bias"] = src.take(f"{jax_prefix}/attn/out/bias")

    out[f"{torch_prefix}.ln_2.weight"] = src.take(f"{jax_prefix}/ln_2/scale")
    out[f"{torch_prefix}.ln_2.bias"] = src.take(f"{jax_prefix}/ln_2/bias")

    out[f"{torch_prefix}.mlp.c_fc.weight"] = src.take(f"{jax_prefix}/mlp/c_fc/kernel").t()
    out[f"{torch_prefix}.mlp.c_fc.bias"] = src.take(f"{jax_prefix}/mlp/c_fc/bias")
    out[f"{torch_prefix}.mlp.c_proj.weight"] = src.take(f"{jax_prefix}/mlp/c_proj/kernel").t()
    out[f"{torch_prefix}.mlp.c_proj.bias"] = src.take(f"{jax_prefix}/mlp/c_proj/bias")


def _convert_ml_attention(src: _Source, jax_prefix: str, torch_prefix: str,
                          input_dim: int, out: Dict[str, torch.Tensor]) -> None:
    """MagicLens DotProductAttention -> our MLMultiHeadAttention."""
    for jax_name, torch_name in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
        kernel = src.take(f"{jax_prefix}/{jax_name}/w")  # (D, N, H)
        bias = src.take(f"{jax_prefix}/{jax_name}/b")    # (N, H)
        out[f"{torch_prefix}.{torch_name}.weight"] = kernel.reshape(input_dim, -1).t()
        out[f"{torch_prefix}.{torch_name}.bias"] = bias.reshape(-1)

    # MagicLens stores `post` as (D, N, H) -- reshape only, NO transpose.
    post = src.take(f"{jax_prefix}/post/w")
    out[f"{torch_prefix}.out_proj.weight"] = post.reshape(input_dim, -1)
    out[f"{torch_prefix}.out_proj.bias"] = src.take(f"{jax_prefix}/post/b")


def convert(flat: Dict[str, np.ndarray], embed_dim: int,
            vision_width: int) -> Tuple[Dict[str, torch.Tensor], _Source]:
    src = _Source(flat)
    out: Dict[str, torch.Tensor] = {}

    out["clip.logit_scale"] = src.take("clip/logit_scale")

    # ---- CLIP text tower ----
    out["clip.token_embedding.weight"] = src.take("clip/text/token_embedding/embedding")
    out["clip.positional_embedding"] = src.take("clip/text/positional_embedding")
    out["clip.ln_final.weight"] = src.take("clip/text/ln_final/scale")
    out["clip.ln_final.bias"] = src.take("clip/text/ln_final/bias")
    # open_clip applies this as `x @ text_projection`, matching Flax's (in, out).
    out["clip.text_projection"] = src.take("clip/text/text_projection/kernel")

    n_text = src.count_indexed("clip/text/transformer/", "resblocks_")
    for i in range(n_text):
        _convert_clip_resblock(
            src,
            f"clip/text/transformer/resblocks_{i}",
            f"clip.transformer.resblocks.{i}",
            embed_dim,
            out,
        )

    # ---- CLIP vision tower ----
    # Flax conv kernels are (KH, KW, Cin, Cout); torch wants (Cout, Cin, KH, KW).
    out["clip.visual.conv1.weight"] = src.take("clip/visual/conv1/kernel").permute(3, 2, 0, 1).contiguous()
    out["clip.visual.class_embedding"] = src.take("clip/visual/class_embedding")
    out["clip.visual.positional_embedding"] = src.take("clip/visual/positional_embedding")
    out["clip.visual.ln_pre.weight"] = src.take("clip/visual/ln_pre/scale")
    out["clip.visual.ln_pre.bias"] = src.take("clip/visual/ln_pre/bias")
    out["clip.visual.ln_post.weight"] = src.take("clip/visual/ln_post/scale")
    out["clip.visual.ln_post.bias"] = src.take("clip/visual/ln_post/bias")
    out["clip.visual.proj"] = src.take("clip/visual/proj/kernel")

    n_vision = src.count_indexed("clip/visual/transformer/", "resblocks_")
    for i in range(n_vision):
        _convert_clip_resblock(
            src,
            f"clip/visual/transformer/resblocks_{i}",
            f"clip.visual.transformer.resblocks.{i}",
            vision_width,
            out,
        )

    # ---- MagicLens multimodal encoder ----
    n_layers = src.count_indexed("multimodal_encoder/", "x_layers_")
    for i in range(n_layers):
        jax_prefix = f"multimodal_encoder/x_layers_{i}"
        torch_prefix = f"multimodal_encoder.layers.{i}"

        out[f"{torch_prefix}.layer_norm.scale"] = src.take(f"{jax_prefix}/layer_norm/scale")
        out[f"{torch_prefix}.layer_norm.bias"] = src.take(f"{jax_prefix}/layer_norm/bias")

        _convert_ml_attention(
            src, f"{jax_prefix}/self_attention", f"{torch_prefix}.self_attention", embed_dim, out
        )

        out[f"{torch_prefix}.ff_layer.ln.scale"] = src.take(f"{jax_prefix}/ff_layer/layer_norm/scale")
        out[f"{torch_prefix}.ff_layer.ln.bias"] = src.take(f"{jax_prefix}/ff_layer/layer_norm/bias")
        out[f"{torch_prefix}.ff_layer.ffn1.weight"] = src.take(f"{jax_prefix}/ff_layer/ffn_layer1/linear/w").t()
        out[f"{torch_prefix}.ff_layer.ffn1.bias"] = src.take(f"{jax_prefix}/ff_layer/ffn_layer1/bias/b")
        out[f"{torch_prefix}.ff_layer.ffn2.weight"] = src.take(f"{jax_prefix}/ff_layer/ffn_layer2/linear/w").t()
        out[f"{torch_prefix}.ff_layer.ffn2.bias"] = src.take(f"{jax_prefix}/ff_layer/ffn_layer2/bias/b")

    # ---- MagicLens pooler ----
    out["contrastive_multimodal_pooler.pooling_attn_query"] = src.take(
        "contrastive_multimodal_pooler/pooling_attn_query"
    )
    out["contrastive_multimodal_pooler.pool_attn.per_dim_scale"] = src.take(
        "contrastive_multimodal_pooler/pool_attn/per_dim_scale/per_dim_scale"
    )
    _convert_ml_attention(
        src,
        "contrastive_multimodal_pooler/pool_attn",
        "contrastive_multimodal_pooler.pool_attn",
        embed_dim,
        out,
    )
    out["contrastive_multimodal_pooler.pool_attn_ln.scale"] = src.take(
        "contrastive_multimodal_pooler/pool_attn_ln/scale"
    )
    out["contrastive_multimodal_pooler.pool_attn_ln.bias"] = src.take(
        "contrastive_multimodal_pooler/pool_attn_ln/bias"
    )

    return out, src


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/magiclens/magic_lens_clip_base.pkl"),
                        help="Official MagicLens .pkl checkpoint.")
    parser.add_argument("--model-size", choices=["base", "large"], default="base")
    parser.add_argument("--output", type=Path, default=None,
                        help="Destination .pt file (defaults to the checkpoint path with a .pt suffix).")
    args = parser.parse_args()

    output = args.output or args.checkpoint.with_suffix(".pt")

    print(f"Loading {args.checkpoint} ...")
    with open(args.checkpoint, "rb") as handle:
        model_bytes = pickle.load(handle)
    flat = _flatten(serialization.msgpack_restore(model_bytes)["params"])
    print(f"  {len(flat)} source tensors, "
          f"{sum(int(np.prod(v.shape)) for v in flat.values()):,} parameters")

    print(f"Building PyTorch MagicLens('{args.model_size}') for reference ...")
    model = MagicLens(args.model_size)
    reference = model.state_dict()
    embed_dim = model.hidden_dim
    vision_width = reference["clip.visual.class_embedding"].shape[0]

    print("Converting ...")
    converted, src = convert(flat, embed_dim, vision_width)

    # ---- strict validation, before anything touches disk ----
    problems = []

    leftover = src.unused()
    if leftover:
        problems.append(f"{len(leftover)} source tensor(s) never consumed, e.g. {leftover[:5]}")

    missing = sorted(set(reference) - set(converted))
    if missing:
        problems.append(f"{len(missing)} destination key(s) never filled, e.g. {missing[:5]}")

    unexpected = sorted(set(converted) - set(reference))
    if unexpected:
        problems.append(f"{len(unexpected)} converted key(s) absent from the model, e.g. {unexpected[:5]}")

    mismatched = [
        (key, tuple(converted[key].shape), tuple(reference[key].shape))
        for key in sorted(set(converted) & set(reference))
        if converted[key].shape != reference[key].shape
    ]
    if mismatched:
        problems.append(f"{len(mismatched)} shape mismatch(es), e.g. {mismatched[:3]}")

    if problems:
        print("\nCONVERSION FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    # Round-trip through the real module so load_state_dict itself signs off.
    model.load_state_dict(converted, strict=True)

    total = sum(int(np.prod(v.shape)) for v in converted.values())
    print(f"  all {len(flat)} source tensors consumed")
    print(f"  all {len(reference)} destination keys filled, shapes match")
    print(f"  load_state_dict(strict=True) OK -- {total:,} parameters")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    size_mb = output.stat().st_size / 1024 ** 2
    print(f"\nWrote {output} ({size_mb:.1f} MB)")
    print("Next: run the parity check against the original JAX model before trusting these weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
