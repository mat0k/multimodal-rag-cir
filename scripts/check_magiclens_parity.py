"""Parity check: our PyTorch MagicLens vs. the original JAX/Flax model.

The two halves need incompatible dependency sets (the reference model pulls in
Scenic, which pins its own JAX/TensorFlow versions), so this runs as separate
subcommands against separate interpreters and exchanges plain `.npz` files.
Imports are deliberately lazy so each subcommand only loads what it needs.

    # 1. fixed inputs (main env -- uses open_clip's tokenizer)
    python3 scripts/check_magiclens_parity.py emit --output /tmp/ml_inputs.npz

    # 2. reference forward pass (isolated env with jax + flax + scenic)
    /path/to/jaxenv/bin/python scripts/check_magiclens_parity.py jax \
        --inputs /tmp/ml_inputs.npz --output /tmp/ml_jax.npz

    # 3. our forward pass + comparison (main env)
    python3 scripts/check_magiclens_parity.py torch \
        --inputs /tmp/ml_inputs.npz --jax /tmp/ml_jax.npz

Both models receive *identical* token ids (tokenised once in step 1) and, for the
embedding comparison, identical pre-processed pixels (taken from the reference's
own pipeline), so the comparison isolates the model from tokeniser and resize
differences. Preprocessing is checked separately, against the reference's output.

Google's README notes that converted weights may differ slightly, so the pass
threshold is cosine >= 0.999 rather than exact equality.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "magiclens_reference"

# CLIP's standard normalisation constants, asserted against the reference in `torch`.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

PROMPTS = [
    "make it blue",
    "a photo of a dog",
    "same but with stripes",
    "",  # candidate/gallery path: the empty instruction
]


def cmd_emit(args: argparse.Namespace) -> int:
    """Build fixed inputs. Runs in the main env (needs open_clip's tokenizer)."""
    import open_clip

    rng = np.random.default_rng(args.seed)
    batch = len(PROMPTS)
    # NHWC in [0, 1]: the layout and range the reference's preprocessing expects.
    images = rng.uniform(0.0, 1.0, size=(batch, 224, 224, 3)).astype(np.float32)

    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    ids = tokenizer(PROMPTS).numpy().astype(np.int32)

    np.savez(args.output, images=images, ids=ids, prompts=np.array(PROMPTS, dtype=object))
    print(f"Wrote {args.output}")
    print(f"  images {images.shape} {images.dtype} in [{images.min():.3f}, {images.max():.3f}]")
    print(f"  ids    {ids.shape} {ids.dtype}")
    return 0


def cmd_jax(args: argparse.Namespace) -> int:
    """Reference forward pass. Runs in the isolated jax+flax+scenic env."""
    import pickle

    import jax.numpy as jnp
    from flax import serialization

    if not REFERENCE_DIR.is_dir():
        print(f"missing reference clone: {REFERENCE_DIR}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(REFERENCE_DIR))

    import jax
    from model import MagicLens  # from magiclens_reference/
    from scenic.projects.baselines.clip import model as clip_model

    data = np.load(args.inputs, allow_pickle=True)
    images = jnp.asarray(data["images"])
    ids = jnp.asarray(data["ids"])

    model = MagicLens(args.model_size)
    params = model.init(
        jax.random.PRNGKey(0),
        {"ids": jnp.ones((1, 1, 77), dtype=jnp.int32),
         "image": jnp.ones((1, 224, 224, 3), dtype=jnp.float32)},
    )
    with open(args.checkpoint, "rb") as handle:
        params = serialization.from_bytes(params, pickle.load(handle))
    print("reference model loaded")

    embeds = model.apply(params, {"ids": ids, "image": images})["multimodal_embed_norm"]

    # Also export what the reference's own preprocessing produced, so the torch
    # side can compare against identical pixels rather than re-deriving them.
    preprocessed = clip_model.normalize_image(images)

    np.savez(args.output,
             embeds=np.asarray(embeds, dtype=np.float32),
             preprocessed=np.asarray(preprocessed, dtype=np.float32))
    print(f"Wrote {args.output}")
    print(f"  embeds {embeds.shape}")
    return 0


def _report(name: str, ours: np.ndarray, theirs: np.ndarray, threshold: float) -> bool:
    """Per-row cosine + max abs diff. Returns True if every row clears threshold."""
    a = ours.astype(np.float64)
    b = theirs.astype(np.float64)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    max_abs = np.abs(a - b).max(-1)

    print(f"\n{name}")
    print(f"  {'row':>4}  {'cosine':>12}  {'max|diff|':>12}   prompt")
    for i, (c, m) in enumerate(zip(cos, max_abs)):
        flag = "ok" if c >= threshold else "FAIL"
        label = PROMPTS[i] if i < len(PROMPTS) else ""
        shown = f'"{label}"' if label else '"" (candidate path)'
        print(f"  {i:>4}  {c:>12.8f}  {m:>12.3e}   {flag:<4} {shown}")
    print(f"  min cosine {cos.min():.8f}   max|diff| {max_abs.max():.3e}")
    return bool((cos >= threshold).all())


def cmd_torch(args: argparse.Namespace) -> int:
    """Our forward pass + comparison. Runs in the main env."""
    import torch

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.retrievers.backbones.magiclens.modeling import MagicLens

    data = np.load(args.inputs, allow_pickle=True)
    reference = np.load(args.jax)
    raw_images = data["images"]              # NHWC, [0, 1]
    ids = data["ids"]
    ref_embeds = reference["embeds"]
    ref_preprocessed = reference["preprocessed"]  # NHWC, normalised by the reference

    ok = True

    # (a) Does our normalisation match the reference's preprocessing?
    ours_preprocessed = (raw_images - CLIP_MEAN) / CLIP_STD
    delta = np.abs(ours_preprocessed - ref_preprocessed).max()
    print(f"preprocessing: max|ours - reference| = {delta:.3e}", end="  ")
    if delta < 1e-4:
        print("(CLIP constants match)")
    else:
        print("(MISMATCH -- our normalisation constants differ from the reference)")
        ok = False

    # (b) Model parity, on the reference's own pixels so only the model differs.
    model = MagicLens(args.model_size)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=True)
    model.eval()

    images_nchw = torch.from_numpy(np.ascontiguousarray(ref_preprocessed.transpose(0, 3, 1, 2)))
    with torch.no_grad():
        ours = model.encode_mm(images_nchw, torch.from_numpy(ids.astype(np.int64))).numpy()

    ok &= _report("embeddings (identical pixels + token ids)", ours, ref_embeds, args.threshold)

    # (c) The candidate path must agree with encode_image on the empty instruction.
    empty_rows = [i for i, p in enumerate(PROMPTS) if p == ""]
    if empty_rows:
        with torch.no_grad():
            candidates = model.encode_image(images_nchw[empty_rows]).numpy()
        gap = np.abs(candidates - ours[empty_rows]).max()
        print(f"\nencode_image vs encode_mm(img, \"\"): max|diff| = {gap:.3e}", end="  ")
        print("(consistent)" if gap < 1e-6 else "(INCONSISTENT)")
        ok &= gap < 1e-6

    print("\nPARITY PASSED" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_emit = sub.add_parser("emit", help="write fixed inputs (main env)")
    p_emit.add_argument("--output", type=Path, required=True)
    p_emit.add_argument("--seed", type=int, default=0)
    p_emit.set_defaults(func=cmd_emit)

    p_jax = sub.add_parser("jax", help="reference forward pass (jax+scenic env)")
    p_jax.add_argument("--inputs", type=Path, required=True)
    p_jax.add_argument("--output", type=Path, required=True)
    p_jax.add_argument("--checkpoint", type=Path,
                       default=Path("models/magiclens/magic_lens_clip_base.pkl"))
    p_jax.add_argument("--model-size", choices=["base", "large"], default="base")
    p_jax.set_defaults(func=cmd_jax)

    p_torch = sub.add_parser("torch", help="our forward pass + comparison (main env)")
    p_torch.add_argument("--inputs", type=Path, required=True)
    p_torch.add_argument("--jax", type=Path, required=True)
    p_torch.add_argument("--weights", type=Path,
                         default=Path("models/magiclens/magic_lens_clip_base.pt"))
    p_torch.add_argument("--model-size", choices=["base", "large"], default="base")
    p_torch.add_argument("--threshold", type=float, default=0.999)
    p_torch.set_defaults(func=cmd_torch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
