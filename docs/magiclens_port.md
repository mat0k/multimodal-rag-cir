# Porting MagicLens from JAX/Flax to PyTorch

Working notes on reimplementing MagicLens (Zhang et al., ICML'24) in PyTorch and
converting the official checkpoint. Google released the model as JAX/Flax with a
Scenic CLIP backbone; to our knowledge no PyTorch port existed when this work
started, so these notes record the parameter mapping and the non-obvious details
in enough depth to be reproducible.

- Upstream: <https://github.com/google-deepmind/magiclens>
- Reference clone (gitignored, read-only): `magiclens_reference/`
- Our port: `src/retrievers/backbones/magiclens/{layers,modeling}.py`
- Converter: `scripts/convert_magiclens_weights.py`

## Why port it

MagicLens is a candidate *student* for our similarity-preserving (SP) relation
distillation experiments: a CLIP-based dual encoder with the same retrieval shape
as VISTA, trained on 36.7M web-mined triplets, and — unlike LamRA-Ret — with no
Fashion-IQ/CIRR contamination. Running our pipeline in PyTorch means the model has
to exist in PyTorch.

## Architecture

Query and candidate both flow through the *same* three stages; only the text input
differs.

```
image ──> CLIP vision tower ──┐
                              ├─> [B,2,D] ─> multimodal_encoder ─> pooler ─> L2 ─> [B,D]
text  ──> CLIP text tower ────┘             (4 transformer layers)  (attn pool)
```

A candidate image is encoded with an **empty instruction** (`tokenizer("")`), not
as a bare CLIP image embedding. This mirrors the official eval protocol
(`magiclens_reference/data_utils.py`); shortcutting it to a vision-only path breaks
both parity and protocol alignment. Our `encode_image()` therefore routes through
the identical fusion path as `encode_mm()`.

Sizes (`base`): `embed_dim=512`, 4 fusion layers, 8 heads, 1 query token,
CLIP ViT-B/16 (vision width 768). Total **166,433,025** parameters.

## Three non-standard details

These are genuine deviations from stock transformer components. Each is a silent
correctness trap: the model runs either way and only the numbers are wrong.

1. **LayerNorm applies `(1 + scale)`, not `scale`.** See `LayerNorm` in
   `magiclens_reference/layers.py`. Our `MLLayerNorm` stores the raw `scale` and
   applies the `1 +` at runtime, so checkpoint values copy across unchanged.
   Note this applies to *MagicLens's own* blocks only — the Scenic CLIP backbone
   uses standard LayerNorm.
2. **Attention logits are soft-capped before softmax**: `50 * tanh(logits / 50)`.
   `torch.nn.MultiheadAttention` has no such option, which is why the attention is
   hand-written rather than delegated.
3. **The pooler (only) uses a learned per-dimension query scale** instead of the
   usual `1/sqrt(dim_per_head)`: `r_softplus_0 / sqrt(dim) * softplus(per_dim_scale)`
   with `r_softplus_0 = 1.442695041`. The four fusion layers use ordinary scaling.

Also note `dim_per_head` is **not** `input_dim // num_heads` everywhere: the pooler
projects to `4 * input_dim` total (`2048 = 8 heads x 256`), so it must be passed
explicitly rather than derived.

## Parameter mapping

Checkpoint format: a pickled msgpack blob. `pickle.load()` yields raw `bytes`
(this is expected, not corruption); those bytes decode via
`flax.serialization.msgpack_restore()`. Instantiating the real Flax model — and
therefore installing Scenic — is *not* required just to read the parameter tree.

474 source tensors map onto 378 destination keys. The difference is exactly
`24 blocks x 4` : `open_clip` fuses q/k/v into a single `in_proj_weight`
(3 tensors -> 1) and likewise for biases, across 12 text + 12 vision blocks.

| Source (Flax) | Shape | Destination (PyTorch) | Transform |
|---|---|---|---|
| `Dense` kernel | `(in, out)` | `Linear.weight` `(out, in)` | transpose |
| attn `query/key/value` kernel | `(D, N, H)` | fused `in_proj_weight` / `q_proj.weight` | `reshape(D, -1).t()` |
| attn `query/key/value` bias | `(N, H)` | matching bias | `reshape(-1)` |
| **CLIP** `attn/out/kernel` | `(N, H, D)` | `out_proj.weight` `(D, N*H)` | `reshape(-1, D).t()` |
| **MagicLens** `self_attention/post/w` | `(D, N, H)` | `out_proj.weight` `(D, N*H)` | `reshape(D, -1)`, **no transpose** |
| conv kernel | `(KH, KW, Cin, Cout)` | `conv1.weight` `(Cout, Cin, KH, KW)` | `permute(3, 2, 0, 1)` |
| CLIP LayerNorm `scale` | `(D,)` | `.weight` | direct |
| MagicLens LayerNorm `scale` | `(D,)` | `.scale` | direct (runtime `1 +`) |
| `text_projection` / `visual/proj` kernel | `(in, out)` | `nn.Parameter` | direct, **no transpose** |

### The output-projection trap

The single most dangerous detail. The two halves of the model store attention
output projections in **different axis orders**, because they come from different
codebases:

- Scenic CLIP uses Flax's standard `DenseGeneral` convention: `(N, H, D)` with
  einsum `...NH,NHD->...D`.
- MagicLens's own `AttentionProjection` (`output_proj=True`) keeps the input-dim
  axis first: `(D, N, H)` with einsum `...NH,DNH->...D`.

Both land on `(D, N*H)`, but via different routes. Applying one rule uniformly
produces a `state_dict` that loads cleanly under `strict=True`, runs without error,
and returns silently wrong embeddings. Projection weights must be handled
per-module, and the model must be validated behaviourally, not just structurally.

`text_projection` and `visual.proj` are the mirror-image trap: they *look* like
they need a transpose but do not, because `open_clip` applies them as `x @ P`,
matching Flax's `(in, out)`.

## Validation

### Parity against the original JAX model

`scripts/check_magiclens_parity.py` runs both models on identical token ids and
identical pre-processed pixels and compares the resulting embeddings. The two
halves need incompatible dependency sets, so it runs as three subcommands across
two interpreters, exchanging `.npz` files:

```bash
# isolated env, so Scenic's TensorFlow/JAX pins stay out of the project env
python3 -m venv /tmp/jaxenv
/tmp/jaxenv/bin/pip install "jax[cpu]" flax git+https://github.com/google-research/scenic.git

python3 scripts/check_magiclens_parity.py emit  --output /tmp/ml_in.npz
/tmp/jaxenv/bin/python scripts/check_magiclens_parity.py jax \
    --inputs /tmp/ml_in.npz --output /tmp/ml_jax.npz
python3 scripts/check_magiclens_parity.py torch \
    --inputs /tmp/ml_in.npz --jax /tmp/ml_jax.npz
```

Result, reproduced over three seeds (4 prompts each, including the empty
candidate instruction):

```
preprocessing: max|ours - reference| = 2.384e-07  (CLIP constants match)
min cosine 1.00000000   max|diff| 4.263e-07
PARITY PASSED
```

Cosine similarity is `1.00000000` on every row, with residuals of ~4e-07 — float32
rounding, not approximation. Upstream's README cautions that "due to the weight
conversion, performance may be slightly different"; that caveat does not apply
here, since the port reproduces the reference numerically rather than
approximately. The check also confirms our CLIP normalisation constants match
Scenic's `normalize_image` to 2.4e-07.

Tokenisation is deliberately held fixed (ids are produced once by `open_clip` and
fed to both models) so the comparison isolates the model itself. Whether
`open_clip`'s tokeniser agrees with Scenic's on arbitrary text is a separate
question, untested here.

### Behavioural checks

Structural checks alone are insufficient — see the output-projection trap above.
The converter refuses to write unless all four hold: every source tensor consumed
exactly once, every destination key filled, all shapes match, and
`load_state_dict(strict=True)` succeeds. That catches omissions but *not*
transposition errors.

These two ran before Scenic was available, and remain useful as fast smoke tests
that need neither JAX nor Scenic:

1. **CLIP tower, zero-shot colours.** Four solid-colour images against four
   colour prompts: 4/4 correct diagonal. Exercises both towers, both
   out-projections, and the conv permute.
2. **End-to-end composed retrieval.** The actual task:

   | Query | red candidate | blue candidate | picks |
   |---|---|---|---|
   | red image + "make it blue" | 0.441 | **0.758** | blue |
   | blue image + "make it red" | **0.754** | 0.378 | red |

   Incorrect fusion or pooler weights cannot produce coherent compositional
   behaviour at these margins, so this covers the custom head that check 1 does not.

3. **Protocol self-consistency.** `encode_mm(img, "")` equals `encode_image(img)`
   to `0.0e+00`.

Embeddings are L2-normalised to 1.0 and not collapsed (query-pair cosine 0.169).

## Preprocessing (a second trap)

The official evaluation preprocessing is **not** standard CLIP preprocessing, and getting
it wrong degrades real-image results silently while leaving parity on synthetic inputs
intact. From `magiclens_reference/data_utils.py::process_img`:

```python
ima = jnp.array(img)[jnp.newaxis, ...]
ima = ima / (ima.max() + 1e-12)                              # per-image max, NOT /255
ima = jax.image.resize(ima, (1, size, size, 3), 'bilinear')  # squashes aspect ratio
```

Two departures from the obvious implementation:

1. **Pixels are scaled by each image's own maximum**, not by 255. A photo whose brightest
   pixel is 200 gets brightened, not merely rescaled.
2. **The image is resized straight to a square**, so the aspect ratio is squashed rather
   than cropped. The `largest_square_crop` inside `model.py::_preprocess_images` is a
   *no-op* on the eval path, because `process_img` has already produced a square — so
   reimplementing that crop would **not** match the official pipeline.

Our `MagicLensImagePreprocess` reproduces both. Verified against the reference on real
Fashion-IQ photos at native resolution (296x445, 229x445 portraits, so both the squashing
resize and the max-scaling actually fire): **max abs difference 7.8e-06**, the residual
between JAX's and PyTorch's bilinear-antialias implementations.

Note the original synthetic parity check could not have caught an error here: it fed
224x224 inputs, making the resize a no-op, and handed PyTorch the reference's own
pre-processed pixels in order to isolate the model. Preprocessing needed its own test on
real images.

## Harness integration

`MagicLens` deliberately satisfies the same duck-typed contract as this repo's
`Visualized_BGE`, so the SP dataset, collator, batch sampler, loss, and optimiser run
unmodified:

- `MagicLensTokenizer` adapts open_clip's tokenizer to the HuggingFace-style call the
  datasets make (`padding=`, `max_length=77`, `truncation=`, `return_tensors="pt"`,
  then `["input_ids"]` / `["attention_mask"]`). `attention_mask` is derived as
  `input_ids != 0`, and `max_length != 77` raises rather than silently mis-padding.
- `encode_mm` accepts the `{input_ids, attention_mask}` dict alongside strings and raw
  ids; the three paths are bit-identical (0.0e+00). The mask is ignored on purpose --
  CLIP locates EOT by `argmax` over ids, so padding cannot affect pooling.
- `preprocess_train` / `preprocess_val` are exposed under the same names VISTA uses.
- `src/retrievers/magiclens_retriever.py` wraps the backbone as a `TwoEncoderVLM`.
  `.text` raises: MagicLens has no text-only tower, so only
  `query_embedding_mode='vista_mm'` (native multimodal) is supported.
- `MagicLens.forward()` reproduces `Visualized_BGE.forward`'s contrastive step —
  same similarity, temperature scaling, and target construction — returning an
  object with `.loss`. This is required by the contrastive training path, which
  calls the backbone directly rather than computing the loss in the trainer.
  Reimplementing the objective differently here would leave the two contrastive
  baselines silently non-comparable, which is the whole point of running one.

The SP loss is Gram-based, so MagicLens' 512-d embedding trains against the 4096-d
BGE-VL teacher cache with no projection layer.

## Zero-shot benchmark (end-to-end validation)

Run on the repo's own evaluation pipeline (`val_split`, `vista_mm`, fusion `sum`), i.e.
the same protocol as the VISTA baselines, so the comparison is apples-to-apples.

| Model | FIQ avg R@10 | CIRR summary |
|---|---|---|
| VISTA zero-shot | 24.00 | 48.61 |
| VISTA contrastive fine-tuned | 25.71 | 53.97 |
| VISTA + SP distillation (best, clean teacher) | 27.46 | 56.46 |
| **MagicLens-B zero-shot (this port)** | **25.90** | **64.11** |

Full: FIQ avg R@5/10/50 = 19.36 / 25.90 / 48.41; CIRR global R@1/5/10 = 29.99 / 60.27 /
73.02, subset R@1 = 67.95, summary = 64.11.

Two observations:

1. **CIRR is a decisive win.** Zero-shot MagicLens (64.11) exceeds the *fully SP-distilled*
   VISTA (56.46) by 7.65 points, and zero-shot VISTA by 15.5.
2. **Fashion-IQ is much weaker in relative terms.** 25.90 only matches contrastively
   fine-tuned VISTA (25.71) and sits *below* distilled VISTA (27.46). The asymmetry is
   consistent with the training data: MagicLens' 36.7M web-mined triplets are open-domain,
   which matches CIRR's domain but not Fashion-IQ's narrow fashion-attribute domain.

## Reproduction against the published numbers

Confirmed against Zhang et al. (ICML'24), Table 12 (Fashion-IQ) and Table 13 (CIRR), row
**MagicLens-B / CLIP-B, 166M params** — matching our checkpoint's 166,433,025 exactly.
Full data in `results/magiclens/paper_reproduction/`.

| Benchmark | Metric | Paper | This port |
|---|---|---|---|
| Fashion-IQ (val) | overall R@10 | 26.3 | 25.90 |
| Fashion-IQ (val) | overall R@50 | 47.4 | 48.41 |
| CIRR (test1) | R@1 | 27.0 | 29.52 |
| CIRR (test1) | R@5 | 58.0 | 59.61 |
| CIRR (test1) | R@10 | 70.9 | 72.63 |
| CIRR (test1) | R@50 | 91.1 | 91.74 |
| CIRR (test1) | R_subset@1 | 66.7 | 67.35 |

CIRR was scored by the official server (<https://cirr.cecs.anu.edu.au/>), so our harness is
out of that loop entirely. Mean absolute difference: 0.77 on Fashion-IQ (8 metrics),
1.11 on CIRR (7 metrics).

All seven CIRR deltas are positive — systematic rather than scatter, and worth stating
plainly. Since the model is provably identical to the original, the difference has to come
from the evaluation pipeline: MagicLens' released code contains no CIRR evaluation at all
(only FIQ, CIRCO, DTIN), so the paper's CIRR numbers came from an internal pipeline that
cannot be inspected or matched. We have not isolated the cause and do not claim to have.

## Reproducing

```bash
pip install open_clip_torch "jax[cpu]" flax
python3 scripts/convert_magiclens_weights.py \
    --checkpoint models/magiclens/magic_lens_clip_base.pkl \
    --model-size base \
    --output models/magiclens/magic_lens_clip_base.pt
```

Checkpoint: `gsutil cp -R gs://gresearch/magiclens/models ./` or the Drive link in
the upstream README. Verify the download — a truncated file still unpickles far
enough to look plausible. The `base` checkpoint is ~635 MB (665,747,853 bytes);
an early truncated copy at 82 MB failed only at `pickle.load()`.

## Standalone repository

The port is also packaged as a self-contained repo, staged at `magiclens-pytorch/`
(gitignored here, published separately as `mat0k/magiclens-pytorch`).

Files were **copied, never moved** — this repo keeps and uses its own versions.
Copied: `layers.py`, `modeling.py`, `convert_magiclens_weights.py`,
`check_magiclens_parity.py`, and this document as the basis for its README.
Not copied: `magiclens_retriever.py`, configs, jobs — all depend on this repo's
`TwoEncoderVLM` and are thesis glue.

Two deliberate divergences in the copies: relative imports, and `auto_device` defaults to
**False** there (a library must not seize the GPU on construction — the opposite of what
this repo's trainer needs, and the source of the CPU-training incident above).

Licensing is permissive and settled: upstream releases code under **Apache 2.0** and "all
other materials", including checkpoints, under **CC-BY 4.0** — so redistributing both the
port and the converted weights is allowed with attribution.

Weights (`models/magiclens/magic_lens_clip_base.pt`, 635 MB, plain `torch.save`
state_dict) go to HuggingFace Hub rather than Drive: versioned, fetchable in code, and
free at this size.

## Results as a distillation student

Baseline is zero-shot MagicLens-B: Fashion-IQ avg R@10 **25.90**, CIRR summary **64.11**.
Full data in `results/magiclens/summary/`.

| Method | Setting | Best epoch | Fashion-IQ | Δ | CIRR | Δ |
|---|---|---|---|---|---|---|
| Contrastive fine-tuning | batch 32, lr 1e-6 | 5 / 1 | 25.69 | −0.21 | 64.27 | +0.16 |
| Contrastive fine-tuning | batch 256, lr 3e-6 | 1 | 25.83 | −0.07 | 63.74 | −0.37 |
| SP distillation | λ_ce 1.0 | 1 | 25.88 | −0.02 | 65.68 | +1.57 |
| **SP distillation** | **λ_ce 0.1** | **1** | **27.52** | **+1.62** | **66.18** | **+2.07** |

### Contrastive fine-tuning fails, and not because of batch size

Fine-tuning on LaSCo never beat zero-shot. The initial hypothesis was that the negative
pool was too small: InfoNCE contrasts each query against the other targets *in its batch*,
MagicLens was pretrained at batch 2048, and we used 32 (gradient accumulation does not
enlarge the pool). Retesting at batch 256 with a scaled learning rate **refuted this** —
Fashion-IQ improved only −0.21 → −0.07, still negative, and CIRR got worse.

Caveats: 256 is the hardware ceiling here, not 2048, so a non-linear effect near the
original scale cannot be excluded; and the retest varied batch size and learning rate
together (to avoid an optimizer-step confound), leaving ~half as many updates.

### What actually transfers

On *identical data with an identical student*, two independent contrastive configurations
fail while relation-based SP distillation gains +1.62 / +2.07. **LaSCo's images carry
useful signal; LaSCo's own labels do not.** What transfers is the teacher's relational
geometry over those images, not the dataset's query-to-target supervision — the same
conclusion the VISTA phase reached from a different direction.

### Hyperparameters do not survive a change of student

λ_ce = 1.0 was inherited from VISTA, where contrastive training on LaSCo *helped*, making
CE a free anchor. For MagicLens it *hurts*, so the same value suppressed the gain outright
(Fashion-IQ +0.0 at λ=1.0 versus +1.62 at λ=0.1). The diagnostic was that CE fell
2.75 → 1.65 across epochs while SP stayed flat (0.0007 → 0.0005), with Fashion-IQ
collapsing over exactly that window.

The anchor was weakened rather than removed: SP constrains only candidate–candidate
geometry and never sees queries, so CE is the only term aligning a query with its own
target. λ=0 remains untested. Only {1.0, 0.1} were tried, so the optimum is unmeasured.

Both SP runs **peak at epoch 1** and decay after; λ=0.1 decays more slowly and holds CIRR
about a point above baseline through epoch 10, but Fashion-IQ still falls below baseline
from epoch 3. The deliverable is the epoch-1 checkpoint. Note that `magiclens_best.pth` is
*not* the best model — checkpoint selection compares only trained epochs and never the
untrained starting point.

## Status / open items

- [x] Architecture ported
- [x] Weight conversion, structurally validated
- [x] Behavioural validation (colour probe, composed retrieval, protocol consistency)
- [x] Formal parity against the original JAX model — cosine 1.00000000, 3 seeds
- [x] Preprocessing parity on real images at native resolution — 7.8e-06
- [x] Trainer integration (tokenizer adapter, preprocess, retriever wrapper, backbone
      dispatch); VISTA path verified unchanged
- [x] Zero-shot Fashion-IQ / CIRR benchmark — **gate passed** (see below)
- [ ] Contrastive baseline, then SP distillation run
- [ ] `large` variant not yet converted (config exists; mapping should be identical)
