"""
Build HARD groups for relation-based (group-matrix) distillation.

Instead of random groups, we cluster queries whose TARGET images are mutual
nearest neighbours in retriever space. Within such a group every query's
off-diagonal candidates are look-alike targets that the retriever genuinely
confuses with the true target — i.e. dense, uniform hard negatives — while the
matrix stays square (G x G), so the teacher precompute / dataset / loss are
unchanged.

Pipeline position:
    build_relation_groups.py   (this script — retriever, fast)
        -> writes data/lasco/relation_subset.json in GROUPED order
    precompute_teacher_scores_relation.py   (teacher Qwen, scores G x G)
    train_retriever.py --config .../lasco_distill_relation.yaml

The output subset is consumed as-is by the teacher precompute: group g is the
contiguous slice [g*G : (g+1)*G].

Usage
-----
  python scripts/build_relation_groups.py \\
      --config configs/distillation/precompute_qwen_relation.yaml \\
      --retriever_checkpoint results/training/vista_lasco_finetune_contrastive_v2/checkpoints/vista_epoch01.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.backbones.vista.modeling import Visualized_BGE
from src.retrievers.vista_retriever import VistaImageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target-image dataset (for embedding)
# ---------------------------------------------------------------------------

class _TargetDataset(Dataset):
    def __init__(self, triplets: list[dict], images_dir: Path, image_transform):
        self._triplets = triplets
        self._images_dir = images_dir
        self._tf = image_transform

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, idx):
        rel = self._triplets[idx]["target-image"][1]
        pil = Image.open(self._images_dir / rel).convert("RGB")
        return {"image": self._tf(pil, return_tensors="pt")["pixel_values"][0], "idx": idx}


def _load_retriever(checkpoint: str | None) -> Visualized_BGE:
    backbone = Visualized_BGE(
        model_name_bge="BAAI/bge-base-en-v1.5",
        model_weight=str(PROJECT_ROOT / "models/Visualized_BGE/Visualized_base_en_v1.5.pth"),
        negatives_cross_device=False,
        from_pretrained=None,
    )
    if checkpoint is not None:
        ckpt = PROJECT_ROOT / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
        backbone.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        logger.info(f"Retriever checkpoint loaded: {ckpt}")
    else:
        logger.info("Retriever: zero-shot (base VISTA weights)")
    if torch.cuda.is_available():
        backbone = backbone.cuda()
    backbone.eval()
    return backbone


# ---------------------------------------------------------------------------
# Greedy nearest-neighbour grouping
# ---------------------------------------------------------------------------

def _greedy_groups(embeddings: torch.Tensor, group_size: int) -> list[list[int]]:
    """Greedily form groups of `group_size` mutually-near targets.

    Pick the first unused index as a seed, take its top (G-1) nearest unused
    neighbours (cosine; embeddings are L2-normalised), emit the group, remove
    them, and repeat. The trailing remainder (< G) is dropped.
    """
    device = embeddings.device
    N = embeddings.shape[0]
    used = torch.zeros(N, dtype=torch.bool, device=device)
    order = torch.arange(N, device=device)
    groups: list[list[int]] = []

    n_full = N // group_size
    for _ in tqdm(range(n_full), desc="Grouping"):
        seed = int(order[~used][0].item())
        sims = embeddings @ embeddings[seed]           # [N] cosine sims
        sims[used] = -float("inf")
        sims[seed] = float("inf")                       # ensure seed is picked first
        top = torch.topk(sims, k=group_size).indices    # G nearest (incl. seed)
        members = [int(i) for i in top.tolist()]
        used[top] = True
        groups.append(members)

    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dc = cfg["data"]

    subset_size = args.subset_size or int(dc["subset_size"])
    group_size = args.group_size or int(dc["group_size"])
    seed = int(dc["seed"])

    lasco_path = PROJECT_ROOT / dc["lasco_annotations"]
    images_dir = PROJECT_ROOT / dc["images_dir"]
    subset_output = PROJECT_ROOT / dc["subset_output"]

    if subset_output.exists() and not args.overwrite:
        logger.info(f"{subset_output} already exists — pass --overwrite to rebuild. Aborting.")
        return

    logger.info("=" * 60)
    logger.info(f"Building HARD relation groups | subset={subset_size:,}  G={group_size}  seed={seed}")
    logger.info("=" * 60)

    with open(lasco_path) as f:
        all_triplets = json.load(f)
    logger.info(f"Total LaSCo triplets: {len(all_triplets):,}")

    # Sample which triplets to use (same seed semantics as the random path)
    rng = np.random.default_rng(seed)
    size = min(subset_size, len(all_triplets))
    idx = rng.choice(len(all_triplets), size=size, replace=False)
    subset = [all_triplets[i] for i in idx]
    logger.info(f"Sampled {len(subset):,} triplets")

    # Encode target images
    backbone = _load_retriever(args.retriever_checkpoint)
    device = next(backbone.parameters()).device
    image_tf = VistaImageProcessor(backbone.preprocess_val)

    ds = _TargetDataset(subset, images_dir, image_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    logger.info("Encoding target images …")
    t0 = time.time()
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Targets"):
            feats.append(backbone.encode_image(batch["image"].to(device)).cpu())
    emb = torch.cat(feats, dim=0)            # [S, d], already L2-normalised
    logger.info(f"  Encoded {emb.shape[0]:,} targets in {(time.time()-t0)/60:.1f} min")

    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Greedy grouping on GPU if available
    emb_dev = emb.to(device) if torch.cuda.is_available() else emb
    groups = _greedy_groups(emb_dev, group_size)
    logger.info(f"Formed {len(groups):,} hard groups of size {group_size} "
                f"({len(groups)*group_size:,} triplets; dropped {len(subset) - len(groups)*group_size} remainder)")

    # Flatten in grouped order and write
    grouped_subset = [subset[i] for g in groups for i in g]
    subset_output.parent.mkdir(parents=True, exist_ok=True)
    with open(subset_output, "w") as f:
        json.dump(grouped_subset, f)
    logger.info(f"Grouped relation subset saved → {subset_output} ({len(grouped_subset):,} triplets)")
    logger.info("Next: run scripts/precompute_teacher_scores_relation.py with the same config.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build hard target-neighbour groups for relation KD.")
    parser.add_argument("--config", type=str,
                        default="configs/distillation/precompute_qwen_relation.yaml")
    parser.add_argument("--retriever_checkpoint", type=str,
                        default="results/training/vista_lasco_finetune_contrastive_v2/checkpoints/vista_epoch01.pth",
                        help="Fine-tuned VISTA .pth used to embed targets. Omit/none for zero-shot.")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild even if relation_subset.json exists.")
    main(parser.parse_args())
