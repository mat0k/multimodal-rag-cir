"""
Held-out candidate-candidate SP-vs-teacher metric, per epoch checkpoint of an SP
run, reported next to Recall@K so we can read the CE<->SP tension against results.

For each epoch checkpoint of the student:
  c   = VISTA.encode_image(heldout)              [B, d_s]  (L2-normed)
  G_s = rownorm_L2(c @ c^T),  G_t = rownorm_L2(t @ t^T)     [B, B]
  sp_loss         = || G_t - G_s ||_F^2 / B^2      (same objective as training)
  offdiag_pearson = corr( offdiag(c@c^T), offdiag(t@t^T) )  (interpretable agreement)

Recall (fashioniq avg R@10, cirr summary avg) is read from the run's train_log.json.
Output: <run-dir>/heldout_sp_metric.json  + a printed table.

Usage:
  python scripts/eval_heldout_sp.py --run-dir results/lamra_ret/distill_lamra_ret_sp_moderate
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.backbones.vista.modeling import Visualized_BGE  # noqa: E402
from src.retrievers.vista_retriever import VistaImageProcessor  # noqa: E402


@torch.no_grad()
def _student_embed(backbone, proc, paths, images_dir, device, bs=128) -> torch.Tensor:
    embs = []
    for s in range(0, len(paths), bs):
        chunk = paths[s : s + bs]
        px = torch.stack([
            proc(Image.open(images_dir / p).convert("RGB"), return_tensors="pt")["pixel_values"][0]
            for p in chunk
        ]).to(device)
        embs.append(backbone.encode_image(px).float().cpu())
    return F.normalize(torch.cat(embs, 0), dim=-1)


def _sp_and_corr(c: torch.Tensor, t: torch.Tensor):
    B = c.shape[0]
    Gs_raw, Gt_raw = c @ c.T, t @ t.T
    sp = float(((F.normalize(Gt_raw, p=2, dim=1) - F.normalize(Gs_raw, p=2, dim=1)) ** 2).sum() / (B * B))
    off = ~torch.eye(B, dtype=torch.bool)
    a, b = Gs_raw[off].numpy(), Gt_raw[off].numpy()
    pearson = float(np.corrcoef(a, b)[0, 1])
    return sp, pearson


def main(a: argparse.Namespace) -> None:
    run_dir = PROJECT_ROOT / a.run_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_dir = PROJECT_ROOT / a.images_dir

    teacher = F.normalize(torch.load(PROJECT_ROOT / a.heldout_emb, map_location="cpu").float(), dim=-1)
    paths = json.load(open(PROJECT_ROOT / a.heldout_paths))
    print(f"held-out set: {len(paths)} images")

    cfg = json.load(open(run_dir / "run_config.json"))
    model_name = cfg["model"]["model_name_or_path"]
    ckpts = sorted((run_dir / "checkpoints").glob("vista_epoch*.pth"))
    if not ckpts:
        print("no epoch checkpoints yet."); return

    backbone = Visualized_BGE(model_name_bge=model_name, model_weight=str(ckpts[0]))
    backbone.to(device).eval()
    proc = VistaImageProcessor(backbone.preprocess_val)

    train_log = json.load(open(run_dir / "train_log.json"))
    recall_by_ep = {
        e["epoch"]: (
            e.get("eval", {}).get("fashioniq_val_avg_recall_at10"),
            e.get("eval", {}).get("cirr_val_summary_average"),
        ) for e in train_log
    }

    rows = {}
    for ck in ckpts:
        ep = int(ck.stem.replace("vista_epoch", ""))
        backbone.load_state_dict(torch.load(ck, map_location=device))
        c = _student_embed(backbone, proc, paths, images_dir, device)
        sp, pearson = _sp_and_corr(c, teacher)
        fiq, cirr = recall_by_ep.get(ep, (None, None))
        rows[ep] = {"sp_loss": round(sp, 5), "offdiag_pearson": round(pearson, 4),
                    "fiq_r10": fiq, "cirr_summary": cirr}
        print(f"ep{ep}: sp_loss={sp:.5f}  offdiag_pearson={pearson:.3f}  "
              f"FIQ R@10={fiq}  CIRR summ={cirr}")

    out = run_dir / "heldout_sp_metric.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--heldout_emb", default="data/lasco/sp_heldout_teacher_emb.pt")
    p.add_argument("--heldout_paths", default="data/lasco/sp_heldout_paths.json")
    p.add_argument("--images_dir", default="data/lasco/images")
    main(p.parse_args())
