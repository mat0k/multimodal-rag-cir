"""
Precompute LamRA-Ret TEACHER embeddings for a HELD-OUT set of LaSCo candidate
images (images NOT in the 40K distill subset). Used by the candidate-candidate
SP-vs-teacher eval metric to read the CE<->SP tension against Recall@K.

GPU, run once.

Output:
  data/lasco/sp_heldout_paths.json        [B] relative image paths
  data/lasco/sp_heldout_teacher_emb.pt    [B, d_teacher] L2-normalized
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.lamra_ret_retriever import LamRARetRetriever  # noqa: E402


def main(a: argparse.Namespace) -> None:
    subset = json.load(open(PROJECT_ROOT / a.subset_path))
    train = json.load(open(PROJECT_ROOT / a.lasco_train))
    subset_qids = {t["qid"] for t in subset}
    subset_paths = {t["target-image"][1] for t in subset}

    # held-out candidate images: not in subset (by qid) and a distinct target path
    seen, heldout = set(), []
    import random
    rng = random.Random(a.seed)
    rng.shuffle(train)
    for t in train:
        if t["qid"] in subset_qids:
            continue
        p = t["target-image"][1]
        if p in subset_paths or p in seen:
            continue
        seen.add(p)
        heldout.append(p)
        if len(heldout) >= a.n:
            break
    print(f"held-out candidate images: {len(heldout)}")

    images_dir = PROJECT_ROOT / a.images_dir
    retriever = LamRARetRetriever(
        model_path=str(PROJECT_ROOT / a.model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
    )

    embs = []
    for s in range(0, len(heldout), a.batch_size):
        chunk = heldout[s : s + a.batch_size]
        imgs = [Image.open(images_dir / p).convert("RGB") for p in chunk]
        embs.append(retriever.encode_images(imgs).cpu())
        if (s // a.batch_size) % 20 == 0:
            print(f"  {min(s+a.batch_size,len(heldout))}/{len(heldout)} encoded")
    emb = torch.nn.functional.normalize(torch.cat(embs, 0), dim=-1)

    out_emb = PROJECT_ROOT / a.out_emb
    out_paths = PROJECT_ROOT / a.out_paths
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, out_emb)
    json.dump(heldout, open(out_paths, "w"))
    print(f"saved {tuple(emb.shape)} -> {out_emb}\nsaved paths -> {out_paths}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lasco_train", default="data/lasco/lasco_train.json")
    p.add_argument("--subset_path", default="data/lasco/distill_subset.json")
    p.add_argument("--images_dir", default="data/lasco/images")
    p.add_argument("--model_path", default="models/LamRA-Ret-Qwen2.5VL-7b")
    p.add_argument("--out_emb", default="data/lasco/sp_heldout_teacher_emb.pt")
    p.add_argument("--out_paths", default="data/lasco/sp_heldout_paths.json")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    main(p.parse_args())
