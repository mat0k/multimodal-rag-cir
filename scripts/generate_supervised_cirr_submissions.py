"""
CIRR test1 submissions for the SUPERVISED (in-domain) distillation runs, selected
automatically by BEST VALIDATION EPOCH.

Each training run evaluates on CIRR val every epoch, so the best epoch is already in
its train_log.json. This reads that log, resolves the matching checkpoint, and drives
the same generator every other submission used (generate_cirr_test_submission.py), so
the protocol is identical -- only the model selection is automated.

Selection metric: cirr_val_summary_average = mean(global R@5, subset R@1), CIRR's
official summary. Ties go to the EARLIER epoch (less overfitting, same score).

Output, with the epoch in the folder name as agreed:

    results/cirr_test_submissions/supervised/<student>_<arm>_ep<N>/
        <student>_<arm>_ep<N>_recall_top50.json
        <student>_<arm>_ep<N>_recall_subset_top3.json

Writes the files only; uploading to https://cirr.cecs.anu.edu.au/ stays manual.

    python scripts/generate_supervised_cirr_submissions.py                 # every run found
    python scripts/generate_supervised_cirr_submissions.py --only sp,crd   # by arm
"""
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_cirr_test_submission import generate_for_model  # noqa: E402

METRIC = "cirr_val_summary_average"


def discover(runs_root: Path, students: list[str], only: set[str] | None) -> list[dict]:
    """One entry per finished CIRR run, pinned to its best-val epoch checkpoint."""
    entries = []
    for student in students:
        for run_dir in sorted((runs_root / student).glob(f"distill_{student}_cirr_*")):
            arm = run_dir.name.split("_cirr_", 1)[1]
            if only and arm not in only:
                continue
            log_path = run_dir / "train_log.json"
            if not log_path.exists():
                print(f"  SKIP {run_dir.name}: no train_log.json (run unfinished?)")
                continue

            scored = [(e["epoch"], e["eval"][METRIC]) for e in json.load(open(log_path))
                      if METRIC in e.get("eval", {})]
            if not scored:
                print(f"  SKIP {run_dir.name}: no {METRIC} in log")
                continue

            # max() keeps the first maximum, i.e. the earliest best epoch.
            best_epoch, best_score = max(scored, key=lambda r: r[1])
            ckpt = run_dir / "checkpoints" / f"{student}_epoch{best_epoch:02d}.pth"
            if not ckpt.exists():
                print(f"  SKIP {run_dir.name}: {ckpt.name} missing")
                continue

            short = f"{student}_{arm}_ep{best_epoch}"
            entries.append({
                "name": f"{student.upper()} + {arm} distillation, supervised on CIRR (ep{best_epoch})",
                "short": short,
                "checkpoint": str(ckpt.resolve().relative_to(PROJECT_ROOT)),
                "_val": best_score,
                "_epochs": len(scored),
                **({"type": "magiclens", "model_size": "base"} if student == "magiclens" else {}),
            })
            print(f"  {run_dir.name:<32} best ep{best_epoch}/{len(scored)}  val {best_score:.2f}")
    return entries


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — this will be very slow.", flush=True)

    only = {s.strip() for s in a.only.split(",")} if a.only else None
    print("Selecting best epoch per run:")
    entries = discover(PROJECT_ROOT / a.runs_root, a.students, only)
    if not entries:
        raise SystemExit("no runs to submit — nothing matched.")

    gen_args = Namespace(batch_size=a.batch_size, num_workers=a.num_workers,
                         query_embedding_mode="vista_mm", fusion_type="sum",
                         top_k=50, subset_top_k=3, out_dir=a.out_dir)
    results = [generate_for_model(e, gen_args, device) for e in entries]
    for r, e in zip(results, entries):
        r["val_summary_average"] = round(e["_val"], 4)
        r["selected_by"] = f"best of {e['_epochs']} epochs on {METRIC}"

    out_dir = PROJECT_ROOT / a.out_dir
    manifest_path = out_dir / "submission_manifest.json"
    manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "split": "test1",
                "dataset_version": "rc2", "server": "https://cirr.cecs.anu.edu.au/",
                "setting": "supervised (trained on CIRR train)", "models": results}
    if manifest_path.exists():
        try:
            previous = json.load(open(manifest_path))
            keep = [m for m in previous.get("models", [])
                    if m.get("short") not in {r["short"] for r in results}]
            manifest["models"] = keep + results
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: could not merge manifest ({exc})")
    json.dump(manifest, open(manifest_path, "w"), indent=2)

    print(f"\n{'='*70}\nSUBMIT TO https://cirr.cecs.anu.edu.au/\n{'='*70}")
    for r in results:
        print(f"  {r['name']}\n     {r['files']['recall']['path']}")
    print(f"\nManifest -> {manifest_path.relative_to(PROJECT_ROOT)}")
    print("NOTE: the server limits test submissions. Submit deliberately.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="results/supervised")
    p.add_argument("--students", nargs="+", default=["vista", "magiclens"])
    p.add_argument("--out-dir", default="results/cirr_test_submissions/supervised")
    p.add_argument("--only", default=None, help="comma-separated arms, e.g. sp,crd")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    main(p.parse_args())
