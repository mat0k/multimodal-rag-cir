"""
Aggregate multi-seed runs into a mean/std summary (SP vs matched no-SP control),
so we can report "27.x +/- y" instead of a single number.

For each run: best-epoch by FashionIQ avg R@10, reporting FIQ + CIRR at that epoch.
Then mean/std across seeds per group, plus the SP - control gap.

Usage:
  python scripts/analyze_seeds.py \
    --sp results/lamra_ret/distill_lamra_ret_sp_random_long_w3000 \
         results/lamra_ret/distill_lamra_ret_sp_w3000_seed43 \
         results/lamra_ret/distill_lamra_ret_sp_w3000_seed44 \
    --control results/lamra_ret/distill_lamra_ret_control_seed42 \
              results/lamra_ret/distill_lamra_ret_control_seed43 \
              results/lamra_ret/distill_lamra_ret_control_seed44 \
    --out results/lamra_ret/seed_analysis/seed_summary.json
"""
import argparse
import json
import statistics as st
from pathlib import Path


def _best(run_dir: str):
    p = Path(run_dir) / "train_log.json"
    if not p.exists():
        return None
    log = json.load(open(p))
    best = None
    for e in log:
        ev = e.get("eval", {})
        f = ev.get("fashioniq_val_avg_recall_at10")
        c = ev.get("cirr_val_summary_average")
        if f is not None and (best is None or f > best["fiq"]):
            best = {"epoch": e["epoch"], "fiq": f, "cirr": c, "seed": json.load(open(Path(run_dir)/"run_config.json")).get("seed")}
    return best


def _agg(runs):
    per = {r: _best(r) for r in runs}
    done = {r: b for r, b in per.items() if b is not None}
    fiq = [b["fiq"] for b in done.values()]
    cirr = [b["cirr"] for b in done.values()]
    out = {"n_done": len(done), "n_total": len(runs), "per_run": per}
    if fiq:
        out["fiq_mean"] = round(st.mean(fiq), 3)
        out["fiq_std"] = round(st.pstdev(fiq), 3) if len(fiq) > 1 else 0.0
        out["cirr_mean"] = round(st.mean(cirr), 3)
        out["cirr_std"] = round(st.pstdev(cirr), 3) if len(cirr) > 1 else 0.0
    return out


def main(a):
    summary = {
        "sp_w3000": _agg(a.sp),
        "control_noSP": _agg(a.control),
        "reference_contrastive_v2": {"fiq": 25.71, "cirr": 53.97, "note": "fixed single-run baseline"},
    }
    sp, ct = summary["sp_w3000"], summary["control_noSP"]
    if "fiq_mean" in sp and "fiq_mean" in ct:
        summary["gap_sp_minus_control"] = {
            "fiq": round(sp["fiq_mean"] - ct["fiq_mean"], 3),
            "cirr": round(sp["cirr_mean"] - ct["cirr_mean"], 3),
        }
        summary["verdict"] = (
            "SP beats control beyond seed spread"
            if sp["fiq_mean"] - sp.get("fiq_std", 0) > ct["fiq_mean"] + ct.get("fiq_std", 0)
            else "overlap within seed spread — inconclusive, need more seeds"
        )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sp", nargs="+", required=True)
    p.add_argument("--control", nargs="+", required=True)
    p.add_argument("--out", default="results/lamra_ret/seed_analysis/seed_summary.json")
    main(p.parse_args())
