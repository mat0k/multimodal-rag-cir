"""
Project the BGE-VL-MLLM-S1 teacher embeddings 4096 -> 768 ONCE, offline, frozen.

Purpose: the ABSORPTION ABLATION for feature-based distillation. The default
feature-KD design puts a trainable Linear(768 -> d_teacher) head on the student
and aligns in the teacher's space. That head has ~3.1M params per side and can
do the alignment on its own while the student's real 768-d retrieval embedding
barely moves — the exact failure we already hit with the CRD MLP projector.

This script removes that confound: the teacher is brought DOWN to 768 with a
fixed, non-trainable basis, so training can align the student's NATIVE embedding
(the one retrieval actually uses) with zero trainable projection anywhere.
  - plateaus too  -> absorption is ruled out; the ceiling is real.
  - does better   -> absorption was the problem.

Basis: uncentered truncated SVD (top-768 right singular vectors), fit JOINTLY on
queries + candidates. Two details that matter:
  * uncentered — centering shifts the space and distorts cosine geometry, which
    is the only thing retrieval cares about;
  * joint basis — queries and candidates MUST land in the same subspace or
    query-candidate cosine becomes meaningless.

Outputs (L2-normalized, aligned to distill_subset.json order):
  data/lasco/bge_vl_cand_emb_pca768.pt    [N, 768]
  data/lasco/bge_vl_query_emb_pca768.pt   [N, 768]
  data/lasco/bge_vl_pca768_basis.pt       {V: [4096, 768], stats}

Usage:
  python scripts/precompute_teacher_pca768.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def _cosine_preservation(orig: torch.Tensor, proj: torch.Tensor, n: int, seed: int) -> dict:
    """How well does the 768-d teacher preserve the 4096-d teacher's geometry?

    This is the number that says whether the ablation is even meaningful: if the
    down-projected teacher no longer carries the original similarity structure,
    a null result would be uninformative rather than evidence about absorption.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(orig.shape[0], generator=g)[:n]
    a, b = _norm(orig[idx].float()), _norm(proj[idx].float())
    ca = (a @ a.T).flatten()
    cb = (b @ b.T).flatten()
    va, vb = ca - ca.mean(), cb - cb.mean()
    pearson = float((va @ vb) / (va.norm() * vb.norm() + 1e-12))
    return {
        "pearson_cosine_4096_vs_768": round(pearson, 4),
        "mean_abs_cosine_delta": round(float((ca - cb).abs().mean()), 4),
        "sample_n": n,
    }


def main(a) -> None:
    cand_p = PROJECT_ROOT / a.cand_emb
    query_p = PROJECT_ROOT / a.query_emb
    C = _norm(torch.load(cand_p, map_location="cpu").float())
    Q = _norm(torch.load(query_p, map_location="cpu").float())
    print(f"loaded cand {tuple(C.shape)} | query {tuple(Q.shape)}", flush=True)
    assert C.shape == Q.shape, "cand/query caches must align"
    d_in = C.shape[-1]
    if a.dim >= d_in:
        raise SystemExit(f"--dim {a.dim} must be < teacher dim {d_in}")

    # Joint uncentered SVD -> shared top-k right singular basis.
    X = torch.cat([Q, C], dim=0)  # [2N, 4096]
    print(f"fitting uncentered SVD on {tuple(X.shape)} (joint query+candidate) …", flush=True)
    # economy SVD via the 4096x4096 Gram — far cheaper than full SVD on 80000x4096.
    G = X.T @ X                                        # [4096, 4096]
    evals, evecs = torch.linalg.eigh(G)                # ascending
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp(min=0), evecs[:, order]
    V = evecs[:, : a.dim].contiguous()                 # [4096, dim]

    energy = float(evals[: a.dim].sum() / evals.sum())
    print(f"  energy retained in top-{a.dim}: {energy*100:.2f}%", flush=True)

    Cp, Qp = _norm(C @ V), _norm(Q @ V)

    stats = {
        "candidates": _cosine_preservation(C, Cp, a.probe_n, a.seed),
        "queries": _cosine_preservation(Q, Qp, a.probe_n, a.seed),
        "energy_retained": round(energy, 4),
        "dim_in": d_in,
        "dim_out": a.dim,
    }
    print("\ngeometry preservation (4096-d vs projected):")
    for k in ("candidates", "queries"):
        s = stats[k]
        print(f"  {k:11s} pearson={s['pearson_cosine_4096_vs_768']:.4f}  "
              f"mean|Δcos|={s['mean_abs_cosine_delta']:.4f}")

    torch.save(Cp, PROJECT_ROOT / a.out_cand)
    torch.save(Qp, PROJECT_ROOT / a.out_query)
    torch.save({"V": V, "stats": stats}, PROJECT_ROOT / a.out_basis)
    print(f"\nsaved {tuple(Cp.shape)} -> {a.out_cand}")
    print(f"saved {tuple(Qp.shape)} -> {a.out_query}")
    print(f"saved basis            -> {a.out_basis}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cand_emb", default="data/lasco/bge_vl_cand_emb.pt")
    p.add_argument("--query_emb", default="data/lasco/bge_vl_query_emb.pt")
    p.add_argument("--out_cand", default="data/lasco/bge_vl_cand_emb_pca768.pt")
    p.add_argument("--out_query", default="data/lasco/bge_vl_query_emb_pca768.pt")
    p.add_argument("--out_basis", default="data/lasco/bge_vl_pca768_basis.pt")
    p.add_argument("--dim", type=int, default=768, help="student embedding dim")
    p.add_argument("--probe_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
