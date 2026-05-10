"""
LaSCo re-ranker fine-tuning dataset.

Returns flat (ref_img_path, text_edit, cand_img_path, label) tuples.
Label = 1 for the ground-truth positive, 0 for a hard negative.

Derives positives and hard negatives from the same distill_subset.json /
teacher_scores_qwen.json files used for retriever distillation — no extra
precomputation needed.

Each query contributes exactly:
  - 1 positive pair  (query → target image,  label = 1)
  - neg_per_query negative pairs  (query → hard-negative image, label = 0)

Total samples = max_queries × (1 + neg_per_query).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


class LaSCoReranker(Dataset):
    """Flat pairs dataset for binary relevance fine-tuning of Qwen3-VL."""

    def __init__(
        self,
        subset_path: str,
        scores_path: str,
        images_dir: str,
        neg_per_query: int = 1,
        max_queries: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.neg_per_query = neg_per_query

        with open(subset_path) as f:
            all_triplets: list[dict] = json.load(f)
        with open(scores_path) as f:
            raw: dict = json.load(f)

        teacher_scores: dict[int, dict] = {int(k): v for k, v in raw.items()}

        # Keep only triplets that have teacher scores
        triplets = [t for t in all_triplets if t["qid"] in teacher_scores]

        # qid → triplet lookup (abs covers negative-signed qids in older score files)
        by_qid: dict[int, dict] = {}
        for t in all_triplets:
            by_qid[t["qid"]] = t
            by_qid[-t["qid"]] = t

        # Optionally subsample queries
        rng = random.Random(seed)
        if max_queries is not None and max_queries < len(triplets):
            triplets = rng.sample(triplets, max_queries)

        # Build flat pair list
        self._pairs: list[tuple[str, str, str, int]] = []  # (ref, text, cand, label)
        for t in triplets:
            qid = t["qid"]
            ref_path = str(self.images_dir / t["query-image"][1])
            pos_path = str(self.images_dir / t["target-image"][1])
            text = t["query-text"]

            self._pairs.append((ref_path, text, pos_path, 1))

            neg_qids: list[int] = teacher_scores[qid]["neg_qids"]
            chosen = rng.sample(neg_qids, min(neg_per_query, len(neg_qids)))
            for nq in chosen:
                neg_triplet = by_qid.get(nq)
                if neg_triplet is None:
                    continue
                neg_path = str(self.images_dir / neg_triplet["target-image"][1])
                self._pairs.append((ref_path, text, neg_path, 0))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict:
        ref_path, text, cand_path, label = self._pairs[idx]
        return {
            "ref_path": ref_path,
            "text": text,
            "cand_path": cand_path,
            "label": label,
        }
