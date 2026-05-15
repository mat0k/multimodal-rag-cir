"""
LaSCo re-ranker dataset using retriever-mined hard negatives.

Drop-in replacement for LaSCoReranker, but negatives come from
precomputed retriever similarity search instead of teacher score neg_qids.

Hard negatives file format (from scripts/precompute_hard_negatives.py):
  {"<qid>": ["train2014/img1.jpg", ...], ...}   (relative to images_dir)
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LaSCoRerankerHardNeg(Dataset):
    """Flat (ref, text, candidate, label) pairs using retriever hard negatives."""

    def __init__(
        self,
        subset_path: str,
        hard_neg_path: str,
        images_dir: str,
        neg_per_query: int = 1,
        max_queries: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.images_dir = Path(images_dir)

        with open(subset_path) as f:
            all_triplets: list[dict] = json.load(f)
        with open(hard_neg_path) as f:
            hard_neg_map: dict[str, list[str]] = json.load(f)

        triplets = [t for t in all_triplets if str(t["qid"]) in hard_neg_map]

        rng = random.Random(seed)
        if max_queries is not None and max_queries < len(triplets):
            triplets = rng.sample(triplets, max_queries)

        self._pairs: list[tuple[str, str, str, int]] = []
        missing = 0
        for t in triplets:
            qid = str(t["qid"])
            ref_path = str(self.images_dir / t["query-image"][1])
            pos_path = str(self.images_dir / t["target-image"][1])
            text = t["query-text"]

            self._pairs.append((ref_path, text, pos_path, 1))

            negs = hard_neg_map.get(qid, [])
            if not negs:
                missing += 1
                continue
            chosen = rng.sample(negs, min(neg_per_query, len(negs)))
            for neg_rel in chosen:
                self._pairs.append((ref_path, text, str(self.images_dir / neg_rel), 0))

        if missing:
            logger.warning(f"{missing} queries had no hard negatives and were skipped.")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict:
        ref_path, text, cand_path, label = self._pairs[idx]
        return {"ref_path": ref_path, "text": text, "cand_path": cand_path, "label": label}
