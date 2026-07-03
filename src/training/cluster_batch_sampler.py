"""
Batch sampler that forms each batch from ONE semantic cluster, so the SP
candidate-candidate similarity matrix has real structure to distil.

- cluster_ids=None  -> random floor ablation (all items in one pseudo-cluster).
- moderate/tight    -> pass the k-means labels; batches are drawn within a cluster.

Per epoch it reshuffles both the within-cluster item order and the visiting
order of the batches. Per-cluster remainders (< batch_size) are dropped so every
batch is full and single-cluster.
"""
import random
from collections import defaultdict
from typing import Iterator, Optional


class ClusterBatchSampler:
    def __init__(
        self,
        cluster_ids: Optional[list[int]],
        batch_size: int,
        n_items: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self._seed = seed

        if cluster_ids is None:
            if n_items is None:
                raise ValueError("n_items required when cluster_ids is None (random mode).")
            self.clusters = {0: list(range(n_items))}
        else:
            buckets: dict[int, list[int]] = defaultdict(list)
            for idx, c in enumerate(cluster_ids):
                buckets[c].append(idx)
            self.clusters = dict(buckets)

        self._num_batches = sum(len(v) // batch_size for v in self.clusters.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed + self.epoch)
        batches: list[list[int]] = []
        for members in self.clusters.values():
            items = list(members)
            if self.shuffle:
                rng.shuffle(items)
            n_full = len(items) // self.batch_size
            for b in range(n_full):
                batches.append(items[b * self.batch_size : (b + 1) * self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
