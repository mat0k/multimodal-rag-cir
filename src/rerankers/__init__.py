try:
	from .lamra_rank import LamRARanker
except ModuleNotFoundError:
	# lamra_rank.py may be added later during staged implementation.
	LamRARanker = None

__all__ = ["LamRARanker"]
