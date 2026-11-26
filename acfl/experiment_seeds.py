"""Standard FL seed lists for journal and scaling benchmarks."""

from __future__ import annotations

from typing import Iterable, List

# Primary journal protocol: ten independent FL seeds (42–51 inclusive).
JOURNAL_SEEDS: List[int] = list(range(42, 52))

# Seeds added when extending an existing three-seed (42–44) run to ten seeds.
EXTEND_SEEDS: List[int] = list(range(45, 52))

LEGACY_THREE_SEEDS: List[int] = [42, 43, 44]


def seeds_csv(seeds: Iterable[int] | None = None) -> str:
    """Comma-separated seed string for argparse ``--seeds`` defaults."""
    return ",".join(str(s) for s in (seeds if seeds is not None else JOURNAL_SEEDS))


def extend_seeds_csv() -> str:
    return seeds_csv(EXTEND_SEEDS)
