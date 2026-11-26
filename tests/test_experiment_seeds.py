"""Tests for standard FL seed lists."""
from acfl.experiment_seeds import EXTEND_SEEDS, JOURNAL_SEEDS, extend_seeds_csv, seeds_csv


def test_journal_seeds_are_ten():
    assert JOURNAL_SEEDS == list(range(42, 52))
    assert len(JOURNAL_SEEDS) == 10


def test_extend_seeds_start_at_45():
    assert EXTEND_SEEDS == list(range(45, 52))
    assert seeds_csv() == "42,43,44,45,46,47,48,49,50,51"
    assert extend_seeds_csv() == "45,46,47,48,49,50,51"
