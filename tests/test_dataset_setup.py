"""Dataset preparation invariants that affect reportable experiments."""

from acdan.datasets.bfcl import stratified_bfcl_split


def test_bfcl_split_is_stable_stratified_and_disjoint():
    rows = [
        {"task_id": f"{category}-{i}", "category": category}
        for category in ("simple", "parallel", "multiple")
        for i in range(10)
    ]
    dev_a, test_a = stratified_bfcl_split(rows, dev_fraction=0.2)
    dev_b, test_b = stratified_bfcl_split(list(reversed(rows)), dev_fraction=0.2)
    assert dev_a == dev_b
    assert test_a == test_b
    assert len(dev_a) == 6
    assert len(test_a) == 24
    assert {row["task_id"] for row in dev_a}.isdisjoint(
        row["task_id"] for row in test_a
    )
