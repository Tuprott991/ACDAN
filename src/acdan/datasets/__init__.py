"""Dataset adapters: turn local files into ACDAN tasks + outcome checkers.

Nothing here downloads data. Point each adapter at a local JSONL you have
already prepared on the VM (schema documented in ``base.py``). A synthetic
fallback adapter lets the whole real pipeline run offline for smoke tests.
"""

from acdan.datasets.base import (
    RawTask,
    ANSWER_SELECTION_DATASETS,
    MATH_DATASETS,
    ROADMAP_ANSWER_DATASETS,
    SyntheticRawDataset,
    JSONLRawDataset,
    outcome_exact,
    outcome_tool_sequence,
    build_dataset,
    build_outcome_checker,
)

__all__ = [
    "RawTask",
    "ANSWER_SELECTION_DATASETS",
    "MATH_DATASETS",
    "ROADMAP_ANSWER_DATASETS",
    "SyntheticRawDataset",
    "JSONLRawDataset",
    "outcome_exact",
    "outcome_tool_sequence",
    "build_dataset",
    "build_outcome_checker",
]
