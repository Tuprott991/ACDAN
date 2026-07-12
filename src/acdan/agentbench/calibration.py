"""Small dependency-free Platt calibrator for held-out selector confidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class PlattCalibrator:
    slope: float = 1.0
    intercept: float = 0.0
    epsilon: float = 1e-6

    def predict(self, confidence: float) -> float:
        p = float(np.clip(confidence, self.epsilon, 1.0 - self.epsilon))
        logit = np.log(p / (1.0 - p))
        z = np.clip(self.slope * logit + self.intercept, -30.0, 30.0)
        return float(1.0 / (1.0 + np.exp(-z)))

    def fit(
        self,
        confidences: Sequence[float],
        correct: Sequence[bool],
        *,
        epochs: int = 1000,
        lr: float = 0.05,
        l2: float = 1e-3,
    ) -> "PlattCalibrator":
        if len(confidences) != len(correct) or not confidences:
            raise ValueError("calibration requires equally sized non-empty inputs")
        p = np.clip(np.asarray(confidences, dtype=np.float64), self.epsilon, 1.0 - self.epsilon)
        x = np.log(p / (1.0 - p))
        y = np.asarray(correct, dtype=np.float64)
        a, b = float(self.slope), float(self.intercept)
        for _ in range(epochs):
            z = np.clip(a * x + b, -30.0, 30.0)
            pred = 1.0 / (1.0 + np.exp(-z))
            error = pred - y
            a -= lr * (float(np.mean(error * x)) + l2 * a)
            b -= lr * float(np.mean(error))
        self.slope, self.intercept = a, b
        return self

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
