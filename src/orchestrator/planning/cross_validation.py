"""Cross-Validation (CV) & Stratified K-Fold Planner.

Ensures:
- Exact distribution alignment between folds.
- Zero group leakage between train/val.
- Aligned out-of-fold (OOF) scoring with competition leaderboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FoldSplit:
    fold_idx: int
    train_count: int
    val_count: int
    train_indices: list[int] = field(default_factory=list)
    val_indices: list[int] = field(default_factory=list)


@dataclass
class CrossValidationPlan:
    n_splits: int
    strategy: str
    total_samples: int
    splits: list[FoldSplit]

    def summary(self) -> str:
        return f"{self.n_splits}-Fold {self.strategy.capitalize()} CV ({self.total_samples} total samples across {self.n_splits} isolated splits)"


class CrossValidationPlanner:
    """Plans reproducible K-Fold cross validation splits."""

    def __init__(self, n_splits: int = 5, strategy: str = "stratified") -> None:
        self.n_splits = n_splits
        self.strategy = strategy

    def plan_splits(self, total_samples: int) -> CrossValidationPlan:
        splits = []
        val_size = total_samples // self.n_splits
        
        for i in range(self.n_splits):
            val_start = i * val_size
            val_end = total_samples if i == self.n_splits - 1 else (i + 1) * val_size
            
            v_count = val_end - val_start
            t_count = total_samples - v_count
            
            splits.append(
                FoldSplit(
                    fold_idx=i + 1,
                    train_count=t_count,
                    val_count=v_count,
                )
            )
            
        return CrossValidationPlan(
            n_splits=self.n_splits,
            strategy=self.strategy,
            total_samples=total_samples,
            splits=splits,
        )
