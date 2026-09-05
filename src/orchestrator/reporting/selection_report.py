"""Formatter for model / compute selection output."""

from __future__ import annotations

from typing import Iterable

from ..models.ranking import RankedCandidate
from ..models.selection import SelectionReport
from ..planning.compute import ComputeOption


def render_selection_report(ranked: Iterable[RankedCandidate], selection: SelectionReport) -> str:
    lines = ["Candidate ranking:"]
    for i, r in enumerate(list(ranked)[:10], start=1):
        f = "FEASIBLE" if r.feasibility is None or r.feasibility.feasible else "INFEASIBLE"
        lines.append(f"  {i}. {r.candidate.identifier[:35]:<35} score={r.score:5.1f}  {f}")
    lines.append("")
    lines.append(selection.render())
    return "\n".join(lines)


def render_compute_options(options: Iterable[ComputeOption], winner: ComputeOption | None, reason: str) -> str:
    lines = ["Compute options", ""]
    for o in options:
        lines.append(o.render())
        lines.append("")
    if winner:
        lines.append(f"Recommendation: {winner.name}")
        lines.append(f"Reason: {reason}")
    else:
        lines.append("No feasible compute option.")
    return "\n".join(lines)
