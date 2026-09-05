"""Model discovery, filtering, ranking, and selection (PRD section 10-11)."""

from .candidate import ModelCandidate, ModelMetadata, TrainingStage
from .discovery import ModelDiscovery, ModelSource
from .sources import (
    HuggingFaceSource,
    ModelScopeSource,
    UnslothSource,
    NgcSource,
    LocalRegistrySource,
    UserSpecifiedSource,
    OfficialProviderSource,
)
from .filters import CandidateFilter, FilterCriteria
from .ranking import ModelRanker, RankingWeights, RankedCandidate
from .selection import SelectionPolicy, SelectionReport, Objective
from .dedupe import deduplicate

__all__ = [
    "ModelCandidate",
    "ModelMetadata",
    "TrainingStage",
    "ModelDiscovery",
    "ModelSource",
    "HuggingFaceSource",
    "ModelScopeSource",
    "UnslothSource",
    "NgcSource",
    "LocalRegistrySource",
    "UserSpecifiedSource",
    "OfficialProviderSource",
    "CandidateFilter",
    "FilterCriteria",
    "ModelRanker",
    "RankingWeights",
    "RankedCandidate",
    "SelectionPolicy",
    "SelectionReport",
    "Objective",
    "deduplicate",
]
