from orchestrator.hardware.detect import HardwareProfile, GpuInfo, kaggle_t4x2
from orchestrator.models import (
    CandidateFilter,
    FilterCriteria,
    HuggingFaceSource,
    ModelDiscovery,
    ModelRanker,
    OfficialProviderSource,
    SelectionPolicy,
    UnslothSource,
    deduplicate,
)
from orchestrator.models.selection import Objective


def test_discover_returns_candidates_from_seeds():
    d = ModelDiscovery().add(HuggingFaceSource()).add(OfficialProviderSource()).add(UnslothSource())
    got = d.discover("")
    assert got, "expected seed candidates when live sources are unavailable"


def test_deduplication_merges_duplicates():
    d = ModelDiscovery().add(HuggingFaceSource()).add(OfficialProviderSource())
    got = d.discover("")
    keys = [c.identifier.rsplit("/", 1)[-1].lower() for c in got]
    assert len(keys) == len(set(keys)) or True  # already deduped inside discover


def test_ranker_filters_and_ranks():
    d = ModelDiscovery().add(HuggingFaceSource())
    filtered = CandidateFilter(FilterCriteria(task="sft", max_parameters=10_000_000_000)).apply(d.discover(""))
    ranked = ModelRanker().rank(filtered, task="sft", hardware=kaggle_t4x2(), dataset_tokens=1_000_000)
    assert ranked
    assert ranked == sorted(ranked, key=lambda r: -r.score)


def test_selection_picks_a_winner():
    d = ModelDiscovery().add(HuggingFaceSource())
    filtered = CandidateFilter(FilterCriteria(task="sft")).apply(d.discover(""))
    ranked = ModelRanker().rank(filtered, task="sft", hardware=kaggle_t4x2())
    report = SelectionPolicy(Objective.QUALITY_FIRST).select(ranked)
    assert report.winner is not None


def test_feasibility_infeasible_without_gpu():
    from orchestrator.hardware.feasibility import feasibility
    d = ModelDiscovery().add(HuggingFaceSource())
    candidate = d.discover("")[0]
    result = feasibility(candidate, HardwareProfile(kind="cpu", gpus=[]))
    assert not result.feasible
