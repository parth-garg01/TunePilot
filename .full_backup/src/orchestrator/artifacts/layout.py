"""Local project and experiment directory layout (PRD section 18)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    """
    projects/
      project-name/
        data/
        configs/
        experiments/
        jobs/
        checkpoints/
        models/
        evaluations/
        logs/
    """

    root: Path
    name: str

    @property
    def project_dir(self) -> Path:
        return self.root / self.name

    @property
    def data(self) -> Path:
        return self.project_dir / "data"

    @property
    def configs(self) -> Path:
        return self.project_dir / "configs"

    @property
    def experiments(self) -> Path:
        return self.project_dir / "experiments"

    @property
    def jobs(self) -> Path:
        return self.project_dir / "jobs"

    @property
    def checkpoints(self) -> Path:
        return self.project_dir / "checkpoints"

    @property
    def models(self) -> Path:
        return self.project_dir / "models"

    @property
    def evaluations(self) -> Path:
        return self.project_dir / "evaluations"

    @property
    def logs(self) -> Path:
        return self.project_dir / "logs"

    @property
    def db(self) -> Path:
        return self.project_dir / "orchestrator.sqlite"

    def ensure(self) -> None:
        for d in (
            self.project_dir,
            self.data,
            self.configs,
            self.experiments,
            self.jobs,
            self.checkpoints,
            self.models,
            self.evaluations,
            self.logs,
        ):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentLayout:
    """
    experiments/<experiment-id>/
      config/
      checkpoints/
      logs/
      model/
      evaluation/
    """

    root: Path
    experiment_id: str

    @property
    def base(self) -> Path:
        return self.root / self.experiment_id

    @property
    def config(self) -> Path:
        return self.base / "config"

    @property
    def checkpoints(self) -> Path:
        return self.base / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.base / "logs"

    @property
    def model(self) -> Path:
        return self.base / "model"

    @property
    def evaluation(self) -> Path:
        return self.base / "evaluation"

    def ensure(self) -> None:
        for d in (self.base, self.config, self.checkpoints, self.logs, self.model, self.evaluation):
            d.mkdir(parents=True, exist_ok=True)
