#!/usr/bin/env python3
"""
Minimale Pfad-Konfiguration für bl-cox (Standalone-Repo)
OUTBOX_ROOT (ENV) oder fallback auf project_root/dynamic_system_outputs/outbox.
"""

import os
from pathlib import Path


class ProjectPaths:
    _project_root: Path = None

    @classmethod
    def _initialize_root(cls) -> Path:
        if cls._project_root is None:
            cls._project_root = Path(__file__).resolve().parent.parent
        return cls._project_root

    @classmethod
    def project_root(cls) -> Path:
        return cls._initialize_root()

    @classmethod
    def config_directory(cls) -> Path:
        shared = cls.project_root() / "config" / "shared" / "config"
        return shared if shared.exists() else cls.project_root() / "config"

    @classmethod
    def dynamic_system_outputs_directory(cls) -> Path:
        return cls.project_root() / "dynamic_system_outputs"

    # OUTBOX
    @classmethod
    def outbox_directory(cls) -> Path:
        env_root = os.environ.get("OUTBOX_ROOT")
        if env_root:
            try:
                p = Path(env_root).resolve()
                p.mkdir(parents=True, exist_ok=True)
                return p
            except Exception:
                pass
        return cls.dynamic_system_outputs_directory() / "outbox"

    @classmethod
    def outbox_cox_directory(cls) -> Path:
        return cls.outbox_directory() / "cox"

    @classmethod
    def outbox_cox_experiment_directory(cls, experiment_id: int) -> Path:
        return cls.outbox_cox_directory() / f"experiment_{int(experiment_id)}"

    # Optional: für JSON-DB-Kompatibilität
    @classmethod
    def outbox_churn_experiment_directory(cls, experiment_id: int) -> Path:
        return (cls.outbox_directory() / "churn") / f"experiment_{int(experiment_id)}"

    @classmethod
    def outbox_counterfactuals_directory(cls) -> Path:
        return cls.outbox_directory() / "counterfactuals"

    @classmethod
    def data_dictionary_file(cls) -> Path:
        return cls.config_directory() / "data_dictionary_optimized.json"

    @classmethod
    def feature_mapping_file(cls) -> Path:
        return cls.config_directory() / "feature_mapping.json"


