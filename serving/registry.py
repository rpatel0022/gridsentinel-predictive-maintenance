"""A file-based model registry — stages, promotion, rollback, and an audit trail.

Production ML needs to answer "which model is live, and how do I put the last one
back?" without a redeploy. This is a lightweight registry that does exactly that on
disk: each model version is a saved bundle, a ``registry.json`` tracks which version
is in each stage, and every transition is appended to an **audit log** (who/what
moved which version when) — the governance trail the role asks for.

The stage-transition logic (:func:`promote`, :func:`rollback`) is pure and operates
on a plain dict, so it is fully unit-testable without joblib or a model. The
:class:`ModelRegistry` class wraps it with bundle IO.
"""

from __future__ import annotations

import copy
import json
import os

PRODUCTION = "production"
CANDIDATE = "candidate"


def empty_state() -> dict:
    """A fresh registry state."""
    return {
        "stages": {PRODUCTION: None, CANDIDATE: None},
        "versions": {},
        "production_history": [],
        "audit": [],
    }


def _audit(state: dict, action: str, version: str | None, at: str, **extra) -> None:
    entry = {"action": action, "version": version, "at": at}
    entry.update(extra)
    state["audit"].append(entry)


def register(state: dict, version: str, at: str, *, metrics: dict | None = None) -> dict:
    """Record a new version in the CANDIDATE stage (pure)."""
    state = copy.deepcopy(state)
    if version in state["versions"]:
        raise ValueError(f"version already registered: {version}")
    state["versions"][version] = {"metrics": metrics or {}, "created": at}
    state["stages"][CANDIDATE] = version
    _audit(state, "register", version, at, stage=CANDIDATE)
    return state


def promote(state: dict, version: str, at: str) -> dict:
    """Promote ``version`` to PRODUCTION, remembering the prior prod for rollback."""
    state = copy.deepcopy(state)
    if version not in state["versions"]:
        raise ValueError(f"unknown version: {version}")
    previous = state["stages"][PRODUCTION]
    state["stages"][PRODUCTION] = version
    state["production_history"].append(version)
    _audit(state, "promote", version, at, **{"from": previous})
    return state


def rollback(state: dict, at: str) -> dict:
    """Restore the previously-promoted production version.

    Raises:
        ValueError: If there is no prior production version to roll back to.
    """
    state = copy.deepcopy(state)
    history = state["production_history"]
    if len(history) < 2:
        raise ValueError("no previous production version to roll back to")
    rolled_from = history.pop()
    restored = history[-1]
    state["stages"][PRODUCTION] = restored
    _audit(state, "rollback", restored, at, **{"from": rolled_from})
    return state


class ModelRegistry:
    """Directory-backed registry: ``<root>/registry.json`` + ``<root>/<version>.joblib``."""

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._path = os.path.join(root, "registry.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path) as fh:
                return json.load(fh)
        return empty_state()

    def _save_state(self) -> None:
        with open(self._path, "w") as fh:
            json.dump(self.state, fh, indent=2)

    def _bundle_path(self, version: str) -> str:
        return os.path.join(self.root, f"{version}.joblib")

    def register(self, bundle, at: str, *, metrics: dict | None = None) -> str:
        """Save ``bundle`` and record it as the candidate. Returns its version."""
        import joblib

        version = bundle.version
        joblib.dump(bundle, self._bundle_path(version))
        self.state = register(self.state, version, at, metrics=metrics or bundle.metrics)
        self._save_state()
        return version

    def promote(self, version: str, at: str) -> None:
        self.state = promote(self.state, version, at)
        self._save_state()

    def rollback(self, at: str) -> None:
        self.state = rollback(self.state, at)
        self._save_state()

    def stage_version(self, stage: str) -> str | None:
        return self.state["stages"].get(stage)

    def load(self, stage: str = PRODUCTION):
        """Load the bundle currently in ``stage``."""
        import joblib

        version = self.stage_version(stage)
        if version is None:
            raise FileNotFoundError(f"no model in stage {stage}")
        return joblib.load(self._bundle_path(version))

    @property
    def audit(self) -> list[dict]:
        return self.state["audit"]
