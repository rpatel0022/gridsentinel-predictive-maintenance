"""Validate the AWS task definition — structure + no leaked secret. Stdlib only."""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK_DEF = ROOT / "infra/aws/task-definition.json"


def test_task_definition_valid_and_fargate():
    td = json.loads(TASK_DEF.read_text())
    assert "FARGATE" in td["requiresCompatibilities"]
    assert td["containerDefinitions"], "no containers defined"


def test_container_exposes_and_healthchecks_the_api():
    td = json.loads(TASK_DEF.read_text())
    api = next(c for c in td["containerDefinitions"] if c["name"] == "api")
    assert any(p["containerPort"] == 8000 for p in api["portMappings"])
    assert "/health" in " ".join(api["healthCheck"]["command"])


def test_secret_via_ssm_not_hardcoded():
    raw = TASK_DEF.read_text()
    td = json.loads(raw)
    api = next(c for c in td["containerDefinitions"] if c["name"] == "api")
    # EIA key is injected from SSM, never as a plaintext value.
    assert any(s["name"] == "EIA_API_KEY" and "valueFrom" in s for s in api["secrets"])
    assert "GV4Q0" not in raw  # no real key ever lands here
    env_names = {e["name"] for e in api.get("environment", [])}
    assert "EIA_API_KEY" not in env_names  # not a plaintext env var
