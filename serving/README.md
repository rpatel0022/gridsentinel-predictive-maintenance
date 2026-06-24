# serving/

The production inference service (Phase 3+): FastAPI app with pydantic
request/response validation, Dockerfile, docker-compose stack, and the registry
rollback/canary wiring. Serves cost-optimised predictions (see
`src/gridsentinel/cost.py`).
