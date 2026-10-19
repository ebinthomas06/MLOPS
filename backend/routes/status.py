
from fastapi import APIRouter, HTTPException

from routes.repos import repositories

router = APIRouter(prefix="/status", tags=["Status"])

# Temporary in-memory run history.
# Replace with persistent storage if history must survive restarts.
pipeline_runs = {}


def record_pipeline_run(repo_id: int, status: str, result=None):
    """Record the latest pipeline state for a repository."""
    pipeline_runs[repo_id] = {
        "status": status,
        "result": result,
    }


@router.get("/{repo_id}")
def get_repository_status(repo_id: int):
    repository = next(
        (repo for repo in repositories if repo["id"] == repo_id),
        None,
    )

    if repository is None:
        raise HTTPException(
            status_code=404,
            detail="Repository not found",
        )

    run = pipeline_runs.get(repo_id)

    return {
        "repository": repository,
        "pipeline": {
            "status": run["status"] if run else "not_started",
            "last_run": None,
            "result": run["result"] if run else None,
        },
        "deployment": {
            "status": "not_deployed",
            "url": None,
        },
        "evidently": {
            "report_url": None,
        },
    }
