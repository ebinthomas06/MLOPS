
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from routes.repos import repositories
from routes.status import record_pipeline_run
from services.validator import validate_repository

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


def get_repository(repo_id: int) -> dict:
    """Find a connected repository by its ID."""
    repository = next(
        (repo for repo in repositories if repo["id"] == repo_id),
        None,
    )

    if repository is None:
        raise HTTPException(
            status_code=404,
            detail=f"Repository with ID {repo_id} was not found.",
        )

    return repository


def get_local_repository_path(repo_id: int) -> Path:
    """
    Resolve the local checkout path for a repository.

    Set its local_path when connecting the repository, or configure
    AUTOMLOPS_REPO_ROOT to use a root directory containing repo folders.
    """
    repository = get_repository(repo_id)

    configured_path = repository.get("local_path")

    if configured_path:
        repo_path = Path(configured_path).expanduser().resolve()
    else:
        repo_root = os.getenv("AUTOMLOPS_REPO_ROOT")

        if not repo_root:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No local checkout is configured for this repository. "
                    "Connect or clone the repository locally and set its "
                    "local_path, or configure AUTOMLOPS_REPO_ROOT."
                ),
            )

        repo_path = (Path(repo_root).expanduser() / repository["name"]).resolve()

    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Local repository directory does not exist: {repo_path}",
        )

    return repo_path


@router.post("/{repo_id}/validate")
def validate_pipeline(repo_id: int):
    """
    Validate the connected repository's mlops.yaml configuration.

    This endpoint does not start model training or deployment.
    """
    repository = get_repository(repo_id)

    try:
        repo_path = get_local_repository_path(repo_id)
        result = validate_repository(str(repo_path))

    except HTTPException as exc:
        record_pipeline_run(
            repo_id,
            "validation_error",
            {"valid": False, "errors": [exc.detail]},
        )
        raise

    except Exception:
        record_pipeline_run(
            repo_id,
            "validation_error",
            {
                "valid": False,
                "errors": ["An unexpected error occurred during validation."],
            },
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during validation.",
        )

    status = "validation_passed" if result["valid"] else "validation_failed"

    record_pipeline_run(repo_id, status, result)

    return {
        "repository_id": repo_id,
        "repository": repository["full_name"],
        "pipeline_status": status,
        "validation": result,
        "training_started": False,
        "deployment_started": False,
    }


@router.get("/{repo_id}/validation")
def get_validation_status(
    repo_id: int,
    include_details: bool = Query(default=True),
):
    """Return the latest recorded validation result for a repository."""
    repository = get_repository(repo_id)

    # Read the in-memory status tracker.
    from routes.status import pipeline_runs

    run = pipeline_runs.get(repo_id)

    if run is None:
        return {
            "repository_id": repo_id,
            "repository": repository["full_name"],
            "pipeline_status": "not_started",
            "validation": None,
        }

    result = run.get("result")

    return {
        "repository_id": repo_id,
        "repository": repository["full_name"],
        "pipeline_status": run["status"],
        "validation": result if include_details else None,
    }
