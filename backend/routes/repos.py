
from fastapi import APIRouter

router = APIRouter(prefix="/repos", tags=["Repositories"])

# Temporary in-memory store until GitHub integration is added.
repositories = []


@router.post("/connect")
def connect_repository(
    name: str,
    full_name: str,
    clone_url: str,
):
    repository = {
        "id": len(repositories) + 1,
        "name": name,
        "full_name": full_name,
        "clone_url": clone_url,
        "status": "connected",
    }

    repositories.append(repository)
    return repository


@router.get("")
def list_repositories():
    return {"repositories": repositories}
