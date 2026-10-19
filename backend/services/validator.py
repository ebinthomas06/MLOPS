
from pathlib import Path

import yaml


REQUIRED_FILES = ["mlops.yaml"]

REQUIRED_SECTIONS = [
    "version",
    "project",
    "model",
    "data",
    "training",
    "evaluation",
    "quality_gate",
    "deployment",
    "monitoring",
]


def validate_repository(repo_path: str) -> dict:
    path = Path(repo_path)

    if not path.exists():
        return {
            "valid": False,
            "errors": [f"Repository path does not exist: {repo_path}"],
        }

    if not path.is_dir():
        return {
            "valid": False,
            "errors": [f"Repository path is not a directory: {repo_path}"],
        }

    errors = []

    for filename in REQUIRED_FILES:
        if not (path / filename).is_file():
            errors.append(f"Missing required file: {filename}")

    if errors:
        return {"valid": False, "errors": errors}

    config_path = path / "mlops.yaml"

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        return {
            "valid": False,
            "errors": [f"Invalid YAML syntax: {exc}"],
        }
    except OSError as exc:
        return {
            "valid": False,
            "errors": [f"Could not read mlops.yaml: {exc}"],
        }

    if not isinstance(config, dict):
        return {
            "valid": False,
            "errors": ["mlops.yaml must contain a YAML object."],
        }

    for section in REQUIRED_SECTIONS:
        if section not in config:
            errors.append(f"Missing required field: {section}")

    if errors:
        return {"valid": False, "errors": errors}

    for section in [
        "project",
        "model",
        "data",
        "training",
        "evaluation",
        "quality_gate",
        "deployment",
        "monitoring",
    ]:
        if not isinstance(config[section], dict):
            errors.append(f"'{section}' must be an object.")

    if errors:
        return {"valid": False, "errors": errors}

    def require_string(section, field):
        value = config[section].get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"'{section}.{field}' must be a non-empty string."
            )

    require_string("project", "name")
    require_string("model", "task")
    require_string("model", "framework")
    require_string("model", "entrypoint")
    require_string("model", "artifact")
    require_string("data", "path")
    require_string("data", "target_column")
    require_string("training", "command")
    require_string("evaluation", "primary_metric")
    require_string("quality_gate", "metric")
    require_string("quality_gate", "comparison")
    require_string("deployment", "entrypoint")
    require_string("monitoring", "report_type")

    if not isinstance(config["evaluation"].get("metrics"), list):
        errors.append("'evaluation.metrics' must be a list.")

    if not isinstance(config["quality_gate"].get("min_value"), (int, float)) \
            or isinstance(config["quality_gate"].get("min_value"), bool):
        errors.append("'quality_gate.min_value' must be a number.")

    else:
        threshold = config["quality_gate"]["min_value"]
        if not 0 <= threshold <= 1:
            errors.append(
                "'quality_gate.min_value' must be between 0 and 1."
            )

    if config["quality_gate"].get("comparison") not in [
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
    ]:
        errors.append("Invalid 'quality_gate.comparison'.")

    test_size = config["data"].get("test_size")
    if not isinstance(test_size, (int, float)) \
            or isinstance(test_size, bool) \
            or not 0 < test_size < 1:
        errors.append("'data.test_size' must be between 0 and 1.")

    port = config["deployment"].get("port")
    if not isinstance(port, int) or isinstance(port, bool) \
            or not 1 <= port <= 65535:
        errors.append(
            "'deployment.port' must be an integer from 1 to 65535."
        )

    for section in ["deployment", "monitoring"]:
        if not isinstance(config[section].get("enabled"), bool):
            errors.append(f"'{section}.enabled' must be true or false.")

    if not isinstance(config["monitoring"].get("drift_detection"), bool):
        errors.append("'monitoring.drift_detection' must be true or false.")

    if not isinstance(config["version"], str):
        errors.append("'version' must be a string, such as '1.0'.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
