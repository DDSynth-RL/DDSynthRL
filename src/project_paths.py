from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".project-root").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root marker .project-root starting from {start}"
    )


_PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def get_project_root() -> Path:
    return _PROJECT_ROOT


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def require_project_relative_path(
    path_like: str | Path,
    *,
    label: str,
    source: Path,
) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        raise ValueError(
            f"Recipe {source} must define `{label}` as a project-relative path, "
            f"not absolute path {path}."
        )
    return (_PROJECT_ROOT / path).resolve()


def project_relative_string(path_like: str | Path) -> str:
    resolved = resolve_project_path(path_like).resolve()
    try:
        return str(resolved.relative_to(_PROJECT_ROOT))
    except ValueError as exc:
        raise ValueError(
            f"Path {resolved} must remain inside project root {_PROJECT_ROOT}."
        ) from exc
