#!/bin/bash

find_project_root() {
  local source_path="$1"
  local dir
  dir="$(cd "$(dirname "$source_path")" && pwd)"

  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/.project-root" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done

  echo "Could not locate project root marker .project-root starting from $source_path" >&2
  return 1
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  echo "Python interpreter not found. Activate the project environment or set PYTHON_BIN." >&2
  return 1
}
