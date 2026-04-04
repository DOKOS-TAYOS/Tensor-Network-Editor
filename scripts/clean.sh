#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

FAILED=0

printf 'Cleaning generated artifacts in "%s"\n' "$PROJECT_ROOT"

remove_dir() {
  target_path=$1
  if [ -d "$target_path" ]; then
    printf 'Removing directory "%s"\n' "$target_path"
    rm -rf -- "$target_path" || FAILED=1
    if [ -e "$target_path" ]; then
      printf 'Failed to remove directory "%s"\n' "$target_path"
      FAILED=1
    fi
  fi
}

remove_dir_warn() {
  target_path=$1
  if [ -d "$target_path" ]; then
    printf 'Removing directory "%s"\n' "$target_path"
    rm -rf -- "$target_path" || true
    if [ -e "$target_path" ]; then
      printf 'Warning: could not remove directory "%s"\n' "$target_path"
    fi
  fi
}

remove_file_pattern() {
  pattern=$1
  found_match=0
  for target_path in $pattern; do
    if [ -e "$target_path" ]; then
      found_match=1
      printf 'Removing file "%s"\n' "$target_path"
      rm -f -- "$target_path" || FAILED=1
      if [ -e "$target_path" ]; then
        printf 'Failed to remove file "%s"\n' "$target_path"
        FAILED=1
      fi
    fi
  done
  if [ "$found_match" -eq 0 ]; then
    :
  fi
}

remove_glob_dirs() {
  pattern=$1
  for target_path in $pattern; do
    if [ -d "$target_path" ]; then
      remove_dir "$target_path"
    fi
  done
}

remove_glob_dirs_warn() {
  pattern=$1
  for target_path in $pattern; do
    if [ -d "$target_path" ]; then
      remove_dir_warn "$target_path"
    fi
  done
}

remove_named_dirs() {
  base_dir=$1
  dir_name=$2
  if [ ! -d "$base_dir" ]; then
    return
  fi

  find "$base_dir" -type d -name "$dir_name" -prune 2>/dev/null | while IFS= read -r target_path; do
    remove_dir "$target_path"
  done
}

remove_dir ".pytest_cache"
remove_dir ".mypy_cache"
remove_dir ".ruff_cache"
remove_dir ".test_output"
remove_dir "build"
remove_dir "dist"
remove_dir "htmlcov"

remove_glob_dirs "./*.egg-info"
remove_glob_dirs "./src/*.egg-info"
remove_glob_dirs_warn "./pytest-cache-files-*"

remove_file_pattern "./.coverage"
remove_file_pattern "./.coverage.*"
remove_file_pattern "./coverage.xml"

remove_dir "__pycache__"
remove_named_dirs "./src" "__pycache__"
remove_named_dirs "./tests" "__pycache__"
remove_named_dirs "./examples" "__pycache__"
remove_named_dirs "./scripts" "__pycache__"

if [ "$FAILED" -ne 0 ]; then
  printf 'Clean finished with errors.\n'
  exit 1
fi

printf 'Clean finished successfully.\n'
