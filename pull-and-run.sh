#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: bash pull-and-run.sh multihopspatial {smoke|full}\n'
}

if [[ "$#" -eq 1 && "$1" == --help ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 2 || "$1" != multihopspatial || ( "$2" != smoke && "$2" != full ) ]]; then
  usage >&2
  exit 2
fi

remote=https://github.com/21cimu/VLM-Reasoning.git
export VLM_ROOT_DIR="${VLM_ROOT_DIR:-/home/ubuntu/vLM-reasoning}"
checkout="${VLM_REPO_DIR:-$VLM_ROOT_DIR/repo}"
mkdir -p "$(dirname -- "$checkout")"
if [[ -e "$checkout/.git" ]]; then
  if [[ "$(git -C "$checkout" remote get-url origin)" != "$remote" ||
        "$(git -C "$checkout" branch --show-current)" != main ]]; then
    printf 'Expected a main checkout of %s at %s\n' "$remote" "$checkout" >&2
    exit 1
  fi
  git -C "$checkout" diff --quiet HEAD -- || {
    printf 'Tracked files have local changes in %s\n' "$checkout" >&2
    exit 1
  }
  git -C "$checkout" pull --ff-only origin main
else
  git clone --branch main "$remote" "$checkout"
fi

exec bash "$checkout/experiments/multihopspatial/run.sh" "$2"
