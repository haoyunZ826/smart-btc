#!/bin/bash
# Publish site/ to the gh-pages branch that GitHub Pages serves.
#
# Used by .github/workflows/monitor.yml and runnable by hand when CI is not
# available. Uses a detached worktree so the current checkout is never
# switched underneath you — a branch checkout mid-run is how a publish script
# ends up committing the wrong tree.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
WT=$(mktemp -d)

if [ ! -f site/data.json ]; then
  echo "site/data.json missing — run scripts/monitor.py first" >&2
  exit 1
fi

if [ -n "${CI:-}" ]; then
  git config user.name  "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
fi

cleanup() { git worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"; }
trap cleanup EXIT

git fetch origin gh-pages --depth 1 2>/dev/null || true
if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
  git worktree add -q "$WT" origin/gh-pages
  git -C "$WT" checkout -q -B gh-pages
else
  git worktree add -q --detach "$WT"
  git -C "$WT" checkout -q --orphan gh-pages
  git -C "$WT" rm -rq --cached . 2>/dev/null || true
  find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
fi

cp "$ROOT/site/index.html" "$ROOT/site/data.json" "$WT/"
touch "$WT/.nojekyll"

git -C "$WT" add -A
# A run with no new bar produces no diff. That is success, not failure.
if git -C "$WT" diff --cached --quiet; then
  echo "no dashboard change"
  exit 0
fi

git -C "$WT" -c commit.gpgsign=false commit -q -m "dashboard $(date -u '+%Y-%m-%d %H:%M') UTC"
git -C "$WT" push -q origin gh-pages
echo "published to gh-pages"
