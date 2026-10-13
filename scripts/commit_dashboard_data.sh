#!/usr/bin/env bash
set -euo pipefail

message="$1"
shift

branch="${GITHUB_REF_NAME:-main}"

git add "$@"
if git diff --cached --quiet; then
  echo "No dashboard data changes."
  exit 0
fi

git commit -m "$message"

for attempt in 1 2 3; do
  git fetch origin "$branch"
  git rebase "origin/$branch"
  if git push origin "HEAD:$branch"; then
    exit 0
  fi
  echo "Push attempt ${attempt} failed; retrying after syncing ${branch}."
  sleep 5
done

echo "Unable to push dashboard data after retries."
exit 1
