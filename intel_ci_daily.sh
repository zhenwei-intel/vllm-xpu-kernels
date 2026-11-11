#!/usr/bin/env bash
# Daily: pick the newest finished failing "Full intel CI-daily" build on main,
# fetch failed logs via ci-fetch-log.sh, and run copilot with /intel-ci-fix.
#
# Usage: intel_ci_daily.sh [--dry-run] [build_number]
# Env (or $ENV_FILE): REPO, HF_TOKEN, HF_HOME, GPUS, COPILOT_BIN,
#   COPILOT_MODEL_ID, STATE_DIR, AGENT_TIMEOUT
set -euo pipefail

DRY_RUN=0
[[ ${1:-} == --dry-run ]] && { DRY_RUN=1; shift; }
BUILD=${1:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SKILL_DIR=$SCRIPT_DIR/.claude/skills/intel-ci-fix
ENV_FILE=${ENV_FILE:-$HOME/.config/intel-ci-fixer.env}
# shellcheck disable=SC1090
[[ -f $ENV_FILE ]] && source "$ENV_FILE"

REPO=${REPO:-/data/zhenweil/vllm}
COPILOT_BIN=${COPILOT_BIN:-copilot}
MODEL=${COPILOT_MODEL_ID:-claude-opus-5.5}
STATE_DIR=${STATE_DIR:-$HOME/.cache/intel-ci-fixer}
GPUS=${GPUS:-0,1,2,3}
HF_HOME=${HF_HOME:-/data/hf_home}
AGENT_TIMEOUT=${AGENT_TIMEOUT:-3h}
DAILY_MSG='Full intel CI-daily'
PIPELINE=https://buildkite.com/vllm/intel-ci
LIST_URL="$PIPELINE/builds?branch=main&created_from_days=7&query=Full+intel+CI-daily"

((DRY_RUN)) || : "${HF_TOKEN:?set HF_TOKEN in $ENV_FILE}"
mkdir -p "$STATE_DIR/reports" "$STATE_DIR/logs" "$HOME/.copilot/skills"
touch "$STATE_DIR/done.txt"
ln -sfn "$SKILL_DIR" "$HOME/.copilot/skills/intel-ci-fix"

exec 9>"$STATE_DIR/lock"
flock -n 9 || { echo "previous run still active"; exit 0; }

bk() { curl -fsSL --retry 3 --max-time 120 -A Mozilla/5.0 "$@"; }

if [[ -z $BUILD ]]; then
  for b in $(bk "$LIST_URL" | grep -o 'intel-ci/builds/[0-9]\+' | grep -o '[0-9]\+$' | sort -rnu); do
    info=$(bk "$PIPELINE/builds/$b.json")
    jq -e --arg m "$DAILY_MSG" '.branch_name == "main" and (.message | contains($m))' \
      <<<"$info" >/dev/null || continue
    [[ -n $(jq -r '.finished_at // empty' <<<"$info") ]] || continue
    BUILD=$b
    break
  done
fi
[[ -n $BUILD ]] || { echo "no finished daily build found"; exit 0; }
if [[ $# -eq 0 ]] && grep -qx "$BUILD" "$STATE_DIR/done.txt"; then
  echo "build $BUILD already processed"
  exit 0
fi

BUILD_URL="$PIPELINE/builds/$BUILD"
JOBS=$("$SKILL_DIR/scripts/ci-fetch-log.sh" --metadata "$BUILD_URL")
if [[ -z $JOBS ]]; then
  echo "build $BUILD has no failed jobs"
  echo "$BUILD" >>"$STATE_DIR/done.txt"
  exit 0
fi

echo "=== build $BUILD ($BUILD_URL) ==="
echo "$JOBS"
((DRY_RUN)) && exit 0

STAMP=$(date +%F-%H%M)
RUN_ID=icf-$STAMP-$$
REPORT=$STATE_DIR/reports/$BUILD.md
SESSION=$STATE_DIR/logs/$BUILD-$STAMP.session.md
trap 'docker ps -aq --filter "label=intel-ci-fixer=$RUN_ID" | xargs -r docker rm -f >/dev/null 2>&1 || true' EXIT

DENY_ARGS=()
for t in \
  'shell(git push)' 'shell(gh pr create)' 'shell(gh pr merge)' 'shell(gh pr ready)' \
  'shell(gh pr comment)' 'shell(gh pr review)' 'shell(gh pr close)' \
  'shell(gh issue create)' 'shell(gh issue comment)' \
  'shell(docker rmi)' 'shell(docker system)' 'shell(docker image prune)'; do
  DENY_ARGS+=(--deny-tool "$t")
done

PROMPT="Use the /intel-ci-fix skill to analyze $BUILD_URL.
Pre-fetched failed job logs (<log_path>\t<job_name>\t<metadata_json>):
$JOBS
Use only GPUs $GPUS (ZE_AFFINITY_MASK=$GPUS) and add '--label intel-ci-fixer=$RUN_ID' to every docker run.
Write the final markdown report to $REPORT and print a Chinese summary (中文回答)."

cd "$REPO"
ZE_AFFINITY_MASK=$GPUS HF_TOKEN=$HF_TOKEN HF_HOME=$HF_HOME \
  timeout -k 2m "$AGENT_TIMEOUT" "$COPILOT_BIN" --model "$MODEL" -p "$PROMPT" \
  --allow-all-tools "${DENY_ARGS[@]}" --allow-all-paths --allow-all-urls --no-ask-user \
  --share "$SESSION"

echo "$BUILD" >>"$STATE_DIR/done.txt"
echo "done: report=$REPORT session=$SESSION"
