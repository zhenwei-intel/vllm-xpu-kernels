#!/usr/bin/env bash
# Run the /intel-ci-fix skill on new "Full intel CI-daily" failures on main.
# Picks the newest finished failing daily build, caches stripped failed-job logs
# and a 7-day signature history table, runs copilot in local-only mode, and saves
# the report under $STATE_DIR/reports/<build>.md.
#
# Usage: intel_ci_daily.sh [--dry-run] [build_number]
# Env (or $ENV_FILE): REPO, HF_TOKEN, HF_HOME, GPUS, COPILOT_BIN,
#   COPILOT_MODEL_ID, STATE_DIR, AGENT_TIMEOUT
set -euo pipefail

DRY_RUN=0
[[ ${1:-} == --dry-run ]] && { DRY_RUN=1; shift; }
FORCE_BUILD=${1:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
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
LOG_BASE=https://buildkite.com/organizations/vllm/pipelines/intel-ci/builds
LIST_URL="$PIPELINE/builds?branch=main&created_from_days=7&query=Full+intel+CI-daily"
CACHE=$STATE_DIR/builds
COOKIE=$STATE_DIR/bk.cookie

((DRY_RUN)) || : "${HF_TOKEN:?set HF_TOKEN in $ENV_FILE}"
mkdir -p "$STATE_DIR/logs" "$STATE_DIR/reports" "$CACHE" "$HOME/.copilot/skills"
touch "$STATE_DIR/done.txt"
[[ -d $SCRIPT_DIR/.claude/skills/intel-ci-fix ]] && \
  ln -sfn "$SCRIPT_DIR/.claude/skills/intel-ci-fix" "$HOME/.copilot/skills/intel-ci-fix"

exec 9>"$STATE_DIR/lock"
flock -n 9 || { echo "previous run still active"; exit 0; }

STAMP=$(date +%F-%H%M)
RUN_ID=icf-$STAMP-$$
LOG=$STATE_DIR/logs/$STAMP.log

say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
bk() { curl -fsSL --retry 3 --max-time 300 -A Mozilla/5.0 -b "$COOKIE" -c "$COOKIE" "$@"; }
is_done() { grep -qx "$1" "$STATE_DIR/done.txt"; }
mark_done() { is_done "$1" || echo "$1" >>"$STATE_DIR/done.txt"; }
strip_log() {
  sed -E 's/\x1b?_?bk;t=[0-9]+\x07?//g; s/\x1b\[[0-9;?]*[A-Za-z]//g; s/[\x1b\r]//g'
}

# Print "<signature>\t<detail>" per failure in a stripped log. The signature is
# the pytest node id without parameters and without the leading "tests/".
log_signatures() {
  local out
  out=$(grep -E '^(FAILED|ERROR) [^ ]+::' "$1" | awk '!seen[$0]++ {
      sig = $2; sub(/\[.*/, "", sig); sub(/^tests\//, "", sig)
      print sig "\t" substr($0, 1, 300) }') || true
  if [[ -z $out ]]; then
    out=$(grep -E 'Error|Exception|Timeout|timed out' "$1" \
      | grep -vE '🚨|user command error|HTTP Error 401|DeprecationWarning' \
      | tail -1 | cut -c1-300) || true
    out="NO_PYTEST_SUMMARY"$'\t'"${out:-unknown}"
  fi
  printf '%s\n' "$out"
}

# Cache a build. Logs are fetched only once the build has finished (marked by
# .complete); running builds are re-fetched on the next run.
fetch_build() {
  local b=$1 d=$CACHE/$1 id name dev kind agent lf
  mkdir -p "$d"
  [[ -f $d/.complete ]] && return 0
  bk "$PIPELINE/builds/$b.json" -o "$d/build.json"
  [[ -n $(jq -r '.finished_at // empty' "$d/build.json") ]] || return 0
  bk "$PIPELINE/builds/$b/data/jobs" -o "$d/jobs.json"
  # Final attempt per job name; parallel shards share a name, any failure wins.
  jq -r '.records[] | select(.type == "script" and .retried_in_job_uuid == null
        and (.state == "finished" or .state == "timed_out"))
      | [.name, (if (.passed or .soft_failed) then "pass" else "fail" end)] | @tsv' \
    "$d/jobs.json" \
    | awk -F'\t' '{ if (s[$1] != "fail") s[$1] = $2 } END { for (k in s) print k "\t" s[k] }' \
      >"$d/job_status.tsv"
  : >"$d/failures.tsv"
  while IFS=$'\t' read -r id name dev kind agent; do
    lf=$d/$id.log
    if [[ ! -s $lf ]]; then
      bk "$LOG_BASE/$b/jobs/$id/download" | strip_log >"$lf.tmp"
      mv "$lf.tmp" "$lf"
    fi
    while IFS=$'\t' read -r sig detail; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$b" "$name" "$kind" "$sig" "$dev" "$agent" "$lf" "$detail"
    done < <(log_signatures "$lf") >>"$d/failures.tsv"
  done < <(jq -r '
      (.records | map({key: .id, value: .}) | from_entries) as $m
      | def final: if .retried_in_job_uuid and $m[.retried_in_job_uuid]
          then $m[.retried_in_job_uuid] | final else . end;
      .records[] | select(.type == "script"
          and (.state == "finished" or .state == "timed_out")
          and ((.passed or .soft_failed) | not))
      | (final) as $f
      | (if .retried_in_job_uuid == null then "fail"
         elif ($f.passed or $f.soft_failed) then "retry-pass" else empty end) as $k
      | [.id, .name,
         ((.agent_meta_data // []) | map(select(startswith("device="))) | join(",")),
         $k, (.agent_name // "")] | @tsv' "$d/jobs.json")
  touch "$d/.complete"
}

# Per-signature history of the target's failures over all finished builds.
history_table() {
  local t=$1 b
  for b in "${FINISHED[@]}"; do sed "s/^/$b\t/" "$CACHE/$b/job_status.tsv"; done >"$WORK/status.tsv"
  for b in "${FINISHED[@]}"; do printf '%s\t%s\n' "$b" "$(jq -r .commit_id "$CACHE/$b/build.json")"; done >"$WORK/commits.tsv"
  for b in "${FINISHED[@]}"; do cat "$CACHE/$b/failures.tsv"; done >"$WORK/failures.tsv"
  awk -v builds="${FINISHED[*]}" -v target="$t" '
    BEGIN { FS = "\t"; nb = split(builds, B, " ") }
    FILENAME == ARGV[1] { st[$1, $2] = $3; next }
    FILENAME == ARGV[2] { c[$1] = $2; next }
    {
      k = $1 SUBSEP $2 SUBSEP $4
      if (!(k in fs) || $3 == "fail") fs[k] = $3
      if ($1 == target && $3 == "fail" && !(($2, $4) in seen)) {
        seen[$2, $4] = 1; job[++n] = $2; sig[n] = $4
      }
    }
    END {
      print "| job | signature | history | fails/runs | streak | last pass .. first fail of streak |"
      print "|---|---|---|---|---|---|"
      for (i = 1; i <= n; i++) {
        h = ""; f = 0; r = 0; ti = 0
        for (j = 1; j <= nb; j++) {
          b = B[j]
          if (!((b, job[i]) in st)) s = "-"
          else {
            r++; x = fs[b, job[i], sig[i]]
            if (x == "fail") { s = "F"; f++ }
            else if (x == "retry-pass") s = "R"
            else if (st[b, job[i]] == "fail") s = "x"
            else s = "P"
          }
          S[j] = s; h = h (b == target ? "[" s "]" : s)
          if (b == target) ti = j
        }
        streak = 0; bad = ""; good = ""
        for (j = ti; j >= 1; j--) {
          if (S[j] == "-") continue
          if (S[j] != "F") break
          streak++; bad = B[j]
        }
        for (; j >= 1; j--) if (S[j] == "P") { good = B[j]; break }
        rng = (good == "" ? "none in window" : good " (" substr(c[good], 1, 10) ")") \
          " .. " bad " (" substr(c[bad], 1, 10) ")"
        printf "| %s | `%s` | `%s` | %d/%d | %d | %s |\n", job[i], sig[i], h, f, r, streak, rng
      }
    }' "$WORK/status.tsv" "$WORK/commits.tsv" "$WORK/failures.tsv"
}

write_context() {
  local t=$1 d=$CACHE/$1 commit
  commit=$(jq -r .commit_id "$d/build.json")
  {
    echo "# intel-ci build $t"
    echo "- URL: $PIPELINE/builds/$t"
    echo "- commit: $commit (branch main, \"$DAILY_MSG\")"
    echo "- image: public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:$commit-xpu"
    echo "- finished daily builds in window (oldest to newest): ${FINISHED[*]}"
    echo
    echo "## Failures in $t (final attempts only)"
    echo "| job | device | agent | signature | log | pytest line |"
    echo "|---|---|---|---|---|---|"
    awk -F'\t' '$3 == "fail" { printf "| %s | %s | %s | `%s` | %s | %s |\n", $2, $5, $6, $4, $7, $8 }' "$d/failures.tsv"
    echo
    echo "## Signature history"
    echo "One char per finished build in window order, target in brackets."
    echo "F = this signature failed, R = it failed but the job passed on retry,"
    echo "x = the job failed for another reason, P = the job passed, - = the job did not run."
    echo
    history_table "$t"
    echo
    echo "## Failed-then-passed-on-retry in $t (flaky evidence, not build failures)"
    awk -F'\t' '$3 == "retry-pass" { printf "- %s (%s, %s): `%s` log %s\n", $2, $5, $6, $4, $7; n++ }
      END { if (!n) print "none" }' "$d/failures.tsv"
    echo
    echo "## Local ci-fix branches in $REPO"
    git -C "$REPO" for-each-ref --format='- %(refname:short) %(committerdate:short) %(subject)' 'refs/heads/ci-fix/*' | grep . || echo "none"
  } >"$CTX"
}

say "run $RUN_ID: listing daily builds"
mapfile -t BUILDS < <(bk "$LIST_URL" | grep -o 'intel-ci/builds/[0-9]\+' | grep -o '[0-9]\+$' | sort -nu)
[[ -n $FORCE_BUILD ]] && mapfile -t BUILDS < <(printf '%s\n' "${BUILDS[@]}" "$FORCE_BUILD" | sort -nu)

FINISHED=()
for b in "${BUILDS[@]}"; do
  fetch_build "$b"
  if [[ $b != "$FORCE_BUILD" ]] && ! jq -e --arg m "$DAILY_MSG" \
      '.branch_name == "main" and (.message | contains($m))' "$CACHE/$b/build.json" >/dev/null; then
    continue
  fi
  if [[ ! -f $CACHE/$b/.complete ]]; then
    say "build $b still running, skipped"
    continue
  fi
  FINISHED+=("$b")
done

TARGET=$FORCE_BUILD
if [[ -n $TARGET && ! -f $CACHE/$TARGET/.complete ]]; then
  say "build $TARGET has not finished"; exit 1
fi
if [[ -z $TARGET ]]; then
  last_done=$(sort -n "$STATE_DIR/done.txt" | tail -1)
  for ((i = ${#FINISHED[@]} - 1; i >= 0; i--)); do
    b=${FINISHED[i]}
    ((b > ${last_done:-0})) || break
    is_done "$b" && continue
    if grep -q $'^[0-9]*\t[^\t]*\tfail\t' "$CACHE/$b/failures.tsv"; then TARGET=$b; break; fi
  done
fi
if [[ -z $TARGET ]]; then say "nothing to do"; exit 0; fi

WORK=$(mktemp -d)
CTX=$CACHE/$TARGET/context.md
REPORT=$STATE_DIR/reports/$TARGET.md
cleanup() {
  docker ps -aq --filter "label=intel-ci-fixer=$RUN_ID" | xargs -r docker rm -f >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT
write_context "$TARGET"
say "target build $TARGET, context $CTX"
if ((DRY_RUN)); then cat "$CTX"; exit 0; fi

DENY_ARGS=()
for t in \
  'shell(git push)' 'shell(gh pr create)' 'shell(gh pr merge)' 'shell(gh pr ready)' \
  'shell(gh pr comment)' 'shell(gh pr review)' 'shell(gh pr close)' \
  'shell(gh issue create)' 'shell(gh issue comment)' \
  'shell(docker rmi)' 'shell(docker system)' 'shell(docker image prune)'; do
  DENY_ARGS+=(--deny-tool "$t")
done
GUARD=(
  ZE_AFFINITY_MASK="$GPUS" HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME"
  GIT_CONFIG_COUNT=3
  GIT_CONFIG_KEY_0=url.push-disabled://.pushInsteadOf GIT_CONFIG_VALUE_0=https://github.com/
  GIT_CONFIG_KEY_1=url.push-disabled://.pushInsteadOf GIT_CONFIG_VALUE_1=git@github.com:
  GIT_CONFIG_KEY_2=url.push-disabled://.pushInsteadOf GIT_CONFIG_VALUE_2=ssh://git@github.com/
)

PROMPT="Use the /intel-ci-fix skill to analyze $PIPELINE/builds/$TARGET. \
All pre-fetched inputs are in $CTX (failed jobs, stripped log paths, 7-day signature history \
with <good>..<bad> commits, retry-pass evidence, and local ci-fix branches); read $CTX first \
instead of re-fetching logs. vllm repo: $REPO. GPUs: use only $GPUS (ZE_AFFINITY_MASK=$GPUS), \
and add '--label intel-ci-fixer=$RUN_ID' to every docker run. Save the final report to $REPORT \
and also print a Chinese summary (中文回答)."

say "running /intel-ci-fix on build $TARGET (model: $MODEL, GPUs: $GPUS)"
set +e
(cd "$REPO" && env "${GUARD[@]}" timeout -k 2m "$AGENT_TIMEOUT" "$COPILOT_BIN" \
  -p "$PROMPT" --model "$MODEL" \
  --allow-all-tools "${DENY_ARGS[@]}" --allow-all-paths --allow-all-urls --no-ask-user \
  --share "$STATE_DIR/logs/session-$STAMP.md" 2>&1) | tee "$WORK/stdout.md" >>"$LOG"
rc=$?
set -e
say "copilot exited with $rc"

[[ -s $REPORT ]] || cp "$WORK/stdout.md" "$REPORT"
if [[ ! -s $REPORT ]] || ((rc != 0)); then
  say "build $TARGET incomplete (rc=$rc); will retry next run"
  exit 1
fi

for b in "${FINISHED[@]}"; do
  ((b <= TARGET)) && mark_done "$b"
done
say "build $TARGET done, report: $REPORT"
