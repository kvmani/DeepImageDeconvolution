#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_ID=""
DRYRUN="false"
CONDA_ENV=""
LOG_ROOT="./logs/train_runs"
KILL_PYTHON="true"
GPU_WATCH_SEC="60"

usage() {
  cat <<'USAGE'
Usage: bash ./run_train_jobs.sh [options]

Options:
  --cuda_id "0,1"         Set CUDA_VISIBLE_DEVICES
  --dryrun true|false      Print commands only (default: false)
  --conda_env <envname>    Optional conda environment to activate
  --log_root <path>        Log root directory (default: ./logs/train_runs)
  --kill_python true|false Kill all python3 before each run (default: true)
  --gpu_watch_sec <sec>    GPU polling interval seconds (default: 60)
USAGE
}

parse_bool() {
  local value
  value="${1,,}"
  case "$value" in
    true|false)
      echo "$value"
      ;;
    *)
      echo "Invalid boolean: $1 (expected true or false)" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda_id)
      CUDA_ID="$2"
      shift 2
      ;;
    --dryrun)
      DRYRUN="$(parse_bool "$2")"
      shift 2
      ;;
    --conda_env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --kill_python)
      KILL_PYTHON="$(parse_bool "$2")"
      shift 2
      ;;
    --gpu_watch_sec)
      GPU_WATCH_SEC="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$CUDA_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_ID"
fi

if [[ ! "$GPU_WATCH_SEC" =~ ^[0-9]+$ ]]; then
  echo "--gpu_watch_sec must be an integer (got: $GPU_WATCH_SEC)" >&2
  exit 1
fi

if [[ -n "$CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH; cannot activate env '$CONDA_ENV'" >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

JobList=(
  "configs/train_default.yaml baseline_01 --set train.lr=0.0002 --set train.batch_size=4"
  "configs/train_default.yaml baseline_02 --set train.lr=0.00002 --set train.batch_size=4"
)

mkdir -p "$LOG_ROOT"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
STARTUP_LOG="$LOG_ROOT/${RUN_TS}_startup.log"

log_startup() {
  echo "$1" | tee -a "$STARTUP_LOG"
}

log_startup "Run started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
log_startup "Host: $(hostname)"
log_startup "User: $(whoami)"
log_startup "PWD: $(pwd)"
log_startup "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
log_startup "Python: $(command -v python3 || echo '<missing>')"
log_startup "Python version: $(python3 --version 2>&1 || echo '<missing>')"

if command -v nvidia-smi >/dev/null 2>&1; then
  log_startup "nvidia-smi:"
  nvidia-smi | tee -a "$STARTUP_LOG"
else
  log_startup "nvidia-smi: <not available>"
fi

if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  log_startup "Git branch: $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
  log_startup "Git commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
  log_startup "Git status:"
  git -C "$REPO_ROOT" status --porcelain | tee -a "$STARTUP_LOG"
else
  log_startup "Git repo: <not found>"
fi

gpu_watch_pid=""
start_gpu_watch() {
  local log_file="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while true; do
        echo "---- GPU snapshot @ $(date -u +"%Y-%m-%dT%H:%M:%SZ") ----"
        nvidia-smi
        sleep "$GPU_WATCH_SEC"
      done
    ) >>"$log_file" 2>&1 &
    gpu_watch_pid=$!
  fi
}

stop_gpu_watch() {
  if [[ -n "$gpu_watch_pid" ]]; then
    kill "$gpu_watch_pid" >/dev/null 2>&1 || true
    wait "$gpu_watch_pid" >/dev/null 2>&1 || true
    gpu_watch_pid=""
  fi
}

job_index=0
for job in "${JobList[@]}"; do
  job_index=$((job_index + 1))
  read -r -a job_parts <<<"$job"
  config_path="${job_parts[0]}"
  run_tag="${job_parts[1]}"
  overrides=("${job_parts[@]:2}")

  if [[ ! -f "$config_path" ]]; then
    echo "Config file not found: $config_path" >&2
    exit 1
  fi

  job_dir="$LOG_ROOT/${RUN_TS}_job${job_index}_${run_tag}"
  mkdir -p "$job_dir"

  cp "$config_path" "$job_dir/config.yaml"

  git -C "$REPO_ROOT" rev-parse HEAD >"$job_dir/git_commit.txt" 2>/dev/null || true
  git -C "$REPO_ROOT" status --porcelain >"$job_dir/git_status_porcelain.txt" 2>/dev/null || true
  git -C "$REPO_ROOT" diff >"$job_dir/git_diff.patch" 2>/dev/null || true

  train_log="$job_dir/train.log"
  cmd=(python3 scripts/run_train.py --config "$config_path" --run-tag "$run_tag")
  if [[ ${#overrides[@]} -gt 0 ]]; then
    cmd+=("${overrides[@]}")
  fi

  echo "==== Job ${job_index}: ${run_tag} ====" | tee -a "$train_log"
  echo "Config: $config_path" | tee -a "$train_log"
  echo "Command: ${cmd[*]}" | tee -a "$train_log"

  if [[ "$DRYRUN" == "true" ]]; then
    echo "DRYRUN: ${cmd[*]}" | tee -a "$train_log"
    continue
  fi

  if [[ "$KILL_PYTHON" == "true" ]]; then
    killall -9 python3 >/dev/null 2>&1 || true
  fi

  start_gpu_watch "$train_log"
  "${cmd[@]}" 2>&1 | tee -a "$train_log"
  stop_gpu_watch
  echo "==== Job ${job_index} complete ====" | tee -a "$train_log"
done
