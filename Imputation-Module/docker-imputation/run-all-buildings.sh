#!/usr/bin/env bash
# Reconstruct all four buildings plus the campus total for a given date.
# Writes one CSV + one PNG per target into ./io/.
#
# Usage:
#   CASSANDRA_HOSTS=10.64.253.14 ./run-all-buildings.sh [TARGET_DATE] \
#       [--test-gap START END]... [--overlay-prior-week] [--overlay-actual] \
#       [--no-clear]
#
# TARGET_DATE defaults to today (UTC). The 7-day window ending the day
# before TARGET_DATE is what actually gets reconstructed.
#
# --test-gap can be passed multiple times. When at least one is given, each
# building's run also emits a test_report CSV (MAE/RMSE/max) and the output
# filenames are suffixed with _test so clean reconstructions aren't
# overwritten. START/END are naive Europe/Paris datetimes.
#
# --overlay-prior-week adds a dashed purple curve showing the 7 days before
# the reconstructed window, shifted +7 days. It's the "just copy last week"
# baseline, useful for a visual sanity check.
#
# --overlay-actual adds a solid black curve with the actual pre-imputation
# values. In --test-gap mode that's the ground truth; otherwise it's just
# the raw input with sensor gaps showing as breaks.
#
# By default we wipe reconstructed*.csv/png and test_report_*.csv in ./io/
# before running, so the folder only holds the latest run. Pass --no-clear
# to keep old outputs around (handy for diffing). input*.csv and anything
# from other workflows is left alone.

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  echo "Usage: $0 [TARGET_DATE] [--test-gap START END]... [--overlay-prior-week] [--overlay-actual] [--no-clear]" >&2
  exit 1
}

if [[ $# -gt 0 && "$1" != -* ]]; then
  TARGET_DATE="$1"
  shift
else
  TARGET_DATE="$(date -u +%Y-%m-%d)"
fi

GAP_ARGS=()
OVERLAY_ARGS=()
CLEAR=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-gap)
      [[ $# -ge 3 ]] || die "--test-gap requires 2 arguments (START END)"
      GAP_ARGS+=(--test-gap "$2" "$3")
      shift 3
      ;;
    --overlay-prior-week)
      OVERLAY_ARGS+=(--overlay-prior-week)
      shift
      ;;
    --overlay-actual)
      OVERLAY_ARGS+=(--overlay-actual)
      shift
      ;;
    --no-clear)
      CLEAR=false
      shift
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

TEST_MODE=false
SUFFIX=""
if [[ ${#GAP_ARGS[@]} -gt 0 ]]; then
  TEST_MODE=true
  SUFFIX="_test"
fi

BUILDINGS=(Ptot_HA Ptot_HEI_13RT Ptot_HEI_5RNS Ptot_RIZOMM Ptot_Campus)

if $CLEAR; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  IO_DIR="${SCRIPT_DIR}/io"
  echo "=== Clearing previous reconstruction outputs in ${IO_DIR} ==="
  shopt -s nullglob
  stale=( "${IO_DIR}"/reconstructed*.csv \
          "${IO_DIR}"/reconstructed*.png \
          "${IO_DIR}"/test_report_*.csv )
  shopt -u nullglob
  if (( ${#stale[@]} > 0 )); then
    rm -f -- "${stale[@]}"
  fi
fi

for B in "${BUILDINGS[@]}"; do
  echo "=== Reconstructing ${B} for ${TARGET_DATE} ==="

  cmd_args=(
    --source cassandra
    --target-date "${TARGET_DATE}"
    --building "${B}"
    --output "/io/reconstructed${SUFFIX}_${B}_${TARGET_DATE}.csv"
    --plot   "/io/reconstructed${SUFFIX}_${B}_${TARGET_DATE}.png"
  )

  if $TEST_MODE; then
    cmd_args+=("${GAP_ARGS[@]}")
    cmd_args+=(--test-report "/io/test_report_${B}_${TARGET_DATE}.csv")
  fi

  if [[ ${#OVERLAY_ARGS[@]} -gt 0 ]]; then
    cmd_args+=("${OVERLAY_ARGS[@]}")
  fi

  docker compose run --rm imputer impute_cli.py "${cmd_args[@]}"
done

if $TEST_MODE; then
  echo "=== Done. CSVs + PNGs + test reports in docker-imputation/io/ ==="
else
  echo "=== Done. CSVs + PNGs in docker-imputation/io/ ==="
fi
