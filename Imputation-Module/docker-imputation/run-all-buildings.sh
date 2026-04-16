#!/usr/bin/env bash
# Reconstruct the 4 buildings + the campus total for a given target date,
# writing one CSV + one PNG per building into ./io/.
#
# Usage:
#   CASSANDRA_HOSTS=10.64.253.14 ./run-all-buildings.sh [TARGET_DATE] \
#     [--test-gap START END]...
#
# TARGET_DATE defaults to today (UTC). The 7-day window ending the day
# before TARGET_DATE is what actually gets reconstructed.
#
# --test-gap is repeatable. When at least one is given, each building's
# run also emits a test_report CSV (MAE/RMSE/max) and output filenames
# are suffixed with _test so clean reconstructions are not overwritten.
# START/END are naive Europe/Paris datetimes, e.g. '2026-04-14 08:00'.

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  echo "Usage: $0 [TARGET_DATE] [--test-gap START END]..." >&2
  exit 1
}

if [[ $# -gt 0 && "$1" != -* ]]; then
  TARGET_DATE="$1"
  shift
else
  TARGET_DATE="$(date -u +%Y-%m-%d)"
fi

GAP_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-gap)
      [[ $# -ge 3 ]] || die "--test-gap requires 2 arguments (START END)"
      GAP_ARGS+=(--test-gap "$2" "$3")
      shift 3
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

  docker compose run --rm imputer "${cmd_args[@]}"
done

if $TEST_MODE; then
  echo "=== Done. CSVs + PNGs + test reports in docker-imputation/io/ ==="
else
  echo "=== Done. CSVs + PNGs in docker-imputation/io/ ==="
fi
