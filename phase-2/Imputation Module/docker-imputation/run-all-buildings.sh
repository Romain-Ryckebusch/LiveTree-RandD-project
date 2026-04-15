#!/usr/bin/env bash
# Reconstruct the 4 buildings + the campus total for a given target date,
# writing one CSV + one PNG per building into ./io/.
#
# Usage:
#   CASSANDRA_HOSTS=10.64.253.14 ./run-all-buildings.sh [TARGET_DATE]
#
# TARGET_DATE defaults to today (UTC). The 7-day window ending the day
# before TARGET_DATE is what actually gets reconstructed.

set -euo pipefail

TARGET_DATE="${1:-$(date -u +%Y-%m-%d)}"

BUILDINGS=(Ptot_HA Ptot_HEI_13RT Ptot_HEI_5RNS Ptot_RIZOMM Ptot_Campus)

for B in "${BUILDINGS[@]}"; do
  echo "=== Reconstructing ${B} for ${TARGET_DATE} ==="
  docker compose run --rm imputer \
    --source cassandra \
    --target-date "${TARGET_DATE}" \
    --building "${B}" \
    --output  "/io/reconstructed_${B}_${TARGET_DATE}.csv" \
    --plot    "/io/reconstructed_${B}_${TARGET_DATE}.png"
done

echo "=== Done. CSVs + PNGs in docker-imputation/io/ ==="
