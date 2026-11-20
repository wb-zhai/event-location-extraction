#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(gcloud config get-value project 2>/dev/null)"

ZONE="$(
  gcloud compute instances list \
    --filter='name=zhai-bastion' \
    --format='value(zone.basename())' \
  | head -1
)"

if [ -z "$ZONE" ]; then
    echo "Could not find zhai-bastion"
    exit 1
fi

python3 - <<'PY' > /tmp/zhai_db_target.env
from pathlib import Path
from dotenv import dotenv_values

e = dotenv_values(Path(".env"))

host = e.get("SQL_HOST") or e.get("PGHOST")
port = e.get("SQL_PORT") or e.get("PGPORT") or "5432"

if not host:
    raise SystemExit("No SQL_HOST/PGHOST found in .env")

print(f'DB_HOST={host!r}')
print(f'DB_PORT={port!r}')
PY

source /tmp/zhai_db_target.env

echo "project : $PROJECT"
echo "bastion : zhai-bastion"
echo "zone    : $ZONE"
echo "remote  : $DB_HOST:$DB_PORT"
echo "local   : localhost:5439"
echo
echo "Keep this terminal open."

exec gcloud compute ssh zhai-bastion \
  --project="$PROJECT" \
  --zone="$ZONE" \
  -- \
  -N \
  -L "5439:${DB_HOST}:${DB_PORT}"
