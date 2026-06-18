#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
OUT="${OUT:-/tmp/habibi-smoke.wav}"

curl -fsS "$BASE_URL/health"
echo

curl -fsS \
  -H 'Content-Type: application/json' \
  -X POST "$BASE_URL/infer" \
  -d '{"text":"هلا والله، كيف أقدر أخدمك اليوم؟","dialect":"SAU","model":"Unified","output_file":"smoke.wav"}' \
  -o "$OUT"

file "$OUT" || true
ls -lh "$OUT"
