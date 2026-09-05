#!/usr/bin/env bash
#
# Start the LiteLLM gateway with provider keys from POC/.env.
#
#   ./run.sh          # start (or restart) the stack
#   ./run.sh down     # stop it
#   ./run.sh logs     # follow the gateway log
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$HERE/../../.env"

# Export only well-formed KEY=VALUE lines: .env holds hand-edited content, and a
# stray line would otherwise abort the whole script under `set -e`.
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
        export "${line?}"
    done < "$ENV_FILE"
else
    echo "warning: $ENV_FILE not found; the gateway will start without provider keys" >&2
fi

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
else
    COMPOSE=(docker-compose)
fi

cd "$HERE"
case "${1:-up}" in
    up)
        "${COMPOSE[@]}" up -d
        echo -n "waiting for the gateway"
        for _ in $(seq 1 30); do
            if curl -sf -m 2 http://localhost:4000/health/liveliness >/dev/null; then
                echo " — ready at http://localhost:4000"
                curl -sf -H "Authorization: Bearer ${LITELLM_MASTER_KEY:-sk-dummypass}" \
                     http://localhost:4000/v1/models || true
                exit 0
            fi
            echo -n "."
            sleep 2
        done
        echo " — timed out; check: ./run.sh logs" >&2
        exit 1
        ;;
    down) "${COMPOSE[@]}" down ;;
    logs) "${COMPOSE[@]}" logs -f litellm ;;
    *) echo "usage: run.sh [up|down|logs]" >&2; exit 1 ;;
esac
