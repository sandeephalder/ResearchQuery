#!/usr/bin/env bash
#
# Start, resume, or retry the Open RAG Benchmark data ingestion.
#
#   ./run_ingestion.sh            # start (skips PDFs already on disk)
#   ./run_ingestion.sh retry      # re-attempt only ids in failed_downloads.json
#   ./run_ingestion.sh force      # re-download everything, ignoring local files
#   ./run_ingestion.sh status     # report progress without downloading anything
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INGESTION_DIR="$SCRIPT_DIR/Ingestion"
PDF_DIR="$INGESTION_DIR/data/raw_dataset/pdf/raw_pdf"
FAILURES_JSON="$INGESTION_DIR/data/raw_dataset/pdf/failed_downloads.json"
URLS_JSON="$INGESTION_DIR/data/dataset/pdf/arxiv/pdf_urls.json"
LOG_DIR="$SCRIPT_DIR/logs"

usage() {
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}


if command -v uv >/dev/null 2>&1; then
    PY=(uv run --project "$SCRIPT_DIR" python)
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PY=("$SCRIPT_DIR/.venv/bin/python")
else
    echo "error: neither uv nor $SCRIPT_DIR/.venv/bin/python is available" >&2
    echo "hint: install uv, or create the venv with 'uv sync'" >&2
    exit 1
fi

count_pdfs() {
    [[ -d "$PDF_DIR" ]] || { echo 0; return; }
    find "$PDF_DIR" -maxdepth 1 -name '*.pdf' -type f | wc -l | tr -d ' '
}

count_json_keys() {
    [[ -f "$1" ]] || { echo 0; return; }
    "${PY[@]}" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0
}

show_status() {
    local have want failed
    have="$(count_pdfs)"
    want="$(count_json_keys "$URLS_JSON")"
    failed="$(count_json_keys "$FAILURES_JSON")"

    echo "PDFs on disk : $have${want:+ / $want}"
    echo "Failed ids   : $failed"
    if [[ "$failed" != "0" ]]; then
        echo "Log          : $FAILURES_JSON"
        echo
        echo "Failure reasons:"
        "${PY[@]}" -c "
import collections, json, sys
data = json.load(open(sys.argv[1]))
for reason, n in collections.Counter(v.get('reason', 'unknown') for v in data.values()).most_common():
    print(f'  {n:>5}  {reason}')
" "$FAILURES_JSON"
        echo
        echo "Re-run with: $(basename "${BASH_SOURCE[0]}") retry"
    fi
}

MODE="${1:-start}"
case "$MODE" in
    start)  ARGS=() ;;

    retry)  ARGS=(--retry-failed --skip-dataset) ;;
    force)  ARGS=(--force) ;;
    status) show_status; exit 0 ;;
    -h|--help|help) usage 0 ;;
    *) echo "error: unknown mode '$MODE'" >&2; echo >&2; usage 1 >&2 ;;
esac

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ingestion_$(date +%Y%m%d_%H%M%S)_${MODE}.log"

echo "Mode    : $MODE"
echo "Runner  : ${PY[*]}"
echo "Log     : $LOG_FILE"
echo "Before  : $(count_pdfs) PDFs on disk"
echo

cd "$INGESTION_DIR"
# tee so the run stays interactive (tqdm) while keeping a full record on disk
set +e
"${PY[@]}" ingestion.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS="${PIPESTATUS[0]}"
set -e

echo
echo "After   : $(count_pdfs) PDFs on disk"
if [[ -f "$FAILURES_JSON" ]]; then
    echo "Failed  : $(count_json_keys "$FAILURES_JSON") id(s) — see $FAILURES_JSON, then re-run with: $(basename "${BASH_SOURCE[0]}") retry"
else
    echo "Failed  : 0"
fi

exit "$STATUS"
