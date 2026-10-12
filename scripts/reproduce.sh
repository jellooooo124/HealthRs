#!/usr/bin/env bash
# One-command reproduction (run from anywhere; uses the active Python environment or $PYTHON).
#
#   bash scripts/reproduce.sh --sample      # ~1 min, no download: whole offline pipeline on data/sample/
#                                           #   -> outputs_sample/ (small data: numbers differ from the report)
#   bash scripts/reproduce.sh --full        # ~40 min: needs data/raw/foodcom/{recipes,reviews}.csv;
#                                           #   rebuilds data/processed/ and every offline result in a fresh
#                                           #   outputs_reproduced/, then compares it with the committed outputs/
#   bash scripts/reproduce.sh --agent-eval  # rebuild the agent-evaluation tables from the logged dialogs and
#                                           #   judge scores in outputs/agent_eval/final/ (no API calls)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
MODE=${1:---sample}

offline() {
  $PY scripts/prepare_data.py
  $PY scripts/data_report.py
  $PY scripts/run_offline.py --stage all --jobs "${JOBS:-4}"
}

case "$MODE" in
  --sample)
    export HEALTHREC_DATA_CONFIG=data_sample HEALTHREC_OUTPUTS=outputs_sample
    offline
    echo "Done. Results: outputs_sample/tables/ and outputs_sample/figures/" ;;
  --full)
    if [ ! -f data/raw/foodcom/recipes.csv ] || [ ! -f data/raw/foodcom/reviews.csv ]; then
      echo "Missing data. Download https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews"
      echo "and put recipes.csv and reviews.csv into data/raw/foodcom/."; exit 1
    fi
    export HEALTHREC_OUTPUTS=${HEALTHREC_OUTPUTS:-outputs_reproduced}
    offline
    $PY scripts/check_reproduction.py "$HEALTHREC_OUTPUTS/tables" outputs/tables ;;
  --agent-eval)
    $PY scripts/run_agent_eval.py --run final --stage report
    $PY scripts/run_agent_eval.py --run final --stage agreement ;;
  *)
    echo "usage: bash scripts/reproduce.sh [--sample | --full | --agent-eval]"; exit 2 ;;
esac
