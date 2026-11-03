#!/usr/bin/env bash
# Pipeline completo, do download à interface. Uso: ./run.sh  (ou ./run.sh app para só subir a interface)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
if [ "${1:-}" != "app" ]; then
  $PY src/data_loader.py          # 1. aquisição (EuroSAT MS ≈2 GB + cenas do INPE + WorldCover)
  $PY src/train.py                # 5–7. amostras, treino e avaliação no teste
  $PY src/predict.py              # 8. classificação por região
  $PY src/evaluation.py           # 9. figuras e análise de erros
  $PY presentation/scripts/exportar_dados.py   # números e gráficos dos slides
  $PY -m pytest tests -q
fi
$PY src/app.py
