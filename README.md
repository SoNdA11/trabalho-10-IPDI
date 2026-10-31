# Trabalho 10 — Classificação de Imagens de Satélite

Sistema completo de classificação de cobertura do solo com backend Python (Flask), interface web e apresentação acadêmica (25 slides).

## Execução

```bash
cd trab-10
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/data_loader.py
.venv/bin/python src/train.py
.venv/bin/python src/evaluation.py
.venv/bin/python src/app.py
```

Ou com Docker:
```bash
docker compose up --build
```

## Testes

```bash
python -m pytest tests/ -q
```

## Endpoints adicionais

- `POST /api/classify_batch` — classifica múltiplas imagens (campo `files`).

Acesse `http://localhost:5000` (interface) e `presentation/slides/index.html` (slides).

## Estrutura

Ver plano em `task.md`. Resultados em `results/` (figuras, métricas, mapas). Modelo em `models/random_forest.joblib`.

## Resultados

- Acurácia (teste): 77,7%
- F1-macro: 77,4%
- 10 classes EuroSAT, 5.000 amostras, 63 features por imagem.