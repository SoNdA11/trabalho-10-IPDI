# data/

- `raw/` — dataset EuroSAT baixado (zip + extraído).
- `processed/` — arrays `.npy` prontos para treino (`X_features.npy`, `y_features.npy`, `X_rgb.npy`, `y_rgb.npy`).

O download é feito por `src/data_loader.py`. O dataset não é versionado (arquivo grande).