# Metodologia

## Dataset

EuroSAT RGB (Sentinel-2 MSI), 10 classes, 5.000 amostras (500 por classe), imagens 64×64 px.

## Pré-processamento

Conversão para float32, correção de valores inválidos, normalização min-max por canal.

## Extração de características

63 features: estatísticas espectrais (15), histograma (24), índices NDVI/NDWI/brilho/RI (20), textura (4).

## Classificador

RandomForestClassifier (scikit-learn), 200 árvores, min_samples_leaf=2, max_features=sqrt, class_weight=balanced.

## Avaliação

Divisão 70/15/15 estratificada. Métricas: acurácia, precisão, recall, F1. Matriz de confusão.

## Resultados

Acurácia teste 77,7%, F1-macro 77,4%. Classe mais difícil: Highway (F1 0,40).
