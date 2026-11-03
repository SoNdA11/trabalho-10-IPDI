# data/

Nada aqui é versionado: tudo é baixado e gerado por `src/data_loader.py` e `src/train.py`.

## raw/ — dados originais

| caminho | fonte | tamanho | conteúdo |
|---|---|---|---|
| `EuroSAT_MS/<Classe>/*.tif` | Zenodo, registro 7711810 (`EuroSAT_MS.zip`) | ≈2,9 GB extraído | 27.000 GeoTIFF 64×64, 13 bandas uint16 (Sentinel-2 L1C) |
| `inpe/S2B_MSIL1C_20240901T125309_…zip` | INPE · BDC, coleção `S2_L1C_BUNDLE-1` | ≈700 MB | cena SAFE completa, tile 24MXV |
| `worldcover/ESA_WorldCover_10m_2021_v200_S06W039_Map.tif` | ESA, bucket público na AWS | 60 MB | mapa de cobertura 10 m |
| `EuroSAT_RGB/` (opcional) | Zenodo (`EuroSAT_RGB.zip`) | 95 MB | JPGs usados só para calibrar o envio de JPG/PNG na interface |

A cena CBERS-4 não é baixada inteira: `prepare_cbers_scene` lê a janela da área de
estudo direto dos COGs do INPE (`/vsicurl/`).

## processed/ — dados preparados

| arquivo | gerado por | descrição |
|---|---|---|
| `s2_mossoro_l1c_10m.tif` | `data_loader.prepare_s2_scene` | 13 bandas DN, 2442×2558 px, 10 m, UTM 24S |
| `s2_mossoro_worldcover_10m.tif` | `data_loader.prepare_reference` | WorldCover na grade do Sentinel-2 (vizinho mais próximo) |
| `cbers4_mux_mossoro_20m.tif` | `data_loader.prepare_cbers_scene` | 4 bandas SR + CMASK, 1221×1279 px, 20 m |
| `cbers4_mux_worldcover_20m.tif` | `data_loader.prepare_reference` | WorldCover agregado a 20 m pela **moda** |
| `eurosat_features.npz` | `train.build_eurosat_features` | 97 características (13 bandas) e 27 (RGB) por recorte, rótulos, coordenadas |
| `eurosat_demo_samples.npz` | `train.save_demo_samples` | 240 recortes do **teste** usados na interface |
| `s2_split.npy`, `cbers_split.npy` | `train.train_scene` | mapa de blocos treino/validação/teste |
| `rgb_jpg_calibration.json` | `train.calibrate_rgb_jpg` | ajuste linear JPG → reflectância (r ≥ 0,993) |
| `scenes.json` | `data_loader` | metadados das cenas |

## Área de estudo

Mossoró/RN e entorno: lon −37,45 a −37,22; lat −5,30 a −5,08 (25,6 × 24,4 km).
As duas cenas usam a mesma grade UTM 24S alinhada a 20 m: um pixel CBERS = 2×2 pixels Sentinel-2.

## Condições de uso

- **Sentinel-2:** dados Copernicus, acesso livre e aberto (Copernicus Sentinel Data Terms and Conditions).
  Distribuído pelo INPE no Brazil Data Cube.
- **CBERS-4:** política de dados abertos do INPE: imagens gratuitas, com citação da fonte.
- **EuroSAT:** Helber et al. (2019), licença MIT (registro Zenodo 7711810); citar o artigo ao usar.
- **ESA WorldCover:** licença CC BY 4.0, "© ESA WorldCover project 2021 / Contains modified
  Copernicus Sentinel data (2021) processed by ESA WorldCover consortium".
