# Referências

## Dados

- **INPE — Brazil Data Cube.** Catálogo STAC: <https://data.inpe.br/bdc/stac/v1>.
  Coleções `S2_L1C_BUNDLE-1` (Sentinel-2 L1C) e `CB4-MUX-L4-SR-1` (CBERS-4 MUX L4, reflectância de superfície).
  Portal: <https://data.inpe.br/bdc/>.
- **INPE — Catálogo de imagens / CBERS.** <https://www.gov.br/inpe/pt-br> · <http://www.cbers.inpe.br/>.
- **ESA — Sentinel-2 MSI.** *Sentinel-2 User Handbook* e documentação do produto L1C
  (bandas, resoluções e mudança de baseline 04.00 com `RADIO_ADD_OFFSET = −1000`):
  <https://sentiwiki.copernicus.eu/web/s2-mission>.
- **HELBER, P.; BISCHKE, B.; DENGEL, A.; BORTH, D.** EuroSAT: A Novel Dataset and Deep Learning
  Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied
  Earth Observations and Remote Sensing*, v. 12, n. 7, p. 2217–2226, 2019. DOI: 10.1109/JSTARS.2019.2918242.
  Dados: <https://zenodo.org/records/7711810> (licença MIT).
- **ZANAGA, D. et al.** *ESA WorldCover 10 m 2021 v200*. Zenodo, 2022. DOI: 10.5281/zenodo.7254221.
  <https://esa-worldcover.org> (licença CC BY 4.0).

## Sensoriamento remoto e índices espectrais

- **FERREIRA, K. R. et al.** Earth Observation Data Cubes for Brazil: Requirements, Methodology and
  Products. *Remote Sensing*, v. 12, n. 24, 4033, 2020. DOI: 10.3390/rs12244033.
- **ROUSE, J. W.; HAAS, R. H.; SCHELL, J. A.; DEERING, D. W.** Monitoring vegetation systems in the
  Great Plains with ERTS. In: *Third Earth Resources Technology Satellite-1 Symposium*. NASA SP-351,
  p. 309–317, 1974. (NDVI)
- **McFEETERS, S. K.** The use of the Normalized Difference Water Index (NDWI) in the delineation of
  open water features. *International Journal of Remote Sensing*, v. 17, n. 7, p. 1425–1432, 1996.
- **XU, H.** Modification of normalised difference water index (NDWI) to enhance open water features
  in remotely sensed imagery. *International Journal of Remote Sensing*, v. 27, n. 14, p. 3025–3033, 2006. (MNDWI)
- **ZHA, Y.; GAO, J.; NI, S.** Use of normalized difference built-up index in automatically mapping
  urban areas from TM imagery. *International Journal of Remote Sensing*, v. 24, n. 3, p. 583–594, 2003. (NDBI)
- **JENSEN, J. R.** *Sensoriamento remoto do ambiente: uma perspectiva em recursos terrestres*.
  Tradução da 2. ed. São José dos Campos: Parêntese, 2009.

## Processamento de imagens e textura

- **GONZALEZ, R. C.; WOODS, R. E.** *Processamento digital de imagens*. 3. ed. São Paulo: Pearson, 2010.
  (filtros espaciais, Sobel, filtro de moda/mediana, histogramas)
- **HARALICK, R. M.; SHANMUGAM, K.; DINSTEIN, I.** Textural features for image classification.
  *IEEE Transactions on Systems, Man, and Cybernetics*, v. SMC-3, n. 6, p. 610–621, 1973. (GLCM)

## Classificação e avaliação

- **BREIMAN, L.** Random forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.
- **BELGIU, M.; DRĂGUŢ, L.** Random forest in remote sensing: a review of applications and future
  directions. *ISPRS Journal of Photogrammetry and Remote Sensing*, v. 114, p. 24–31, 2016.
- **CORTES, C.; VAPNIK, V.** Support-vector networks. *Machine Learning*, v. 20, p. 273–297, 1995.
- **COHEN, J.** A coefficient of agreement for nominal scales. *Educational and Psychological
  Measurement*, v. 20, n. 1, p. 37–46, 1960. (kappa)
- **LANDIS, J. R.; KOCH, G. G.** The measurement of observer agreement for categorical data.
  *Biometrics*, v. 33, n. 1, p. 159–174, 1977. (escala de interpretação do kappa)
- **PONTIUS, R. G.; MILLONES, M.** Death to Kappa: birth of quantity disagreement and allocation
  disagreement for accuracy assessment. *International Journal of Remote Sensing*, v. 32, n. 15,
  p. 4407–4429, 2011. (críticas ao kappa)
- **CONGALTON, R. G.; GREEN, K.** *Assessing the Accuracy of Remotely Sensed Data: Principles and
  Practices*. 3. ed. Boca Raton: CRC Press, 2019. (matriz de confusão em mapeamento)
- **ROBERTS, D. R. et al.** Cross-validation strategies for data with temporal, spatial, hierarchical,
  or phylogenetic structure. *Ecography*, v. 40, n. 8, p. 913–929, 2017. (validação por blocos espaciais)
- **PLOTON, P. et al.** Spatial validation reveals poor predictive performance of large-scale ecological
  mapping models. *Nature Communications*, v. 11, 4540, 2020. (vazamento espacial)

## Software

- **PEDREGOSA, F. et al.** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning
  Research*, v. 12, p. 2825–2830, 2011.
- **GILLIES, S. et al.** Rasterio: geospatial raster I/O for Python. <https://rasterio.readthedocs.io>.
- **HARRIS, C. R. et al.** Array programming with NumPy. *Nature*, v. 585, p. 357–362, 2020.
- **VIRTANEN, P. et al.** SciPy 1.0. *Nature Methods*, v. 17, p. 261–272, 2020.
- **Flask:** <https://flask.palletsprojects.com> · **Reveal.js 5.1:** <https://revealjs.com> ·
  **Apache ECharts:** <https://echarts.apache.org>.
