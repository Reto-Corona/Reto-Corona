# Reto-Corona

Pipeline del reto de analytics de Corona. Genera cuatro versiones de la
segmentación de consumidores finales:

|                                      | Sin PCA        | Con PCA        |
|--------------------------------------|----------------|----------------|
| Valor original (sin corregir)        | versión 1      | versión 2      |
| Valor corregido (factor 2,65)        | versión 3      | versión 4      |

La versión del valor se controla con la variable de entorno `CORONA_VERSION`
(`original` o `corregido`) y la variante de la segmentación con `CORONA_PCA`
(`sin_pca` o `con_pca`). Los scripts no se editan a mano para cambiar entre
versiones.

## Estructura de salidas

```
data/
├── base_transaccional.csv            (no se toca)
├── maestro_clientes.csv              (no se toca)
├── clientes_limpio.parquet
├── transaccional_original.parquet
├── transaccional_corregido.parquet
├── features_clientes_original.parquet
├── features_clientes_corregido.parquet
├── segmentos_original_sin_pca.parquet
├── segmentos_original_con_pca.parquet
├── segmentos_corregido_sin_pca.parquet
└── segmentos_corregido_con_pca.parquet

salidas/
├── original/
│   ├── descriptivos/          (graficas.py y los CSV descriptivos)
│   │   └── graficas/
│   ├── sin_pca/               (segmentación sin PCA)
│   │   ├── graficas/
│   │   ├── segmentacion_metricas.csv
│   │   ├── segmentacion_perfil_kmeans.csv
│   │   ├── segmentacion_perfil_kmedoids.csv
│   │   ├── segmentacion_medoides.csv
│   │   └── resumen_version.csv
│   └── con_pca/               (idem + pca_varianza.csv y pca_cargas.csv)
├── corregido/                  (misma estructura que original)
└── comparacion_versiones.csv   (consolidado de las 4 versiones + Rand ajustado)
```

## Dependencias

Python:

```powershell
pip install duckdb pandas pyarrow matplotlib
```

R (en una consola de R):

```r
install.packages(c("nanoparquet", "cluster", "ggplot2", "scales"))
```

## Correr todo el pipeline

Un solo comando corre las 4 versiones en orden y escribe la comparación:

```powershell
python scripts/run_pipeline.py
```

El script busca `Rscript` en `PATH` y, si no lo encuentra, en
`C:\Program Files\R\...\bin\Rscript.exe`. Si está en otra ruta, usar:

```powershell
$env:RSCRIPT = "C:\ruta\a\Rscript.exe"; python scripts/run_pipeline.py
```

Para correr solo algunos pasos, usar la variable `SOLO`:

```powershell
$env:SOLO = "segmentacion,comparacion"; python scripts/run_pipeline.py
```

Valores posibles de `SOLO`: `clientes`, `transaccional`, `variables`,
`graficas`, `segmentacion`, `comparacion`.

## Correr una sola variante manualmente

```powershell
$env:CORONA_VERSION = "corregido"
python scripts/limpieza/limpieza_clientes.py
python scripts/limpieza/limpieza_transaccional.py
python scripts/limpieza/variables_clientes.py
python scripts/limpieza/graficas.py

$env:CORONA_PCA = "sin_pca"
Rscript scripts/segmentacion/segmentacion.R
$env:CORONA_PCA = "con_pca"
Rscript scripts/segmentacion/segmentacion.R
```

En bash sería `CORONA_VERSION=corregido CORONA_PCA=sin_pca Rscript ...`.

## Qué hace cada script

- `scripts/limpieza/limpieza_clientes.py`: deduplica `maestro_clientes.csv` y
  escribe `data/clientes_limpio.parquet`.
- `scripts/limpieza/limpieza_transaccional.py`: con `CORONA_VERSION=original`
  no corrige el valor; con `CORONA_VERSION=corregido` aplica la regla del
  factor 2,65 a las líneas detectadas. Escribe
  `data/transaccional_<version>.parquet`.
- `scripts/limpieza/variables_clientes.py`: construye las variables por
  cliente y los descriptivos. Escribe
  `data/features_clientes_<version>.parquet` y
  `salidas/<version>/descriptivos/*.csv`.
- `scripts/limpieza/graficas.py`: gráficas descriptivas en
  `salidas/<version>/descriptivos/graficas/`.
- `scripts/segmentacion/segmentacion.R`: segmentación de consumidores
  finales con K-means y K-medoids. Elige k ∈ [2, 8] por silhouette.
  - `CORONA_PCA=sin_pca` (comportamiento histórico): agrupa sobre las 8
    variables estandarizadas.
  - `CORONA_PCA=con_pca`: aplica PCA sobre esas 8 variables y agrupa sobre
    las componentes necesarias para explicar al menos el 80% de la varianza.
    Los perfiles y el mapa de calor siguen calculándose con las 8 variables
    originales (interpretables). Guarda además `pca_varianza.csv` y
    `pca_cargas.csv`.
- `scripts/segmentacion/comparacion.py`: consolida las 4 variantes en
  `salidas/comparacion_versiones.csv`, con `k`, silhouette, estabilidad,
  acuerdo K-means vs K-medoids, número de componentes del PCA (si aplica) y
  Rand ajustado entre versiones (con PCA vs sin PCA y original vs corregido).
