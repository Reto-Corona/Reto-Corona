# Reto-Corona

Pipeline del reto de analytics de Corona. Genera seis versiones de la
segmentación de consumidores finales:

|                                      | Sin PCA   | Con PCA (80%) | Con PCA (90%) |
|--------------------------------------|-----------|---------------|---------------|
| Valor original (sin corregir)        | versión 1 | versión 2     | versión 5     |
| Valor corregido (factor 2,65)        | versión 3 | versión 4     | versión 6     |

La versión del valor se controla con la variable de entorno `CORONA_VERSION`
(`original` o `corregido`) y la variante de la segmentación con `CORONA_PCA`
(`sin_pca`, `con_pca` o `con_pca_90`). Los scripts no se editan a mano para
cambiar entre versiones.

Diferencia entre las tres variantes de segmentación:

- `sin_pca`: agrupa sobre las 8 variables estandarizadas. k lo elige el
  silhouette entre 2 y 8.
- `con_pca`: PCA con corte de 80% de varianza (4 componentes). k lo elige el
  silhouette entre 2 y 8.
- `con_pca_90`: PCA con corte de 90% de varianza (5 componentes), con k fijado
  en 6. La quinta componente (descartada en `con_pca`) carga sobre `log_visitas`
  frente a `log_categorias` y `log_ticket`, es decir, separa clientes
  recurrentes de los que compran mucho en una sola visita; k = 6 coincide con el
  codo de la inercia.

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
├── segmentos_original_con_pca_90.parquet
├── segmentos_corregido_sin_pca.parquet
├── segmentos_corregido_con_pca.parquet
└── segmentos_corregido_con_pca_90.parquet

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
│   ├── con_pca/               (idem + pca_varianza.csv y pca_cargas.csv)
│   └── con_pca_90/            (idem; PCA al 90% y k = 6 fijo)
├── corregido/                  (misma estructura que original)
└── comparacion_versiones.csv   (consolidado de las 6 versiones + Rand ajustado)
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

Un solo comando corre las 6 versiones en orden y escribe la comparación:

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
$env:CORONA_PCA = "con_pca_90"
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
  finales con K-means y K-medoids. Siempre calcula las métricas para k ∈ [2, 8].
  - `CORONA_PCA=sin_pca` (comportamiento histórico): agrupa sobre las 8
    variables estandarizadas. k lo elige el silhouette.
  - `CORONA_PCA=con_pca`: aplica PCA sobre esas 8 variables y agrupa sobre
    las componentes necesarias para explicar al menos el 80% de la varianza.
    k lo elige el silhouette. Guarda `pca_varianza.csv` y `pca_cargas.csv`.
  - `CORONA_PCA=con_pca_90`: igual que `con_pca`, pero con umbral de 90% de
    varianza (una componente más que conserva la señal de visitas recurrentes)
    y k fijado en 6.
  Los perfiles y el mapa de calor siempre se calculan sobre las 8 variables
  originales estandarizadas para que sean interpretables. En las tres variantes
  `resumen_version.csv` incluye `silhouette_kmeans_std` y
  `silhouette_kmedoids_std`, el silhouette medido sobre las 8 variables (con la
  misma muestra), para poder comparar las variantes en la misma escala.
- `scripts/segmentacion/comparacion.py`: consolida las 6 variantes en
  `salidas/comparacion_versiones.csv`, con `k`, silhouette (en el espacio de
  agrupación y sobre las 8 variables estandarizadas), estabilidad, acuerdo
  K-means vs K-medoids, número de componentes del PCA (si aplica) y Rand
  ajustado entre variantes (con_pca vs sin_pca, con_pca_90 vs sin_pca,
  con_pca_90 vs con_pca y original vs corregido para cada variante de PCA).
