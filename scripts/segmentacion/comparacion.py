"""
Consolidación de las 4 versiones de la segmentación en
salidas/comparacion_versiones.csv.

Requiere haber corrido antes la segmentación de las 4 variantes:
    original/sin_pca, original/con_pca, corregido/sin_pca, corregido/con_pca.

Para cada variante toma:
    - k, silhouette K-means y K-medoids, estabilidad y acuerdo K-means vs
      K-medoids desde salidas/<version>/<pca>/resumen_version.csv.
    - % de clientes y % de ventas de cada segmento calculándolo con el parquet
      de segmentos y el parquet de features de esa versión.

Después añade filas de acuerdo (Rand ajustado) entre versiones:
    - con PCA vs sin PCA  (dentro de cada versión del valor)
    - original vs corregido (dentro de cada variante de PCA)

Uso:
    pip install pandas pyarrow
    python scripts/segmentacion/comparacion.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parent.parent.parent
DATA = RAIZ / "data"
OUT = RAIZ / "salidas"

VERSIONES = ("original", "corregido")
PCAS = ("sin_pca", "con_pca")


def ruta_segmentos(version: str, pca: str) -> Path:
    return DATA / f"segmentos_{version}_{pca}.parquet"


def ruta_features(version: str) -> Path:
    return DATA / f"features_clientes_{version}.parquet"


def ruta_resumen(version: str, pca: str) -> Path:
    return OUT / version / pca / "resumen_version.csv"


def indice_rand_ajustado(a: np.ndarray, b: np.ndarray) -> float:
    """
    Rand ajustado: 1 si las agrupaciones son idénticas, 0 si coincidirían por
    azar. Misma fórmula que la implementación en R, pensada para dos
    particiones de los mismos clientes.
    """
    tabla = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy()
    def comb2(x):
        return x * (x - 1) / 2
    juntos = comb2(tabla).sum()
    en_a = comb2(tabla.sum(axis=1)).sum()
    en_b = comb2(tabla.sum(axis=0)).sum()
    n = tabla.sum()
    esperado = en_a * en_b / comb2(n)
    denominador = (en_a + en_b) / 2 - esperado
    if denominador == 0:
        return float("nan")
    return float((juntos - esperado) / denominador)


def cargar_version(version: str, pca: str) -> dict:
    seg = pd.read_parquet(ruta_segmentos(version, pca))
    seg = seg.sort_values("cliente").reset_index(drop=True)
    feat = pd.read_parquet(ruta_features(version), columns=["cliente", "monto"])
    df = seg.merge(feat, on="cliente", how="left")
    resumen = pd.read_csv(ruta_resumen(version, pca)).iloc[0].to_dict()
    return {"version": version, "pca": pca, "seg": seg, "df": df, "resumen": resumen}


def perfil_segmentos(df: pd.DataFrame, columna: str) -> tuple[str, str]:
    """Devuelve pct_clientes y pct_ventas por segmento como 'seg1=xx,x;seg2=...'. """
    total_clientes = len(df)
    total_ventas = df["monto"].sum()
    agrupado = df.groupby(columna).agg(
        n=("cliente", "size"), ventas=("monto", "sum")
    ).sort_index()
    pct_cli = ";".join(
        f"seg{int(s)}={100 * n / total_clientes:.1f}".replace(".", ",")
        for s, n in zip(agrupado.index, agrupado["n"])
    )
    pct_ven = ";".join(
        f"seg{int(s)}={100 * v / total_ventas:.1f}".replace(".", ",")
        for s, v in zip(agrupado.index, agrupado["ventas"])
    )
    return pct_cli, pct_ven


def fila_version(datos: dict) -> dict:
    r = datos["resumen"]
    df = datos["df"]
    pct_cli_km, pct_ven_km = perfil_segmentos(df, "segmento_kmeans")
    pct_cli_kmed, pct_ven_kmed = perfil_segmentos(df, "segmento_kmedoids")
    return {
        "tipo": "version",
        "version": datos["version"],
        "pca": datos["pca"],
        "k": int(r["k"]),
        "n_componentes_pca": (
            int(r["n_componentes_pca"])
            if pd.notna(r.get("n_componentes_pca")) else ""
        ),
        "silhouette_kmeans": round(r["silhouette_kmeans"], 4),
        "silhouette_kmedoids": round(r["silhouette_kmedoids"], 4),
        "estabilidad_media": round(r["estabilidad_media"], 4),
        "estabilidad_min": round(r["estabilidad_min"], 4),
        "acuerdo_kmeans_kmedoids": round(r["acuerdo_kmeans_kmedoids"], 4),
        "n_clientes": int(r["n_clientes"]),
        "pct_clientes_segmentos_kmeans": pct_cli_km,
        "pct_ventas_segmentos_kmeans": pct_ven_km,
        "pct_clientes_segmentos_kmedoids": pct_cli_kmed,
        "pct_ventas_segmentos_kmedoids": pct_ven_kmed,
    }


def fila_comparacion(etiqueta: str, a: dict, b: dict) -> dict:
    # Las dos variantes se evalúan sobre los mismos clientes dentro de cada
    # versión del valor, así que podemos hacer merge por cliente antes de
    # comparar los segmentos.
    comp = a["seg"][["cliente", "segmento_kmeans", "segmento_kmedoids"]].merge(
        b["seg"][["cliente", "segmento_kmeans", "segmento_kmedoids"]],
        on="cliente", how="inner", suffixes=("_a", "_b"),
    )
    ari_km = indice_rand_ajustado(comp["segmento_kmeans_a"].to_numpy(),
                                   comp["segmento_kmeans_b"].to_numpy())
    ari_kmed = indice_rand_ajustado(comp["segmento_kmedoids_a"].to_numpy(),
                                     comp["segmento_kmedoids_b"].to_numpy())
    return {
        "tipo": "comparacion",
        "version": etiqueta,
        "pca": f"{a['version']}/{a['pca']}  vs  {b['version']}/{b['pca']}",
        "acuerdo_kmeans_kmedoids": "",
        "n_clientes": len(comp),
        "pct_clientes_segmentos_kmeans": f"ari_kmeans={ari_km:.4f}".replace(".", ","),
        "pct_clientes_segmentos_kmedoids": f"ari_kmedoids={ari_kmed:.4f}".replace(".", ","),
    }


def main() -> None:
    print("Leyendo las 4 versiones...")
    datos = {}
    for v in VERSIONES:
        for p in PCAS:
            if not ruta_resumen(v, p).exists():
                raise FileNotFoundError(
                    f"Falta {ruta_resumen(v, p)}. Corre la segmentación "
                    f"con CORONA_VERSION={v} CORONA_PCA={p}."
                )
            datos[(v, p)] = cargar_version(v, p)

    filas = [fila_version(datos[(v, p)]) for v in VERSIONES for p in PCAS]

    # con PCA vs sin PCA (misma versión del valor)
    for v in VERSIONES:
        filas.append(fila_comparacion(
            f"con_pca_vs_sin_pca_{v}", datos[(v, "sin_pca")], datos[(v, "con_pca")]
        ))
    # original vs corregido (misma variante de PCA)
    for p in PCAS:
        filas.append(fila_comparacion(
            f"original_vs_corregido_{p}", datos[("original", p)], datos[("corregido", p)]
        ))

    tabla = pd.DataFrame(filas)
    salida = OUT / "comparacion_versiones.csv"
    tabla.to_csv(salida, index=False)
    print(f"Escrito {salida}")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
