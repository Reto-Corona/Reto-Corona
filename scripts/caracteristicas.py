"""
 Ingeniería de Características
Requiere haber corrido antes:
    limpieza_transaccional.py  limpieza_clientes.py union_bases.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUTA_ENTRADA = DATA_DIR / "datos_completos.parquet"
RUTA_SALIDA = DATA_DIR / "features_clientes.parquet"

# Percentiles usados para sub-segmentar clientes tipo "Empresa" por volumen de cantidad comprada 

PERCENTIL_EMPRESA_PEQUENA = 0.50
PERCENTIL_EMPRESA_MEDIANA = 0.90


def cargar() -> pd.DataFrame:
    if not RUTA_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_ENTRADA}. Corre primero limpieza_transaccional.py, "
            "limpieza_clientes.py y union_bases.py"
        )
    return pd.read_parquet(RUTA_ENTRADA)


# ------------------------------------------------------------------
# RFM
# ------------------------------------------------------------------
def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recencia: días entre la última compra del cliente y la fecha MÁXIMA
    del dataset 

    Frecuencia: número de combinaciones DISTINTAS (cliente, fecha_pedido).

    Valor Monetario: suma de 'valor'
    """
    fecha_referencia = df["fecha_pedido"].max()

    rfm = (
        df.groupby("cliente")
        .agg(
            ultima_compra=("fecha_pedido", "max"),
            valor_monetario=("valor", "sum"),
        )
        .reset_index()
    )
    rfm["recencia_dias"] = (fecha_referencia - rfm["ultima_compra"]).dt.days

    frecuencia = (
        df.drop_duplicates(subset=["cliente", "fecha_pedido"])
        .groupby("cliente")
        .size()
        .rename("frecuencia_visitas")
        .reset_index()
    )

    rfm = rfm.merge(frecuencia, on="cliente", how="left")
    rfm["ticket_promedio"] = rfm["valor_monetario"] / rfm["frecuencia_visitas"]

    return rfm[["cliente", "recencia_dias", "frecuencia_visitas", "valor_monetario", "ticket_promedio"]]


def extraer_demograficas(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cliente", "tipo_cliente", "edad", "genero", "tratamiento_datos", "es_contactable"]
    demograficas = df[cols].drop_duplicates(subset="cliente").reset_index(drop=True)
    return demograficas


def calcular_comportamiento(df: pd.DataFrame) -> pd.DataFrame:
    """
    cantidad_total: suma libre de 'cantidad' entre categorías (ver si lo podemos hacer

    negocio_principal: el negocio donde el cliente concentra más 'valor'.
    n_categorias_distintas: diversidad de categorías compradas.
    """
    cantidad_total = df.groupby("cliente")["cantidad"].sum().rename("cantidad_total")

    valor_por_negocio = df.groupby(["cliente", "negocio"])["valor"].sum().reset_index()
    idx_max_negocio = valor_por_negocio.groupby("cliente")["valor"].idxmax()
    negocio_principal = (
        valor_por_negocio.loc[idx_max_negocio, ["cliente", "negocio"]]
        .rename(columns={"negocio": "negocio_principal"})
        .set_index("cliente")
    )

    n_categorias = df.groupby("cliente")["categoria"].nunique().rename("n_categorias_distintas")
    n_negocios = df.groupby("cliente")["negocio"].nunique().rename("n_negocios_distintos")

    comportamiento = pd.concat(
        [cantidad_total, negocio_principal, n_categorias, n_negocios], axis=1
    ).reset_index()

    return comportamiento


def calcular_geograficas(df: pd.DataFrame) -> pd.DataFrame:
    """
    region_principal / cluster_principal: la moda (valor más frecuente)
    por cliente, ya que un cliente puede comprar en más de un punto de
    venta/región a lo largo del tiempo.
    """
    def moda(serie: pd.Series):
        m = serie.mode()
        return m.iloc[0] if len(m) > 0 else np.nan

    geograficas = (
        df.groupby("cliente")
        .agg(
            region_principal=("region", moda),
            cluster_principal=("cluster", moda),
        )
        .reset_index()
    )
    return geograficas


def segmentar_empresas(features: pd.DataFrame) -> pd.DataFrame:
    """
    Solo aplica a tipo_cliente == 'Empresa'. Usa percentiles de
    cantidad_total para diferenciar tamaños
    Para 'Consumidor final' queda como 'No aplica'.
    """
    es_empresa = features["tipo_cliente"] == "Empresa"
    cantidad_empresas = features.loc[es_empresa, "cantidad_total"]

    p_pequena = cantidad_empresas.quantile(PERCENTIL_EMPRESA_PEQUENA)
    p_mediana = cantidad_empresas.quantile(PERCENTIL_EMPRESA_MEDIANA)

    def clasificar(cantidad):
        if cantidad <= p_pequena:
            return "Empresa pequeña"
        elif cantidad <= p_mediana:
            return "Empresa mediana"
        else:
            return "Empresa grande / distribuidor"

    features["segmento_empresa"] = "No aplica"
    features.loc[es_empresa, "segmento_empresa"] = cantidad_empresas.apply(clasificar)

    return features


def reporte_calidad(features: pd.DataFrame) -> None:
    print("REPORTE - features_clientes")
    print(f"Clientes: {features.shape[0]:,} | Variables: {features.shape[1]}")
    print(f"\nRFM - resumen:\n{features[['recencia_dias','frecuencia_visitas','valor_monetario']].describe()}")
    print(f"\nNegocio principal:\n{features['negocio_principal'].value_counts()}")
    print(f"\nCluster principal:\n{features['cluster_principal'].value_counts()}")
    print(f"\nSegmento empresa:\n{features['segmento_empresa'].value_counts()}")
    print(f"\nNulos por columna:\n{features.isna().sum()[features.isna().sum() > 0]}")



def main() -> pd.DataFrame:
    df = cargar()
    rfm = calcular_rfm(df)
    demograficas = extraer_demograficas(df)
    comportamiento = calcular_comportamiento(df)
    geograficas = calcular_geograficas(df)
    features = (
        rfm.merge(demograficas, on="cliente", how="left")
        .merge(comportamiento, on="cliente", how="left")
        .merge(geograficas, on="cliente", how="left")
    )
    features = segmentar_empresas(features)
    reporte_calidad(features)
    print(f"\nGuardando en {RUTA_SALIDA} ...")
    features.to_parquet(RUTA_SALIDA, index=False)
    print("Listo.")

    return features


if __name__ == "__main__":
    main()
