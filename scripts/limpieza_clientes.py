"""
 Limpieza de maestro_clientes

Carga maestro_clientes.csv, deduplica, corrige tipos, agrega la variable
es_contactable, y guarda el resultado en data/clientes_limpio.parquet

"""

import pandas as pd
from pathlib import Path



DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUTA_ENTRADA = DATA_DIR / "maestro_clientes.csv"
RUTA_SALIDA = DATA_DIR / "clientes_limpio.parquet"

CSV_KWARGS = {"sep": ";", "encoding": "utf-8-sig", "decimal": "."}


def cargar() -> pd.DataFrame:
    return pd.read_csv(RUTA_ENTRADA, **CSV_KWARGS)


def limpiar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres de columna y convierte tipos de dato."""
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    df["primera_fecha"] = pd.to_datetime(
        df["primera_fecha"], format="%Y-%m-%d", errors="coerce"
    )
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    return df


def deduplicar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Solo guarda el registro con la compra más antigua por cliente. Devuelve el df deduplicado y el df original
    """
    df_raw = df.copy()
    idx_mas_antigua = df.groupby("cliente")["primera_fecha"].idxmin()
    df_dedup = df.loc[idx_mas_antigua].reset_index(drop=True)
    return df_dedup, df_raw


def agregar_variable_contactable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Más rápido para ver si es contactable o no
    """
    df["es_contactable"] = df["tratamiento_datos"] == "SÍ"
    return df


def reporte_calidad(df_dedup: pd.DataFrame, df_raw: pd.DataFrame) -> None:

    print(f"Filas originales: {df_raw.shape[0]:,}")
    print(f"Filas tras deduplicar por cliente (fecha más antigua): {df_dedup.shape[0]:,}")
    print(f"Duplicados eliminados: {df_raw.shape[0] - df_dedup.shape[0]:,}")
    print(f"\nNulos por columna:\n{df_dedup.isna().sum()[df_dedup.isna().sum() > 0]}")
    print(f"\nDistribución tipo_cliente:\n{df_dedup['tipo_cliente'].value_counts()}")
    print(f"\nDistribución genero:\n{df_dedup['genero'].value_counts()}")
    print(f"\nDistribución tratamiento_datos:\n{df_dedup['tratamiento_datos'].value_counts()}")
    print(f"\nClientes contactables (es_contactable=True): {df_dedup['es_contactable'].sum():,} "
          f"({df_dedup['es_contactable'].mean()*100:.1f}%)")


def main() -> pd.DataFrame:
    df = cargar()
    df = limpiar_tipos(df)
    df, df_raw = deduplicar(df)
    df = agregar_variable_contactable(df)
    reporte_calidad(df, df_raw)
    print(f"\nGuardando en {RUTA_SALIDA} ...")
    df.to_parquet(RUTA_SALIDA, index=False)

    return df


if __name__ == "__main__":
    main()
