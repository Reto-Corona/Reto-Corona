"""
 Limpieza de base_transaccional

"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUTA_ENTRADA = DATA_DIR / "base_transaccional.csv"
RUTA_SALIDA = DATA_DIR / "transaccional_limpio.parquet"

CSV_KWARGS = {"sep": ";", "encoding": "utf-8-sig", "decimal": "."}


def cargar() -> pd.DataFrame:
    return pd.read_csv(RUTA_ENTRADA, **CSV_KWARGS)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cambios aplicados (sin borrar ninguna fila):
    1. Corrige el typo 'Subategoria' a 'Subcategoria'
    2. Estandariza nombres de columna 
    3. Convierte '(No Value)' en region a NaN
    4. Convierte tipos: fecha_pedido a fecha, cantidad/precio/valor a numérico

    """
    df = df.rename(columns={"Subategoria": "Subcategoria"})
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    df["region"] = df["region"].replace("(No Value)", np.nan)

    df["fecha_pedido"] = pd.to_datetime(
        df["fecha_pedido"], format="%Y-%m-%d", errors="coerce"
    )

    for col in ["cantidad", "precio", "valor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def reporte_calidad(df: pd.DataFrame) -> None:
    
    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    print(
        f"Rango de fechas: {df['fecha_pedido'].min().date()} "
        f"a {df['fecha_pedido'].max().date()}"
    )
    print(f"Nulos por columna:\n{df.isna().sum()[df.isna().sum() > 0]}")
    print(f"Duplicados exactos: {df.duplicated().sum()}")
    print(f"Filas con valor = 0: {(df['valor'] == 0).sum():,} ")
    print(f"Filas con valor < 0: {(df['valor'] < 0).sum():,} ")
    print(f"Clientes distintos: {df['cliente'].nunique():,}")
    print(f"Pedidos distintos (id_pedido): {df['id_pedido'].nunique():,}")


def main() -> pd.DataFrame:
    df = cargar()
    df = limpiar(df)
    reporte_calidad(df)
    print(f"\nGuardando en {RUTA_SALIDA} ...")
    df.to_parquet(RUTA_SALIDA, index=False)
    print("Listo.")

    return df


if __name__ == "__main__":
    main()
