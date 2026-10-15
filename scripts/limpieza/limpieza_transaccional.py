"""
 Limpieza de base_transaccional

Carga base_transaccional.csv, limpia nombres y tipos, agrega valor_corregido
y guarda el resultado en data/transaccional_<version>.parquet.

El interruptor de corrección se controla con la variable de entorno
CORONA_VERSION:
    - CORONA_VERSION=original  -> CORREGIR_VALOR = False
    - CORONA_VERSION=corregido -> CORREGIR_VALOR = True   (valor por defecto)

Así el script puede correrse en paralelo para las dos variantes sin pisarse,
sin tener que editarlo a mano.

"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RUTA_ENTRADA = DATA_DIR / "base_transaccional.csv"

CSV_KWARGS = {"sep": ";", "encoding": "utf-8-sig", "decimal": "."}

# En la base, valor = precio * cantidad / 2.65 (factor de escala de Corona)
FACTOR_ESCALA = 2.65
# Margen para decidir que el desajuste es una potencia de 10 exacta
TOLERANCIA_ESCALA = 0.01


def obtener_version() -> str:
    """
    Lee CORONA_VERSION del entorno. Debe ser 'original' o 'corregido'.
    Si no está definida, se usa 'corregido' (comportamiento histórico).
    """
    version = os.environ.get("CORONA_VERSION", "corregido").strip().lower()
    if version not in ("original", "corregido"):
        raise ValueError(
            f"CORONA_VERSION debe ser 'original' o 'corregido' (recibí '{version}')."
        )
    return version


def ruta_salida(version: str) -> Path:
    return DATA_DIR / f"transaccional_{version}.parquet"


def cargar() -> pd.DataFrame:
    return pd.read_csv(RUTA_ENTRADA, **CSV_KWARGS)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """
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


def detectar_valor_reducido(df: pd.DataFrame) -> pd.Series:
    """
    r = precio * cantidad / valor / 2.65. En una línea normal r vale 1.
    Si r es 10, 100 o 1.000 (±1%), el valor quedó dividido por esa escala.
    Devuelve True en esas líneas.

    Se comprobó por producto que en esas líneas la cantidad es igual a la de
    las líneas normales y el valor es el que viene reducido.
    """
    calculable = (df["valor"] != 0) & (df["cantidad"] > 0) & (df["precio"] > 0)
    r = (df["precio"] * df["cantidad"] / df["valor"] / FACTOR_ESCALA).where(calculable)
    escala = np.round(np.log10(r))
    return escala.between(1, 3) & (
        (r / np.power(10, escala) - 1).abs() <= TOLERANCIA_ESCALA
    )


def corregir_valor(df: pd.DataFrame, corregir_valor_flag: bool) -> pd.DataFrame:
    """
    Con corregir_valor_flag = True, las líneas detectadas se recalculan como
    precio * cantidad / 2.65. Con False no se corrige ninguna.

    Nunca se modifica 'valor': el resultado queda en 'valor_corregido' y las
    líneas tocadas se marcan con valor_corregido_flag = 1.
    'fecha' es fecha_pedido sin hora.
    """
    corregir = detectar_valor_reducido(df) & corregir_valor_flag

    df["valor_corregido_flag"] = corregir.astype("int32")
    df["valor_corregido"] = df["valor"].where(
        ~corregir, df["precio"] * df["cantidad"] / FACTOR_ESCALA
    )
    df["fecha"] = df["fecha_pedido"].dt.date

    return df


def reporte_calidad(df: pd.DataFrame, corregir_valor_flag: bool) -> None:

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
    detectadas = detectar_valor_reducido(df).sum()
    corregidas = df["valor_corregido_flag"].sum()
    print(f"Corrección de valor: {'ENCENDIDA' if corregir_valor_flag else 'APAGADA'}")
    print(f"Líneas detectadas con valor reducido: {detectadas:,} ({detectadas / len(df) * 100:.2f}%)")
    print(f"Líneas con valor corregido: {corregidas:,} ({corregidas / len(df) * 100:.2f}%)")
    print(
        f"Valor total original: {df['valor'].sum():,.0f} | "
        f"corregido: {df['valor_corregido'].sum():,.0f} "
        f"({(df['valor_corregido'].sum() / df['valor'].sum() - 1) * 100:+.2f}%)"
    )


def main() -> pd.DataFrame:
    version = obtener_version()
    corregir_flag = version == "corregido"
    salida = ruta_salida(version)
    print(f"== Versión: {version} (CORREGIR_VALOR = {corregir_flag}) ==")

    df = cargar()
    df = limpiar(df)
    df = corregir_valor(df, corregir_flag)
    reporte_calidad(df, corregir_flag)
    print(f"\nGuardando en {salida} ...")
    df.to_parquet(salida, index=False)
    print("Listo.")

    return df


if __name__ == "__main__":
    main()
