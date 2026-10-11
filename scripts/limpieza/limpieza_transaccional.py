"""
 Limpieza de base_transaccional

Carga base_transaccional.csv, limpia nombres y tipos, agrega valor_corregido
y guarda el resultado en data/transaccional_corregido.parquet

"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RUTA_ENTRADA = DATA_DIR / "base_transaccional.csv"
RUTA_SALIDA = DATA_DIR / "transaccional_corregido.parquet"

CSV_KWARGS = {"sep": ";", "encoding": "utf-8-sig", "decimal": "."}

# Interruptor de la corrección de valor. Está ENCENDIDO (11 oct 2026); falta
# confirmarlo con Corona, que pidió no modificar los valores.
# En True, se recalcula el valor de las líneas detectadas (ver corregir_valor).
# En False, valor_corregido es igual a valor y ninguna línea queda marcada.
CORREGIR_VALOR = True

# En la base, valor = precio * cantidad / 2.65 (factor de escala de Corona)
FACTOR_ESCALA = 2.65
# Margen para decidir que el desajuste es una potencia de 10 exacta
TOLERANCIA_ESCALA = 0.01


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


def corregir_valor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Con CORREGIR_VALOR = True, las líneas detectadas se recalculan como
    precio * cantidad / 2.65. Con False no se corrige ninguna.

    Nunca se modifica 'valor': el resultado queda en 'valor_corregido' y las
    líneas tocadas se marcan con valor_corregido_flag = 1.
    'fecha' es fecha_pedido sin hora.
    """
    corregir = detectar_valor_reducido(df) & CORREGIR_VALOR

    df["valor_corregido_flag"] = corregir.astype("int32")
    df["valor_corregido"] = df["valor"].where(
        ~corregir, df["precio"] * df["cantidad"] / FACTOR_ESCALA
    )
    df["fecha"] = df["fecha_pedido"].dt.date

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
    detectadas = detectar_valor_reducido(df).sum()
    corregidas = df["valor_corregido_flag"].sum()
    print(f"Corrección de valor: {'ENCENDIDA' if CORREGIR_VALOR else 'APAGADA'}")
    print(f"Líneas detectadas con valor reducido: {detectadas:,} ({detectadas / len(df) * 100:.2f}%)")
    print(f"Líneas con valor corregido: {corregidas:,} ({corregidas / len(df) * 100:.2f}%)")
    print(
        f"Valor total original: {df['valor'].sum():,.0f} | "
        f"corregido: {df['valor_corregido'].sum():,.0f} "
        f"({(df['valor_corregido'].sum() / df['valor'].sum() - 1) * 100:+.2f}%)"
    )


def main() -> pd.DataFrame:
    df = cargar()
    df = limpiar(df)
    df = corregir_valor(df)
    reporte_calidad(df)
    print(f"\nGuardando en {RUTA_SALIDA} ...")
    df.to_parquet(RUTA_SALIDA, index=False)
    print("Listo.")

    return df


if __name__ == "__main__":
    main()
