"""
Unión de bases (por Cliente)
IMPORTANTE: correr primero limpieza_transaccional.py y limpieza_clientes.py

"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUTA_TRANSACCIONAL_LIMPIA = DATA_DIR / "transaccional_limpio.parquet"
RUTA_CLIENTES_LIMPIA = DATA_DIR / "clientes_limpio.parquet"
RUTA_SALIDA = DATA_DIR / "datos_completos.parquet"


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RUTA_TRANSACCIONAL_LIMPIA.exists() or not RUTA_CLIENTES_LIMPIA.exists():
        raise FileNotFoundError(
            "Faltan los archivos limpios. Corre primero:\n"
            "  python limpieza_transaccional.py\n"
            "  python limpieza_clientes.py"
        )
    transaccional = pd.read_parquet(RUTA_TRANSACCIONAL_LIMPIA)
    clientes = pd.read_parquet(RUTA_CLIENTES_LIMPIA)
    return transaccional, clientes


def unir(transaccional: pd.DataFrame, clientes: pd.DataFrame) -> pd.DataFrame:
    """
    'cliente' es la ÚNICA columna que comparten ambas bases 
    """
    return transaccional.merge(clientes, on="cliente", how="left")


def reporte_calidad(transaccional: pd.DataFrame, clientes: pd.DataFrame, datos_completos: pd.DataFrame) -> None:
    
    coincide = datos_completos.shape[0] == transaccional.shape[0]
    print(f"Filas transaccional: {transaccional.shape[0]:,}")
    print(f"Filas datos_completos: {datos_completos.shape[0]:,} (sin inflar: {coincide})")
    print(f"Columnas resultantes: {datos_completos.shape[1]}")

    c_t = set(transaccional["cliente"].unique())
    c_m = set(clientes["cliente"].unique())
    huerfanos = c_t - c_m
    print(f"\nClientes distintos en transaccional: {len(c_t):,}")
    print(f"Clientes distintos en maestro: {len(c_m):,}")
    print(f"Clientes huérfanos (en transaccional, sin maestro): {len(huerfanos)}")
    if huerfanos:
        print(f"  -> {list(huerfanos)}")
        print("  -> quedan con NaN en las columnas de cliente (tipo_cliente, edad, genero, etc.)")

    # Filas cuyas columnas de cliente quedaron vacías tras el merge (huérfanos)
    filas_sin_cliente_info = datos_completos["tipo_cliente"].isna().sum()
    print(f"\nFilas de transaccional sin info de cliente tras el merge: {filas_sin_cliente_info}")


def main() -> pd.DataFrame:
    transaccional, clientes = cargar()
    datos_completos = unir(transaccional, clientes)
    reporte_calidad(transaccional, clientes, datos_completos)
    print(f"\nGuardando en {RUTA_SALIDA} ...")
    datos_completos.to_parquet(RUTA_SALIDA, index=False)
    print("Listo.")

    return datos_completos


if __name__ == "__main__":
    main()
