"""
Funciones y constantes compartidas por los modelos de scripts/modelos/.

- Rutas (calculadas desde la ubicación de este archivo, como en el resto del proyecto).
- Versión del valor (CORONA_VERSION, por defecto 'corregido') y segmentación de trabajo
  (con_pca_90, 6 segmentos de consumidores finales).
- Semilla, nombres de segmentos y carga de datos.
- Estilo de gráficas: reutiliza la paleta y la configuración de scripts/limpieza/graficas.py.

No se corre solo; lo importan los scripts mNN_*.py.
"""
import os
import sys
import unicodedata
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent.parent
DATA = RAIZ / "data"
DATA_MODELOS = DATA / "modelos"
DATA_EXTERNOS = DATA / "externos"
SALIDAS_MODELOS = RAIZ / "salidas" / "modelos"

# Reutiliza el estilo de las gráficas descriptivas (paleta, fuente, formatos)
sys.path.insert(0, str(RAIZ / "scripts" / "limpieza"))
import graficas as g  # noqa: E402  (importarlo no genera gráficas: main() está protegido)

# La consola de Windows a veces no es UTF-8; así los acentos no rompen los print
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ---------------------------------------------------------------------------
# Parámetros generales
# ---------------------------------------------------------------------------
SEMILLA = 2026
SEGMENTACION = "con_pca_90"  # variante de la segmentación de trabajo
# Los CSV se escriben con 10 cifras significativas: las sumas en paralelo de DuckDB varían en el
# último decimal entre corridas y así los archivos salen idénticos.
FORMATO_CSV = "%.10g"

NOMBRES_SEGMENTOS = {
    1: "Proyecto recurrente",
    2: "Proyecto grande en una visita",
    3: "Solo Baños",
    4: "Solo Pinturas",
    5: "Solo Cocinas",
    6: "Solo Revestimientos",
}
# Grupos de segmentos que usa el MDP (m11)
GRUPOS_SEGMENTO = {
    1: "Recurrente",
    2: "Proyecto grande",
    3: "Un solo negocio",
    4: "Un solo negocio",
    5: "Un solo negocio",
    6: "Un solo negocio",
}


def obtener_version() -> str:
    """
    Versión del valor: 'original' o 'corregido' (por defecto 'corregido').

    Se toma del argumento --version del script (--version original o --version=original) y, si
    no viene, de la variable CORONA_VERSION. La versión elegida se deja en CORONA_VERSION para que
    los procesos hijos (por ejemplo, el relanzamiento de m04) usen la misma.
    """
    version = os.environ.get("CORONA_VERSION", "corregido")
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--version" and i + 1 < len(args):
            version = args[i + 1]
        elif arg.startswith("--version="):
            version = arg.split("=", 1)[1]
    version = version.strip().lower()
    if version not in ("original", "corregido"):
        raise ValueError(
            f"La versión debe ser 'original' o 'corregido' (recibí '{version}')."
        )
    os.environ["CORONA_VERSION"] = version
    return version


VERSION = obtener_version()
TRX_PARQUET = DATA / f"transaccional_{VERSION}.parquet"
FEATURES_PARQUET = DATA / f"features_clientes_{VERSION}.parquet"
SEGMENTOS_PARQUET = DATA / f"segmentos_{VERSION}_{SEGMENTACION}.parquet"

# Con la versión original todo va a salidas/modelos_original/ (los puntajes por cliente, en su
# subcarpeta puntajes/) para no tocar las salidas de la versión corregida.
if VERSION == "original":
    SALIDAS_MODELOS = RAIZ / "salidas" / "modelos_original"
    DATA_MODELOS = SALIDAS_MODELOS / "puntajes"
# Rutas relativas para citarlas en los resumen.md
RUTA_SALIDAS_TXT = SALIDAS_MODELOS.relative_to(RAIZ).as_posix()
RUTA_PUNTAJES_TXT = DATA_MODELOS.relative_to(RAIZ).as_posix()
VALOR_TXT = "valor corregido" if VERSION == "corregido" else "valor original"


# ---------------------------------------------------------------------------
# Carpetas y datos
# ---------------------------------------------------------------------------
def carpeta_salida(nombre_modelo: str) -> Path:
    """Crea y devuelve salidas/modelos/<nombre_modelo>/."""
    carpeta = SALIDAS_MODELOS / nombre_modelo
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def ruta_puntajes(nombre_modelo: str) -> Path:
    """Ruta del parquet de puntajes por cliente: data/modelos/<nombre_modelo>.parquet."""
    DATA_MODELOS.mkdir(parents=True, exist_ok=True)
    return DATA_MODELOS / f"{nombre_modelo}.parquet"


def verificar_insumos(*archivos: Path) -> None:
    """Falla con un mensaje claro si falta algún archivo del pipeline."""
    for archivo in archivos:
        if not Path(archivo).exists():
            raise FileNotFoundError(
                f"No se encontró {archivo}. Corre primero python scripts/run_pipeline.py."
            )


def conexion() -> duckdb.DuckDBPyConnection:
    """Conexión DuckDB con vistas 'trx', 'features' y 'segmentos' de la versión activa."""
    verificar_insumos(TRX_PARQUET, FEATURES_PARQUET, SEGMENTOS_PARQUET)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW trx AS SELECT * FROM '{TRX_PARQUET}'")
    con.execute(f"CREATE VIEW features AS SELECT * FROM '{FEATURES_PARQUET}'")
    con.execute(f"CREATE VIEW segmentos AS SELECT * FROM '{SEGMENTOS_PARQUET}'")
    return con


def cargar_clientes() -> pd.DataFrame:
    """Ficha de consumidores finales con su segmento (segmento_kmeans de con_pca_90)."""
    con = conexion()
    df = con.execute("""
        SELECT f.*, s.segmento_kmeans AS segmento
        FROM features f
        JOIN segmentos s USING (cliente)
        WHERE f.tipo_cliente = 'Consumidor final'
    """).df()
    df["nombre_segmento"] = df["segmento"].map(NOMBRES_SEGMENTOS)
    df["es_contactable"] = df["es_contactable"].fillna(False).astype(bool)
    return df


def normalizar(texto) -> str:
    """Mayúsculas sin tildes ni espacios repetidos (para cruzar nombres de territorios)."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(t.upper().replace(",", " ").replace(".", " ").split())


# ---------------------------------------------------------------------------
# Gráficas (mismo estilo de scripts/limpieza/graficas.py)
# ---------------------------------------------------------------------------
plt = g.plt
configurar_estilo = g.configurar_estilo
miles = g.miles
pct = g.pct
rotular_barras = g.rotular_barras
SUPERFICIE, TINTA, TINTA_2, TENUE = g.SUPERFICIE, g.TINTA, g.TINTA_2, g.TENUE
REJILLA, EJE, AZUL, NARANJA, ROJO = g.REJILLA, g.EJE, g.AZUL, g.NARANJA, g.ROJO
GRIS_MEDIO, SERIES = g.GRIS_MEDIO, g.SERIES
VERDE, AMARILLO, ROSA = g.SERIES[2], g.SERIES[3], g.SERIES[4]
FUENTE = g.FUENTE_CORREGIDA if VERSION == "corregido" else g.FUENTE


def guardar_figura(fig, carpeta: Path, nombre: str, titulo: str, subtitulo: str,
                   nota: str | None = None) -> None:
    """Igual que graficas.guardar, pero escribe en la carpeta del modelo."""
    nota = FUENTE if nota is None else nota
    alto = fig.get_figheight()
    fig.tight_layout(rect=[0, 0.42 / alto, 1, 1 - 0.78 / alto])
    fig.text(0.012, 1 - 0.14 / alto, titulo, fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 1 - 0.44 / alto, subtitulo, fontsize=10.5, color=TINTA_2, va="top")
    fig.text(0.012, 0.10 / alto, nota, fontsize=8.5, color=TENUE, va="bottom")
    fig.savefig(carpeta / f"{nombre}.png", dpi=200)
    plt.close(fig)
    print(f"  {nombre}.png")


def num(x: float, dec: int = 1) -> str:
    """0.1234 -> '0,1' (decimal con coma)."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def escribir_resumen(carpeta: Path, texto: str) -> None:
    """Escribe resumen.md en la carpeta del modelo."""
    (carpeta / "resumen.md").write_text(texto.strip() + "\n", encoding="utf-8")
    print("  resumen.md")
