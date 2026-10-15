"""
Corre el pipeline completo del reto para las 4 versiones de la segmentación:

    original/sin_pca   original/con_pca
    corregido/sin_pca  corregido/con_pca

y después consolida los resultados en salidas/comparacion_versiones.csv.

Pasos por cada versión del valor (original/corregido):
    1. limpieza_transaccional.py  -> data/transaccional_<version>.parquet
    2. variables_clientes.py      -> data/features_clientes_<version>.parquet
                                     salidas/<version>/descriptivos/*.csv
    3. graficas.py                -> salidas/<version>/descriptivos/graficas/*.png
    4. segmentacion.R (sin_pca y con_pca) -> data/segmentos_<version>_<pca>.parquet
                                             salidas/<version>/<pca>/*

limpieza_clientes.py se corre una sola vez al principio (no depende del valor).

Uso:
    pip install duckdb pandas pyarrow matplotlib
    # En R:  install.packages(c("nanoparquet", "cluster", "ggplot2", "scales"))
    python scripts/run_pipeline.py

Variables opcionales:
    RSCRIPT   ruta al ejecutable Rscript (si no está en PATH)
    SOLO      lista separada por comas de pasos a ejecutar, útil para depurar.
              Valores: clientes, transaccional, variables, graficas,
              segmentacion, comparacion. Si no se define se corren todos.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


RAIZ = Path(__file__).resolve().parent.parent
SCRIPTS = RAIZ / "scripts"

VERSIONES = ("original", "corregido")
PCAS = ("sin_pca", "con_pca")


def encontrar_rscript() -> str:
    # 1) respeta RSCRIPT si está definida y existe
    desde_env = os.environ.get("RSCRIPT")
    if desde_env and Path(desde_env).exists():
        return desde_env
    # 2) PATH
    en_path = shutil.which("Rscript")
    if en_path:
        return en_path
    # 3) rutas típicas de Windows
    candidatos = [
        Path("C:/Program Files/R"),
        Path("C:/Program Files (x86)/R"),
    ]
    for base in candidatos:
        if not base.exists():
            continue
        for sub in sorted(base.iterdir(), reverse=True):  # R-4.6.1, R-4.5.0, ...
            exe = sub / "bin" / "Rscript.exe"
            if exe.exists():
                return str(exe)
    raise FileNotFoundError(
        "No encontré Rscript. Instala R y pon su carpeta bin en PATH, "
        "o define la variable de entorno RSCRIPT con la ruta al ejecutable."
    )


def correr(cmd: list[str], env: dict | None = None, descripcion: str = "") -> None:
    if descripcion:
        print(f"\n>>> {descripcion}")
    print("    " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(
            f"Falló: {' '.join(str(c) for c in cmd)} (código {r.returncode})"
        )


def entorno(**extra) -> dict:
    e = os.environ.copy()
    e.update({k: v for k, v in extra.items() if v is not None})
    return e


def pasos_seleccionados() -> set[str]:
    todos = {"clientes", "transaccional", "variables", "graficas", "segmentacion", "comparacion"}
    pedido = os.environ.get("SOLO", "").strip()
    if not pedido:
        return todos
    seleccion = {p.strip() for p in pedido.split(",") if p.strip()}
    desconocido = seleccion - todos
    if desconocido:
        raise SystemExit(f"SOLO: valores no válidos: {sorted(desconocido)}")
    return seleccion


def main() -> None:
    pasos = pasos_seleccionados()
    rscript = encontrar_rscript() if "segmentacion" in pasos else None
    py = sys.executable

    if "clientes" in pasos:
        correr(
            [py, str(SCRIPTS / "limpieza" / "limpieza_clientes.py")],
            env=entorno(),
            descripcion="Limpieza maestro de clientes (una vez para las dos versiones)",
        )

    for version in VERSIONES:
        print(f"\n============================================================")
        print(f"==  Versión: {version}")
        print(f"============================================================")
        env_v = entorno(CORONA_VERSION=version)

        if "transaccional" in pasos:
            correr(
                [py, str(SCRIPTS / "limpieza" / "limpieza_transaccional.py")],
                env=env_v,
                descripcion=f"Limpieza transaccional ({version})",
            )
        if "variables" in pasos:
            correr(
                [py, str(SCRIPTS / "limpieza" / "variables_clientes.py")],
                env=env_v,
                descripcion=f"Variables por cliente y descriptivos ({version})",
            )
        if "graficas" in pasos:
            correr(
                [py, str(SCRIPTS / "limpieza" / "graficas.py")],
                env=env_v,
                descripcion=f"Gráficas descriptivas ({version})",
            )
        if "segmentacion" in pasos:
            for pca in PCAS:
                env_vp = entorno(CORONA_VERSION=version, CORONA_PCA=pca)
                correr(
                    [rscript, "--vanilla", str(SCRIPTS / "segmentacion" / "segmentacion.R")],
                    env=env_vp,
                    descripcion=f"Segmentación ({version}, {pca})",
                )

    if "comparacion" in pasos:
        correr(
            [py, str(SCRIPTS / "segmentacion" / "comparacion.py")],
            env=entorno(),
            descripcion="Comparación entre versiones",
        )

    print("\nPipeline completo.")


if __name__ == "__main__":
    main()
