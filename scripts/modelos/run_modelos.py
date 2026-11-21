"""
Corre todos los modelos de scripts/modelos/ en un orden que respeta sus dependencias y al final
escribe salidas/modelos/resumen_modelos.md.

    m02_gmm.py            mezclas gaussianas como comparación de la segmentación
    m03_empresas.py       segmentación de empresas (ABC + K-means)
    m05_canasta.py        reglas de asociación (canasta)
    m06_secuencias.py     minería de secuencias (PrefixSpan) y transición entre negocios
    m07_elasticidad.py    elasticidad precio
    m08_recompra.py       probabilidad de recompra a 90 días
    m04_red_bayesiana.py  red bayesiana de la recompra (usa la base y las métricas de m08)
    m09_supervivencia.py  tiempo hasta la segunda compra (usa la base de m08)
    m10_clv.py            valor de vida del cliente (BG/NBD + Gamma-Gamma)
    m11_mdp.py            cadena de Markov y MDP del ciclo de vida
    m12_territorio.py     potencial por departamento con datos del DANE
    m13_mochila.py        selección de clientes a contactar (usa m08, m10 y m11)
    m14_piloto.py         diseño del piloto (usa m08 y los costos de m11)
    resumen_modelos.py    tabla resumen de todos los modelos

Requiere haber corrido antes el pipeline (python scripts/run_pipeline.py) y tener los archivos
del DANE en data/externos/. La versión del valor se elige con --version original|corregido (o con
CORONA_VERSION); por defecto es 'corregido'.

Con --version original corre solo los modelos que dependen del valor (MODELOS_ORIGINAL) y escribe
en salidas/modelos_original/<script>/ (puntajes en salidas/modelos_original/puntajes/). No toca
salidas/modelos/ ni data/modelos/. No escribe resumen_modelos.md (incluye m02, m10 y m13, que no
se corren en esa versión).

Uso:
    pip install -r requirements.txt
    pip install --no-deps lifelines==0.30.3 autograd-gamma==0.5.0
    python scripts/modelos/run_modelos.py
    python scripts/modelos/run_modelos.py --version original

Variable opcional:
    SOLO_MODELOS  lista separada por comas con los modelos a correr (p. ej. "m08,m14").
                  El resumen se vuelve a escribir siempre al final.
"""
import os
import subprocess
import sys
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
MODELOS = ["m02_gmm", "m03_empresas", "m05_canasta", "m06_secuencias", "m07_elasticidad", "m08_recompra",
           "m04_red_bayesiana", "m09_supervivencia", "m10_clv", "m11_mdp", "m12_territorio", "m13_mochila",
           "m14_piloto"]
# Modelos que usan el valor (ticket, monto, valor de la visita, negocio principal por valor o costo
# como % del ticket). m05 (solo productos) y m07 (precio y cantidad) no cambian con la versión;
# m02, m10 y m13 se descartaron para la versión original.
MODELOS_ORIGINAL = ["m03_empresas", "m06_secuencias", "m08_recompra", "m04_red_bayesiana",
                    "m09_supervivencia", "m11_mdp", "m12_territorio", "m14_piloto"]


def version_pedida() -> str:
    """Versión del argumento --version (o de CORONA_VERSION); por defecto 'corregido'."""
    version = os.environ.get("CORONA_VERSION", "corregido")
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--version" and i + 1 < len(args):
            version = args[i + 1]
        elif arg.startswith("--version="):
            version = arg.split("=", 1)[1]
    version = version.strip().lower()
    if version not in ("original", "corregido"):
        raise SystemExit(f"--version debe ser 'original' o 'corregido' (recibí '{version}').")
    return version


VERSION = version_pedida()


def seleccion() -> list[str]:
    """Modelos a correr según la versión y SOLO_MODELOS (prefijo o nombre completo)."""
    base = MODELOS_ORIGINAL if VERSION == "original" else MODELOS
    pedido = [p.strip() for p in os.environ.get("SOLO_MODELOS", "").split(",") if p.strip()]
    if not pedido:
        return base
    elegidos = [m for m in base if any(m == p or m.startswith(p + "_") for p in pedido)]
    if not elegidos:
        raise SystemExit(f"SOLO_MODELOS no coincide con ningún modelo: {pedido}. Opciones: {base}")
    return elegidos


def correr(script: str) -> None:
    """Corre un script de la carpeta y se detiene si falla."""
    print(f"\n>>> {script}")
    r = subprocess.run([sys.executable, str(CARPETA / f"{script}.py"), "--version", VERSION], cwd=CARPETA)
    if r.returncode != 0:
        raise SystemExit(f"Falló {script} (código {r.returncode})")


def main() -> None:
    print(f"Versión del valor: {VERSION}")
    for modelo in seleccion():
        correr(modelo)
    if VERSION == "corregido":
        correr("resumen_modelos")
    print("\nModelos completos.")


if __name__ == "__main__":
    main()
