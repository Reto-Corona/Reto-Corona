"""
m04 - Red bayesiana para la recompra a 90 días

Mismos clientes, objetivo y partición temporal que m08 (se reutiliza m08_recompra.construir_base).
Las variables continuas se discretizan en terciles calculados con la muestra de entrenamiento.

Compara dos estructuras (pgmpy):
    - aprendida: búsqueda hill climbing con puntaje BIC
    - experta: aristas definidas por criterio de negocio (ESTRUCTURA_EXPERTA)
Los parámetros se estiman con un prior BDeu. La probabilidad de recompra de cada cliente se calcula
con la manta de Markov de 'recompra' (todas las demás variables están observadas).

Reporta: gráfica de cada red, AUC frente a m08 y consultas interpretables con eliminación de variables.

Escribe:
    data/modelos/m04_red_bayesiana.parquet     probabilidad de recompra con cada red
    salidas/modelos/m04_red_bayesiana/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m04_red_bayesiana.py
"""
import logging
import warnings

# pgmpy 1.1 avisa que BayesianEstimator y StructureScore se eliminarán en la 1.3. Con la versión
# fijada en requirements.txt (1.1.2) funcionan igual; al actualizar pgmpy hay que migrar a
# pgmpy.parameter_estimator.DiscreteBayesianEstimator.
warnings.filterwarnings("ignore", category=FutureWarning, module="pgmpy")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pgmpy.*")

import networkx as nx  # noqa: E402
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import average_precision_score, roc_auc_score

import comun as c
from m08_recompra import construir_base

NOMBRE = "m04_red_bayesiana"
OBJETIVO = "recompra"
MUESTRA_ESTRUCTURA = 150_000   # clientes de entrenamiento para buscar la estructura
MAX_PADRES = 3
TAMANO_PRIOR = 10              # tamaño de muestra equivalente del prior BDeu
REGIONES = ["BOGOTÁ", "VALLE", "ANTIOQUIA", "CUNDINAMARCA", "ATLÁNTICO", "SANTANDER"]
CANALES = ["cluster_1", "cluster_2", "cluster_4"]
ETIQUETAS = {
    "valor": "Valor visita", "productos": "Productos", "categorias": "Categorías", "negocio": "Negocio",
    "canal": "Canal", "region": "Región", "trimestre": "Trimestre", "generacion": "Generación",
    "genero": "Género", "contactable": "Contactable", "recompra": "RECOMPRA 90d",
}
# Criterio experto: el perfil explica qué compra; región explica el canal; lo que se compra y
# dónde explica el valor; y la recompra depende de qué, cuánto y dónde se compró, de la época y
# de si se le puede contactar.
ESTRUCTURA_EXPERTA = [
    ("generacion", "negocio"), ("genero", "negocio"), ("region", "canal"),
    ("canal", "valor"), ("negocio", "valor"), ("negocio", "categorias"), ("categorias", "productos"),
    ("productos", "valor"),
    ("valor", "recompra"), ("negocio", "recompra"), ("canal", "recompra"), ("region", "recompra"),
    ("trimestre", "recompra"), ("contactable", "recompra"), ("categorias", "recompra"),
]
logging.getLogger("pgmpy").setLevel(logging.ERROR)


def discretizar(df: pd.DataFrame, cortes: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Variables discretas; los terciles de valor y productos salen del entrenamiento."""
    if cortes is None:
        cortes = {"valor": df["valor"].quantile([1 / 3, 2 / 3]).tolist(),
                  "productos": df["productos"].quantile([1 / 3, 2 / 3]).tolist()}
    d = pd.DataFrame(index=df.index)
    d["valor"] = pd.cut(df["valor"], [-np.inf, *cortes["valor"], np.inf], labels=["bajo", "medio", "alto"])
    cp = sorted(set(cortes["productos"]))
    etiquetas_p = ["bajo", "medio", "alto"][:len(cp) + 1]
    d["productos"] = pd.cut(df["productos"], [-np.inf, *cp, np.inf], labels=etiquetas_p)
    d["categorias"] = np.where(df["categorias"] >= 3, "3+", df["categorias"].astype(int).astype(str))
    d["negocio"] = df["negocio_visita"].astype(str)
    d["canal"] = np.where(df["canal"].isin(CANALES), df["canal"], "otros")
    d["region"] = np.where(df["region"].isin(REGIONES), df["region"],
                           np.where(df["region"] == "SIN REGIÓN", "SIN REGIÓN", "Otras"))
    d["trimestre"] = "T" + ((df["fecha_primera_visita"].dt.month - 1) // 3 + 1).astype(str)
    d["generacion"] = df["generacion"].astype(str)
    d["genero"] = df["genero"].astype(str)
    d["contactable"] = df["contactable"].astype(str)
    d[OBJETIVO] = df["recompra_90d"].map({1.0: "1", 0.0: "0"})
    return d.astype(str), cortes


def ajustar(estructura: list, datos: pd.DataFrame, estados: dict) -> DiscreteBayesianNetwork:
    """Red con la estructura dada y parámetros BDeu."""
    modelo = DiscreteBayesianNetwork(estructura)
    modelo.add_nodes_from(datos.columns)
    estimador = BayesianEstimator(modelo, datos, state_names=estados)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpds = estimador.get_parameters(prior_type="BDeu", equivalent_sample_size=TAMANO_PRIOR)
    modelo.add_cpds(*cpds)
    modelo.check_model()
    return modelo


def aprender_estructura(datos: pd.DataFrame) -> list:
    """Hill climbing con BIC sobre una muestra del entrenamiento."""
    muestra = datos.sample(min(MUESTRA_ESTRUCTURA, len(datos)), random_state=c.SEMILLA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag = HillClimbSearch(muestra).estimate(scoring_method="bic-d", max_indegree=MAX_PADRES, show_progress=False)
    return sorted(dag.edges())


def prob_objetivo(modelo: DiscreteBayesianNetwork, datos: pd.DataFrame) -> np.ndarray:
    """P(recompra = 1 | resto) con la manta de Markov (vectorizado sobre las CPD)."""
    hijos = list(modelo.successors(OBJETIVO))
    log_p = {}
    for r in ("0", "1"):
        total = np.zeros(len(datos))
        for var in [OBJETIVO] + hijos:
            cpd = modelo.get_cpds(var)
            idx = []
            for v in cpd.variables:
                estados = cpd.state_names[v]
                valores = np.full(len(datos), r) if v == OBJETIVO else datos[v].to_numpy()
                mapa = {s: i for i, s in enumerate(estados)}
                idx.append(np.array([mapa[x] for x in valores]))
            total += np.log(cpd.values[tuple(idx)])
        log_p[r] = total
    m = np.maximum(log_p["0"], log_p["1"])
    p1, p0 = np.exp(log_p["1"] - m), np.exp(log_p["0"] - m)
    return p1 / (p0 + p1)


def graficar_red(carpeta, modelo: DiscreteBayesianNetwork, nombre: str, titulo: str, subtitulo: str) -> None:
    """DAG en capas (orden topológico), con la recompra resaltada."""
    g = nx.DiGraph(modelo.edges())
    g.add_nodes_from(modelo.nodes())
    capas = list(nx.topological_generations(g))
    pos = {}
    for i, capa in enumerate(capas):
        capa = sorted(capa)
        for j, n in enumerate(capa):
            pos[n] = (j - (len(capa) - 1) / 2, -i)
    fig, ax = c.plt.subplots(figsize=(12, 1.6 * len(capas) + 2))
    manta = set(modelo.get_markov_blanket(OBJETIVO))
    for u, v in g.edges():
        en_obj = OBJETIVO in (u, v)
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>", color=c.NARANJA if en_obj else c.EJE, lw=1.8 if en_obj else 1.1,
                                    shrinkA=22, shrinkB=22, connectionstyle="arc3,rad=0.06"))
    for n, (x, y) in pos.items():
        color = c.NARANJA if n == OBJETIVO else (c.AZUL if n in manta else c.TENUE)
        ax.scatter([x], [y], s=2600, color=color, zorder=3)
        ax.text(x, y, ETIQUETAS.get(n, n).replace(" ", "\n"), ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", zorder=4)
    ax.set_axis_off()
    ax.margins(0.12)
    c.guardar_figura(fig, carpeta, nombre, titulo, subtitulo,
                     nota=c.FUENTE + " Naranja: objetivo; azul: su manta de Markov; gris: resto.")


def consultas(modelo: DiscreteBayesianNetwork) -> pd.DataFrame:
    """Consultas interpretables P(recompra = 1 | evidencia) y una diagnóstica."""
    ve = VariableElimination(modelo)
    lista = [
        ("Base (sin evidencia)", {}),
        ("Millennial, primera compra en Baños, canal Centro Corona C1",
         {"generacion": "Millennials", "negocio": "BAÑOS", "canal": "cluster_2"}),
        ("Primera compra en Revestimientos", {"negocio": "REVESTIMIENTOS E INSTALACIÓN"}),
        ("Primera compra en Baños", {"negocio": "BAÑOS"}),
        ("Primera compra en Cocinas", {"negocio": "COCINAS"}),
        ("Valor de la visita alto (tercil superior)", {"valor": "alto"}),
        ("Valor de la visita bajo (tercil inferior)", {"valor": "bajo"}),
        ("Valor alto en Revestimientos, canal Centro Corona C1",
         {"valor": "alto", "negocio": "REVESTIMIENTOS E INSTALACIÓN", "canal": "cluster_2"}),
        ("Valor bajo en Cocinas, canal Centro Corona C2", {"valor": "bajo", "negocio": "COCINAS", "canal": "cluster_1"}),
        ("3 o más categorías en la visita", {"categorias": "3+"}),
        ("Bogotá, contactable", {"region": "BOGOTÁ", "contactable": "Sí"}),
        ("Región pequeña (Otras)", {"region": "Otras"}),
    ]
    filas = []
    for texto, ev in lista:
        q = ve.query([OBJETIVO], evidence=ev or None, show_progress=False)
        filas.append({"consulta": f"P(recompra | {texto})", "probabilidad": float(q.get_value(**{OBJETIVO: "1"}))})
    # Diagnóstica (al revés): entre quienes recompran, ¿qué tan probable es que la visita fuera de valor alto?
    q = ve.query(["valor"], evidence={OBJETIVO: "1"}, show_progress=False)
    filas.append({"consulta": "P(valor alto | recompra) [diagnóstica]", "probabilidad": float(q.get_value(valor="alto"))})
    return pd.DataFrame(filas)


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    base, _ = construir_base()
    entrena = base[base["muestra"] == "entrenamiento"]
    prueba = base[base["muestra"] == "prueba"]
    d_tr, cortes = discretizar(entrena)
    d_te, _ = discretizar(prueba, cortes)
    estados = {col: sorted(set(d_tr[col]) | set(d_te[col])) for col in d_tr.columns}
    print(f"  Entrenamiento: {len(d_tr):,} | prueba: {len(d_te):,} | cortes valor: {[round(x) for x in cortes['valor']]}")

    aristas = aprender_estructura(d_tr)
    print(f"  Estructura aprendida: {len(aristas)} aristas")
    red_aprendida = ajustar(aristas, d_tr, estados)
    red_experta = ajustar(ESTRUCTURA_EXPERTA, d_tr, estados)

    y = (d_te[OBJETIVO] == "1").astype(int).to_numpy()
    redes = {"Estructura aprendida (hill climbing + BIC)": red_aprendida, "Estructura experta": red_experta}
    probs = {n: prob_objetivo(m, d_te) for n, m in redes.items()}
    m08 = pd.read_csv(c.SALIDAS_MODELOS / "m08_recompra" / "metricas.csv") \
        if (c.SALIDAS_MODELOS / "m08_recompra" / "metricas.csv").exists() else pd.DataFrame()
    filas = [{"modelo": n, "auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
              "aristas": red.number_of_edges(), "padres_recompra": ", ".join(sorted(red.get_parents(OBJETIVO))),
              "hijos_recompra": ", ".join(sorted(red.get_children(OBJETIVO))),
              "manta_markov": len(red.get_markov_blanket(OBJETIVO))}
             for (n, p), red in zip(probs.items(), redes.values())]
    for r in m08.itertuples():
        filas.append({"modelo": f"m08 {r.modelo}", "auc": r.auc, "pr_auc": r.pr_auc})
    metricas = pd.DataFrame(filas)
    metricas.to_csv(carpeta / "metricas.csv", index=False, float_format=c.FORMATO_CSV)
    print(metricas[["modelo", "auc", "pr_auc"]].to_string(index=False))
    pd.DataFrame(aristas, columns=["desde", "hacia"]).to_csv(carpeta / "aristas_aprendidas.csv", index=False)
    pd.DataFrame(ESTRUCTURA_EXPERTA, columns=["desde", "hacia"]).to_csv(carpeta / "aristas_expertas.csv", index=False)

    graficar_red(carpeta, red_aprendida, "01_red_aprendida", "Red bayesiana aprendida (hill climbing + BIC)",
                 f"{len(aristas)} aristas, máximo {MAX_PADRES} padres por nodo; estructura buscada con "
                 f"{c.miles(MUESTRA_ESTRUCTURA)} clientes de entrenamiento.")
    graficar_red(carpeta, red_experta, "02_red_experta", "Red bayesiana con estructura experta",
                 f"{len(ESTRUCTURA_EXPERTA)} aristas definidas por criterio de negocio.")

    mejor_nombre = max(probs, key=lambda n: roc_auc_score(y, probs[n]))
    q = consultas(redes[mejor_nombre]).rename(columns={"probabilidad": "prob_mejor_red"})
    q["prob_otra_red"] = consultas([r for n, r in redes.items() if n != mejor_nombre][0])["probabilidad"]
    q.insert(1, "red", mejor_nombre)
    q.to_csv(carpeta / "consultas.csv", index=False, float_format=c.FORMATO_CSV)
    print(q.to_string(index=False))

    # Probabilidad para todos los clientes observables y no observables (las redes se entrenaron en 'entrenamiento')
    d_todos, _ = discretizar(base.assign(recompra_90d=base["recompra_90d"].fillna(0)), cortes)
    salida = base[["cliente", "segmento", "muestra", "recompra_90d"]].copy()
    salida["prob_red_aprendida"] = prob_objetivo(red_aprendida, d_todos)
    salida["prob_red_experta"] = prob_objetivo(red_experta, d_todos)
    salida.to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(salida):,} clientes)")
    escribir_resumen(carpeta, metricas, q, aristas, red_aprendida, red_experta, cortes, len(d_tr), len(d_te), mejor_nombre)


def escribir_resumen(carpeta, metricas, q, aristas, red_a, red_e, cortes, n_tr, n_te, mejor) -> None:
    """resumen.md del modelo."""
    filas_m = "\n".join(f"| {x.modelo} | {c.num(x.auc, 3)} | {c.num(x.pr_auc, 3)} |" for x in metricas.itertuples())
    filas_q = "\n".join(f"| {x.consulta} | {c.pct(x.prob_mejor_red * 100)} | {c.pct(x.prob_otra_red * 100)} |"
                        for x in q.itertuples())
    lista_a = ", ".join(f"{ETIQUETAS[u]} → {ETIQUETAS[v]}" for u, v in aristas)
    auc = metricas.set_index("modelo")["auc"]
    m08_lgbm = auc.get("m08 LightGBM", np.nan)
    base = q.iloc[0].prob_mejor_red
    texto = f"""
# m04 - Red bayesiana para la recompra a 90 días

## Qué hace
Modela la recompra a 90 días como una red bayesiana discreta (`pgmpy`), con los mismos clientes,
objetivo y partición temporal de m08. Compara una estructura **aprendida** (hill climbing con
puntaje BIC, máximo {MAX_PADRES} padres por nodo) con una estructura **experta** definida por criterio de
negocio. Los parámetros se estiman con un prior BDeu (tamaño equivalente {TAMANO_PRIOR}). La red permite
hacer consultas del tipo P(recompra | evidencia), incluidas consultas "al revés" (diagnósticas).

## Datos usados
- Base de m08 (`m08_recompra.construir_base`): {c.miles(n_tr)} clientes de entrenamiento y {c.miles(n_te)} de prueba.
- Discretización (cortes calculados en el entrenamiento): valor de la visita en terciles (cortes
  {c.miles(cortes['valor'][0])} y {c.miles(cortes['valor'][1])} en unidades de la base), productos distintos en
  terciles, categorías (1, 2, 3+), negocio, canal (C2 = cluster_1, C1 = cluster_2, Tienda Cerámica =
  cluster_4, otros), región (6 principales, Otras y SIN REGIÓN), trimestre de la primera visita,
  generación, género y contactabilidad.

## Métricas (muestra de prueba)
| Modelo | AUC | PR-AUC |
|---|---|---|
{filas_m}

Estructura aprendida ({len(aristas)} aristas): {lista_a}.
Padres de la recompra: aprendida = {', '.join(ETIQUETAS[p] for p in red_a.get_parents(OBJETIVO)) or 'ninguno'};
experta = {', '.join(ETIQUETAS[p] for p in red_e.get_parents(OBJETIVO))}.

## Consultas (red con mejor AUC: {mejor})
| Consulta | Mejor red | Otra red |
|---|---|---|
{filas_q}

## Hallazgos principales
- La mejor red alcanza AUC {c.num(auc[mejor], 3)}, frente a {c.num(m08_lgbm, 3)} de LightGBM en m08: pierde algo de
  capacidad predictiva por discretizar y limitar las dependencias, pero gana interpretabilidad.
- Las consultas confirman lo de m08 con probabilidades directas: el negocio de la primera compra y el
  valor de la visita son lo que más mueve la recompra (base {c.pct(base * 100)}).
- La red aprendida es más parsimoniosa: la recompra solo depende directamente de
  {", ".join(ETIQUETAS[x] for x in sorted(red_a.get_parents(OBJETIVO)))} ({len(red_a.get_parents(OBJETIVO))} padres, frente a
  {len(red_e.get_parents(OBJETIVO))} en la experta), y aun así predice {"mejor" if auc.iloc[0] > auc.iloc[1] else "peor"}
  (AUC {c.num(auc.iloc[0], 3)} frente a {c.num(auc.iloc[1], 3)}). Canal, región, contactabilidad y época no aportan
  información sobre la recompra una vez se sabe qué y cuánto se compró; el BIC las deja fuera de la
  manta de Markov.
- Las variables del perfil (generación, género) quedan conectadas con el negocio y la región, no con
  la recompra: explican qué compra el cliente, no si vuelve.

## Limitaciones
- Discretizar pierde información (por eso el AUC es menor que en m08).
- Las aristas de la red aprendida son dependencias estadísticas, no causales; el BIC elige la
  estructura más simple que explica los datos, no la "verdadera".
- La estructura se busca con una muestra de {c.miles(MUESTRA_ESTRUCTURA)} clientes por tiempo de cómputo.
- Mismas limitaciones de datos que m08 (fechas desplazadas, clientes previos sin marcar).
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    # Hill climbing desempata estructuras equivalentes según el orden de iteración de conjuntos,
    # que en Python depende de PYTHONHASHSEED. Se fija a la semilla del proyecto (relanzando el
    # script si hace falta) para que la red aprendida sea siempre la misma.
    import os
    import subprocess
    import sys
    if os.environ.get("PYTHONHASHSEED") != str(c.SEMILLA):
        entorno = {**os.environ, "PYTHONHASHSEED": str(c.SEMILLA)}
        sys.exit(subprocess.run([sys.executable, *sys.argv], env=entorno).returncode)
    main()
