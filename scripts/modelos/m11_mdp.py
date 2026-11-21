"""
m11 - Proceso de decisión de Markov (MDP) del ciclo de vida del cliente

Paso 1. Cadena de Markov con datos reales.
    Cada consumidor final tiene un estado al cierre de cada mes, desde el mes de su primera
    compra hasta el último mes de la base:
        Nuevo       1 visita acumulada y menos de 3 meses sin comprar
        Recurrente  2-3 visitas acumuladas y menos de 3 meses sin comprar
        Leal        4+ visitas acumuladas y menos de 3 meses sin comprar
        Dormido     de 3 a 6 meses sin comprar
        Perdido     más de 6 meses sin comprar
    Se estima la matriz de transición mensual (total y por grupo de segmento: Recurrente,
    Proyecto grande, Un solo negocio). Recompensa de cada estado = gasto esperado del mes
    siguiente para un cliente que está en ese estado.

Paso 2. MDP.
    Acciones: no hacer nada, contacto de venta cruzada, descuento. El efecto de las acciones
    NO está en los datos: son supuestos explícitos (diccionario EFECTOS más abajo) que el
    piloto (m14) debe medir. Las acciones solo aplican a clientes contactables.
    Se resuelve con iteración de valor (descuento mensual 0,95) y se hace un análisis de
    sensibilidad con el efecto de las acciones a la mitad y al doble.

Escribe:
    data/modelos/m11_mdp.parquet               estado actual, acción recomendada y valor por cliente
    salidas/modelos/m11_mdp/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m11_mdp.py
"""
from functools import lru_cache

import numpy as np
import pandas as pd

import comun as c

NOMBRE = "m11_mdp"
ESTADOS = ["Nuevo", "Recurrente", "Leal", "Dormido", "Perdido"]
ACCIONES = ["No hacer nada", "Venta cruzada", "Descuento"]
GRUPOS = ["Total", "Recurrente", "Proyecto grande", "Un solo negocio"]
GAMMA = 0.95          # factor de descuento mensual
TOLERANCIA = 1e-6     # parada de la iteración de valor
MIN_OBSERVACIONES = 1_000  # con menos transiciones, un estado del grupo usa los parámetros del total
FACTORES_SENSIBILIDAD = {"Mitad del efecto": 0.5, "Efecto base": 1.0, "Doble del efecto": 2.0}
# Sensibilidad al costo de contacto (proporción del ticket mediano por visita), cruzada con la del efecto
COSTOS_SENSIBILIDAD = {"1% del ticket": 0.01, "2% del ticket": 0.02, "5% del ticket": 0.05}

# ---------------------------------------------------------------------------
# SUPUESTOS DE LAS ACCIONES (no salen de los datos; los debe medir el piloto)
# Valores conservadores:
# - aumento_prob_compra: puntos (en proporción) que se suman a la probabilidad de comprar
#   en el mes siguiente. Las campañas de CRM en retail suelen lograr entre 1 y 5 pp; se usan
#   valores en la parte baja y menores para clientes dormidos o perdidos, que responden menos.
# - costo_contacto_pct_ticket: costo por cliente contactado (mensaje + tiempo del asesor) como
#   proporción del ticket mediano por visita de los consumidores finales. Así el costo queda en
#   las mismas unidades (escaladas) de la base que las ventas. 2% del ticket es conservador para un
#   contacto de CRM (mensaje o llamada corta). Para cambiarlo, editar este valor.
# - perdida_margen: fracción del gasto del mes que se regala con el descuento. Se aplica a
#   TODA la compra del mes, incluida la que el cliente habría hecho sin descuento
#   (canibalización), lo que hace al descuento caro a propósito.
# - MARGEN: la recompensa es el gasto (ventas), como pide el diseño. Si se quiere decidir
#   sobre margen de contribución, cambiar MARGEN (p. ej. 0,30); los costos no se escalan.
# ---------------------------------------------------------------------------
MARGEN = 1.0
EFECTOS = {
    "No hacer nada": {
        "aumento_prob_compra": {e: 0.0 for e in ESTADOS},
        "costo_contacto_pct_ticket": 0.0,
        "perdida_margen": 0.0,
    },
    "Venta cruzada": {
        "aumento_prob_compra": {"Nuevo": 0.015, "Recurrente": 0.015, "Leal": 0.010,
                                "Dormido": 0.010, "Perdido": 0.005},
        "costo_contacto_pct_ticket": 0.02,
        "perdida_margen": 0.0,
    },
    "Descuento": {
        "aumento_prob_compra": {"Nuevo": 0.030, "Recurrente": 0.030, "Leal": 0.020,
                                "Dormido": 0.020, "Perdido": 0.010},
        "costo_contacto_pct_ticket": 0.02,
        "perdida_margen": 0.10,
    },
}


@lru_cache(maxsize=1)
def ticket_mediano_visita() -> float:
    """Ticket mediano por visita (cliente + fecha + punto de venta) de los consumidores finales, en unidades de la base."""
    con = c.conexion()
    return float(con.execute("""
        SELECT MEDIAN(v) FROM (
          SELECT t.cliente, t.fecha, t.punto_de_venta, SUM(t.valor_corregido) AS v
          FROM trx t JOIN features f USING (cliente)
          WHERE f.tipo_cliente = 'Consumidor final'
          GROUP BY ALL)
    """).fetchone()[0])


def costo_contacto(accion: str, pct_ticket: float | None = None) -> float:
    """Costo por contacto de una acción, en unidades de la base (proporción x ticket mediano por visita).

    pct_ticket reemplaza el supuesto de EFECTOS (para la sensibilidad); "No hacer nada" siempre cuesta 0.
    """
    pct = EFECTOS[accion]["costo_contacto_pct_ticket"]
    if pct_ticket is not None and pct > 0:
        pct = pct_ticket
    return pct * ticket_mediano_visita()


def aumento(accion: str, estado, factor: float = 1.0):
    """Aumento supuesto (en proporción, p. ej. 0,015 = 1,5 pp) de la probabilidad de comprar, por estado."""
    tabla = EFECTOS[accion]["aumento_prob_compra"]
    if isinstance(estado, str):
        return tabla[estado] * factor
    return np.array([tabla[e] for e in estado]) * factor


def ganancia_inmediata(accion: str, estado, valor, prob_base=0.0, factor: float = 1.0,
                       pct_ticket: float | None = None):
    """Valor esperado de aplicar la acción frente a no hacer nada, en el periodo del contacto.

        aumento en puntos x valor de la compra x (1 - pérdida de margen)
        - pérdida de margen x probabilidad base x valor   (canibalización del descuento)
        - costo por contacto

    Es la definición única del efecto de una acción: el MDP la usa para construir sus recompensas
    y m13 la importa para valorar cada contacto, así que los dos modelos no pueden divergir.
    """
    ef = EFECTOS[accion]
    d = aumento(accion, estado, factor)
    pm = ef["perdida_margen"]
    return MARGEN * (d * valor * (1 - pm) - pm * prob_base * valor) - costo_contacto(accion, pct_ticket)


def construir_panel() -> pd.DataFrame:
    """Transiciones mensuales agregadas: grupo, contactable, estado, compra y estado siguiente."""
    con = c.conexion()
    grupos = ", ".join(f"({k}, '{v}')" for k, v in c.GRUPOS_SEGMENTO.items())
    con.execute(f"""
    CREATE TEMP TABLE panel AS
    WITH visitas AS (
      SELECT cliente, fecha::DATE AS fecha, punto_de_venta, SUM(valor_corregido) AS valor
      FROM trx GROUP BY ALL
    ),
    vm AS (   -- visitas y gasto por cliente y mes (mes como índice entero)
      SELECT cliente, year(fecha) * 12 + month(fecha) - 1 AS m,
             COUNT(*) AS visitas, SUM(valor) AS gasto
      FROM visitas GROUP BY ALL
    ),
    cli AS (
      SELECT f.cliente, g.grupo, COALESCE(f.es_contactable, FALSE) AS contactable,
             MIN(vm.m) AS m0
      FROM features f
      JOIN segmentos s USING (cliente)
      JOIN (VALUES {grupos}) g(segmento, grupo) ON g.segmento = s.segmento_kmeans
      JOIN vm USING (cliente)
      WHERE f.tipo_cliente = 'Consumidor final' AND f.posible_cliente_previo = 0
      GROUP BY ALL
    ),
    meses AS (
      SELECT cliente, grupo, contactable,
             UNNEST(range(m0, (SELECT MAX(m) FROM vm) + 1)) AS m
      FROM cli
    ),
    p AS (
      SELECT ms.*, COALESCE(vm.visitas, 0) AS visitas, COALESCE(vm.gasto, 0) AS gasto
      FROM meses ms LEFT JOIN vm USING (cliente, m)
    ),
    acum AS (
      SELECT *,
             SUM(visitas) OVER w AS visitas_acum,
             m - MAX(CASE WHEN visitas > 0 THEN m END) OVER w AS meses_sin_compra
      FROM p
      WINDOW w AS (PARTITION BY cliente ORDER BY m ROWS UNBOUNDED PRECEDING)
    ),
    est AS (
      SELECT *,
             CASE WHEN meses_sin_compra > 6 THEN 'Perdido'
                  WHEN meses_sin_compra >= 3 THEN 'Dormido'
                  WHEN visitas_acum <= 1 THEN 'Nuevo'
                  WHEN visitas_acum <= 3 THEN 'Recurrente'
                  ELSE 'Leal' END AS estado
      FROM acum
    )
    SELECT cliente, grupo, contactable, m, estado, gasto, visitas,
           LEAD(estado) OVER (PARTITION BY cliente ORDER BY m) AS estado_sig,
           LEAD(gasto) OVER (PARTITION BY cliente ORDER BY m) AS gasto_sig,
           LEAD(visitas) OVER (PARTITION BY cliente ORDER BY m) > 0 AS compra_sig
    FROM est
    """)
    trans = con.execute("""
        SELECT grupo, contactable, estado, compra_sig, estado_sig,
               COUNT(*) AS n, SUM(gasto_sig) AS gasto_sig
        FROM panel WHERE estado_sig IS NOT NULL GROUP BY ALL
        ORDER BY ALL
    """).df()
    # Estado al cierre del último mes de la base (para recomendar una acción por cliente)
    actual = con.execute("""
        SELECT cliente, grupo, contactable, estado AS estado_actual
        FROM panel QUALIFY ROW_NUMBER() OVER (PARTITION BY cliente ORDER BY m DESC) = 1
    """).df()
    meses = con.execute("SELECT COUNT(DISTINCT m) FROM panel").fetchone()[0]
    print(f"  Transiciones cliente-mes: {trans['n'].sum():,} | clientes: {len(actual):,} | meses: {meses}")
    return trans, actual


def estimar_cadena(trans: pd.DataFrame) -> dict:
    """Por grupo: P(s'|s), probabilidad de compra, P(s'|s, compra), P(s'|s, no compra) y gasto."""
    cadenas = {}
    for grupo in GRUPOS:
        t = trans if grupo == "Total" else trans[trans["grupo"] == grupo]
        tabla = t.groupby(["estado", "compra_sig", "estado_sig"])[["n", "gasto_sig"]].sum()

        def matriz(filtro=None) -> np.ndarray:
            sub = tabla if filtro is None else tabla.xs(filtro, level="compra_sig")
            conteo = sub["n"].groupby(level=["estado", "estado_sig"]).sum().unstack(fill_value=0)
            conteo = conteo.reindex(index=ESTADOS, columns=ESTADOS, fill_value=0)
            filas = conteo.sum(axis=1).replace(0, np.nan)
            m = conteo.div(filas, axis=0).fillna(0).to_numpy(copy=True)
            for i in np.where(m.sum(axis=1) == 0)[0]:  # estado sin datos: se queda donde está
                m[i, i] = 1.0
            return m

        por_estado = t.groupby("estado").agg(n=("n", "sum"), gasto=("gasto_sig", "sum")).reindex(ESTADOS)
        compran = t[t["compra_sig"]].groupby("estado").agg(n=("n", "sum"), gasto=("gasto_sig", "sum")) \
            .reindex(ESTADOS).fillna(0)
        cadenas[grupo] = {
            "P": matriz(),
            "P_compra": matriz(True),
            "P_no_compra": matriz(False),
            "prob_compra": (compran["n"] / por_estado["n"]).fillna(0).to_numpy(copy=True),
            "gasto_si_compra": (compran["gasto"] / compran["n"].replace(0, np.nan)).fillna(0).to_numpy(copy=True),
            "recompensa": (por_estado["gasto"] / por_estado["n"]).fillna(0).to_numpy(copy=True),
            "observaciones": por_estado["n"].fillna(0).to_numpy(copy=True),
            "usa_total": np.zeros(len(ESTADOS), dtype=bool),
        }
        # Un estado con muy pocas observaciones en el grupo toma los parámetros del total
        if grupo != "Total":
            pocos = cadenas[grupo]["observaciones"] < MIN_OBSERVACIONES
            for clave in ("P", "P_compra", "P_no_compra", "prob_compra", "gasto_si_compra", "recompensa"):
                cadenas[grupo][clave][pocos] = cadenas["Total"][clave][pocos]
            cadenas[grupo]["usa_total"] = pocos
            for e in np.array(ESTADOS)[pocos]:
                print(f"  Aviso: {grupo}/{e} tiene menos de {MIN_OBSERVACIONES:,} observaciones; usa el total.")
    return cadenas


def modelo_acciones(cad: dict, factor: float = 1.0, pct_ticket: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Matrices P[a, s, s'] y recompensas R[a, s] bajo los supuestos de EFECTOS.

    R(no hacer nada) = gasto esperado; R(acción) = R(no hacer nada) + ganancia_inmediata(acción).
    """
    n = len(ESTADOS)
    P = np.zeros((len(ACCIONES), n, n))
    R = np.zeros((len(ACCIONES), n))
    p0 = cad["prob_compra"]
    r_nada = MARGEN * p0 * cad["gasto_si_compra"]
    for a, accion in enumerate(ACCIONES):
        p = np.clip(p0 + aumento(accion, ESTADOS, factor), 0, 1)
        # La acción solo cambia la probabilidad de comprar; dado que compra (o no), el estado
        # siguiente se mueve como en los datos.
        P[a] = p[:, None] * cad["P_compra"] + (1 - p)[:, None] * cad["P_no_compra"]
        R[a] = r_nada + ganancia_inmediata(accion, ESTADOS, cad["gasto_si_compra"], p0, factor, pct_ticket)
    return P, R


def iteracion_de_valor(P: np.ndarray, R: np.ndarray, acciones_validas: list[int]) -> tuple:
    """V(s) = max_a R(a,s) + γ Σ P(a,s,s') V(s'). Devuelve V, política e iteraciones."""
    V = np.zeros(P.shape[1])
    for it in range(1, 10_000):
        Q = R[acciones_validas] + GAMMA * P[acciones_validas] @ V
        V_nuevo = Q.max(axis=0)
        if np.max(np.abs(V_nuevo - V)) < TOLERANCIA:
            V = V_nuevo
            break
        V = V_nuevo
    Q = R[acciones_validas] + GAMMA * P[acciones_validas] @ V
    politica = np.array(acciones_validas)[Q.argmax(axis=0)]
    return V, politica, Q, it


def resolver(cadenas: dict, factor: float, pct_ticket: float | None = None) -> pd.DataFrame:
    """Política óptima (contactables) y valor de no actuar (no contactables) por grupo y estado."""
    filas = []
    for grupo, cad in cadenas.items():
        P, R = modelo_acciones(cad, factor, pct_ticket)
        V_c, pol, Q, it = iteracion_de_valor(P, R, list(range(len(ACCIONES))))
        V_nc, _, _, _ = iteracion_de_valor(P, R, [0])
        for i, estado in enumerate(ESTADOS):
            filas.append({
                "grupo": grupo, "estado": estado, "factor_efecto": factor,
                "accion_optima": ACCIONES[pol[i]],
                "valor_contactable": V_c[i], "valor_no_contactable": V_nc[i],
                "ganancia_por_actuar": V_c[i] - V_nc[i],
                **{f"Q_{a}": Q[k, i] for k, a in enumerate(ACCIONES)},
                "iteraciones": it,
            })
    return pd.DataFrame(filas)


def graficar_matrices(carpeta, cadenas: dict) -> None:
    """Mapa de calor de la matriz de transición mensual por grupo."""
    fig, ejes = c.plt.subplots(2, 2, figsize=(12, 10))
    cmap = c.g.LinearSegmentedColormap.from_list("azul", [c.SUPERFICIE, c.AZUL])
    for ax, grupo in zip(ejes.flat, GRUPOS):
        m = cadenas[grupo]["P"] * 100
        ax.imshow(m, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(ESTADOS)):
            for j in range(len(ESTADOS)):
                ax.text(j, i, c.num(m[i, j], 1), ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > 55 else c.TINTA)
        ax.set_xticks(range(len(ESTADOS)), ESTADOS, fontsize=9)
        ax.set_yticks(range(len(ESTADOS)), ESTADOS, fontsize=9)
        ax.set_title(grupo, fontsize=11, loc="left")
        ax.set_xlabel("Estado del mes siguiente")
        ax.grid(False)
    c.guardar_figura(fig, carpeta, "01_matrices_transicion", "Matriz de transición mensual por grupo",
                     "Porcentaje de clientes de cada estado (fila) que pasa a cada estado (columna) al mes siguiente.")


def graficar_recompensas(carpeta, cadenas: dict) -> None:
    """Gasto mensual esperado por estado y grupo."""
    fig, ax = c.plt.subplots(figsize=(11, 5))
    x = np.arange(len(ESTADOS))
    ancho = 0.2
    for k, grupo in enumerate(GRUPOS):
        v = cadenas[grupo]["recompensa"] / 1000
        barras = ax.bar(x + (k - 1.5) * ancho, v, ancho * 0.92, color=c.SERIES[k], label=grupo)
        c.rotular_barras(ax, barras, [c.miles(b) for b in v], size=7.5)
    ax.set_xticks(x, ESTADOS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Miles por cliente y mes (unidades de la base)")
    ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.08))
    c.guardar_figura(fig, carpeta, "02_recompensa_por_estado", "Recompensa: gasto esperado del mes siguiente",
                     "Promedio por cliente (incluye los meses sin compra), en miles de unidades de la base.")


def graficar_politicas(carpeta, sens: pd.DataFrame) -> None:
    """Política óptima para contactables por grupo, estado y escenario de efecto."""
    colores = {"No hacer nada": c.GRIS_MEDIO, "Venta cruzada": c.AZUL, "Descuento": c.NARANJA}
    escenarios = list(FACTORES_SENSIBILIDAD.items())
    fig, ejes = c.plt.subplots(1, len(escenarios), figsize=(13, 4.6), sharey=True)
    for ax, (nombre, f) in zip(ejes, escenarios):
        s = sens[sens["factor_efecto"] == f]
        for i, grupo in enumerate(GRUPOS):
            for j, estado in enumerate(ESTADOS):
                accion = s[(s["grupo"] == grupo) & (s["estado"] == estado)]["accion_optima"].iloc[0]
                ax.add_patch(c.plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, color=colores[accion]))
                ax.text(j, i, accion.replace(" ", "\n"), ha="center", va="center", fontsize=7.5,
                        color="white" if accion != "No hacer nada" else c.TINTA_2)
        ax.set_xlim(-0.5, len(ESTADOS) - 0.5)
        ax.set_ylim(len(GRUPOS) - 0.5, -0.5)
        ax.set_xticks(range(len(ESTADOS)), ESTADOS, fontsize=8.5)
        ax.set_yticks(range(len(GRUPOS)), GRUPOS)
        ax.set_title(nombre, fontsize=11, loc="left")
        ax.grid(False)
    c.guardar_figura(fig, carpeta, "03_politica_optima", "Política óptima para clientes contactables",
                     "Acción que maximiza el valor esperado (γ = 0,95) con el efecto supuesto, su mitad y su doble.",
                     nota=c.FUENTE + " Los efectos de las acciones son supuestos, no datos.")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")

    trans, actual = construir_panel()
    trans.to_csv(carpeta / "transiciones_conteo.csv", index=False, float_format=c.FORMATO_CSV)
    cadenas = estimar_cadena(trans)

    # Paso 1: matrices y recompensas
    filas_m, filas_r = [], []
    for grupo, cad in cadenas.items():
        for i, e in enumerate(ESTADOS):
            filas_r.append({"grupo": grupo, "estado": e, "observaciones": int(cad["observaciones"][i]),
                            "prob_compra_mes_siguiente": cad["prob_compra"][i],
                            "gasto_si_compra": cad["gasto_si_compra"][i],
                            "recompensa_gasto_esperado": cad["recompensa"][i],
                            "usa_parametros_total": bool(cad["usa_total"][i])})
            for j, e2 in enumerate(ESTADOS):
                filas_m.append({"grupo": grupo, "estado": e, "estado_siguiente": e2, "probabilidad": cad["P"][i, j]})
    matrices = pd.DataFrame(filas_m)
    recompensas = pd.DataFrame(filas_r)
    matrices.to_csv(carpeta / "matrices_transicion.csv", index=False, float_format=c.FORMATO_CSV)
    recompensas.to_csv(carpeta / "recompensas_por_estado.csv", index=False, float_format=c.FORMATO_CSV)
    print(recompensas.to_string(index=False))
    graficar_matrices(carpeta, cadenas)
    graficar_recompensas(carpeta, cadenas)

    # Paso 2: MDP con el efecto base y sensibilidad
    sens = pd.concat([resolver(cadenas, f) for f in FACTORES_SENSIBILIDAD.values()], ignore_index=True)
    sens.to_csv(carpeta / "politica_sensibilidad.csv", index=False, float_format=c.FORMATO_CSV)
    base = sens[sens["factor_efecto"] == 1.0]
    base.to_csv(carpeta / "politica_optima.csv", index=False, float_format=c.FORMATO_CSV)
    print(base[["grupo", "estado", "accion_optima", "valor_contactable", "valor_no_contactable"]].to_string(index=False))
    graficar_politicas(carpeta, sens)
    pd.DataFrame([{"accion": a, "estado": e, "aumento_prob_compra": EFECTOS[a]["aumento_prob_compra"][e],
                   "costo_contacto_pct_ticket": EFECTOS[a]["costo_contacto_pct_ticket"],
                   "costo_contacto_unidades_base": costo_contacto(a), "perdida_margen": EFECTOS[a]["perdida_margen"]}
                  for a in ACCIONES for e in ESTADOS]).to_csv(carpeta / "supuestos_acciones.csv", index=False, float_format=c.FORMATO_CSV)

    # Acción recomendada por cliente (estado al cierre de la base)
    pol = base.set_index(["grupo", "estado"])
    claves = list(zip(actual["grupo"], actual["estado_actual"]))
    actual["accion_recomendada"] = np.where(
        actual["contactable"], [pol.loc[k, "accion_optima"] for k in claves], "No hacer nada (no contactable)")
    actual["valor_esperado"] = np.where(
        actual["contactable"], [pol.loc[k, "valor_contactable"] for k in claves],
        [pol.loc[k, "valor_no_contactable"] for k in claves])
    actual.to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(actual):,} clientes)")
    distrib = actual.groupby(["grupo", "estado_actual", "accion_recomendada"]).size().rename("clientes").reset_index()
    distrib.to_csv(carpeta / "clientes_por_estado_y_accion.csv", index=False, float_format=c.FORMATO_CSV)

    # Sensibilidad cruzada: costo de contacto x efecto de las acciones
    contact = actual[actual["contactable"]].groupby(["grupo", "estado_actual"]).size().rename("clientes")
    filas_s = []
    for nombre_c, pct in COSTOS_SENSIBILIDAD.items():
        for nombre_e, f in FACTORES_SENSIBILIDAD.items():
            r = resolver(cadenas, f, pct).set_index(["grupo", "estado"])
            accion = r["accion_optima"].reindex(contact.index)
            filas_s.append({"costo": nombre_c, "pct_ticket": pct, "costo_unidades_base": pct * ticket_mediano_visita(),
                            "efecto": nombre_e, "factor_efecto": f,
                            "pct_contactables_con_accion": contact[accion.values != "No hacer nada"].sum() / contact.sum() * 100,
                            "accion_segmento2_nuevo": r.loc[("Proyecto grande", "Nuevo"), "accion_optima"]})
    sens_costo = pd.DataFrame(filas_s)
    sens_costo.to_csv(carpeta / "sensibilidad_costo_efecto.csv", index=False, float_format=c.FORMATO_CSV)
    print(sens_costo[["costo", "efecto", "pct_contactables_con_accion", "accion_segmento2_nuevo"]].to_string(index=False))

    escribir_resumen(carpeta, cadenas, recompensas, sens, actual, sens_costo)


def escribir_resumen(carpeta, cadenas, recompensas, sens, actual, sens_costo) -> None:
    """resumen.md del modelo."""
    P = cadenas["Total"]["P"]
    idx = {e: i for i, e in enumerate(ESTADOS)}
    r = recompensas.set_index(["grupo", "estado"])
    filas_r = "\n".join(
        f"| {e} | " + " | ".join(f"{c.miles(r.loc[(g, e), 'recompensa_gasto_esperado'])}" for g in GRUPOS)
        + f" | {c.pct(r.loc[('Total', e), 'prob_compra_mes_siguiente'] * 100)} |"
        for e in ESTADOS)
    filas_p = []
    for g in GRUPOS:
        for nombre, f in FACTORES_SENSIBILIDAD.items():
            pol = sens[(sens["grupo"] == g) & (sens["factor_efecto"] == f)].set_index("estado")["accion_optima"]
            filas_p.append(f"| {g} | {nombre} | " + " | ".join(pol[e] for e in ESTADOS) + " |")
    filas_p = "\n".join(filas_p)
    sup = "\n".join(
        f"| {a} | " + " / ".join(c.num(EFECTOS[a]['aumento_prob_compra'][e] * 100, 1) for e in ESTADOS)
        + f" | {c.pct(EFECTOS[a]['costo_contacto_pct_ticket'] * 100, 0)} ({c.miles(costo_contacto(a))}) | "
        f"{c.pct(EFECTOS[a]['perdida_margen'] * 100, 0)} |"
        for a in ACCIONES)
    # Cambios de política entre escenarios
    piv = sens.pivot_table(index=["grupo", "estado"], columns="factor_efecto", values="accion_optima", aggfunc="first")
    cambios_mitad = int((piv[0.5] != piv[1.0]).sum())
    cambios_doble = int((piv[2.0] != piv[1.0]).sum())
    base = sens[sens["factor_efecto"] == 1.0]
    n_actuar = int((base["accion_optima"] != "No hacer nada").sum())
    contact = actual[actual["contactable"]]
    pct_accion = (contact["accion_recomendada"] != "No hacer nada").mean() * 100

    # Hallazgos calculados a partir de la política base
    rb = base.merge(recompensas, on=["grupo", "estado"])
    lineas = []
    for accion in ACCIONES[1:]:
        sub = rb[rb["accion_optima"] == accion]
        if len(sub):
            lista = ", ".join(f"{x.grupo}/{x.estado}" for x in sub.itertuples())
            pmin, pmax = sub["prob_compra_mes_siguiente"].min() * 100, sub["prob_compra_mes_siguiente"].max() * 100
            rango = c.pct(pmin) if len(sub) == 1 else f"entre {c.pct(pmin)} y {c.pct(pmax)}"
            lineas.append(
                f"- **{accion}** es óptima en: {lista} (probabilidad base de comprar el mes siguiente: {rango}).")
        else:
            lineas.append(f"- **{accion}** no es óptima en ningún estado con el efecto base.")
    alta = rb[(rb["accion_optima"] != "Descuento") & (rb["prob_compra_mes_siguiente"] >= 0.15)]
    if len(alta):
        lineas.append(
            "- Donde la probabilidad base de compra es alta ("
            + ", ".join(f"{x.grupo}/{x.estado}" for x in alta.itertuples())
            + ") el descuento no conviene: regala margen sobre compras que se harían igual "
              "(canibalización). El descuento solo gana donde casi nadie compraría sin él.")
    uno = rb[rb["grupo"] == "Un solo negocio"]
    if (uno["accion_optima"] == "No hacer nada").all():
        lineas.append(
            f"- En \"Un solo negocio\" ninguna acción se paga: el gasto cuando compran va de "
            f"{c.miles(uno['gasto_si_compra'].min())} a {c.miles(uno['gasto_si_compra'].max())} (unidades de la base), "
            "muy bajo para cubrir el contacto con aumentos de 1 a 3 pp.")
    pg = sens[(sens["grupo"] == "Proyecto grande") & (sens["estado"] == "Nuevo")] \
        .set_index("factor_efecto")["accion_optima"]
    lineas.append(
        "- Segmento del piloto (Proyecto grande, estado Nuevo): "
        + "; ".join(f"{n.lower()} → {pg[f]}" for n, f in FACTORES_SENSIBILIDAD.items())
        + ". Ver la sección anterior.")
    hallazgos = "\n".join(lineas)

    # Lectura para el segmento 2 (Proyecto grande), estado Nuevo: qué tan cerca están las acciones
    q = base[(base["grupo"] == "Proyecto grande") & (base["estado"] == "Nuevo")].iloc[0]
    qs = {a: q[f"Q_{a}"] for a in ACCIONES}
    orden = sorted(qs, key=qs.get, reverse=True)
    brecha = (qs[orden[0]] - qs[orden[1]]) / abs(qs[orden[1]]) * 100
    cambia = pg.nunique() > 1
    filas_q = "\n".join(f"| {a} | {c.miles(qs[a])} | {c.num((qs[a] / qs['No hacer nada'] - 1) * 100, 1)}% |"
                        for a in ACCIONES)
    escenarios = "; ".join(f"{n.lower()} → {pg[f]}" for n, f in FACTORES_SENSIBILIDAD.items())
    if cambia or brecha < 5:
        conclusion = (
            "**Es una decisión de borde:** con los supuestos actuales no se puede afirmar si contactar a este "
            "segmento crea o destruye valor, porque el resultado depende de un efecto que hoy no está medido. "
            "Por eso el piloto (m14) es necesario antes de escalar: debe medir el aumento real en la recompra "
            "y el costo por contacto; con esos valores se reemplazan los supuestos de `EFECTOS` y se vuelve a "
            "resolver el MDP.")
    else:
        conclusion = ("La decisión es clara con los supuestos actuales, pero el piloto (m14) sigue siendo "
                      "necesario para confirmar el efecto antes de escalar.")
    lectura_seg2 = (
        "## Lectura para el segmento 2 (Proyecto grande en una visita)\n"
        "Valor esperado (Q) de cada acción para un cliente contactable en estado Nuevo, con el efecto base "
        "(unidades de la base):\n\n"
        "| Acción | Valor esperado | Diferencia frente a no hacer nada |\n|---|---|---|\n"
        f"{filas_q}\n\n"
        f"La mejor acción ({orden[0]}) supera a la segunda ({orden[1]}) por solo {c.num(brecha, 1)}%"
        + (f", y la política cambia entre escenarios de efecto ({escenarios})" if cambia else "")
        + f". {conclusion}\n")
    usa_total = recompensas[recompensas["usa_parametros_total"]]
    aviso_total = ("" if usa_total.empty else
                   "- Estados con menos de " + c.miles(MIN_OBSERVACIONES) + " transiciones en su grupo usan los "
                   "parámetros del total: " + ", ".join(f"{x.grupo}/{x.estado}" for x in usa_total.itertuples()) + ".\n")
    # Tabla 3 x 3: costo de contacto x efecto
    efectos = list(FACTORES_SENSIBILIDAD)
    filas_sc = []
    for nombre_c in COSTOS_SENSIBILIDAD:
        celdas = []
        for nombre_e in efectos:
            x = sens_costo[(sens_costo["costo"] == nombre_c) & (sens_costo["efecto"] == nombre_e)].iloc[0]
            celdas.append(f"{c.pct(x.pct_contactables_con_accion)} · {x.accion_segmento2_nuevo}")
        costo_u = sens_costo.loc[sens_costo["costo"] == nombre_c, "costo_unidades_base"].iloc[0]
        filas_sc.append(f"| {nombre_c} ({c.miles(costo_u)}) | " + " | ".join(celdas) + " |")
    s2 = sens_costo["accion_segmento2_nuevo"]
    conteo_s2 = ", ".join(f"{a}: {n}" for a, n in s2.value_counts().items())
    tabla_costo = "\n".join([
        "## Sensibilidad al costo de contacto y al efecto",
        "Cada celda: porcentaje de clientes contactables con alguna acción recomendada · acción óptima en el "
        "segmento 2 (Proyecto grande), estado Nuevo. Las filas cambian el costo por contacto (proporción del "
        "ticket mediano por visita, entre paréntesis en unidades de la base); las columnas, el efecto supuesto "
        "de las acciones.",
        "",
        "| Costo por contacto | " + " | ".join(efectos) + " |",
        "|---|" + "---|" * len(efectos),
        *filas_sc,
        "",
        f"La acción en el segmento 2, estado Nuevo, toma {s2.nunique()} valores distintos en los 9 escenarios "
        f"({conteo_s2}): la decisión sobre ese segmento depende de supuestos que hoy no están medidos.",
        "",
    ])
    texto = f"""
# m11 - MDP del ciclo de vida del cliente

## Qué hace
1. **Cadena de Markov con datos reales.** Cada consumidor final tiene un estado al cierre de cada
   mes, desde su primera compra hasta el último mes de la base: Nuevo (1 visita acumulada),
   Recurrente (2-3), Leal (4+), Dormido (3 a 6 meses sin comprar) y Perdido (más de 6 meses).
   Se estima la matriz de transición mensual, en total y por grupo de segmento (Recurrente =
   segmento 1, Proyecto grande = segmento 2, Un solo negocio = segmentos 3 a 6). La recompensa de
   cada estado es el gasto esperado del mes siguiente (incluye los meses sin compra).
2. **MDP.** Acciones: no hacer nada, contacto de venta cruzada y descuento. Una acción solo cambia
   la probabilidad de comprar el mes siguiente; dado que el cliente compra (o no), el estado se
   mueve como en los datos. Se resuelve con iteración de valor (γ = {c.num(GAMMA, 2)} mensual) y se
   repite con el efecto de las acciones a la mitad y al doble. Las acciones solo aplican a clientes
   contactables; para los demás se reporta el valor de no hacer nada.

> **Importante: los efectos de las acciones son supuestos, no datos.** La base no registra
> contactos ni descuentos, así que el aumento en la probabilidad de compra, el costo por contacto y
> la pérdida de margen están fijados a mano en el diccionario `EFECTOS` al inicio de
> `scripts/modelos/m11_mdp.py`. El piloto (m14) debe medirlos; con esos resultados se reemplazan
> estos valores y se vuelve a resolver el MDP.

## Datos usados
- `data/transaccional_{c.VERSION}.parquet`, `data/features_clientes_{c.VERSION}.parquet` (contactabilidad,
  tipo de cliente y `posible_cliente_previo`) y `data/segmentos_{c.VERSION}_{c.SEGMENTACION}.parquet`.
- Consumidores finales segmentados sin `posible_cliente_previo`: {c.miles(len(actual))} clientes,
  {c.miles(int(cadenas['Total']['observaciones'].sum()))} transiciones cliente-mes.

## Supuestos de las acciones
| Acción | Aumento prob. de compra, pp (Nue / Rec / Leal / Dorm / Perd) | Costo por contacto: % del ticket mediano (unidades de la base) | Pérdida de margen |
|---|---|---|---|
{sup}

## Métricas y resultados de la cadena
Gasto esperado del mes siguiente por estado (por cliente, unidades de la base) y probabilidad de comprar (total):

| Estado | Total | Recurrente | Proyecto grande | Un solo negocio | Prob. de compra (total) |
|---|---|---|---|---|---|
{filas_r}

Persistencia mensual (total): Leal → Leal {c.pct(P[idx['Leal'], idx['Leal']] * 100)},
Dormido → Perdido {c.pct(P[idx['Dormido'], idx['Perdido']] * 100)}, Perdido → Perdido
{c.pct(P[idx['Perdido'], idx['Perdido']] * 100)}. Matrices completas en `matrices_transicion.csv`
y `01_matrices_transicion.png`.

## Política óptima (contactables)
| Grupo | Escenario | Nuevo | Recurrente | Leal | Dormido | Perdido |
|---|---|---|---|---|---|---|
{filas_p}

{tabla_costo}
{lectura_seg2}
## Hallazgos principales
- Con el efecto base, en {n_actuar} de las {len(base)} combinaciones grupo-estado conviene actuar;
  al cierre de la base eso equivale a {c.pct(pct_accion)} de los clientes contactables.
{hallazgos}
- Sensibilidad: con la mitad del efecto cambia la acción en {cambios_mitad} combinaciones y con el
  doble en {cambios_doble}. Las decisiones en el límite son justamente las que el piloto debe aclarar.

## Limitaciones
- Los efectos de las acciones son supuestos; la política es tan buena como esos supuestos.
- El MDP supone que la acción puede repetirse cada mes con el mismo efecto (sin fatiga ni
  saturación del cliente). Esto es optimista sobre todo en Dormido y Perdido, donde el valor de actuar
  se acumula por muchos meses de contacto; conviene limitar la frecuencia de contacto en la operación.
- Costo por contacto = {c.pct(EFECTOS['Venta cruzada']['costo_contacto_pct_ticket'] * 100, 0)} del ticket mediano por visita de los
  consumidores finales ({c.miles(ticket_mediano_visita())} en unidades de la base), para que costos y ventas estén
  en las mismas unidades escaladas. El parámetro `costo_contacto_pct_ticket` es editable en `EFECTOS`.
- La recompensa es el gasto (ventas), no el margen; esto favorece actuar. Cambiar `MARGEN` para
  decidir sobre margen de contribución.
- La cadena supone que la transición solo depende del estado actual (propiedad de Markov) y que es
  estable en el tiempo; la base cubre solo 31 meses (29 meses de panel, porque los clientes de enero y febrero de
  2000 se excluyen como posibles clientes previos), con fechas desplazadas.
- Los grupos de segmento se calcularon con todo el historial; la cadena por grupo es descriptiva
  (por ejemplo, un cliente "Proyecto grande" casi nunca llega a Leal por definición).
{aviso_total}- Los estados "Perdido" tienen poca historia porque la base empieza en 2000-01.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
