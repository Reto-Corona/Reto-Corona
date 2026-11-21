"""
m14 - Diseño del piloto

Segmento objetivo: 2 (Proyecto grande en una visita), solo clientes contactables.
Línea base: tasa real de recompra a 90 días de ese grupo, calculada en m08.

1. Tamaño de muestra por grupo (tratamiento y control) para detectar aumentos de 2, 3 y 5
   puntos porcentuales con alfa = 0,05 (dos colas) y potencia = 0,8 (statsmodels).
2. Asignación aleatoria estratificada por región y canal (ejemplo con la semilla del proyecto)
   y verificación de balance entre grupos.
3. Duración, calendario y KPIs del piloto.

Requiere haber corrido m08_recompra.py (data/modelos/m08_recompra.parquet).

Escribe:
    data/modelos/m14_piloto.parquet               asignación de ejemplo por cliente
    salidas/modelos/m14_piloto/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m14_piloto.py
"""
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

import comun as c
from m11_mdp import EFECTOS, costo_contacto

NOMBRE = "m14_piloto"
SEGMENTO = 2
ALFA = 0.05
POTENCIA = 0.80
AUMENTOS_PP = [2, 3, 5]
AUMENTO_DISENO_PP = 3      # el que se usa para la asignación de ejemplo
DURACION_DIAS = 90
MIN_ESTRATO = 30           # estratos más pequeños se agrupan dentro de su región
# El costo por contacto se importa de m11 (costo_contacto): proporción del ticket mediano por
# visita, en las mismas unidades de la base que las ventas.


def tamano_muestra(p0: float, aumento: float) -> int:
    """Clientes por grupo para detectar p0 -> p0 + aumento (dos colas, grupos iguales)."""
    h = proportion_effectsize(p0 + aumento, p0)
    n = NormalIndPower().solve_power(effect_size=h, alpha=ALFA, power=POTENCIA, ratio=1.0,
                                     alternative="two-sided")
    return int(np.ceil(n))


def efecto_minimo(p0: float, n: int) -> float:
    """Aumento mínimo detectable (en proporción) con n clientes por grupo."""
    h = NormalIndPower().solve_power(nobs1=n, alpha=ALFA, power=POTENCIA, ratio=1.0, alternative="two-sided")
    # h = 2·asin(√p1) − 2·asin(√p0)  ->  p1
    p1 = np.sin(np.arcsin(np.sqrt(p0)) + h / 2) ** 2
    return p1 - p0


def definir_estratos(df: pd.DataFrame) -> pd.Series:
    """Estrato = región x canal; los pequeños se agrupan en 'región|otros' y luego en 'OTROS'."""
    estrato = df["region"].astype(str) + "|" + df["canal"].astype(str)
    tam = estrato.map(estrato.value_counts())
    estrato = estrato.where(tam >= MIN_ESTRATO, df["region"].astype(str) + "|otros")
    tam = estrato.map(estrato.value_counts())
    return estrato.where(tam >= MIN_ESTRATO, "OTROS|otros")


def asignar(df: pd.DataFrame, n_por_grupo: int, rng: np.random.Generator) -> pd.Series:
    """Muestra estratificada proporcional de 2n clientes, alternando tratamiento y control en cada estrato."""
    fraccion = min(1.0, 2 * n_por_grupo / len(df))
    grupo = pd.Series("fuera_del_piloto", index=df.index)
    for _, idx in df.groupby("estrato").groups.items():
        idx = rng.permutation(np.array(idx))
        k = int(round(len(idx) * fraccion))
        k += k % 2  # par, para que el estrato quede balanceado
        elegidos = idx[:min(k, len(idx))]
        grupo.loc[elegidos[0::2]] = "tratamiento"
        grupo.loc[elegidos[1::2]] = "control"
    return grupo


def balance(df: pd.DataFrame) -> pd.DataFrame:
    """Diferencia estandarizada entre tratamiento y control (|d| < 0,1 = balanceado)."""
    t, cc = df[df["grupo"] == "tratamiento"], df[df["grupo"] == "control"]
    filas = []
    variables = {"prob_recompra_90d": "Probabilidad de recompra (m08)"}
    for v, nombre in variables.items():
        d = (t[v].mean() - cc[v].mean()) / np.sqrt((t[v].var() + cc[v].var()) / 2)
        filas.append({"variable": nombre, "tratamiento": t[v].mean(), "control": cc[v].mean(), "dif_estandarizada": d})
    for col, nombre in (("region", "Región"), ("canal", "Canal")):
        for nivel in df.loc[df["grupo"] != "fuera_del_piloto", col].value_counts().head(6).index:
            pt, pc = (t[col] == nivel).mean(), (cc[col] == nivel).mean()
            d = (pt - pc) / np.sqrt((pt * (1 - pt) + pc * (1 - pc)) / 2)
            filas.append({"variable": f"{nombre}: {nivel}", "tratamiento": pt, "control": pc, "dif_estandarizada": d})
    return pd.DataFrame(filas)


def graficar_potencia(carpeta, p0: float, tabla: pd.DataFrame) -> None:
    """Clientes por grupo necesarios según el aumento a detectar."""
    aumentos = np.linspace(1, 6, 60)
    ns = [tamano_muestra(p0, a / 100) for a in aumentos]
    fig, ax = c.plt.subplots(figsize=(9.5, 5.5))
    ax.plot(aumentos, ns, color=c.AZUL, lw=2)
    for r in tabla.itertuples():
        ax.scatter([r.aumento_pp], [r.n_por_grupo], color=c.NARANJA, zorder=3, s=40)
        ax.annotate(f"+{r.aumento_pp} pp: {c.miles(r.n_por_grupo)} por grupo", (r.aumento_pp, r.n_por_grupo),
                    xytext=(8, 6), textcoords="offset points", fontsize=9.5, color=c.TINTA)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.miles(v)))
    ax.set_xlabel("Aumento a detectar en la tasa de recompra a 90 días (puntos porcentuales)")
    ax.set_ylabel("Clientes por grupo (escala log)")
    c.guardar_figura(fig, carpeta, "01_tamano_de_muestra", "Tamaño de muestra del piloto",
                     f"Línea base {c.pct(p0 * 100)} (segmento 2 contactable, m08); α = 0,05 dos colas, potencia 0,8.")


def graficar_balance(carpeta, bal: pd.DataFrame) -> None:
    """Diferencias estandarizadas de la asignación de ejemplo."""
    b = bal.iloc[::-1]
    fig, ax = c.plt.subplots(figsize=(9, 6))
    colores = [c.AZUL if abs(d) < 0.1 else c.ROJO for d in b["dif_estandarizada"]]
    ax.barh(b["variable"], b["dif_estandarizada"], color=colores, height=0.6)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=c.EJE, ls="--", lw=1)
    ax.axvline(0, color=c.TENUE, lw=0.8)
    ax.set_xlim(-0.15, 0.15)
    ax.xaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.num(v, 2)))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Diferencia estandarizada (tratamiento − control)")
    c.guardar_figura(fig, carpeta, "02_balance_asignacion", "Balance de la asignación aleatoria estratificada",
                     f"Muestra de ejemplo para +{AUMENTO_DISENO_PP} pp. Dentro de ±0,1 los grupos se consideran balanceados.")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    ruta_m08 = c.ruta_puntajes("m08_recompra")
    c.verificar_insumos(ruta_m08)
    m08 = pd.read_parquet(ruta_m08)

    objetivo = m08[(m08["segmento"] == SEGMENTO) & m08["es_contactable"].fillna(False).astype(bool)].copy()
    observables = objetivo[objetivo["recompra_90d"].notna()]
    p0 = observables["recompra_90d"].mean()
    print(f"  Segmento {SEGMENTO} contactable: {len(objetivo):,} clientes; "
          f"línea base (observables, n = {len(observables):,}): {p0:.4f}")

    # Flujo mensual de clientes NUEVOS del segmento (primera compra en el mes; sin clientes previos),
    # promedio de los últimos 12 meses completos: define cuánto dura el reclutamiento
    nuevos = objetivo[objetivo["muestra"] != "excluido_cliente_previo"]
    meses = nuevos["fecha_primera_visita"].dt.to_period("M")
    ultimo = meses.max()
    por_mes = meses[(meses > ultimo - 13) & (meses < ultimo)].value_counts().sort_index()
    flujo_mensual = por_mes.mean()
    por_mes.rename("clientes_nuevos").rename_axis("mes").reset_index().assign(mes=lambda d: d["mes"].astype(str)) \
        .to_csv(carpeta / "flujo_mensual_nuevos.csv", index=False)
    costo = costo_contacto("Venta cruzada")
    existentes = alternativa_existentes(objetivo)

    # 1) Tamaño de muestra
    tabla = pd.DataFrame([{
        "aumento_pp": a, "tasa_base": p0, "tasa_tratamiento": p0 + a / 100,
        "efecto_h_cohen": proportion_effectsize(p0 + a / 100, p0),
        "n_por_grupo": tamano_muestra(p0, a / 100),
    } for a in AUMENTOS_PP])
    tabla["n_total"] = 2 * tabla["n_por_grupo"]
    tabla["pct_del_universo"] = tabla["n_total"] / len(objetivo) * 100
    tabla["meses_de_reclutamiento"] = tabla["n_total"] / flujo_mensual
    tabla["recompras_incrementales_esperadas"] = tabla["n_por_grupo"] * tabla["aumento_pp"] / 100
    tabla["costo_contactos_unidades_base"] = tabla["n_por_grupo"] * costo
    tabla["costo_por_recompra_incremental_unidades_base"] = costo / (tabla["aumento_pp"] / 100)
    tabla.to_csv(carpeta / "tamano_muestra.csv", index=False, float_format=c.FORMATO_CSV)
    print(tabla.to_string(index=False))
    mde_todos = efecto_minimo(p0, len(objetivo) // 2)
    graficar_potencia(carpeta, p0, tabla)

    # 2) Asignación aleatoria estratificada (ejemplo con +3 pp)
    rng = np.random.default_rng(c.SEMILLA)
    objetivo = objetivo.sort_values("cliente").reset_index(drop=True)  # orden fijo = asignación reproducible
    objetivo["estrato"] = definir_estratos(objetivo)
    n_diseno = int(tabla.loc[tabla["aumento_pp"] == AUMENTO_DISENO_PP, "n_por_grupo"].iloc[0])
    objetivo["grupo"] = asignar(objetivo, n_diseno, rng)
    conteo = objetivo["grupo"].value_counts()
    print(conteo.to_string())
    estratos = (objetivo.groupby(["estrato", "grupo"]).size().unstack(fill_value=0).reset_index())
    estratos.to_csv(carpeta / "estratos.csv", index=False, float_format=c.FORMATO_CSV)
    bal = balance(objetivo)
    bal.to_csv(carpeta / "balance_asignacion.csv", index=False, float_format=c.FORMATO_CSV)
    graficar_balance(carpeta, bal)
    objetivo[["cliente", "segmento", "region", "canal", "estrato", "grupo", "prob_recompra_90d",
              "fecha_primera_visita"]].to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(objetivo):,} clientes)")

    # 3) KPIs
    kpis = pd.DataFrame([
        {"kpi": "Tasa de segunda compra a 90 días", "tipo": "Principal",
         "definicion": "Clientes con otra visita (fecha posterior) en los 90 días siguientes al inicio / clientes del grupo",
         "comparacion": "Tratamiento − control, prueba de dos proporciones"},
        {"kpi": "Ticket de la segunda compra", "tipo": "Secundario",
         "definicion": f"{c.VALOR_TXT.capitalize()} de la segunda visita, entre quienes recompran",
         "comparacion": "Diferencia de medianas (bootstrap)"},
        {"kpi": "% que compra en un segundo negocio", "tipo": "Secundario",
         "definicion": "Clientes que en los 90 días compran en un negocio distinto al de la primera visita / clientes del grupo",
         "comparacion": "Tratamiento − control, prueba de dos proporciones"},
        {"kpi": "Costo por recompra lograda", "tipo": "Eficiencia",
         "definicion": "Costo total de la campaña / (recompras del tratamiento − tasa del control x clientes del tratamiento)",
         "comparacion": "Frente al margen de la segunda compra"},
    ])
    kpis.to_csv(carpeta / "kpis.csv", index=False, float_format=c.FORMATO_CSV)

    existentes["tabla"].to_csv(carpeta / "tamano_muestra_existentes.csv", index=False, float_format=c.FORMATO_CSV)
    escribir_resumen(carpeta, p0, objetivo, observables, tabla, flujo_mensual, mde_todos, bal, conteo, n_diseno,
                     costo, existentes, por_mes)


def alternativa_existentes(objetivo: pd.DataFrame) -> dict:
    """Alternativa: piloto con clientes existentes del segmento 2 (ya pasaron sus primeros 90 días).

    Línea base: entre los clientes que al inicio de los últimos 90 días de la base ya llevaban más de
    90 días desde su primera compra, proporción que compró en esos 90 días.
    """
    con = c.conexion()
    fin = pd.Timestamp(con.execute("SELECT MAX(fecha) FROM trx").fetchone()[0])
    inicio = fin - pd.Timedelta(days=DURACION_DIAS)
    candidatos = objetivo[(objetivo["muestra"] != "excluido_cliente_previo") &
                          (objetivo["fecha_primera_visita"] <= inicio - pd.Timedelta(days=DURACION_DIAS))]
    con.register("cand", candidatos[["cliente"]])
    compran = set(con.execute(f"""
        SELECT DISTINCT t.cliente FROM trx t JOIN cand USING (cliente)
        WHERE t.fecha::DATE > DATE '{inicio:%Y-%m-%d}'
    """).df()["cliente"])
    p_ex = candidatos["cliente"].isin(compran).mean()
    # Universo hoy: existentes = todos los del segmento 2 contactables con más de 90 días desde la primera compra
    universo = objetivo[(objetivo["muestra"] != "excluido_cliente_previo") &
                        (objetivo["fecha_primera_visita"] <= fin - pd.Timedelta(days=DURACION_DIAS))]
    sin_segunda = universo[universo["recompra_90d"] == 0]
    tabla = pd.DataFrame([{"aumento_pp": a, "tasa_base": p_ex, "n_por_grupo": tamano_muestra(p_ex, a / 100)}
                          for a in AUMENTOS_PP])
    tabla["n_total"] = 2 * tabla["n_por_grupo"]
    tabla["pct_del_universo"] = tabla["n_total"] / len(universo) * 100
    return {"p": p_ex, "n_candidatos_base": len(candidatos), "universo": len(universo),
            "sin_segunda_compra": len(sin_segunda), "tabla": tabla, "inicio": inicio}


def escribir_resumen(carpeta, p0, objetivo, observables, tabla, flujo, mde_todos, bal, conteo, n_diseno,
                     costo, existentes, por_mes) -> None:
    """resumen.md del modelo."""
    filas = "\n".join(
        f"| +{r.aumento_pp} pp | {c.pct(r.tasa_tratamiento * 100)} | {c.miles(r.n_por_grupo)} | {c.miles(r.n_total)} | "
        f"{c.num(r.meses_de_reclutamiento, 1)} | {c.miles(r.costo_por_recompra_incremental_unidades_base)} |"
        for r in tabla.itertuples())
    ex = existentes
    filas_ex = "\n".join(
        f"| +{r.aumento_pp} pp | {c.pct((r.tasa_base + r.aumento_pp / 100) * 100)} | {c.miles(r.n_por_grupo)} | "
        f"{c.miles(r.n_total)} | {c.pct(r.pct_del_universo)} |" for r in ex["tabla"].itertuples())
    r3_ex = ex["tabla"][ex["tabla"]["aumento_pp"] == AUMENTO_DISENO_PP].iloc[0]
    max_d = bal["dif_estandarizada"].abs().max()
    n_estratos = objetivo.loc[objetivo["grupo"] != "fuera_del_piloto", "estrato"].nunique()
    r3 = tabla[tabla["aumento_pp"] == AUMENTO_DISENO_PP].iloc[0]
    # Ticket mediano por visita del segmento, del perfil de la segmentación de trabajo
    perfil = c.RAIZ / "salidas" / c.VERSION / c.SEGMENTACION / "segmentacion_perfil_kmeans.csv"
    ticket = (pd.read_csv(perfil).set_index("segmento").loc[SEGMENTO, "ticket_mediano"]
              if perfil.exists() else np.nan)
    if np.isnan(ticket):
        frase_ticket = "debe compararse con el margen de la segunda compra que mida el piloto."
    else:
        relacion = "por debajo" if r3.costo_por_recompra_incremental_unidades_base < ticket else "por encima"
        veces = ticket / r3.costo_por_recompra_incremental_unidades_base
        frase_ticket = (f"queda {relacion} del ticket mediano por visita del segmento ({c.miles(ticket)}; "
                        f"{c.num(veces, 1)} veces el costo); "
                        "el piloto debe confirmar que el margen de la segunda compra lo cubre.")
    texto = f"""
# m14 - Diseño del piloto

## Qué hace
Diseña un experimento aleatorio para medir si un contacto de venta cruzada aumenta la segunda
compra del segmento **2. Proyecto grande en una visita**, solo con clientes contactables. Calcula
el tamaño de muestra, propone una asignación aleatoria estratificada (con un ejemplo y su
verificación de balance) y define duración y KPIs. Es el experimento que debe medir los efectos
que el MDP (m11) hoy toma como supuestos.

## Datos usados
- `{c.RUTA_PUNTAJES_TXT}/m08_recompra.parquet`: segmento, contactabilidad, región y canal de la primera
  visita, recompra real a 90 días y probabilidad predicha.
- Universo: {c.miles(len(objetivo))} clientes del segmento 2 contactables. Línea base = tasa real
  de recompra a 90 días de los {c.miles(len(observables))} observables: **{c.pct(p0 * 100)}**.
- Costo por contacto: {c.pct(EFECTOS['Venta cruzada']['costo_contacto_pct_ticket'] * 100, 0)} del ticket mediano por visita de los
  consumidores finales = {c.miles(costo)} en unidades de la base (el mismo supuesto de m11, importado de
  `m11_mdp.costo_contacto`). Todas las cifras de costo están en las unidades escaladas de la base.

## Métricas: tamaño de muestra (α = 0,05 dos colas, potencia = 0,8)
| Aumento a detectar | Tasa esperada en tratamiento | Clientes por grupo | Total | Meses de reclutamiento* | Costo por recompra incremental (unidades de la base) |
|---|---|---|---|---|---|
{filas}

\\* Meses de reclutamiento = clientes totales / clientes nuevos que llegan por mes (ver abajo).
Como referencia, si se usara todo el universo histórico dividido en dos grupos, el efecto mínimo
detectable sería de {c.num(mde_todos * 100, 2)} pp.

## Población del piloto: clientes nuevos del segmento 2
El piloto se aplica a **clientes nuevos** del segmento 2, contactables, **justo después de su
primera compra** (la compra grande en una sola visita). Es el momento en que la segunda compra
todavía es probable ({c.pct(p0 * 100)} vuelve en 90 días) y en que la venta cruzada tiene sentido,
porque el proyecto está en marcha. Por eso la población no se toma de una vez: se recluta a medida
que llegan clientes nuevos.

- Llegan en promedio **{c.miles(flujo)} clientes nuevos del segmento 2 contactables por mes** (promedio de
  los últimos {len(por_mes)} meses completos de la base, sin clientes previos; mínimo {c.miles(por_mes.min())},
  máximo {c.miles(por_mes.max())}; detalle en `flujo_mensual_nuevos.csv`).
- Por eso reclutar los {c.miles(r3.n_total)} clientes del diseño de +{AUMENTO_DISENO_PP} pp toma unos
  **{c.num(r3.meses_de_reclutamiento, 1)} meses**, más 90 días de seguimiento del último cliente reclutado.
- En operación, el segmento de un cliente nuevo se conoce con su primera compra (ticket alto y varias
  categorías en una visita); la regla de entrada debe replicar esa definición con los datos del día.

## Alternativa: clientes existentes del segmento 2
También se puede hacer con clientes que **ya** son del segmento 2 y pasaron sus primeros 90 días.
Hay {c.miles(ex['universo'])} contactables en esa situación ({c.miles(ex['sin_segunda_compra'])} de ellos no hicieron la
segunda compra en sus primeros 90 días). La ventaja es que el piloto arranca de inmediato, sin esperar
el reclutamiento. La desventaja es que la línea base es mucho menor: de los {c.miles(ex['n_candidatos_base'])}
clientes existentes al inicio de los últimos 90 días de la base, solo **{c.pct(ex['p'] * 100)}** compró en esos
90 días. El KPI principal pasa a ser "compra en los próximos 90 días" y el tamaño de muestra cambia:

| Aumento a detectar | Tasa esperada en tratamiento | Clientes por grupo | Total | % de los existentes |
|---|---|---|---|---|
{filas_ex}

Con +{AUMENTO_DISENO_PP} pp se necesitan {c.miles(r3_ex.n_por_grupo)} clientes por grupo. Sin embargo, un aumento de
{AUMENTO_DISENO_PP} pp es relativamente mucho más difícil de lograr sobre una base de {c.pct(ex['p'] * 100)} que sobre
{c.pct(p0 * 100)}, y lo que se aprenda con clientes "fríos" no se traslada directamente al momento
posterior a la primera compra. **Recomendación:** clientes nuevos como diseño principal; clientes
existentes solo como prueba rápida complementaria (por ejemplo, para medir la tasa de contacto
efectivo y el costo real por contacto mientras se recluta).

## Diseño propuesto
- **Unidad y población:** cliente nuevo del segmento 2, contactable, que acaba de hacer su primera
  compra grande; entran al piloto a medida que llegan (reclutamiento continuo, ver arriba).
- **Asignación:** aleatoria 1:1, estratificada por región x canal (cluster) de la primera visita. Con
  reclutamiento continuo se aleatoriza dentro de cada estrato en bloques (por ejemplo, de 2 o 4
  clientes) a medida que llegan, para que los grupos queden balanceados en todo momento. El ejemplo de
  abajo aplica el mismo procedimiento al universo histórico para mostrar el balance.
  Estratos con menos de {MIN_ESTRATO} clientes se agrupan en "región|otros" y, si siguen siendo
  pequeños, en "OTROS". En el ejemplo (diseño para +{AUMENTO_DISENO_PP} pp, semilla {c.SEMILLA}) quedan
  {c.miles(conteo.get('tratamiento', 0))} en tratamiento y {c.miles(conteo.get('control', 0))} en control en
  {n_estratos} estratos; la mayor diferencia estandarizada es {c.num(max_d, 3)} (balanceado si < 0,1).
  La asignación está en `{c.RUTA_PUNTAJES_TXT}/m14_piloto.parquet`.
- **Tratamiento:** contacto de venta cruzada (asesor o mensaje) con productos complementarios al
  proyecto. **Control:** sin contacto (operación normal).
- **Duración:** {DURACION_DIAS} días de seguimiento para cada cliente desde su entrada al piloto.
  Diseño recomendado: +{AUMENTO_DISENO_PP} pp, {c.miles(n_diseno)} clientes por grupo, unos
  {c.num(r3.meses_de_reclutamiento, 1)} meses de reclutamiento más 3 meses de seguimiento.

## KPIs
| KPI | Tipo | Definición |
|---|---|---|
| Tasa de segunda compra a 90 días | Principal | Clientes con otra visita en los 90 días siguientes / clientes del grupo |
| Ticket de la segunda compra | Secundario | {c.VALOR_TXT.capitalize()} de la segunda visita, entre quienes recompran |
| % que compra en un segundo negocio | Secundario | Clientes que compran en un negocio distinto al de la primera visita / clientes del grupo |
| Costo por recompra lograda | Eficiencia | Costo de la campaña / recompras incrementales (tratamiento − control) |

## Hallazgos principales
- Con una línea base de {c.pct(p0 * 100)}, detectar +2 pp exige {c.miles(tabla.iloc[0].n_por_grupo)} clientes por
  grupo; +3 pp, {c.miles(tabla.iloc[1].n_por_grupo)}; y +5 pp, {c.miles(tabla.iloc[2].n_por_grupo)}. El universo
  alcanza para los tres, pero +2 pp necesita una fracción grande del segmento y más tiempo de reclutamiento.
- +{AUMENTO_DISENO_PP} pp es un buen equilibrio: es un efecto realista para CRM y su costo por recompra
  incremental ({c.miles(r3.costo_por_recompra_incremental_unidades_base)} en unidades de la base) {frase_ticket}

## Limitaciones
- La línea base es histórica; si cambia la tasa de recompra, hay que recalcular el tamaño.
- El costo por contacto es un supuesto (m11); el piloto debe registrar el costo real.
- Se asume que no hay contaminación entre grupos (el control no recibe otros contactos de la
  campaña) y que todos los asignados a tratamiento son efectivamente contactados; si la tasa de
  contacto efectivo es menor, el efecto medido se diluye (análisis por intención de tratar) y hay
  que aumentar la muestra.
- Con varios KPIs secundarios conviene ajustar por comparaciones múltiples o tratarlos como exploratorios.

## Fase 2: modelo de uplift
Con los datos del piloto (tratamiento aleatorio, respuesta observada y las variables de m08) se
entrena un modelo de **uplift** (por ejemplo, un T-learner o un X-learner con LightGBM, o un
árbol causal) que estima, para cada cliente, cuánto aumenta su probabilidad de recompra **por
recibir el contacto**, no solo su probabilidad de recompra. La siguiente campaña se dirige a los
clientes con mayor uplift estimado (los "persuadibles") y se excluyen los que compran igual sin
contacto o los que reaccionan mal. El modelo se valida con curvas Qini o de uplift sobre una parte
del piloto que no se usa para entrenar, y los efectos medidos reemplazan los supuestos del MDP (m11).
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
