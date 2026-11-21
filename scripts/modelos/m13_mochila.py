"""
m13 - Selección de clientes a contactar con programación entera (problema de la mochila)

Universo: consumidores finales contactables (segmentados).

Valor esperado de contactar a un cliente (acción "Venta cruzada"), con la MISMA definición de m11:
    valor = aumento en puntos de la probabilidad de compra (según el estado del cliente en m11)
            x valor esperado del cliente a 12 meses (m10)
            - costo por contacto
La fórmula se importa de m11 (m11_mdp.ganancia_inmediata), junto con el aumento y el costo, así que
los dos modelos no pueden divergir y todo queda en las mismas unidades de la base.

Para que sea eficiente, los clientes se agrupan y la variable de decisión es cuántos contactar de cada
grupo (entera, entre 0 y el tamaño del grupo):
    principal:   segmento x región x decil del valor de contactar
    comparación: segmento x región x decil de P(recompra) de m08
Restricciones: presupuesto total, tope de presupuesto por región y un mínimo de contactos por
segmento para aprender. Se resuelve con OR-Tools (SCIP).

Se compara con una regla simple (contactar a los de mayor P(recompra) hasta agotar el presupuesto),
se traza la curva de valor esperado según el presupuesto y se mide la coherencia con m11.

Escribe:
    data/modelos/m13_mochila.parquet           valor de contactar y selección por cliente (presupuesto base)
    salidas/modelos/m13_mochila/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m13_mochila.py
"""
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

import comun as c
from m11_mdp import EFECTOS, costo_contacto, ganancia_inmediata

NOMBRE = "m13_mochila"
ACCION = "Venta cruzada"
CONTACTOS_BASE = 10_000                    # presupuesto base, expresado en número de contactos
CURVA_CONTACTOS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
TOPE_REGION = 0.35                          # ninguna región puede llevarse más del 35% del presupuesto
MIN_POR_SEGMENTO = 200                      # contactos mínimos por segmento (para aprender)
ESTADO_SIN_DATO = "Perdido"                 # clientes sin estado en m11 (clientes previos): efecto más bajo
LIMITE_SEGUNDOS = 60
AGRUPACIONES = {
    "valor": ("decil_valor", "Segmento x región x decil del valor de contactar (principal)"),
    "prob": ("decil_prob", "Segmento x región x decil de P(recompra) (comparación)"),
}


def cargar_universo() -> pd.DataFrame:
    """Contactables con P(recompra) de m08, valor a 12 meses de m10 y estado y acción de m11."""
    for nombre in ("m08_recompra", "m10_clv", "m11_mdp"):
        c.verificar_insumos(c.ruta_puntajes(nombre))
    m08 = pd.read_parquet(c.ruta_puntajes("m08_recompra"))
    m10 = pd.read_parquet(c.ruta_puntajes("m10_clv"))[["cliente", "valor_esperado_12m"]]
    m11 = pd.read_parquet(c.ruta_puntajes("m11_mdp"))[["cliente", "estado_actual", "accion_recomendada"]]
    d = m08[m08["es_contactable"].fillna(False).astype(bool)][
        ["cliente", "segmento", "nombre_segmento", "region", "prob_recompra_90d"]]
    d = d.merge(m10, on="cliente", how="inner").merge(m11, on="cliente", how="left")
    d["estado_m11"] = d["estado_actual"].fillna(ESTADO_SIN_DATO)
    d["aumento"] = d["estado_m11"].map(EFECTOS[ACCION]["aumento_prob_compra"])
    d["costo"] = costo_contacto(ACCION)
    d["valor_contacto"] = ganancia_inmediata(ACCION, d["estado_m11"].tolist(), d["valor_esperado_12m"].to_numpy())
    d["decil_prob"] = pd.qcut(d["prob_recompra_90d"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    d["decil_valor"] = pd.qcut(d["valor_contacto"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    return d.sort_values("cliente").reset_index(drop=True)


def agrupar(d: pd.DataFrame, decil: str) -> pd.DataFrame:
    """Grupos segmento x región x decil, con tamaño y valor medio de contactar."""
    g = (d.groupby(["segmento", "region", decil])
         .agg(clientes=("cliente", "size"), valor_medio=("valor_contacto", "mean"),
              prob_media=("prob_recompra_90d", "mean"), valor_12m_medio=("valor_esperado_12m", "mean"))
         .reset_index())
    g["id"] = np.arange(len(g))
    return g


def resolver(grupos: pd.DataFrame, presupuesto: float, costo: float, con_restricciones: bool = True) -> tuple:
    """Programa entero: max Σ valor_g x_g  s.a. presupuesto, tope por región y mínimo por segmento."""
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SetTimeLimit(LIMITE_SEGUNDOS * 1000)
    x = [solver.IntVar(0, int(n), f"x_{i}") for i, n in enumerate(grupos["clientes"])]
    solver.Add(solver.Sum(costo * xi for xi in x) <= presupuesto)
    if con_restricciones:
        for _, idx in grupos.groupby("region").groups.items():
            solver.Add(solver.Sum(costo * x[i] for i in idx) <= TOPE_REGION * presupuesto)
        cupo = int(presupuesto // costo)
        for _, idx in grupos.groupby("segmento").groups.items():
            minimo = min(MIN_POR_SEGMENTO, int(grupos.loc[idx, "clientes"].sum()), cupo // grupos["segmento"].nunique())
            solver.Add(solver.Sum(x[i] for i in idx) >= minimo)
    solver.Maximize(solver.Sum(v * xi for v, xi in zip(grupos["valor_medio"], x)))
    estado = solver.Solve()
    if estado not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"El solver no encontró solución (estado {estado}).")
    sol = np.array([round(xi.solution_value()) for xi in x])
    return sol, solver.Objective().Value(), estado == pywraplp.Solver.OPTIMAL


def regla_simple(d: pd.DataFrame, n: int) -> float:
    """Contactar a los n de mayor P(recompra) (sin restricciones)."""
    return d.nlargest(n, "prob_recompra_90d")["valor_contacto"].sum()


def cota_individual(d: pd.DataFrame, n: int) -> float:
    """Referencia: elegir cliente por cliente los de mayor valor positivo (sin agrupar ni restricciones)."""
    v = d["valor_contacto"].nlargest(n)
    return v[v > 0].sum()


def coherencia_m11(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Compara 'm11 recomienda actuar' con 'm13: contactar tiene valor positivo' cliente por cliente."""
    con_m11 = d[d["accion_recomendada"].notna()].copy()
    con_m11["m11_actuar"] = con_m11["accion_recomendada"] != "No hacer nada"
    con_m11["m13_contactar"] = con_m11["valor_contacto"] > 0
    con_m11["coinciden"] = con_m11["m11_actuar"] == con_m11["m13_contactar"]
    matriz = pd.crosstab(con_m11["m11_actuar"].map({True: "m11: actuar", False: "m11: no hacer nada"}),
                         con_m11["m13_contactar"].map({True: "m13: valor > 0", False: "m13: valor <= 0"}))
    por_estado = (con_m11.groupby("estado_m11")
                  .agg(clientes=("cliente", "size"), pct_m11_actuar=("m11_actuar", "mean"),
                       pct_m13_contactar=("m13_contactar", "mean"), pct_coinciden=("coinciden", "mean"))
                  .reset_index())
    for col in ("pct_m11_actuar", "pct_m13_contactar", "pct_coinciden"):
        por_estado[col] *= 100
    por_accion = (con_m11.groupby("accion_recomendada")
                  .agg(clientes=("cliente", "size"), pct_m13_contactar=("m13_contactar", "mean"),
                       pct_coinciden=("coinciden", "mean")).reset_index())
    por_accion[["pct_m13_contactar", "pct_coinciden"]] *= 100
    mismo = con_m11[con_m11["accion_recomendada"].isin(["No hacer nada", ACCION])]
    resumen = {"pct_coinciden": con_m11["coinciden"].mean() * 100,
               "pct_coinciden_misma_accion": mismo["coinciden"].mean() * 100,
               "clientes_misma_accion": len(mismo),
               "pct_desacuerdo_por_descuento": (~con_m11["coinciden"] & (con_m11["accion_recomendada"] == "Descuento")).sum()
               / max((~con_m11["coinciden"]).sum(), 1) * 100}
    return matriz, por_estado, por_accion, resumen


def graficar_curva(carpeta, curva: pd.DataFrame, costo: float) -> None:
    """Valor esperado según el presupuesto para cada estrategia."""
    fig, ax = c.plt.subplots(figsize=(10.5, 6))
    series = [("optimo_valor", "Mochila: segmento x región x decil de valor (principal)", c.AZUL, "-"),
              ("optimo_prob", "Mochila: segmento x región x decil de probabilidad", c.VERDE, "-"),
              ("regla_simple", "Regla simple: mayor probabilidad primero", c.NARANJA, "-"),
              ("cota_individual", "Referencia: cliente por cliente (sin restricciones)", c.TENUE, "--")]
    for col, nombre, color, ls in series:
        ax.plot(curva["contactos"], curva[col] / 1e6, marker="o", color=color, lw=2, ls=ls, label=nombre, ms=4)
    ax.axhline(0, color=c.EJE, lw=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.miles(v)))
    ax.set_xlabel(f"Presupuesto (número de contactos; 1 contacto = {c.miles(costo)} unidades de la base)")
    ax.set_ylabel("Valor esperado neto (millones, unidades de la base)")
    ax.legend(loc="lower left", fontsize=8.5)
    c.guardar_figura(fig, carpeta, "01_curva_valor_presupuesto", "Valor esperado neto según el presupuesto",
                     f"Mochila con tope de {c.pct(TOPE_REGION * 100, 0)} por región y {MIN_POR_SEGMENTO} contactos mínimos por segmento.")


def graficar_plan(carpeta, plan: pd.DataFrame) -> None:
    """Contactos del plan base por segmento."""
    t = plan.groupby("nombre_segmento", as_index=False)["contactos"].sum().sort_values("contactos")
    fig, ax = c.plt.subplots(figsize=(9, 4.8))
    barras = ax.barh(t["nombre_segmento"], t["contactos"], color=c.AZUL, height=0.6)
    c.rotular_barras(ax, barras, [c.miles(v) for v in t["contactos"]], horizontal=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Clientes a contactar")
    c.guardar_figura(fig, carpeta, "02_plan_por_segmento",
                     f"Plan óptimo con {c.miles(CONTACTOS_BASE)} contactos de presupuesto",
                     f"Agrupación principal (decil de valor). Incluye el mínimo de {MIN_POR_SEGMENTO} por segmento para aprender.")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    d = cargar_universo()
    costo = costo_contacto(ACCION)
    print(f"  Contactables: {len(d):,} | costo por contacto: {costo:,.1f} | "
          f"con valor positivo: {(d['valor_contacto'] > 0).sum():,}")
    grupos = {k: agrupar(d, col) for k, (col, _) in AGRUPACIONES.items()}
    print("  Grupos: " + ", ".join(f"{k} {len(g):,}" for k, g in grupos.items()))

    # Curva según el presupuesto
    filas = []
    for n in CURVA_CONTACTOS:
        presupuesto = n * costo
        fila = {"contactos": n, "presupuesto": presupuesto}
        for k, g in grupos.items():
            _, v, opt = resolver(g, presupuesto, costo)
            fila[f"optimo_{k}"] = v
            fila[f"optimo_probado_{k}"] = opt
        fila["regla_simple"] = regla_simple(d, n)
        fila["cota_individual"] = cota_individual(d, n)
        filas.append(fila)
    curva = pd.DataFrame(filas)
    curva.to_csv(carpeta / "curva_presupuesto.csv", index=False, float_format=c.FORMATO_CSV)
    print(curva.to_string(index=False))
    graficar_curva(carpeta, curva, costo)

    # Plan con el presupuesto base y la agrupación principal
    presupuesto = CONTACTOS_BASE * costo
    g = grupos["valor"]
    x, v_base, _ = resolver(g, presupuesto, costo)
    _, v_sin, _ = resolver(g, presupuesto, costo, con_restricciones=False)
    plan = g.assign(contactos=x)
    plan["nombre_segmento"] = plan["segmento"].map(c.NOMBRES_SEGMENTOS)
    plan = plan[plan["contactos"] > 0].sort_values("valor_medio", ascending=False)
    plan.to_csv(carpeta / "plan_presupuesto_base.csv", index=False, float_format=c.FORMATO_CSV)
    graficar_plan(carpeta, plan)

    # Selección por cliente: dentro de cada grupo elegido, al azar (todos tienen el mismo valor esperado)
    rng = np.random.default_rng(c.SEMILLA)
    d["contactar"] = False
    llave = d.set_index(["segmento", "region", "decil_valor"]).index
    for r in plan.itertuples():
        idx = np.flatnonzero(llave == (r.segmento, r.region, r.decil_valor))
        d.loc[rng.choice(idx, size=int(r.contactos), replace=False), "contactar"] = True

    # Coherencia con m11
    matriz, por_estado, por_accion, coh = coherencia_m11(d)
    pct_coinciden = coh["pct_coinciden"]
    por_accion.to_csv(carpeta / "coherencia_m11_por_accion.csv", index=False, float_format=c.FORMATO_CSV)
    matriz.to_csv(carpeta / "coherencia_m11_matriz.csv", float_format=c.FORMATO_CSV)
    por_estado.to_csv(carpeta / "coherencia_m11_por_estado.csv", index=False, float_format=c.FORMATO_CSV)
    print(matriz.to_string())
    print(f"  Coinciden m11 y m13 en {pct_coinciden:.1f}% de los contactables")

    d.drop(columns=["estado_actual"]).to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(d):,} clientes, {d['contactar'].sum():,} a contactar)")

    por_seg_regla = d.nlargest(CONTACTOS_BASE, "prob_recompra_90d")["nombre_segmento"].value_counts()
    base = {"presupuesto": presupuesto, "contactos": int(x.sum()), "valor_mochila": v_base,
            "valor_mochila_sin_restricciones": v_sin, "valor_regla_simple": regla_simple(d, CONTACTOS_BASE),
            "valor_cota_individual": cota_individual(d, CONTACTOS_BASE), "pct_coinciden_m11": pct_coinciden,
            "pct_coinciden_m11_misma_accion": coh["pct_coinciden_misma_accion"]}
    pd.DataFrame([base]).to_csv(carpeta / "comparacion_presupuesto_base.csv", index=False, float_format=c.FORMATO_CSV)
    escribir_resumen(carpeta, d, costo, curva, plan, base, por_seg_regla, grupos, matriz, por_estado, por_accion, coh)


def escribir_resumen(carpeta, d, costo, curva, plan, base, por_seg_regla, grupos, matriz, por_estado,
                     por_accion, coh) -> None:
    """resumen.md del modelo."""
    aum = EFECTOS[ACCION]["aumento_prob_compra"]
    filas_c = "\n".join(
        f"| {c.miles(r.contactos)} | {c.miles(r.optimo_valor)} | {c.miles(r.optimo_prob)} | "
        f"{c.miles(r.regla_simple)} | {c.miles(r.cota_individual)} |" for r in curva.itertuples())
    seg_plan = plan.groupby("nombre_segmento")["contactos"].sum().sort_values(ascending=False)
    filas_p = "\n".join(f"| {s} | {c.miles(n)} | {c.miles(por_seg_regla.get(s, 0))} |" for s, n in seg_plan.items())
    positivos = (d["valor_contacto"] > 0).sum()
    pos_seg = d.groupby("nombre_segmento")["valor_contacto"].apply(lambda s: (s > 0).mean() * 100).sort_values(ascending=False)
    texto_pos = ", ".join(f"{s} {c.pct(v)}" for s, v in pos_seg.items())
    costo_aprender = base["valor_mochila_sin_restricciones"] - base["valor_mochila"]
    cb = curva.set_index("contactos").loc[CONTACTOS_BASE]
    mejor = curva.loc[curva["optimo_valor"].idxmax()]
    regla_neg = curva.loc[curva["regla_simple"] < 0, "contactos"]
    filas_e = "\n".join(
        f"| {x.estado_m11} | {c.miles(x.clientes)} | {c.pct(x.pct_m11_actuar)} | {c.pct(x.pct_m13_contactar)} | "
        f"{c.pct(x.pct_coinciden)} |" for x in por_estado.itertuples())
    filas_m = "\n".join(f"| {i} | " + " | ".join(c.miles(v) for v in fila) + " |" for i, fila in matriz.iterrows())
    pct = base["pct_coinciden_m11"]
    desacuerdo = por_estado.sort_values("pct_coinciden").iloc[0]
    filas_a = "\n".join(f"| {x.accion_recomendada} | {c.miles(x.clientes)} | {c.pct(x.pct_m13_contactar)} | {c.pct(x.pct_coinciden)} |"
                        for x in por_accion.itertuples())
    texto_regla = (f"desde {c.miles(regla_neg.min())} contactos su valor neto es negativo" if len(regla_neg)
                   else "su valor neto queda por debajo de la mochila en todos los presupuestos")
    texto = f"""
# m13 - Selección de clientes a contactar (programación entera)

## Qué hace
Decide a cuántos clientes contactar de cada grupo para maximizar el valor esperado neto de una
campaña de venta cruzada con un presupuesto dado. Es un problema de la mochila entero resuelto con
OR-Tools (SCIP).

- **Valor esperado de contactar** = aumento en puntos de la probabilidad de compra (según el estado del
  cliente en m11) x valor esperado del cliente a 12 meses (m10) − costo por contacto. Es la misma
  definición del efecto que usa el MDP: se importa de `m11_mdp.ganancia_inmediata`, junto con el
  aumento y el costo, así que m11 y m13 no pueden divergir.
- **Agrupación principal:** segmento x región x decil del valor de contactar ({c.miles(len(grupos['valor']))} grupos).
  **Comparación:** segmento x región x decil de P(recompra) de m08 ({c.miles(len(grupos['prob']))} grupos).
  La variable entera es cuántos clientes contactar de cada grupo.
- Restricciones: presupuesto; ninguna región puede llevarse más de {c.pct(TOPE_REGION * 100, 0)} del presupuesto; al
  menos {MIN_POR_SEGMENTO} contactos por segmento para aprender (o el tamaño del segmento si es menor).
- Comparación con una regla simple (los de mayor P(recompra) primero) y con una referencia cliente
  por cliente sin restricciones.

## Datos y supuestos
- Universo: {c.miles(len(d))} consumidores finales contactables con puntaje en m08 y m10.
- Aumento (importado de m11, acción "{ACCION}"): Nuevo {c.num(aum['Nuevo'] * 100, 1)} pp, Recurrente
  {c.num(aum['Recurrente'] * 100, 1)}, Leal {c.num(aum['Leal'] * 100, 1)}, Dormido {c.num(aum['Dormido'] * 100, 1)}, Perdido
  {c.num(aum['Perdido'] * 100, 1)} pp. Costo por contacto: {c.miles(costo)} unidades de la base
  ({c.pct(EFECTOS[ACCION]['costo_contacto_pct_ticket'] * 100, 0)} del ticket mediano por visita). Clientes sin estado en m11
  (clientes previos) usan el efecto de "{ESTADO_SIN_DATO}", el más conservador.

## Métricas
**{c.miles(positivos)}** contactables ({c.pct(positivos / len(d) * 100)}) tienen valor esperado positivo. Por segmento:
{texto_pos}.

Valor esperado neto según el presupuesto (unidades de la base):

| Contactos | Mochila, decil de valor (principal) | Mochila, decil de probabilidad | Regla simple | Referencia cliente por cliente |
|---|---|---|---|---|
{filas_c}

Presupuesto base ({c.miles(CONTACTOS_BASE)} contactos = {c.miles(base['presupuesto'])}), agrupación principal:
- Mochila: **{c.miles(base['valor_mochila'])}** con {c.miles(base['contactos'])} contactos (sin mínimos ni tope por región:
  {c.miles(base['valor_mochila_sin_restricciones'])}; costo de aprender {c.miles(costo_aprender)}).
- Regla simple: **{c.miles(base['valor_regla_simple'])}**.

| Segmento | Contactos (mochila) | Contactos (regla simple) |
|---|---|---|
{filas_p}

## Coherencia con m11
Cliente por cliente, "m11 recomienda alguna acción" frente a "en m13 contactar tiene valor positivo":
**coinciden en {c.pct(pct)} de los contactables**.

| | {' | '.join(matriz.columns)} |
|---|{'---|' * len(matriz.columns)}
{filas_m}

| Estado en m11 | Clientes | m11 recomienda actuar | m13 valor > 0 | Coinciden |
|---|---|---|---|---|
{filas_e}

| Acción recomendada por m11 | Clientes | m13 valor > 0 | Coinciden |
|---|---|---|---|
{filas_a}

**{c.pct(coh['pct_desacuerdo_por_descuento'])} del desacuerdo viene de clientes para los que m11 recomienda
Descuento**, una acción que m13 no evalúa (m13 valora solo la venta cruzada). Restringido a los
{c.miles(coh['clientes_misma_accion'])} clientes donde m11 recomienda venta cruzada o no hacer nada (las mismas opciones
de m13), **coinciden en {c.pct(coh['pct_coinciden_misma_accion'])}**.

Los dos modelos usan la misma definición del efecto y del costo; la diferencia que queda es de
**acción, horizonte y valor**. m13 valora un solo contacto con el valor a 12 meses del propio cliente; m11
valora una política que se repite cada mes y suma el valor futuro de devolver al cliente a un estado
mejor, con el gasto promedio de su grupo y estado. Por eso el mayor desacuerdo está en el estado
{desacuerdo.estado_m11} (coinciden {c.pct(desacuerdo.pct_coinciden)}).

## Hallazgos principales
- Con el presupuesto base, la mochila principal logra {c.miles(cb.optimo_valor)}, frente a {c.miles(cb.optimo_prob)} con la
  agrupación por decil de probabilidad, {c.miles(cb.regla_simple)} con la regla simple y {c.miles(cb.cota_individual)} de la
  referencia cliente por cliente.
- El valor máximo de la curva ({c.miles(mejor.optimo_valor)}) se alcanza con unos {c.miles(mejor.contactos)} contactos; más
  presupuesto no agrega valor porque no hay más clientes con valor positivo.
- La regla simple elige por probabilidad, no por valor: {texto_regla}. Lo que importa es el aumento
  por el valor del cliente.
- Agrupar por decil de valor deja grupos homogéneos y se acerca a la referencia cliente por cliente;
  agrupar por decil de probabilidad mezcla clientes de valor muy distinto.

## Limitaciones
- El valor depende por completo de los supuestos de efecto y costo de m11; el piloto (m14) debe
  medirlos. La sensibilidad al costo y al efecto está en el resumen de m11.
- El valor a 12 meses de m10 es de ventas, no de margen.
- Dentro de un grupo todos tienen el mismo valor esperado; los clientes concretos se eligen al azar
  dentro del grupo (semilla {c.SEMILLA}).
- La fase 2 de m14 (uplift) reemplazaría el aumento supuesto por estado con un efecto estimado por cliente.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
