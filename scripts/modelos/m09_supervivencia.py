"""
m09 - Tiempo hasta la segunda compra (análisis de supervivencia, lifelines)

Clientes: los mismos de m08 (consumidores finales, sin posible_cliente_previo; se reutiliza
m08_recompra.construir_base). Tiempo = días desde la primera visita hasta la segunda (una visita en
una fecha posterior). Evento = segunda visita; censura al final de la base para quien no volvió.

1. Kaplan-Meier por segmento y por negocio de entrada (prueba log-rank).
2. Modelo de Cox con las variables de la primera visita (las mismas de m08), ajustado con los clientes
   de entrenamiento de m08 y evaluado (índice de concordancia) con los de prueba. Se verifica el
   supuesto de riesgos proporcionales con la prueba de residuos de Schoenfeld.
3. En qué día después de la primera compra se concentra la recompra (para definir cuándo contactar).

Escribe:
    data/modelos/m09_supervivencia.parquet     tiempo, evento y riesgo relativo (Cox) por cliente
    salidas/modelos/m09_supervivencia/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m09_supervivencia.py
"""
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index

import comun as c
from m08_recompra import construir_base

NOMBRE = "m09_supervivencia"
MUESTRA_COX = 120_000        # clientes de entrenamiento para ajustar Cox (tiempo de cómputo)
MUESTRA_PH = 20_000          # clientes para la prueba de riesgos proporcionales
PENALIZADOR = 0.001
MIN_CASOS = 50               # una dummy con menos casos en la muestra de Cox se omite
HORIZONTE = 365
DIAS_TABLA = [7, 14, 30, 60, 90, 180, 365]
REGIONES = ["BOGOTÁ", "VALLE", "ANTIOQUIA", "CUNDINAMARCA", "ATLÁNTICO", "SANTANDER"]
CANALES = ["cluster_1", "cluster_2", "cluster_4"]
ABREV = {"REVESTIMIENTOS E INSTALACIÓN": "Revestimientos", "BAÑOS": "Baños", "COCINAS": "Cocinas",
         "PINTURAS E IMPERMEABILIZANTES": "Pinturas", "SERVICIOS": "Servicios"}


def construir() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Base de m08 con el tiempo hasta la segunda visita y la censura."""
    base, fecha_max = construir_base()
    base = base[base["muestra"] != "excluido_cliente_previo"].copy()
    con = c.conexion()
    con.register("primeras", base[["cliente", "fecha_primera_visita"]])
    segunda = con.execute("""
        SELECT p.cliente, MIN(t.fecha::DATE) AS fecha_segunda
        FROM primeras p JOIN trx t ON t.cliente = p.cliente AND t.fecha::DATE > p.fecha_primera_visita::DATE
        GROUP BY p.cliente
    """).df()
    base = base.merge(segunda, on="cliente", how="left")
    base["fecha_segunda"] = pd.to_datetime(base["fecha_segunda"])
    base["evento"] = base["fecha_segunda"].notna().astype(int)
    fin = np.where(base["evento"] == 1, base["fecha_segunda"], fecha_max)
    base["dias"] = (pd.to_datetime(fin) - base["fecha_primera_visita"]).dt.days.clip(lower=0) + 0.5
    base["negocio_entrada"] = base["negocio_visita"].map(ABREV)
    return base.sort_values("cliente").reset_index(drop=True), fecha_max


def covariables(base: pd.DataFrame) -> pd.DataFrame:
    """Variables de la primera visita (las de m08) en formato para Cox (dummies, categoría base omitida)."""
    x = pd.DataFrame(index=base.index)
    x["log_valor"] = base["log_valor"]
    x["log_productos"] = base["log_productos"]
    x["categorias"] = base["categorias"]
    # log(líneas) se omite: su correlación con log(productos) es 0,99 y Cox no converge con las dos
    x["negocios"] = base["negocios"]
    cat = pd.DataFrame({
        "negocio": base["negocio_entrada"],                       # base: Baños
        "canal": np.where(base["canal"].isin(CANALES), base["canal"], "otros"),
        "region": np.where(base["region"].isin(REGIONES), base["region"], "Otras"),
        "trimestre": "T" + ((base["fecha_primera_visita"].dt.month - 1) // 3 + 1).astype(str),
        "generacion": base["generacion"], "genero": base["genero"], "contactable": base["contactable"],
    }, index=base.index)
    bases = {"negocio": "Baños", "canal": "cluster_1", "region": "BOGOTÁ", "trimestre": "T1",
             "generacion": "Millennials", "genero": "Femenino", "contactable": "No"}
    for col, ref in bases.items():
        niveles = [n for n in sorted(cat[col].astype(str).unique()) if n != ref]
        for n in niveles:
            x[f"{col}_{n}"] = (cat[col].astype(str) == n).astype(float)
    return x


def curva_km(df: pd.DataFrame, grupo: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Kaplan-Meier por grupo: porcentaje acumulado que ya hizo la segunda compra en cada día."""
    filas, tabla = [], []
    ajustes = {}
    for g, d in df.groupby(grupo):
        km = KaplanMeierFitter().fit(d["dias"], d["evento"], label=str(g))
        ajustes[g] = km
        t = np.arange(0, HORIZONTE + 1)
        acum = 1 - km.survival_function_at_times(t).to_numpy()
        filas.append(pd.DataFrame({grupo: g, "dia": t, "pct_recompra_acumulada": acum * 100}))
        tabla.append({grupo: g, "clientes": len(d), **{f"dia_{x}": (1 - km.survival_function_at_times(x).iloc[0]) * 100
                                                       for x in DIAS_TABLA}})
    prueba = multivariate_logrank_test(df["dias"], df[grupo], df["evento"])
    return pd.concat(filas), pd.DataFrame(tabla), {"estadistico": prueba.test_statistic, "p": prueba.p_value}


def graficar_km(carpeta, curvas: pd.DataFrame, grupo: str, etiquetas: dict, nombre: str, titulo: str) -> None:
    """Curvas de recompra acumulada (1 - S(t))."""
    fig, ax = c.plt.subplots(figsize=(10, 6))
    colores = (c.SERIES + [c.ROJO, c.TENUE])
    for (g, d), color in zip(curvas.groupby(grupo), colores):
        ax.plot(d["dia"], d["pct_recompra_acumulada"], color=color, lw=2, label=etiquetas.get(g, g))
    for x in (30, 90):
        ax.axvline(x, color=c.EJE, ls="--", lw=1)
    ax.set_xlim(0, HORIZONTE)
    ax.set_xlabel("Días desde la primera compra")
    ax.set_ylabel("Clientes que ya hicieron la segunda compra (%)")
    ax.yaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.pct(v, 0)))
    ax.legend(loc="lower right", fontsize=8.5)
    c.guardar_figura(fig, carpeta, nombre, titulo, "Kaplan-Meier (1 − supervivencia); líneas punteadas en los días 30 y 90.")


def graficar_concentracion(carpeta, tiempos: pd.Series, riesgo: pd.DataFrame) -> None:
    """Distribución del día de la segunda compra (entre quienes recompran en un año) y riesgo semanal."""
    fig, (a1, a2) = c.plt.subplots(1, 2, figsize=(13, 5))
    a1.hist(tiempos, bins=np.arange(0, HORIZONTE + 7, 7), color=c.AZUL)
    for q, ls in ((0.25, ":"), (0.5, "--"), (0.75, ":")):
        a1.axvline(tiempos.quantile(q), color=c.NARANJA, ls=ls, lw=1.3)
    a1.set_xlabel("Día de la segunda compra (semanas)")
    a1.set_ylabel("Clientes")
    a1.grid(axis="x", visible=False)
    a2.plot(riesgo["semana"], riesgo["riesgo_semanal"] * 100, color=c.AZUL, lw=2, marker="o", ms=3)
    a2.set_xlabel("Semana desde la primera compra")
    a2.set_ylabel("Riesgo semanal de recompra (%)")
    a2.yaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.num(v, 1) + "%"))
    c.guardar_figura(fig, carpeta, "03_concentracion_recompra", "¿Cuándo se hace la segunda compra?",
                     "Izquierda: clientes que recompran en 365 días (naranja: cuartiles). Derecha: recompras de la semana / clientes en riesgo.")


def graficar_cox(carpeta, hr: pd.DataFrame) -> None:
    """Razones de riesgo del modelo de Cox con IC 95%."""
    t = hr.reindex(hr["log_hr"].abs().sort_values(ascending=False).index).head(18).sort_values("hr")
    fig, ax = c.plt.subplots(figsize=(10, 7))
    y = np.arange(len(t))
    ax.errorbar(t["hr"], y, xerr=[t["hr"] - t["ic_inf"], t["ic_sup"] - t["hr"]], fmt="o", color=c.AZUL,
                ecolor=c.EJE, elinewidth=2.5, ms=5)
    ax.axvline(1, color=c.TENUE, lw=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.num(v, 2)))
    ax.set_yticks(y, t.index, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Razón de riesgos (>1 = vuelve antes; escala log)")
    c.guardar_figura(fig, carpeta, "04_cox_razones_de_riesgo", "Modelo de Cox: qué acelera la segunda compra",
                     "18 variables con mayor efecto; IC 95%. Categorías base: Baños, cluster_1, Bogotá, T1, Millennials, Femenino, no contactable.")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    base, fecha_max = construir()
    print(f"  Clientes: {len(base):,} | con segunda compra: {base['evento'].mean():.1%}")

    # 1) Kaplan-Meier
    curvas_seg, tabla_seg, lr_seg = curva_km(base, "segmento")
    curvas_neg, tabla_neg, lr_neg = curva_km(base, "negocio_entrada")
    tabla_seg.insert(1, "nombre_segmento", tabla_seg["segmento"].map(c.NOMBRES_SEGMENTOS))
    tabla_seg.to_csv(carpeta / "km_por_segmento.csv", index=False, float_format=c.FORMATO_CSV)
    tabla_neg.to_csv(carpeta / "km_por_negocio_entrada.csv", index=False, float_format=c.FORMATO_CSV)
    pd.DataFrame([{"grupo": "segmento", **lr_seg}, {"grupo": "negocio_entrada", **lr_neg}]) \
        .to_csv(carpeta / "logrank.csv", index=False, float_format="%.6g")
    graficar_km(carpeta, curvas_seg, "segmento", {k: f"{k}. {v}" for k, v in c.NOMBRES_SEGMENTOS.items()},
                "01_km_por_segmento", "Tiempo hasta la segunda compra, por segmento")
    graficar_km(carpeta, curvas_neg, "negocio_entrada", {}, "02_km_por_negocio_entrada",
                "Tiempo hasta la segunda compra, por negocio de la primera compra")

    # 2) Cox con las variables de la primera visita
    x = covariables(base)
    datos = x.assign(dias=base["dias"], evento=base["evento"])
    entrena = datos[base["muestra"] == "entrenamiento"].sample(MUESTRA_COX, random_state=c.SEMILLA)
    # Se descartan dummies constantes o con menos de MIN_CASOS casos en la muestra (Cox no converge con ellas)
    dummies = [col for col in x.columns if set(np.unique(x[col])) <= {0.0, 1.0}]
    raras = [col for col in dummies if entrena[col].sum() < MIN_CASOS or entrena[col].sum() > len(entrena) - MIN_CASOS]
    if raras:
        print(f"  Se omiten por tener muy pocos casos: {raras}")
    x = x.drop(columns=raras)
    datos = datos.drop(columns=raras)
    entrena = entrena.drop(columns=raras)
    prueba = datos[base["muestra"] == "prueba"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cox = CoxPHFitter(penalizer=PENALIZADOR).fit(entrena, "dias", "evento")
    hr = pd.DataFrame({"hr": np.exp(cox.params_), "log_hr": cox.params_,
                       "ic_inf": np.exp(cox.confidence_intervals_.iloc[:, 0]),
                       "ic_sup": np.exp(cox.confidence_intervals_.iloc[:, 1]),
                       "p": cox.summary["p"]})
    hr.index.name = "variable"
    hr.to_csv(carpeta / "cox_razones_de_riesgo.csv", float_format="%.6g")
    c_tr = cox.concordance_index_
    c_te = concordance_index(prueba["dias"], -cox.predict_partial_hazard(prueba), prueba["evento"])
    print(f"  Cox: concordancia entrenamiento {c_tr:.3f} | prueba {c_te:.3f}")
    graficar_cox(carpeta, hr)

    # Supuesto de riesgos proporcionales (Schoenfeld, transformación de rango del tiempo)
    muestra_ph = entrena.sample(MUESTRA_PH, random_state=c.SEMILLA)
    pocas = [col for col in muestra_ph.columns if col not in ("dias", "evento")
             and set(np.unique(muestra_ph[col])) <= {0.0, 1.0} and muestra_ph[col].sum() < MIN_CASOS // 2]
    muestra_ph = muestra_ph.drop(columns=pocas)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cox_ph = CoxPHFitter(penalizer=PENALIZADOR).fit(muestra_ph, "dias", "evento")
        ph = proportional_hazard_test(cox_ph, muestra_ph, time_transform="rank").summary
    ph = ph.rename(columns={"test_statistic": "estadistico"})[["estadistico", "p"]]
    ph.index.name = "variable"
    ph.to_csv(carpeta / "prueba_riesgos_proporcionales.csv", float_format="%.6g")
    violan = ph[ph["p"] < 0.05].sort_values("estadistico", ascending=False)
    print(f"  Riesgos proporcionales: {len(violan)} de {len(ph)} variables con p < 0,05")

    # 3) Cuándo se concentra la recompra
    vuelven = base.loc[(base["evento"] == 1) & (base["dias"] <= HORIZONTE + 0.5), "dias"] - 0.5
    semanas = np.arange(0, HORIZONTE // 7 + 1)
    en_riesgo = np.array([(base["dias"] > s * 7).sum() for s in semanas])
    eventos = np.array([((base["evento"] == 1) & (base["dias"] > s * 7) & (base["dias"] <= (s + 1) * 7)).sum()
                        for s in semanas])
    riesgo = pd.DataFrame({"semana": semanas + 1, "en_riesgo": en_riesgo, "recompras": eventos,
                           "riesgo_semanal": eventos / np.maximum(en_riesgo, 1)})
    riesgo.to_csv(carpeta / "riesgo_semanal.csv", index=False, float_format=c.FORMATO_CSV)
    graficar_concentracion(carpeta, vuelven, riesgo)
    conc = {"recompran_365": len(vuelven), "mediana": vuelven.median(), "p25": vuelven.quantile(0.25),
            "p75": vuelven.quantile(0.75), "p90": vuelven.quantile(0.9),
            "pct_7": (vuelven <= 7).mean() * 100, "pct_30": (vuelven <= 30).mean() * 100,
            "pct_90": (vuelven <= 90).mean() * 100, "semana_pico": int(riesgo.loc[riesgo["riesgo_semanal"].idxmax(), "semana"])}
    pd.DataFrame([conc]).to_csv(carpeta / "concentracion_recompra.csv", index=False, float_format=c.FORMATO_CSV)
    pd.DataFrame([{"concordancia_entrenamiento": c_tr, "concordancia_prueba": c_te,
                   "variables_violan_ph": len(violan), "variables_total": len(ph),
                   "pct_con_segunda_compra": base["evento"].mean() * 100}]) \
        .to_csv(carpeta / "metricas.csv", index=False, float_format=c.FORMATO_CSV)
    print(conc)

    salida = base[["cliente", "segmento", "negocio_entrada", "fecha_primera_visita", "fecha_segunda", "dias", "evento", "muestra"]].copy()
    salida["dias"] = salida["dias"] - 0.5
    salida["riesgo_relativo_cox"] = cox.predict_partial_hazard(x).to_numpy()
    salida.to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(salida):,} clientes)")
    escribir_resumen(carpeta, base, tabla_seg, tabla_neg, lr_seg, lr_neg, hr, c_tr, c_te, ph, violan, conc, fecha_max)


def escribir_resumen(carpeta, base, tabla_seg, tabla_neg, lr_seg, lr_neg, hr, c_tr, c_te, ph, violan, conc, fecha_max) -> None:
    """resumen.md del modelo."""
    cols = [f"dia_{d}" for d in (7, 30, 90, 180, 365)]
    filas_s = "\n".join(f"| {r.segmento}. {r.nombre_segmento} | {c.miles(r.clientes)} | "
                        + " | ".join(c.pct(getattr(r, k)) for k in cols) + " |" for r in tabla_seg.itertuples())
    filas_n = "\n".join(f"| {r.negocio_entrada} | {c.miles(r.clientes)} | "
                        + " | ".join(c.pct(getattr(r, k)) for k in cols) + " |"
                        for r in tabla_neg.sort_values("dia_90", ascending=False).itertuples())
    top_hr = hr.reindex(hr["log_hr"].abs().sort_values(ascending=False).index).head(8)
    filas_h = "\n".join(f"| `{i}` | {c.num(r.hr, 2)} | [{c.num(r.ic_inf, 2)}; {c.num(r.ic_sup, 2)}] |" for i, r in top_hr.iterrows())
    filas_ph = "\n".join(f"| `{i}` | {c.num(r.estadistico, 1)} | {r.p:.2g} |" for i, r in violan.head(8).iterrows())
    neg = tabla_neg.set_index("negocio_entrada")["dia_90"].sort_values(ascending=False)
    hr_valor = hr.loc["log_valor", "hr"]
    if violan.empty:
        lectura_ph = "Ninguna variable viola el supuesto: las razones de riesgo son constantes en el tiempo."
    elif len(violan) <= max(3, len(ph) // 5):
        lectura_ph = (f"El supuesto se cumple para {len(ph) - len(violan)} de {len(ph)} variables. Solo "
                      f"{', '.join(f'`{v}`' for v in violan.index)} lo violan: para ellas la razón de riesgos es un "
                      "promedio en el tiempo (su efecto cambia entre las primeras semanas y después).")
        principales = [v for v in violan.index if v in ("log_valor", "log_productos") or v.startswith("negocio_")]
        if not principales:
            lectura_ph += (" Las variables principales (valor, productos y negocio de la primera visita) sí cumplen "
                           "el supuesto.")
    else:
        lectura_ph = ("Varias variables violan el supuesto; las razones de riesgo deben leerse como un efecto "
                      "promedio en el tiempo, y convendría estratificar por esas variables o agregar interacciones con el tiempo.")
    texto = f"""
# m09 - Tiempo hasta la segunda compra (supervivencia)

## Qué hace
Modela cuánto tarda un cliente en hacer su segunda compra (`lifelines`):
- **Evento:** segunda visita (una visita en una fecha posterior a la primera).
- **Censura:** el cliente que no volvió se censura al final de la base ({fecha_max:%Y-%m-%d}), así que se usan
  todos los clientes, incluidos los de los últimos 90 días que m08 no podía evaluar.
- **Kaplan-Meier** por segmento y por negocio de entrada (prueba log-rank).
- **Cox** con las variables de la primera visita (las de m08), ajustado con {c.miles(MUESTRA_COX)} clientes de
  entrenamiento de m08 y evaluado con los de prueba; prueba de riesgos proporcionales (residuos de
  Schoenfeld) con {c.miles(MUESTRA_PH)} clientes.

## Datos usados
Base de m08 sin clientes previos: {c.miles(len(base))} clientes; {c.pct(base['evento'].mean() * 100)} hizo una segunda compra antes del
cierre de la base.

## Métricas
### Kaplan-Meier: % que ya hizo la segunda compra
| Segmento | Clientes | Día 7 | Día 30 | Día 90 | Día 180 | Día 365 |
|---|---|---|---|---|---|---|
{filas_s}

| Negocio de entrada | Clientes | Día 7 | Día 30 | Día 90 | Día 180 | Día 365 |
|---|---|---|---|---|---|---|
{filas_n}

Log-rank: por segmento χ² = {c.miles(lr_seg['estadistico'])} (p = {lr_seg['p']:.2g}); por negocio de entrada
χ² = {c.miles(lr_neg['estadistico'])} (p = {lr_neg['p']:.2g}).

### Cox
Índice de concordancia: **{c.num(c_tr, 3)}** en entrenamiento y **{c.num(c_te, 3)}** en prueba (equivalente a un AUC para
datos con censura). Razones de riesgo más grandes (> 1 = vuelve antes):

| Variable | Razón de riesgos | IC 95% |
|---|---|---|
{filas_h}

**Supuesto de riesgos proporcionales:** {len(violan)} de {len(ph)} variables tienen p < 0,05 en la prueba de
Schoenfeld. Las más fuertes:

| Variable | Estadístico | p |
|---|---|---|
{filas_ph}

{lectura_ph}

## Cuándo se concentra la recompra
Entre los {c.miles(conc['recompran_365'])} clientes que vuelven en el primer año:
- **{c.pct(conc['pct_7'])} vuelve en la primera semana, {c.pct(conc['pct_30'])} en 30 días y {c.pct(conc['pct_90'])} en 90 días.**
- Mediana: **día {c.miles(conc['mediana'])}**; cuartiles: día {c.miles(conc['p25'])} y día {c.miles(conc['p75'])}; percentil 90: día
  {c.miles(conc['p90'])}.
- El riesgo semanal de recompra es máximo en la semana {conc['semana_pico']} y cae rápido después.

**Cuándo contactar:** la recompra se concentra en las primeras semanas, así que el contacto de venta
cruzada debe hacerse **entre el día 3 y el día {c.miles(max(14, conc['mediana']))} después de la primera compra**, antes
de la mediana. Un segundo recordatorio cerca del día {c.miles(conc['p75'])} cubre la cola. Esperar hasta el día 90
pierde la mayor parte de la ventana.

## Hallazgos principales
- Por negocio de entrada, a 90 días ya volvió {c.pct(neg.iloc[0])} de quienes empezaron por {neg.index[0]} frente a
  {c.pct(neg.iloc[-1])} de quienes empezaron por {neg.index[-1]}.
- Un mayor valor de la primera visita acelera la segunda compra (razón de riesgos {c.num(hr_valor, 2)} por unidad
  de log del valor), igual que en m08.
- Las curvas por segmento son descriptivas: el segmento se calculó con todo el historial y "Proyecto
  recurrente" vuelve casi siempre por definición.

## Limitaciones
- La censura al final de la base supone que quien no ha vuelto todavía podría volver (censura no
  informativa); clientes previos no marcados pueden adelantar la "segunda" compra.
- Cox se ajusta con una muestra de {c.miles(MUESTRA_COX)} clientes por tiempo de cómputo.
- Fechas desplazadas por Corona; los días entre compras sí son reales.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
