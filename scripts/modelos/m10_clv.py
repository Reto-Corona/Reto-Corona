"""
m10 - Valor de vida del cliente (CLV) con BG/NBD y Gamma-Gamma

Datos: visitas (cliente + fecha + punto de venta) de consumidores finales segmentados. Frecuencia,
recencia y antigüedad en días (librería lifetimes; las visitas del mismo día cuentan como una
compra, convención de lifetimes con freq='D').

1. Validación: se ajusta BG/NBD en un periodo de calibración (hasta 365 días antes del final de la
   base) y se compara el número de compras predicho con el real en los 365 días siguientes.
2. Modelo final: BG/NBD + Gamma-Gamma con toda la base; por cliente se calcula la probabilidad de
   seguir activo, las compras esperadas en 12 meses y el valor esperado en 12 meses
   (compras esperadas x valor promedio esperado por compra, sin descuento financiero).

Se ajusta con clientes sin la marca posible_cliente_previo (su antigüedad real es mayor que la
observada) y se predice para todos.

Escribe:
    data/modelos/m10_clv.parquet               probabilidad de seguir activo, compras y valor a 12 meses
    salidas/modelos/m10_clv/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m10_clv.py
"""
import warnings

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
from scipy.stats import spearmanr

import comun as c

NOMBRE = "m10_clv"
HORIZONTE_DIAS = 365
PENALIZADOR = 0.001
# Gamma-Gamma se ajusta con el valor en miles: con los valores crudos el optimizador de lifetimes
# converge a q < 1 (media infinita y valores esperados negativos); en miles y sin penalizador converge
# bien (q > 1). El penalizador encoge q hacia cero, por eso Gamma-Gamma no se penaliza.
ESCALA_VALOR = 1_000


def cargar_visitas() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Visitas con su valor y ficha de los clientes (segmento, contactabilidad, cliente previo)."""
    con = c.conexion()
    visitas = con.execute("""
        SELECT t.cliente, t.fecha::DATE AS fecha, t.punto_de_venta, SUM(t.valor_corregido) AS valor
        FROM trx t JOIN features f USING (cliente) JOIN segmentos s USING (cliente)
        WHERE f.tipo_cliente = 'Consumidor final'
        GROUP BY ALL ORDER BY ALL
    """).df()
    visitas["fecha"] = pd.to_datetime(visitas["fecha"])
    clientes = c.cargar_clientes()[["cliente", "segmento", "nombre_segmento", "es_contactable", "posible_cliente_previo"]]
    return visitas, clientes, visitas["fecha"].max()


def ajustar_bgnbd(s: pd.DataFrame) -> BetaGeoFitter:
    """BG/NBD sobre frecuencia, recencia y antigüedad (días)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BetaGeoFitter(penalizer_coef=PENALIZADOR).fit(s["frequency"], s["recency"], s["T"])


def validar(carpeta, visitas, clientes, fin) -> dict:
    """Calibración / prueba: compras predichas frente a reales en los 365 días de prueba."""
    fin_cal = fin - pd.Timedelta(days=HORIZONTE_DIAS)
    ajuste = visitas[visitas["cliente"].isin(clientes.loc[clientes["posible_cliente_previo"] == 0, "cliente"])]
    ch = calibration_and_holdout_data(ajuste, "cliente", "fecha", calibration_period_end=fin_cal,
                                      observation_period_end=fin, freq="D")
    bg = ajustar_bgnbd(ch.rename(columns={"frequency_cal": "frequency", "recency_cal": "recency", "T_cal": "T"}))
    ch["predichas"] = bg.conditional_expected_number_of_purchases_up_to_time(
        ch["duration_holdout"], ch["frequency_cal"], ch["recency_cal"], ch["T_cal"])
    ch = ch.join(clientes.set_index("cliente")[["segmento"]])
    ch["grupo_frecuencia"] = ch["frequency_cal"].clip(upper=7).astype(int)
    por_frec = ch.groupby("grupo_frecuencia").agg(clientes=("predichas", "size"), reales=("frequency_holdout", "mean"),
                                                  predichas=("predichas", "mean")).reset_index()
    por_seg = ch.groupby("segmento").agg(clientes=("predichas", "size"), reales=("frequency_holdout", "sum"),
                                         predichas=("predichas", "sum")).reset_index()
    por_seg.insert(1, "nombre_segmento", por_seg["segmento"].map(c.NOMBRES_SEGMENTOS))
    por_frec.to_csv(carpeta / "validacion_por_frecuencia.csv", index=False, float_format=c.FORMATO_CSV)
    por_seg.to_csv(carpeta / "validacion_por_segmento.csv", index=False, float_format=c.FORMATO_CSV)

    # Gráfica predicho frente a real
    fig, (a1, a2) = c.plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(por_frec))
    a1.bar(x - 0.2, por_frec["reales"], 0.38, color=c.AZUL, label="Real")
    a1.bar(x + 0.2, por_frec["predichas"], 0.38, color=c.NARANJA, label="Predicho")
    a1.set_xticks(x, [f"{v}" if v < 7 else "7+" for v in por_frec["grupo_frecuencia"]])
    a1.set_xlabel("Compras repetidas en la calibración")
    a1.set_ylabel("Compras promedio en los 365 días de prueba")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="upper left")
    m = max(por_seg["reales"].max(), por_seg["predichas"].max()) / 1000 * 1.1
    a2.scatter(por_seg["reales"] / 1000, por_seg["predichas"] / 1000, color=c.AZUL, s=40, zorder=3)
    a2.plot([0, m], [0, m], color=c.EJE, ls="--", lw=1)
    for r in por_seg.itertuples():
        a2.annotate(r.nombre_segmento, (r.reales / 1000, r.predichas / 1000), xytext=(4, 3),
                    textcoords="offset points", fontsize=7.5, color=c.TINTA_2)
    a2.set(xlim=(0, m), ylim=(0, m), xlabel="Compras reales (miles)", ylabel="Compras predichas (miles)")
    c.guardar_figura(fig, carpeta, "01_validacion_predicho_real", "Validación del BG/NBD: predicho frente a real",
                     f"Calibración hasta {fin_cal:%Y-%m-%d}; prueba de {HORIZONTE_DIAS} días. Izquierda: por frecuencia; derecha: por segmento.")
    real, pred = ch["frequency_holdout"], ch["predichas"]
    from sklearn.metrics import roc_auc_score
    return {
        "auc_vuelve_a_comprar": roc_auc_score((real > 0).astype(int), pred),
        "pct_compran_en_prueba": (real > 0).mean() * 100,
        "fin_calibracion": fin_cal, "clientes_calibracion": len(ch),
        "compras_reales": real.sum(), "compras_predichas": pred.sum(),
        "error_total_pct": (pred.sum() / real.sum() - 1) * 100,
        "mae_cliente": (pred - real).abs().mean(),
        "correlacion_pearson": np.corrcoef(pred, real)[0, 1],
        "spearman": spearmanr(pred, real).statistic,
        "params_bgnbd_calibracion": bg.params_.to_dict(),
    }


def modelo_final(visitas, clientes, fin) -> tuple[pd.DataFrame, BetaGeoFitter, GammaGammaFitter, float]:
    """BG/NBD + Gamma-Gamma con toda la base; predicción a 12 meses para cada cliente."""
    s = summary_data_from_transaction_data(visitas, "cliente", "fecha", monetary_value_col="valor",
                                           observation_period_end=fin, freq="D")
    previos = set(clientes.loc[clientes["posible_cliente_previo"] == 1, "cliente"])
    ajuste = s[~s.index.isin(previos)]
    bg = ajustar_bgnbd(ajuste)
    vuelven = ajuste[(ajuste["frequency"] > 0) & (ajuste["monetary_value"] > 0)]
    corr = np.corrcoef(vuelven["frequency"], vuelven["monetary_value"])[0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gg = GammaGammaFitter(penalizer_coef=0.0).fit(vuelven["frequency"],
                                                              vuelven["monetary_value"] / ESCALA_VALOR)
    if gg.params_["q"] <= 1:
        raise ValueError(f"Gamma-Gamma con q = {gg.params_['q']:.3f} <= 1: el valor esperado no está definido.")
    s["prob_activo"] = bg.conditional_probability_alive(s["frequency"], s["recency"], s["T"])
    s["compras_esperadas_12m"] = bg.conditional_expected_number_of_purchases_up_to_time(
        HORIZONTE_DIAS, s["frequency"], s["recency"], s["T"])
    monetario = s["monetary_value"].where(s["frequency"] > 0, 0).clip(lower=0)
    s["valor_promedio_esperado"] = gg.conditional_expected_average_profit(
        s["frequency"], monetario / ESCALA_VALOR) * ESCALA_VALOR
    s["valor_esperado_12m"] = s["compras_esperadas_12m"] * s["valor_promedio_esperado"]
    s = s.reset_index().merge(clientes, on="cliente", how="left")
    return s, bg, gg, corr


def graficar_segmentos(carpeta, seg: pd.DataFrame) -> None:
    """Valor esperado a 12 meses por segmento: total y promedio por cliente."""
    t = seg.sort_values("segmento")
    nombres = [n.replace(" en una", "\nen una") for n in t["nombre_segmento"]]
    fig, (a1, a2) = c.plt.subplots(1, 2, figsize=(13, 5))
    b1 = a1.bar(nombres, t["valor_total_12m"] / 1e9, color=c.AZUL, width=0.65)
    c.rotular_barras(a1, b1, [c.num(v / 1e9, 1) for v in t["valor_total_12m"]])
    a1.set_ylabel("Valor esperado total (miles de millones, unidades de la base)")
    b2 = a2.bar(nombres, t["valor_medio_12m"] / 1000, color=c.NARANJA, width=0.65)
    c.rotular_barras(a2, b2, [c.miles(v / 1000) for v in t["valor_medio_12m"]])
    a2.set_ylabel("Valor esperado por cliente (miles)")
    for a in (a1, a2):
        a.grid(axis="x", visible=False)
        a.tick_params(axis="x", labelsize=8)
    c.guardar_figura(fig, carpeta, "02_valor_esperado_por_segmento", "Valor esperado en los próximos 12 meses",
                     "BG/NBD (compras esperadas) x Gamma-Gamma (valor por compra), en unidades de la base.")


def graficar_prob_activo(carpeta, s: pd.DataFrame) -> None:
    """Distribución de la probabilidad de seguir activo por segmento."""
    fig, ax = c.plt.subplots(figsize=(10, 5))
    datos = [s.loc[s["segmento"] == k, "prob_activo"] * 100 for k in c.NOMBRES_SEGMENTOS]
    partes = ax.boxplot(datos, tick_labels=[n.replace(" en una", "\nen una") for n in c.NOMBRES_SEGMENTOS.values()],
                        patch_artist=True, showfliers=False, widths=0.55)
    for caja in partes["boxes"]:
        caja.set(facecolor=c.AZUL, alpha=0.75, edgecolor=c.AZUL)
    for med in partes["medians"]:
        med.set(color=c.NARANJA, lw=2)
    ax.tick_params(axis="x", labelsize=8)
    ax.yaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.pct(v, 0)))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Probabilidad de seguir activo")
    c.guardar_figura(fig, carpeta, "03_prob_activo_por_segmento", "Probabilidad de seguir activo al cierre de la base",
                     "BG/NBD; la caja va del percentil 25 al 75 y la línea naranja es la mediana.")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    visitas, clientes, fin = cargar_visitas()
    print(f"  Visitas: {len(visitas):,} | clientes: {visitas['cliente'].nunique():,} | fin: {fin:%Y-%m-%d}")

    val = validar(carpeta, visitas, clientes, fin)
    print(f"  Validación: reales {val['compras_reales']:,.0f} | predichas {val['compras_predichas']:,.0f} | "
          f"error {val['error_total_pct']:.1f}% | MAE {val['mae_cliente']:.3f} | Spearman {val['spearman']:.3f}")

    s, bg, gg, corr = modelo_final(visitas, clientes, fin)
    params = pd.DataFrame([{"modelo": "BG/NBD", **bg.params_.to_dict()}, {"modelo": "Gamma-Gamma", **gg.params_.to_dict()}])
    params.to_csv(carpeta / "parametros.csv", index=False, float_format=c.FORMATO_CSV)
    pd.DataFrame([{k: v for k, v in val.items() if k != "params_bgnbd_calibracion"}]) \
        .to_csv(carpeta / "metricas_validacion.csv", index=False, float_format=c.FORMATO_CSV)

    cols = ["cliente", "segmento", "nombre_segmento", "es_contactable", "posible_cliente_previo", "frequency",
            "recency", "T", "monetary_value", "prob_activo", "compras_esperadas_12m",
            "valor_promedio_esperado", "valor_esperado_12m"]
    s[cols].to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(s):,} clientes)")

    seg = s.groupby(["segmento", "nombre_segmento"]).agg(
        clientes=("cliente", "size"), prob_activo_media=("prob_activo", "mean"),
        compras_esperadas_12m=("compras_esperadas_12m", "sum"), valor_total_12m=("valor_esperado_12m", "sum"),
        valor_medio_12m=("valor_esperado_12m", "mean"), valor_mediano_12m=("valor_esperado_12m", "median")).reset_index()
    seg["pct_valor"] = seg["valor_total_12m"] / seg["valor_total_12m"].sum() * 100
    seg.to_csv(carpeta / "resumen_por_segmento.csv", index=False, float_format=c.FORMATO_CSV)
    print(seg.to_string(index=False))
    graficar_segmentos(carpeta, seg)
    graficar_prob_activo(carpeta, s)
    escribir_resumen(carpeta, val, s, seg, bg, gg, corr, visitas)


def escribir_resumen(carpeta, val, s, seg, bg, gg, corr, visitas) -> None:
    """resumen.md del modelo."""
    filas = "\n".join(
        f"| {x.segmento}. {x.nombre_segmento} | {c.miles(x.clientes)} | {c.pct(x.prob_activo_media * 100)} | "
        f"{c.num(x.compras_esperadas_12m / x.clientes, 2)} | {c.miles(x.valor_medio_12m)} | {c.pct(x.pct_valor)} |"
        for x in seg.itertuples())
    pb = bg.params_
    pg = gg.params_
    top = seg.sort_values("pct_valor", ascending=False).iloc[0]
    concentracion = s["valor_esperado_12m"].sort_values(ascending=False)
    top10 = concentracion.head(int(len(concentracion) * 0.1)).sum() / concentracion.sum() * 100
    una = s["frequency"] == 0
    pct_una = una.mean() * 100
    prob_una = s.loc[una, "prob_activo"].mean() * 100
    repiten = s[~una].groupby("nombre_segmento")["prob_activo"].mean().sort_values()
    texto_repiten = ", ".join(f"{n} {c.pct(v * 100)}" for n, v in repiten.items())
    texto = f"""
# m10 - Valor de vida del cliente (BG/NBD + Gamma-Gamma)

## Qué hace
- **BG/NBD** estima, con la frecuencia (compras repetidas), la recencia (días entre la primera y la
  última compra) y la antigüedad (días desde la primera compra hasta el cierre), la probabilidad de
  que el cliente siga activo y cuántas compras hará en los próximos 12 meses.
- **Gamma-Gamma** estima el valor promedio esperado de cada compra a partir del valor de las compras
  repetidas.
- Valor esperado en 12 meses = compras esperadas x valor promedio esperado (sin descuento financiero).

Librería: `lifetimes` {__import__('lifetimes').__version__}. Funciona con pandas {pd.__version__} y numpy {np.__version__};
solo su módulo de datos de ejemplo (`lifetimes.datasets`) falla por depender de `pkg_resources`, y no se usa.

## Datos usados
- {c.miles(len(visitas))} visitas (cliente + fecha + punto de venta) de {c.miles(visitas['cliente'].nunique())}
  consumidores finales segmentados (`{c.SEGMENTACION}`). Visitas del mismo día cuentan como una compra.
- El ajuste excluye a los clientes con `posible_cliente_previo` (su antigüedad real es mayor que la
  observada); la predicción se hace para todos.
- Valores en las unidades de la base (valor corregido).

## Métricas (validación)
Calibración hasta {val['fin_calibracion']:%Y-%m-%d} ({c.miles(val['clientes_calibracion'])} clientes) y prueba de
{HORIZONTE_DIAS} días:
- Compras reales en la prueba: {c.miles(val['compras_reales'])}; predichas: {c.miles(val['compras_predichas'])}
  (error total {c.num(val['error_total_pct'], 1)}%).
- Por cliente: MAE {c.num(val['mae_cliente'], 3)} compras; correlación de Pearson {c.num(val['correlacion_pearson'], 3)};
  Spearman {c.num(val['spearman'], 3)} (bajo porque {c.pct(100 - val['pct_compran_en_prueba'])} de los clientes no compra en la
  prueba y quedan empatados en cero). Para ordenar quién vuelve a comprar, el AUC de las compras
  predichas es **{c.num(val['auc_vuelve_a_comprar'], 3)}**.
- Gráfica `01_validacion_predicho_real.png` (por frecuencia en la calibración y por segmento).

Parámetros finales: BG/NBD r = {c.num(pb['r'], 3)}, α = {c.num(pb['alpha'], 1)}, a = {c.num(pb['a'], 3)},
b = {c.num(pb['b'], 3)}; Gamma-Gamma (valor en miles) p = {c.num(pg['p'], 3)}, q = {c.num(pg['q'], 3)},
v = {c.num(pg['v'], 1)}. Gamma-Gamma se ajusta con el valor en miles y sin penalizador porque con los
valores crudos (o penalizado) el optimizador converge a q < 1: media infinita y valores esperados negativos.
Supuesto de Gamma-Gamma (independencia entre frecuencia y valor por compra): correlación =
{c.num(corr, 3)} entre los clientes que vuelven.

## Resultados por segmento
| Segmento | Clientes | Prob. activo (media) | Compras esperadas 12m por cliente | Valor esperado 12m por cliente | % del valor total |
|---|---|---|---|---|---|
{filas}

## Hallazgos principales
- **{top.nombre_segmento}** concentra {c.pct(top.pct_valor)} del valor esperado de los próximos 12 meses.
- El 10% de clientes con mayor valor esperado reúne {c.pct(top10)} del valor total: el CLV está muy concentrado.
- **Cuidado al leer la probabilidad de seguir activo:** BG/NBD asigna {c.pct(prob_una, 0)} a los clientes que
  compraron un solo día ({c.pct(pct_una)} del total), porque nunca tuvieron una compra repetida tras la cual
  "abandonar". Por eso los segmentos de una sola visita parecen muy "activos". Entre los que sí repiten,
  la probabilidad media de seguir activo es: {texto_repiten}.
- Para priorizar, la métrica útil es **compras esperadas y valor esperado a 12 meses**, no la
  probabilidad de seguir activo: los segmentos de una sola visita tienen valor esperado bajo y su valor
  futuro depende de lograr la segunda compra (m08, m14).

## Limitaciones
- BG/NBD supone que el cliente compra a una tasa constante mientras está activo y que, después de cada
  compra, puede "abandonar" para siempre; los proyectos de remodelación son más episódicos.
- Los clientes con una sola compra tienen frecuencia 0: su valor por compra es el promedio de la población.
- Ventana corta (31 meses, fechas desplazadas) y validación con un solo corte temporal.
- Sin descuento financiero ni margen: es valor de ventas esperado, en unidades de la base.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
