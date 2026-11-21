"""
m07 - Elasticidad precio

Paso 1 (filtro): verifica si un mismo producto tiene precios distintos en la base (coeficiente de
variación del precio por producto). Si casi no hay variación, se detiene, lo reporta en
resumen.md y no estima nada.

Paso 2: regresión log(cantidad) ~ log(precio) con efectos fijos de producto y de mes, usando
'precio' (no 'valor') y solo líneas con cantidad > 0 y precio > 0. Los efectos fijos se eliminan
con proyecciones alternadas (se resta la media por producto y por mes hasta converger) y los
errores estándar se agrupan por producto. Se estima la elasticidad total, por negocio, por
subcategoría (las que tienen datos suficientes) y por segmento, con intervalos de confianza al 95%.

Como verificación se repite a nivel producto x mes (cantidad total frente a precio promedio), que
no mezcla descuentos por volumen dentro de una misma compra.

Escribe:
    data/modelos/m07_elasticidad.parquet           elasticidades estimadas
    salidas/modelos/m07_elasticidad/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m07_elasticidad.py
"""
import numpy as np
import pandas as pd
from scipy import stats

import comun as c

NOMBRE = "m07_elasticidad"
MIN_LINEAS_PRODUCTO = 30
# Se considera que "casi no hay variación" si el CV mediano por producto es menor que 1% y menos
# del 10% de las líneas está en productos con CV > 5%.
CV_MEDIANO_MIN = 0.01
PCT_LINEAS_CV5_MIN = 0.10
MIN_LINEAS_SUBCATEGORIA = 5_000
MIN_PRODUCTOS_SUBCATEGORIA = 10
TOLERANCIA_FE = 1e-8
ABREV_NEGOCIO = {
    "REVESTIMIENTOS E INSTALACIÓN": "Revestimientos", "BAÑOS": "Baños", "COCINAS": "Cocinas",
    "PINTURAS E IMPERMEABILIZANTES": "Pinturas", "SERVICIOS": "Servicios",
}


def cargar_lineas() -> pd.DataFrame:
    """Líneas con cantidad > 0 y precio > 0, con su segmento (0 si no es consumidor final segmentado)."""
    con = c.conexion()
    df = con.execute("""
        SELECT t.producto, t.subcategoria, t.negocio, strftime(t.fecha, '%Y-%m') AS mes,
               t.cantidad, t.precio, COALESCE(s.segmento_kmeans, 0) AS segmento
        FROM trx t LEFT JOIN segmentos s USING (cliente)
        WHERE t.cantidad > 0 AND t.precio > 0
        ORDER BY ALL
    """).df()
    df["log_q"] = np.log(df["cantidad"])
    df["log_p"] = np.log(df["precio"])
    return df


def variacion_precio(df: pd.DataFrame) -> pd.DataFrame:
    """Coeficiente de variación del precio por producto (productos con al menos MIN_LINEAS_PRODUCTO líneas)."""
    g = df.groupby("producto")["precio"]
    t = pd.DataFrame({"lineas": g.size(), "precios_distintos": g.nunique(),
                      "cv": g.std() / g.mean(), "precio_medio": g.mean()})
    t = t.merge(df.drop_duplicates("producto").set_index("producto")[["negocio", "subcategoria"]],
                left_index=True, right_index=True)
    return t[t["lineas"] >= MIN_LINEAS_PRODUCTO].reset_index()


def quitar_efectos_fijos(d: pd.DataFrame, columnas: list, grupos: list) -> pd.DataFrame:
    """Proyecciones alternadas: resta medias por cada grupo hasta que no cambie (efectos fijos de 2 vías)."""
    x = d[columnas].astype(float).copy()
    for _ in range(200):
        anterior = x.copy()
        for gcol in grupos:
            x = x - x.groupby(d[gcol].values).transform("mean")
        if np.max(np.abs((x - anterior).to_numpy())) < TOLERANCIA_FE:
            break
    return x


def elasticidad(d: pd.DataFrame, efectos=("producto", "mes"), cluster="producto") -> dict:
    """Pendiente de log_q sobre log_p con efectos fijos y error estándar agrupado."""
    if len(d) < 50 or d["producto"].nunique() < 2:
        return {}
    xt = quitar_efectos_fijos(d, ["log_q", "log_p"], list(efectos))
    y, x = xt["log_q"].to_numpy(), xt["log_p"].to_numpy()
    sxx = (x ** 2).sum()
    if sxx <= 0:
        return {}
    beta = (x * y).sum() / sxx
    e = y - beta * x
    puntaje = pd.Series(x * e).groupby(d[cluster].values).sum()
    G = len(puntaje)
    var = (puntaje ** 2).sum() / sxx ** 2 * G / max(G - 1, 1)
    se = np.sqrt(var)
    tcrit = stats.t.ppf(0.975, max(G - 1, 1))
    return {"elasticidad": beta, "error_estandar": se, "ic95_inf": beta - tcrit * se, "ic95_sup": beta + tcrit * se,
            "lineas": len(d), "productos": d["producto"].nunique(), "clusters": G,
            "varianza_logp_dentro": (x ** 2).mean()}


def graficar(carpeta, t: pd.DataFrame, nombre: str, titulo: str, subtitulo: str, etiqueta: str = "grupo") -> None:
    """Elasticidad con intervalo de confianza al 95%."""
    t = t.sort_values("elasticidad", ascending=False)
    fig, ax = c.plt.subplots(figsize=(10, max(3.5, 0.42 * len(t) + 1.8)))
    y = np.arange(len(t))
    ax.errorbar(t["elasticidad"], y, xerr=[t["elasticidad"] - t["ic95_inf"], t["ic95_sup"] - t["elasticidad"]],
                fmt="o", color=c.AZUL, ecolor=c.EJE, elinewidth=2.5, capsize=0, ms=6)
    for yi, (v, n) in enumerate(zip(t["elasticidad"], t["lineas"])):
        ax.annotate(f"{c.num(v, 2)}  ({c.miles(n)} líneas)", (v, yi), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=c.TINTA_2)
    ax.axvline(0, color=c.TENUE, lw=0.8)
    ax.axvline(-1, color=c.NARANJA, lw=1, ls="--")
    ax.set_yticks(y, t[etiqueta])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Elasticidad precio (cambio % en cantidad por 1% de precio)")
    c.guardar_figura(fig, carpeta, nombre, titulo, subtitulo,
                     nota=c.FUENTE + " Línea naranja: elasticidad -1 (demanda unitaria).")


def graficar_cv(carpeta, var: pd.DataFrame) -> None:
    """Histograma del coeficiente de variación del precio por producto."""
    fig, ax = c.plt.subplots(figsize=(9, 5))
    ax.hist(var["cv"].clip(upper=0.5) * 100, bins=50, color=c.AZUL)
    ax.axvline(var["cv"].median() * 100, color=c.NARANJA, ls="--")
    ax.annotate(f"Mediana {c.pct(var['cv'].median() * 100)}", (var["cv"].median() * 100, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", color=c.NARANJA, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Coeficiente de variación del precio dentro del producto (%; recortado en 50%)")
    ax.set_ylabel("Productos")
    c.guardar_figura(fig, carpeta, "01_variacion_precio", "¿Cambia el precio de un mismo producto?",
                     f"{c.miles(len(var))} productos con al menos {MIN_LINEAS_PRODUCTO} líneas.")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    df = cargar_lineas()
    var = variacion_precio(df)
    var.to_csv(carpeta / "variacion_precio_por_producto.csv", index=False, float_format=c.FORMATO_CSV)
    cv_mediano = var["cv"].median()
    pct_lineas_cv5 = var.loc[var["cv"] > 0.05, "lineas"].sum() / var["lineas"].sum()
    pct_varios = (var["precios_distintos"] > 1).mean()
    print(f"  Productos: {len(var):,} | CV mediano: {cv_mediano:.3f} | líneas en productos con CV>5%: {pct_lineas_cv5:.1%}")
    graficar_cv(carpeta, var)
    por_neg_cv = var.groupby("negocio").agg(productos=("producto", "size"), cv_mediano=("cv", "median")).reset_index()
    por_neg_cv.to_csv(carpeta / "variacion_precio_por_negocio.csv", index=False, float_format=c.FORMATO_CSV)

    if cv_mediano < CV_MEDIANO_MIN and pct_lineas_cv5 < PCT_LINEAS_CV5_MIN:
        print("  Casi no hay variación de precio: no se estima la elasticidad.")
        escribir_resumen_sin_variacion(carpeta, var, cv_mediano, pct_lineas_cv5)
        return

    # Elasticidad total y por grupos (línea)
    filas = [{"nivel": "Total", "grupo": "Total", **elasticidad(df)}]
    for neg, d in df.groupby("negocio"):
        filas.append({"nivel": "Negocio", "grupo": ABREV_NEGOCIO.get(neg, neg), **elasticidad(d)})
    for seg, d in df[df["segmento"] > 0].groupby("segmento"):
        filas.append({"nivel": "Segmento", "grupo": f"{seg}. {c.NOMBRES_SEGMENTOS[seg]}", **elasticidad(d)})
    neg_sub = df.drop_duplicates("subcategoria").set_index("subcategoria")["negocio"]
    for sub, d in df.groupby("subcategoria"):
        if len(d) >= MIN_LINEAS_SUBCATEGORIA and d["producto"].nunique() >= MIN_PRODUCTOS_SUBCATEGORIA:
            etiqueta = sub.replace("SUBCAT_000", "S") + f" ({ABREV_NEGOCIO.get(neg_sub[sub], neg_sub[sub])})"
            filas.append({"nivel": "Subcategoría", "grupo": etiqueta, **elasticidad(d)})
    res = pd.DataFrame(filas).dropna(subset=["elasticidad"])
    res["significativa"] = (res["ic95_sup"] < 0) | (res["ic95_inf"] > 0)

    # Verificación: producto x mes (cantidad total frente a precio promedio ponderado)
    pm = (df.assign(valor=df["precio"] * df["cantidad"])
          .groupby(["producto", "mes", "negocio"], as_index=False)
          .agg(cantidad=("cantidad", "sum"), valor=("valor", "sum")))
    pm["log_q"] = np.log(pm["cantidad"])
    pm["log_p"] = np.log(pm["valor"] / pm["cantidad"])
    filas_pm = [{"nivel": "Total", "grupo": "Total", **elasticidad(pm)}]
    for neg, d in pm.groupby("negocio"):
        filas_pm.append({"nivel": "Negocio", "grupo": ABREV_NEGOCIO.get(neg, neg), **elasticidad(d)})
    res_pm = pd.DataFrame(filas_pm).dropna(subset=["elasticidad"])
    res_pm.to_csv(carpeta / "elasticidad_producto_mes.csv", index=False, float_format=c.FORMATO_CSV)

    res.to_csv(carpeta / "elasticidades.csv", index=False, float_format=c.FORMATO_CSV)
    res.assign(tipo="linea").pipe(lambda r: pd.concat([r, res_pm.assign(tipo="producto_mes")]))\
        .to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(res[res["nivel"] != "Subcategoría"][["nivel", "grupo", "elasticidad", "ic95_inf", "ic95_sup", "lineas"]]
          .to_string(index=False))
    print(res_pm[["grupo", "elasticidad", "ic95_inf", "ic95_sup"]].to_string(index=False))

    graficar(carpeta, res[res["nivel"].isin(["Total", "Negocio"])], "02_elasticidad_negocio",
             "Elasticidad precio por negocio", "Efectos fijos de producto y mes; IC 95% agrupado por producto.")
    graficar(carpeta, res[res["nivel"] == "Segmento"], "03_elasticidad_segmento",
             "Elasticidad precio por segmento", "Líneas de consumidores finales de cada segmento.")
    subs = res[res["nivel"] == "Subcategoría"]
    if len(subs):
        graficar(carpeta, subs, "04_elasticidad_subcategoria", "Elasticidad precio por subcategoría",
                 f"Subcategorías con al menos {c.miles(MIN_LINEAS_SUBCATEGORIA)} líneas y {MIN_PRODUCTOS_SUBCATEGORIA} productos.")
    escribir_resumen(carpeta, var, cv_mediano, pct_lineas_cv5, pct_varios, por_neg_cv, res, res_pm)


def escribir_resumen_sin_variacion(carpeta, var, cv_mediano, pct_lineas_cv5) -> None:
    """resumen.md cuando el filtro detiene el modelo."""
    texto = f"""
# m07 - Elasticidad precio

## Qué hace
Verifica si el precio de un mismo producto cambia en la base. **No se estimó la elasticidad** porque
casi no hay variación de precio: el coeficiente de variación mediano por producto es
{c.pct(cv_mediano * 100, 2)} y solo {c.pct(pct_lineas_cv5 * 100)} de las líneas está en productos con CV > 5%.

## Datos usados
`data/transaccional_{c.VERSION}.parquet`, {c.miles(len(var))} productos con al menos {MIN_LINEAS_PRODUCTO} líneas.

## Limitaciones
Sin variación de precio dentro del producto, la elasticidad no se puede identificar con estos datos.
"""
    c.escribir_resumen(carpeta, texto)


def escribir_resumen(carpeta, var, cv_mediano, pct_lineas_cv5, pct_varios, por_neg_cv, res, res_pm) -> None:
    """resumen.md del modelo."""
    def filas(t):
        return "\n".join(
            f"| {x.grupo} | {c.num(x.elasticidad, 3)} | [{c.num(x.ic95_inf, 3)}; {c.num(x.ic95_sup, 3)}] | "
            f"{c.miles(x.lineas)} | {c.miles(x.productos)} |" for x in t.itertuples())
    tot = res[res["nivel"] == "Total"].iloc[0]
    neg = res[res["nivel"] == "Negocio"].sort_values("elasticidad")
    seg = res[res["nivel"] == "Segmento"].sort_values("elasticidad")
    sub = res[res["nivel"] == "Subcategoría"].sort_values("elasticidad")
    pm = res_pm.set_index("grupo")
    filas_pm = "\n".join(
        f"| {g} | {c.num(res.set_index(['nivel', 'grupo']).loc[('Total' if g == 'Total' else 'Negocio', g), 'elasticidad'], 3)} | "
        f"{c.num(x.elasticidad, 3)} | [{c.num(x.ic95_inf, 3)}; {c.num(x.ic95_sup, 3)}] |"
        for g, x in pm.iterrows())
    mas = neg.iloc[0]
    menos = neg.iloc[-1]
    elasticos = (sub["ic95_sup"] < -1).sum()
    inelasticos = ((sub["elasticidad"] > -1) & (sub["elasticidad"] < 0)).sum()
    positivos = (sub["ic95_inf"] > 0).sum()
    cv_neg = ", ".join(f"{ABREV_NEGOCIO.get(x.negocio, x.negocio)} {c.pct(x.cv_mediano * 100)}" for x in por_neg_cv.itertuples())
    signo_pm = "más cercana a cero" if abs(pm.loc["Total", "elasticidad"]) < abs(tot.elasticidad) else "más negativa"
    comp = neg.set_index("grupo")[["elasticidad"]].join(pm[["elasticidad", "ic95_inf", "ic95_sup"]], rsuffix="_pm", how="inner")
    robustos = comp[(comp["ic95_sup"] < 0) & ((comp["elasticidad"] - comp["elasticidad_pm"]).abs() < 0.3)]
    volumen = comp[(comp["elasticidad"] < -0.3) & (comp["ic95_sup"] >= 0)]
    texto_robusto = ", ".join(f"{g} ({c.num(x.elasticidad, 2)} por línea, {c.num(x.elasticidad_pm, 2)} producto x mes)"
                              for g, x in robustos.iterrows()) or "ninguno"
    texto_volumen = ", ".join(f"{g} ({c.num(x.elasticidad, 2)} por línea, {c.num(x.elasticidad_pm, 2)} producto x mes)"
                              for g, x in volumen.iterrows()) or "ninguno"
    texto = f"""
# m07 - Elasticidad precio

## Qué hace
1. **Filtro:** mide si un mismo producto tiene precios distintos en la base (coeficiente de variación
   del precio por producto).
2. **Estimación:** `log(cantidad) ~ log(precio)` con efectos fijos de producto y de mes, con `precio`
   (no `valor`) y solo líneas con cantidad > 0 y precio > 0. La elasticidad sale de comparar, para el
   mismo producto y descontando el mes, líneas con precio más alto y más bajo. Errores estándar
   agrupados por producto; intervalos de confianza al 95%. Se estima en total, por negocio, por
   subcategoría (≥ {c.miles(MIN_LINEAS_SUBCATEGORIA)} líneas y ≥ {MIN_PRODUCTOS_SUBCATEGORIA} productos) y por segmento.
3. **Verificación:** la misma regresión a nivel producto x mes (cantidad total frente a precio promedio).

## Datos usados
- `data/transaccional_{c.VERSION}.parquet` (todas las líneas con cantidad y precio positivos) y la
  segmentación `{c.SEGMENTACION}` para el corte por segmento.
- {c.miles(res.iloc[0].lineas)} líneas, {c.miles(res.iloc[0].productos)} productos.

## Filtro de variación de precio
- {c.miles(len(var))} productos con al menos {MIN_LINEAS_PRODUCTO} líneas; {c.pct(pct_varios * 100)} tienen más de un
  precio distinto. CV mediano del precio dentro del producto: **{c.pct(cv_mediano * 100)}**; {c.pct(pct_lineas_cv5 * 100)}
  de las líneas está en productos con CV > 5%. Por negocio: {cv_neg}.
- Hay variación suficiente para estimar (el umbral para detenerse era CV mediano < {c.pct(CV_MEDIANO_MIN * 100, 0)}).

## Métricas
**Total:** elasticidad {c.num(tot.elasticidad, 3)} (IC 95% {c.num(tot.ic95_inf, 3)} a {c.num(tot.ic95_sup, 3)}).

### Por negocio
| Negocio | Elasticidad | IC 95% | Líneas | Productos |
|---|---|---|---|---|
{filas(neg)}

### Por segmento
| Segmento | Elasticidad | IC 95% | Líneas | Productos |
|---|---|---|---|---|
{filas(seg)}

### Por subcategoría ({len(sub)} con datos suficientes; detalle en `elasticidades.csv`)
| Subcategoría | Elasticidad | IC 95% | Líneas | Productos |
|---|---|---|---|---|
{filas(pd.concat([sub.head(5), sub.tail(5)]).drop_duplicates())}

### Verificación producto x mes
| Grupo | Elasticidad por línea | Elasticidad producto x mes | IC 95% (producto x mes) |
|---|---|---|---|
{filas_pm}

## Hallazgos principales
- El negocio que más responde al precio es **{mas.grupo}** ({c.num(mas.elasticidad, 2)}) y el que menos
  **{menos.grupo}** ({c.num(menos.elasticidad, 2)}).
- De las {len(sub)} subcategorías estimadas, {elasticos} son claramente elásticas (IC por debajo de -1),
  {inelasticos} tienen elasticidad entre -1 y 0 y {positivos} salen con signo positivo significativo
  (señal de que ahí el precio está capturando otra cosa, como calidad o tipo de cliente).
- A nivel producto x mes la elasticidad total es {c.num(pm.loc['Total', 'elasticidad'], 3)}, {signo_pm} que por
  línea: parte de la respuesta por línea es **descuento por volumen** (quien compra más paga menos por
  unidad), no reacción de la demanda a un cambio de precio.
- **Robusto en las dos especificaciones:** {texto_robusto}. Es el resultado más confiable: ahí el
  volumen responde al precio de verdad.
- **Probable descuento por volumen:** {texto_volumen}. La elasticidad por línea es negativa, pero
  desaparece al agregar por producto y mes; no conviene leerla como respuesta de la demanda.
- Uso práctico: las subcategorías y negocios con elasticidad cercana a cero toleran mejor una subida
  de precio; los más elásticos son donde un descuento mueve más volumen (insumo para el supuesto de
  "descuento" del MDP, m11).

## Limitaciones
- El precio no es experimental: cambia por promociones, descuentos negociados, canal o volumen, que
  también afectan la cantidad (endogeneidad). La elasticidad es una asociación dentro del producto y
  el mes, no un efecto causal limpio.
- La cantidad está en las unidades escaladas de la base (múltiplos de 2,65); en logaritmos el factor
  de escala no afecta la elasticidad.
- Las subcategorías están anonimizadas.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
