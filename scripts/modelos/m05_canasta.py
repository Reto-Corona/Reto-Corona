"""
m05 - Reglas de asociación (análisis de canasta)

Canasta = visita (cliente + fecha + punto de venta) de un consumidor final segmentado.
Ítems = subcategoría (y, en una segunda versión fácil de explicar, negocio).

Algoritmo: FP-Growth (mlxtend) con soporte mínimo SOPORTE_MIN; se conservan las reglas con
lift > LIFT_MIN. Se reportan:
    - las 20 mejores reglas por lift
    - las reglas entre negocios distintos (venta cruzada)
    - las mejores reglas dentro de cada segmento
    - las reglas a nivel de negocio

Escribe:
    data/modelos/m05_canasta.parquet           todas las reglas (subcategoría, negocio y por segmento)
    salidas/modelos/m05_canasta/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m05_canasta.py
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

import comun as c

NOMBRE = "m05_canasta"
# 0,2% de las visitas: con ~890 mil canastas son unas 1.800 visitas por conjunto, suficiente para
# que las reglas sean estables y no tan alto como para dejar fuera subcategorías medianas.
SOPORTE_MIN = 0.002
SOPORTE_MIN_SEGMENTO = 0.004   # dentro de cada segmento (bases más pequeñas)
SOPORTE_MIN_NEGOCIO = 0.001
LIFT_MIN = 1.2
CONFIANZA_MIN = 0.05
LONGITUD_MAX = 3
ABREV_NEGOCIO = {
    "REVESTIMIENTOS E INSTALACIÓN": "Revestimientos",
    "BAÑOS": "Baños",
    "COCINAS": "Cocinas",
    "PINTURAS E IMPERMEABILIZANTES": "Pinturas",
    "SERVICIOS": "Servicios",
}


def cargar_canastas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pares (visita, subcategoría) de consumidores finales y el negocio de cada subcategoría."""
    con = c.conexion()
    pares = con.execute("""
        WITH p AS (
          SELECT DISTINCT t.cliente, t.fecha, t.punto_de_venta, t.subcategoria, t.negocio,
                 s.segmento_kmeans AS segmento
          FROM trx t
          JOIN features f USING (cliente)
          JOIN segmentos s USING (cliente)
          WHERE f.tipo_cliente = 'Consumidor final' AND t.subcategoria IS NOT NULL
        )
        SELECT DENSE_RANK() OVER (ORDER BY cliente, fecha, punto_de_venta) AS visita,
               subcategoria, negocio, segmento
        FROM p ORDER BY visita, subcategoria
    """).df()
    # Negocio de cada subcategoría: el más frecuente (casi todas pertenecen a un solo negocio)
    sub = (pares.groupby(["subcategoria", "negocio"]).size().rename("n").reset_index()
           .sort_values(["subcategoria", "n"], ascending=[True, False])
           .drop_duplicates("subcategoria"))
    total = pares.groupby("subcategoria").size()
    sub["pureza_negocio"] = sub["n"].values / total.loc[sub["subcategoria"]].values
    sub["negocio_corto"] = sub["negocio"].map(ABREV_NEGOCIO)
    sub["etiqueta"] = sub["subcategoria"].str.replace("SUBCAT_000", "S", regex=False) + " (" + sub["negocio_corto"] + ")"
    return pares, sub.drop(columns="n")


def matriz(pares: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Matriz booleana visita x ítem (dispersa para ahorrar memoria)."""
    v = pares[["visita", columna]].drop_duplicates()
    filas = v["visita"].astype("category")
    cols = v[columna].astype("category")
    from scipy.sparse import csr_matrix
    m = csr_matrix((np.ones(len(v), dtype=bool), (filas.cat.codes, cols.cat.codes)),
                   shape=(len(filas.cat.categories), len(cols.cat.categories)))
    return pd.DataFrame.sparse.from_spmatrix(m, columns=list(cols.cat.categories))


def reglas(df: pd.DataFrame, soporte: float) -> pd.DataFrame:
    """FP-Growth + reglas con lift > LIFT_MIN y confianza > CONFIANZA_MIN."""
    frecuentes = fpgrowth(df, min_support=soporte, use_colnames=True, max_len=LONGITUD_MAX)
    if frecuentes.empty:
        return pd.DataFrame()
    r = association_rules(frecuentes, metric="lift", min_threshold=LIFT_MIN, num_itemsets=len(df))
    r = r[r["confidence"] >= CONFIANZA_MIN]
    r["antecedente"] = r["antecedents"].apply(lambda s: sorted(s))
    r["consecuente"] = r["consequents"].apply(lambda s: sorted(s))
    r["visitas_con_regla"] = (r["support"] * len(df)).round().astype(int)
    return ordenar(r[["antecedente", "consecuente", "support", "confidence", "lift", "leverage",
                      "conviction", "visitas_con_regla"]])


def ordenar(r: pd.DataFrame) -> pd.DataFrame:
    """Orden fijo: lift y confianza de mayor a menor; los empates, por el texto de la regla."""
    clave = r.assign(_lift=r["lift"].round(9), _conf=r["confidence"].round(9),
                     _a=r["antecedente"].apply(" + ".join), _b=r["consecuente"].apply(" + ".join))
    clave = clave.sort_values(["_lift", "_conf", "_a", "_b"], ascending=[False, False, True, True], kind="mergesort")
    return clave.drop(columns=["_lift", "_conf", "_a", "_b"]).reset_index(drop=True)


def describir(r: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Agrega etiquetas legibles y marca si la regla cruza negocios."""
    mapa_neg = dict(zip(sub["subcategoria"], sub["negocio_corto"]))
    mapa_et = dict(zip(sub["subcategoria"], sub["etiqueta"]))
    r = r.copy()
    r["regla"] = [" + ".join(mapa_et.get(x, x) for x in a) + "  →  " + " + ".join(mapa_et.get(x, x) for x in b)
                  for a, b in zip(r["antecedente"], r["consecuente"])]
    r["negocios_antecedente"] = r["antecedente"].apply(lambda a: sorted({mapa_neg.get(x, "?") for x in a}))
    r["negocios_consecuente"] = r["consecuente"].apply(lambda b: sorted({mapa_neg.get(x, "?") for x in b}))
    r["cruza_negocios"] = [not set(b) <= set(a) for a, b in zip(r["negocios_antecedente"], r["negocios_consecuente"])]
    return r


def una_direccion(r: pd.DataFrame) -> pd.DataFrame:
    """Deja una sola regla por combinación de ítems (A→B y B→A tienen el mismo lift): la de mayor confianza."""
    if r.empty:
        return r
    clave = [tuple(sorted(a + b)) for a, b in zip(r["antecedente"], r["consecuente"])]
    r = r.assign(_clave=clave, _confianza=r["confidence"].round(9))
    # Dentro de cada combinación gana la mayor confianza; si empatan, la primera en orden de texto
    r = ordenar(r).sort_values(["_clave", "_confianza"], ascending=[True, False], kind="mergesort") \
        .drop_duplicates("_clave")
    return ordenar(r.drop(columns=["_clave", "_confianza"]))


def a_texto(r: pd.DataFrame) -> pd.DataFrame:
    """Listas a texto para guardar en CSV/parquet."""
    r = r.copy()
    for col in ("antecedente", "consecuente", "negocios_antecedente", "negocios_consecuente"):
        if col in r:
            r[col] = r[col].apply(lambda x: " + ".join(x))
    return r


def graficar_reglas(carpeta, r: pd.DataFrame, nombre: str, titulo: str, subtitulo: str) -> None:
    """Barras horizontales del lift de las reglas, con su confianza."""
    top = r.head(15).iloc[::-1]
    fig, ax = c.plt.subplots(figsize=(11, 6.5))
    colores = [c.NARANJA if x else c.AZUL for x in top.get("cruza_negocios", [False] * len(top))]
    barras = ax.barh(top["regla"], top["lift"], color=colores, height=0.65)
    c.rotular_barras(ax, barras, [f"lift {c.num(l, 2)} · conf. {c.pct(cf * 100, 0)}"
                                  for l, cf in zip(top["lift"], top["confidence"])], horizontal=True, size=8)
    ax.set_xlim(0, top["lift"].max() * 1.3)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Lift (veces más que si fueran independientes)")
    c.guardar_figura(fig, carpeta, nombre, titulo, subtitulo)


def graficar_negocios(carpeta, pares: pd.DataFrame) -> pd.DataFrame:
    """Lift entre pares de negocios (mapa de calor)."""
    m = matriz(pares.assign(neg=pares["negocio"].map(ABREV_NEGOCIO)), "neg").sparse.to_dense().astype(float)
    n = len(m)
    p = m.mean()
    conj = (m.T @ m) / n
    valores = (conj / np.outer(p, p)).to_numpy(copy=True)
    np.fill_diagonal(valores, np.nan)
    lift = pd.DataFrame(valores, index=conj.index, columns=conj.columns)
    fig, ax = c.plt.subplots(figsize=(7.5, 6))
    cmap = c.g.LinearSegmentedColormap.from_list("div", [c.AZUL, c.SUPERFICIE, c.NARANJA])
    ax.imshow(lift.values, cmap=cmap, norm=c.g.TwoSlopeNorm(vcenter=1, vmin=0, vmax=np.nanmax(lift.values)))
    for i in range(len(lift)):
        for j in range(len(lift)):
            if i != j:
                ax.text(j, i, c.num(lift.iat[i, j], 2), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(lift)), lift.columns)
    ax.set_yticks(range(len(lift)), lift.index)
    ax.grid(False)
    c.guardar_figura(fig, carpeta, "03_lift_entre_negocios", "¿Qué negocios se compran juntos?",
                     "Lift de comprar los dos negocios en la misma visita (1 = independientes).")
    return lift


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    pares, sub = cargar_canastas()
    n_visitas = pares["visita"].nunique()
    items_por_visita = pares.groupby("visita")["subcategoria"].nunique()
    print(f"  Canastas: {n_visitas:,} | subcategorías: {pares['subcategoria'].nunique()} | "
          f"con 2+ subcategorías: {(items_por_visita > 1).mean():.1%}")
    sub.to_csv(carpeta / "subcategorias.csv", index=False, float_format=c.FORMATO_CSV)

    # 1) Subcategoría, toda la base
    r_sub = describir(reglas(matriz(pares, "subcategoria"), SOPORTE_MIN), sub)
    print(f"  Reglas de subcategoría (lift > {LIFT_MIN}): {len(r_sub):,}")
    unicas = una_direccion(r_sub)
    a_texto(unicas.head(20)).to_csv(carpeta / "top20_reglas.csv", index=False, float_format=c.FORMATO_CSV)
    cruzadas = una_direccion(r_sub[r_sub["cruza_negocios"]])
    a_texto(cruzadas).to_csv(carpeta / "reglas_venta_cruzada.csv", index=False, float_format=c.FORMATO_CSV)
    graficar_reglas(carpeta, unicas, "01_top_reglas", "Reglas de asociación más fuertes (subcategoría)",
                    f"FP-Growth, soporte ≥ {c.pct(SOPORTE_MIN * 100, 1)} de {c.miles(n_visitas)} visitas. "
                    "Naranja = la regla cruza negocios.")
    if len(cruzadas):
        graficar_reglas(carpeta, cruzadas, "02_reglas_venta_cruzada", "Reglas de venta cruzada entre negocios",
                        "Reglas cuyo consecuente pertenece a un negocio que no está en el antecedente.")

    # 2) Por segmento
    por_seg = []
    for seg, nombre in c.NOMBRES_SEGMENTOS.items():
        p = pares[pares["segmento"] == seg]
        r = reglas(matriz(p, "subcategoria"), SOPORTE_MIN_SEGMENTO)
        if r.empty:
            continue
        r = una_direccion(describir(r, sub))
        r.insert(0, "segmento", seg)
        r.insert(1, "nombre_segmento", nombre)
        r["canastas_segmento"] = p["visita"].nunique()
        por_seg.append(r)
    por_seg = pd.concat(por_seg, ignore_index=True) if por_seg else pd.DataFrame()
    top_seg = por_seg.groupby("segmento").head(5)
    a_texto(top_seg).to_csv(carpeta / "reglas_por_segmento.csv", index=False, float_format=c.FORMATO_CSV)

    # 3) Negocio
    pares_neg = pares.assign(negocio_corto=pares["negocio"].map(ABREV_NEGOCIO))
    r_neg = reglas(matriz(pares_neg, "negocio_corto"), SOPORTE_MIN_NEGOCIO)
    if not r_neg.empty:
        r_neg = una_direccion(r_neg)
        r_neg["regla"] = r_neg["antecedente"].apply(" + ".join) + "  →  " + r_neg["consecuente"].apply(" + ".join)
    a_texto(r_neg).to_csv(carpeta / "reglas_negocio.csv", index=False, float_format=c.FORMATO_CSV)
    lift_neg = graficar_negocios(carpeta, pares)
    lift_neg.to_csv(carpeta / "lift_entre_negocios.csv", float_format=c.FORMATO_CSV)

    # Todas las reglas en un parquet
    todas = pd.concat([
        a_texto(r_sub).assign(nivel="subcategoria", segmento=0),
        a_texto(por_seg).assign(nivel="subcategoria_segmento") if len(por_seg) else None,
        a_texto(r_neg).assign(nivel="negocio", segmento=0),
    ], ignore_index=True)
    todas.to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(todas):,} reglas)")

    escribir_resumen(carpeta, n_visitas, items_por_visita, sub, r_sub, unicas, cruzadas, por_seg, top_seg, r_neg, lift_neg)


def escribir_resumen(carpeta, n_visitas, items, sub, r_sub, unicas, cruzadas, por_seg, top_seg, r_neg, lift_neg) -> None:
    """resumen.md del modelo."""
    def tabla(r, n):
        return "\n".join(f"| {x.regla} | {c.pct(x.support * 100, 2)} | {c.pct(x.confidence * 100, 1)} | {c.num(x.lift, 2)} |"
                         for x in r.head(n).itertuples())
    filas_seg = "\n".join(
        f"| {x.segmento}. {x.nombre_segmento} | {x.regla} | {c.pct(x.confidence * 100, 1)} | {c.num(x.lift, 2)} |"
        for x in top_seg.groupby("segmento").head(2).itertuples())
    filas_neg = tabla(r_neg.sort_values("lift", ascending=False), 10) if len(r_neg) else "| (sin reglas) | | | |"
    ln = lift_neg.stack().dropna()
    par_max = ln.idxmax()
    par_min = ln.idxmin()
    pureza = (sub["pureza_negocio"] >= 0.99).mean() * 100
    dentro_top = (~unicas.head(20)["cruza_negocios"]).sum()
    cruce_seg = por_seg.groupby(["segmento", "nombre_segmento"])["cruza_negocios"].mean().mul(100).reset_index()
    texto_cruce = "; ".join(f"{x.nombre_segmento} {c.pct(x.cruza_negocios, 0)}" for x in cruce_seg.itertuples())
    seg_cocinas = por_seg[(por_seg["segmento"] == 5)].head(1)
    nota_cocinas = ""
    if len(seg_cocinas) and not set(seg_cocinas.iloc[0]["negocios_consecuente"]) <= {"Cocinas"}:
        nota_cocinas = (" En \"Solo Cocinas\" la regla más fuerte es de Revestimientos: quien remodela la cocina"
                        " también compra el enchape, una venta cruzada natural.")
    texto = f"""
# m05 - Reglas de asociación (canasta)

## Qué hace
Busca qué subcategorías se compran juntas en la misma visita (cliente + fecha + punto de venta)
con FP-Growth (`mlxtend`) y conserva las reglas con lift > {c.num(LIFT_MIN, 1)} y confianza ≥
{c.pct(CONFIANZA_MIN * 100, 0)}. Repite el análisis por segmento y a nivel de negocio.

## Datos usados
- `data/transaccional_{c.VERSION}.parquet`, solo consumidores finales segmentados (`{c.SEGMENTACION}`).
- {c.miles(n_visitas)} canastas, {len(sub)} subcategorías; {c.pct((items > 1).mean() * 100)} de las visitas tiene 2
  o más subcategorías (las demás no aportan reglas, pero cuentan en el soporte).
- Soporte mínimo {c.pct(SOPORTE_MIN * 100, 1)} (≈ {c.miles(SOPORTE_MIN * n_visitas)} visitas) en la base completa,
  {c.pct(SOPORTE_MIN_SEGMENTO * 100, 1)} dentro de cada segmento y {c.pct(SOPORTE_MIN_NEGOCIO * 100, 1)} a nivel de negocio;
  conjuntos de hasta {LONGITUD_MAX} ítems.
- Las subcategorías vienen anonimizadas (`SUBCAT_000046` se muestra como `S046`); se etiquetan con su
  negocio. {c.pct(pureza, 0)} de las subcategorías pertenece a un solo negocio.

## Métricas
- Reglas de subcategoría con lift > {c.num(LIFT_MIN, 1)}: **{c.miles(len(r_sub))}** ({c.miles(len(unicas))} combinaciones
  distintas de ítems, porque A → B y B → A tienen el mismo lift); combinaciones que cruzan negocios:
  **{c.miles(len(cruzadas))}**. En las tablas se muestra una sola dirección por combinación (la de mayor
  confianza); todas las reglas están en `data/modelos/m05_canasta.parquet`.

### 20 mejores reglas (lift)
| Regla | Soporte | Confianza | Lift |
|---|---|---|---|
{tabla(unicas, 20)}

### Venta cruzada (consecuente en otro negocio)
| Regla | Soporte | Confianza | Lift |
|---|---|---|---|
{tabla(cruzadas, 10) if len(cruzadas) else '| (sin reglas) | | | |'}

### Por segmento (2 mejores de cada uno; todas en `reglas_por_segmento.csv`)
| Segmento | Regla | Confianza | Lift |
|---|---|---|---|
{filas_seg}

### Nivel de negocio
| Regla | Soporte | Confianza | Lift |
|---|---|---|---|
{filas_neg}

## Hallazgos principales
- {dentro_top} de las 20 reglas más fuertes son **dentro del mismo negocio** (subcategorías
  complementarias de un mismo proyecto, sobre todo en Baños y Cocinas); sirven para recomendar "lo que
  falta" en la misma visita.
- Entre negocios, el par que más se compra junto es {par_max[0]} + {par_max[1]} (lift
  {c.num(ln.max(), 2)}) y el que menos {par_min[0]} + {par_min[1]} (lift {c.num(ln.min(), 2)}). Las reglas de
  venta cruzada son las candidatas naturales para el contacto de venta cruzada del piloto (m14).
- Porcentaje de reglas que cruzan negocios dentro de cada segmento: {texto_cruce}.{nota_cocinas}

## Limitaciones
- Las subcategorías están anonimizadas: las reglas dicen qué combinar, pero hay que traducirlas con
  el catálogo de Corona para usarlas.
- La canasta es una visita; compras del mismo proyecto en visitas distintas no se ven como canasta.
- Lift alto con soporte bajo puede ser ruido; por eso se exige soporte mínimo y confianza ≥ {c.pct(CONFIANZA_MIN * 100, 0)}.
- Las reglas son asociaciones, no efectos causales de recomendar un producto.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
