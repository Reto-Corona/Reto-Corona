"""
m08 - Probabilidad de recompra a 90 días (consumidores finales)

Unidad: cliente, a partir de su PRIMERA visita (visita = cliente + fecha + punto de venta).
Objetivo: 1 si el cliente vuelve (otra visita en una fecha posterior) dentro de los 90 días
siguientes a la primera visita; 0 si no.

Para evitar fuga de información solo se usan variables de la primera visita y del perfil
(no el segmento ni variables calculadas con todo el historial).

Partición temporal:
    entrenamiento: primeras visitas anteriores a FECHA_CORTE - 90 días (así su objetivo
                   se observa completo antes del corte)
    prueba:        primeras visitas desde FECHA_CORTE hasta el último día observable

Compara regresión logística contra LightGBM (AUC, PR-AUC, calibración, lift por decil) y
explica el mejor con SHAP.

Escribe:
    data/modelos/m08_recompra.parquet             probabilidad por cliente
    salidas/modelos/m08_recompra/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m08_recompra.py
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import comun as c

NOMBRE = "m08_recompra"
VENTANA_DIAS = 90
FECHA_CORTE = pd.Timestamp("2001-11-01")
MUESTRA_SHAP = 5000

CATEGORICAS = ["negocio_visita", "canal", "region", "mes", "generacion", "genero", "contactable"]
NUMERICAS = ["log_valor", "log_productos", "categorias", "lineas", "negocios"]
ETIQUETAS = {
    "log_valor": "Valor de la visita (log)",
    "log_productos": "Productos distintos (log)",
    "categorias": "Categorías distintas",
    "lineas": "Líneas de la visita",
    "negocios": "Negocios distintos",
    "negocio_visita": "Negocio principal de la visita",
    "canal": "Canal (cluster)",
    "region": "Región",
    "mes": "Mes del año",
    "generacion": "Generación",
    "genero": "Género",
    "contactable": "Contactable",
}


def construir_base() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Una fila por consumidor final con las variables de su primera visita y el objetivo."""
    con = c.conexion()
    fecha_max = pd.Timestamp(con.execute("SELECT MAX(fecha) FROM trx").fetchone()[0])
    df = con.execute(f"""
    WITH visitas AS (
      SELECT cliente, fecha::DATE AS fecha, punto_de_venta,
             SUM(valor_corregido) AS valor,
             COUNT(*) AS lineas,
             COUNT(DISTINCT producto) AS productos,
             COUNT(DISTINCT categoria) AS categorias,
             COUNT(DISTINCT negocio) AS negocios,
             MODE(cluster) AS canal,
             MODE(region) AS region
      FROM trx GROUP BY cliente, fecha::DATE, punto_de_venta
    ),
    neg AS (
      SELECT cliente, fecha::DATE AS fecha, punto_de_venta, negocio,
             ROW_NUMBER() OVER (PARTITION BY cliente, fecha::DATE, punto_de_venta
                                ORDER BY ROUND(SUM(valor_corregido), 2) DESC, negocio) AS rk
      FROM trx GROUP BY cliente, fecha::DATE, punto_de_venta, negocio
    ),
    -- primera visita: la fecha más temprana; si ese día fue a dos tiendas, la de mayor valor
    primera AS (
      SELECT v.*, n.negocio AS negocio_visita,
             ROW_NUMBER() OVER (PARTITION BY v.cliente
                                ORDER BY v.fecha, ROUND(v.valor, 2) DESC, v.punto_de_venta) AS orden
      FROM visitas v
      JOIN neg n ON n.cliente = v.cliente AND n.fecha = v.fecha
                AND n.punto_de_venta = v.punto_de_venta AND n.rk = 1
    ),
    objetivo AS (
      SELECT p.cliente,
             MAX(CASE WHEN v.fecha > p.fecha AND v.fecha <= p.fecha + {VENTANA_DIAS}
                      THEN 1 ELSE 0 END) AS recompra_90d
      FROM primera p JOIN visitas v USING (cliente)
      WHERE p.orden = 1 GROUP BY p.cliente
    )
    SELECT p.cliente, p.fecha AS fecha_primera_visita, p.valor, p.lineas, p.productos,
           p.categorias, p.negocios, p.negocio_visita,
           COALESCE(p.canal, 'Sin canal') AS canal,
           COALESCE(p.region, 'SIN REGIÓN') AS region,
           o.recompra_90d,
           f.generacion, f.genero, f.es_contactable, f.posible_cliente_previo,
           s.segmento_kmeans AS segmento
    FROM primera p
    JOIN objetivo o USING (cliente)
    JOIN features f USING (cliente)
    JOIN segmentos s USING (cliente)
    WHERE p.orden = 1 AND f.tipo_cliente = 'Consumidor final'
    """).df()
    # Orden fijo: DuckDB puede devolver las filas en otro orden y eso cambiaría la partición aleatoria
    df = df.sort_values("cliente").reset_index(drop=True)
    df["fecha_primera_visita"] = pd.to_datetime(df["fecha_primera_visita"])
    df["log_valor"] = np.log1p(df["valor"].clip(lower=0))
    df["log_productos"] = np.log1p(df["productos"])
    df["mes"] = df["fecha_primera_visita"].dt.month.astype(str).str.zfill(2)
    df["contactable"] = np.where(df["es_contactable"].fillna(False).astype(bool), "Sí", "No")
    df["generacion"] = df["generacion"].fillna("Sin edad")
    df["genero"] = df["genero"].fillna("No identificado")
    df["nombre_segmento"] = df["segmento"].map(c.NOMBRES_SEGMENTOS)

    # Muestra a la que pertenece cada cliente
    ultimo_observable = fecha_max - pd.Timedelta(days=VENTANA_DIAS)
    fin_entrenamiento = FECHA_CORTE - pd.Timedelta(days=VENTANA_DIAS)
    f0 = df["fecha_primera_visita"]
    df["muestra"] = np.select(
        [df["posible_cliente_previo"] == 1,
         f0 > ultimo_observable,
         f0 < fin_entrenamiento,
         f0 >= FECHA_CORTE],
        ["excluido_cliente_previo", "no_observable", "entrenamiento", "prueba"],
        default="brecha",
    )
    df.loc[df["muestra"].isin(["excluido_cliente_previo", "no_observable"]), "recompra_90d"] = np.nan
    return df, fecha_max


def modelo_logistico() -> Pipeline:
    """Regresión logística con variables categóricas en one-hot y numéricas estandarizadas."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, drop="first"), CATEGORICAS),
        ("num", StandardScaler(), NUMERICAS),
    ])
    return Pipeline([("prep", prep),
                     ("logit", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))])


def a_categorias(df: pd.DataFrame, categorias: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Variables para LightGBM: categóricas como 'category' con las mismas categorías siempre."""
    x = df[CATEGORICAS + NUMERICAS].copy()
    categorias = categorias or {col: sorted(x[col].astype(str).unique()) for col in CATEGORICAS}
    for col in CATEGORICAS:
        valores = x[col].astype(str).where(x[col].astype(str).isin(categorias[col]))
        x[col] = pd.Categorical(valores, categories=categorias[col])
    return x, categorias


def entrenar_lgbm(x: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """LightGBM con parada temprana sobre un 10% aleatorio del entrenamiento."""
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.10, random_state=c.SEMILLA, stratify=y)
    modelo = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=c.SEMILLA, verbose=-1, n_jobs=-1,
    )
    modelo.fit(x_tr, y_tr, eval_X=(x_val,), eval_y=(y_val,), eval_metric="auc",
               callbacks=[lgb.early_stopping(100, verbose=False)])
    return modelo


def lift_por_decil(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Deciles de probabilidad (1 = mayor): tasa real, tasa predicha y lift frente al promedio."""
    d = pd.DataFrame({"y": y, "p": p})
    d["decil"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    t = d.groupby("decil").agg(clientes=("y", "size"), tasa_real=("y", "mean"), tasa_predicha=("p", "mean"))
    t["lift"] = t["tasa_real"] / d["y"].mean()
    t["captura_acumulada"] = d.groupby("decil")["y"].sum().cumsum() / d["y"].sum()
    return t.reset_index()


def graficar_curvas(carpeta, y, probs: dict) -> None:
    """Curvas ROC y precisión-exhaustividad de los dos modelos."""
    fig, (a1, a2) = c.plt.subplots(1, 2, figsize=(11, 5))
    for (nombre, p), color in zip(probs.items(), [c.AZUL, c.NARANJA]):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=color, lw=2, label=f"{nombre} (AUC {c.num(roc_auc_score(y, p), 3)})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=color, lw=2,
                label=f"{nombre} (PR-AUC {c.num(average_precision_score(y, p), 3)})")
    a1.plot([0, 1], [0, 1], color=c.EJE, ls="--", lw=1)
    a2.axhline(y.mean(), color=c.EJE, ls="--", lw=1)
    a1.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="ROC")
    a2.set(xlabel="Exhaustividad (recall)", ylabel="Precisión", title="Precisión-exhaustividad")
    for a in (a1, a2):
        a.legend(loc="lower right" if a is a1 else "upper right")
    c.guardar_figura(fig, carpeta, "01_curvas_roc_pr", "Capacidad de discriminación en la muestra de prueba",
                     f"Primeras visitas desde {FECHA_CORTE:%Y-%m-%d}; la línea punteada es el azar.")


def graficar_calibracion(carpeta, y, probs: dict) -> pd.DataFrame:
    """Curva de calibración (10 grupos por cuantiles) de cada modelo."""
    fig, ax = c.plt.subplots(figsize=(7, 5.5))
    filas = []
    for (nombre, p), color in zip(probs.items(), [c.AZUL, c.NARANJA]):
        real, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(pred, real, marker="o", color=color, lw=2, label=nombre)
        filas += [{"modelo": nombre, "grupo": i + 1, "prob_predicha": pp, "tasa_real": rr}
                  for i, (pp, rr) in enumerate(zip(pred, real))]
    lim = max(max(r["prob_predicha"] for r in filas), max(r["tasa_real"] for r in filas)) * 1.05
    ax.plot([0, lim], [0, lim], color=c.EJE, ls="--", lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Probabilidad predicha", ylabel="Tasa real de recompra")
    ax.legend(loc="upper left")
    c.guardar_figura(fig, carpeta, "02_calibracion", "Calibración en la muestra de prueba",
                     "Sobre la diagonal, el modelo acierta la tasa de recompra de cada grupo.")
    return pd.DataFrame(filas)


def graficar_lift(carpeta, lift: pd.DataFrame, nombre_modelo: str) -> None:
    """Tasa real de recompra por decil de probabilidad."""
    fig, ax = c.plt.subplots(figsize=(9, 5))
    barras = ax.bar(lift["decil"], lift["tasa_real"] * 100, color=c.AZUL, width=0.7)
    c.rotular_barras(ax, barras, [f"{c.num(v, 2)}x" for v in lift["lift"]])
    promedio = (lift["tasa_real"] * lift["clientes"]).sum() / lift["clientes"].sum() * 100
    ax.axhline(promedio, color=c.NARANJA, ls="--", lw=1.2)
    ax.annotate(f"Promedio {c.pct(promedio)}", (10.4, promedio), va="bottom", ha="right",
                color=c.NARANJA, fontsize=9)
    ax.set_xticks(lift["decil"], [f"D{d}" for d in lift["decil"]])
    ax.yaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.pct(v, 0)))
    ax.grid(axis="x", visible=False)
    ax.set(xlabel="Decil de probabilidad (D1 = mayor probabilidad)", ylabel="Tasa real de recompra a 90 días")
    c.guardar_figura(fig, carpeta, "03_lift_por_decil", f"Lift por decil ({nombre_modelo})",
                     "Encima de cada barra: tasa del decil dividida entre la tasa promedio.")


def explicar_shap(carpeta, modelo, x_prueba: pd.DataFrame) -> pd.DataFrame:
    """Importancia SHAP (media del valor absoluto) sobre una muestra de la prueba."""
    muestra = x_prueba.sample(min(MUESTRA_SHAP, len(x_prueba)), random_state=c.SEMILLA)
    with warnings.catch_warnings():  # aviso informativo de shap sobre el formato de salida
        warnings.simplefilter("ignore", UserWarning)
        valores = shap.TreeExplainer(modelo).shap_values(muestra)
    if isinstance(valores, list):  # según la versión, shap devuelve una matriz por clase
        valores = valores[1]
    imp = (pd.DataFrame({"variable": muestra.columns, "shap_medio_abs": np.abs(valores).mean(axis=0)})
           .sort_values("shap_medio_abs", ascending=False))
    imp["etiqueta"] = imp["variable"].map(ETIQUETAS)

    fig, ax = c.plt.subplots(figsize=(9, 5.5))
    orden = imp.iloc[::-1]
    barras = ax.barh(orden["etiqueta"], orden["shap_medio_abs"], color=c.AZUL, height=0.65)
    c.rotular_barras(ax, barras, [c.num(v, 3) for v in orden["shap_medio_abs"]], horizontal=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Media de |SHAP| (log-odds)")
    ax.xaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.num(v, 1)))
    c.guardar_figura(fig, carpeta, "04_importancia_shap", "Qué variables mueven la probabilidad de recompra",
                     f"LightGBM, SHAP sobre {c.miles(len(muestra))} clientes de la muestra de prueba.")

    # Efecto del valor de la primera visita (dependencia SHAP)
    fig, ax = c.plt.subplots(figsize=(8, 5))
    col = list(muestra.columns).index("log_valor")
    ax.scatter(np.expm1(muestra["log_valor"]), valores[:, col], s=6, alpha=0.35, color=c.AZUL, linewidths=0)
    ax.set_xscale("log")
    ax.axhline(0, color=c.EJE, lw=1)
    ax.set(xlabel="Valor de la primera visita (COP, escala log)", ylabel="Aporte SHAP (log-odds)")
    c.guardar_figura(fig, carpeta, "05_shap_valor_visita", "Efecto del valor de la primera visita",
                     "Cada punto es un cliente; arriba de cero sube la probabilidad de volver.")
    return imp


def graficar_segmentos(carpeta, tabla: pd.DataFrame) -> None:
    """Tasa real frente a predicha por segmento."""
    fig, ax = c.plt.subplots(figsize=(10, 5))
    t = tabla.sort_values("segmento")
    c.g.barras_agrupadas(ax, [c.NOMBRES_SEGMENTOS[s].replace(" en una", "\nen una") for s in t["segmento"]],
                         list(t["tasa_real"] * 100), list(t["tasa_predicha"] * 100),
                         ["Tasa real", "Tasa predicha"])
    c.guardar_figura(fig, carpeta, "06_recompra_por_segmento", "Recompra a 90 días por segmento",
                     "Clientes observables (sin clientes previos ni primeras visitas en los últimos 90 días).")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")

    df, fecha_max = construir_base()
    conteo = df["muestra"].value_counts()
    print(conteo.to_string())
    entrena = df[df["muestra"] == "entrenamiento"]
    prueba = df[df["muestra"] == "prueba"]
    y_tr, y_te = entrena["recompra_90d"].astype(int), prueba["recompra_90d"].astype(int)

    # 1) Regresión logística
    logit = modelo_logistico().fit(entrena[CATEGORICAS + NUMERICAS], y_tr)
    p_logit = logit.predict_proba(prueba[CATEGORICAS + NUMERICAS])[:, 1]
    nombres = logit.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({"variable": nombres, "coeficiente": logit.named_steps["logit"].coef_[0]})
    coefs["odds_ratio"] = np.exp(coefs["coeficiente"])
    coefs = coefs.reindex(coefs["coeficiente"].abs().sort_values(ascending=False).index)
    coefs.to_csv(carpeta / "coeficientes_logistica.csv", index=False, float_format=c.FORMATO_CSV)

    # 2) LightGBM
    x_tr, cats = a_categorias(entrena)
    x_te, _ = a_categorias(prueba, cats)
    lgbm = entrenar_lgbm(x_tr, y_tr)
    p_lgbm = lgbm.predict_proba(x_te)[:, 1]

    probs = {"Regresión logística": p_logit, "LightGBM": p_lgbm}
    metricas = pd.DataFrame([{
        "modelo": n,
        "auc": roc_auc_score(y_te, p),
        "pr_auc": average_precision_score(y_te, p),
        "brier": brier_score_loss(y_te, p),
        "tasa_real_prueba": y_te.mean(),
        "tasa_predicha_prueba": p.mean(),
        "lift_decil_1": lift_por_decil(y_te.values, p)["lift"].iloc[0],
        "n_entrenamiento": len(entrena),
        "n_prueba": len(prueba),
    } for n, p in probs.items()])
    metricas.to_csv(carpeta / "metricas.csv", index=False, float_format=c.FORMATO_CSV)
    print(metricas.to_string(index=False))

    mejor = metricas.sort_values("auc", ascending=False)["modelo"].iloc[0]
    print(f"Mejor modelo por AUC: {mejor}")
    lift = lift_por_decil(y_te.values, probs[mejor])
    lift.to_csv(carpeta / "lift_por_decil.csv", index=False, float_format=c.FORMATO_CSV)

    graficar_curvas(carpeta, y_te.values, probs)
    calib = graficar_calibracion(carpeta, y_te.values, probs)
    calib.to_csv(carpeta / "calibracion.csv", index=False, float_format=c.FORMATO_CSV)
    graficar_lift(carpeta, lift, mejor)
    imp = explicar_shap(carpeta, lgbm, x_te)
    imp.to_csv(carpeta / "importancia_shap.csv", index=False, float_format=c.FORMATO_CSV)

    # 3) Probabilidad para todos los consumidores finales (modelo entrenado en 'entrenamiento')
    x_todos, _ = a_categorias(df, cats)
    df["prob_logistica"] = logit.predict_proba(df[CATEGORICAS + NUMERICAS])[:, 1]
    df["prob_lgbm"] = lgbm.predict_proba(x_todos)[:, 1]
    df["prob_recompra_90d"] = df["prob_lgbm"] if mejor == "LightGBM" else df["prob_logistica"]
    df["modelo"] = mejor
    salida = df[["cliente", "segmento", "nombre_segmento", "fecha_primera_visita", "muestra",
                 "recompra_90d", "prob_recompra_90d", "prob_logistica", "prob_lgbm", "modelo",
                 "es_contactable", "region", "canal"]]
    salida.to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(salida):,} clientes)")

    # 4) Tasa real y predicha por segmento
    obs = df[df["recompra_90d"].notna()]
    por_seg = (obs.groupby("segmento")
               .agg(clientes=("cliente", "size"), tasa_real=("recompra_90d", "mean"),
                    tasa_predicha=("prob_recompra_90d", "mean"))
               .reset_index())
    te = prueba.assign(p=probs[mejor]).groupby("segmento").agg(
        clientes_prueba=("cliente", "size"), tasa_real_prueba=("recompra_90d", "mean"),
        tasa_predicha_prueba=("p", "mean")).reset_index()
    contact = (obs[obs["es_contactable"].fillna(False).astype(bool)].groupby("segmento")
               .agg(clientes_contactables=("cliente", "size"),
                    tasa_real_contactables=("recompra_90d", "mean")).reset_index())
    por_seg = por_seg.merge(te, on="segmento").merge(contact, on="segmento")
    por_seg.insert(1, "nombre_segmento", por_seg["segmento"].map(c.NOMBRES_SEGMENTOS))
    por_seg.to_csv(carpeta / "recompra_por_segmento.csv", index=False, float_format=c.FORMATO_CSV)
    print(por_seg.to_string(index=False))
    graficar_segmentos(carpeta, por_seg)

    escribir_resumen(carpeta, df, conteo, metricas, mejor, lift, imp, coefs, por_seg, fecha_max)


def escribir_resumen(carpeta, df, conteo, metricas, mejor, lift, imp, coefs, por_seg, fecha_max) -> None:
    """resumen.md del modelo."""
    m = metricas.set_index("modelo")
    filas_m = "\n".join(
        f"| {n} | {c.num(r.auc, 3)} | {c.num(r.pr_auc, 3)} | {c.num(r.brier, 4)} | "
        f"{c.pct(r.tasa_predicha_prueba * 100)} | {c.num(r.lift_decil_1, 2)}x |"
        for n, r in m.iterrows())
    filas_s = "\n".join(
        f"| {r.segmento}. {r.nombre_segmento} | {c.miles(r.clientes)} | {c.pct(r.tasa_real * 100)} | "
        f"{c.pct(r.tasa_predicha * 100)} | {c.pct(r.tasa_real_contactables * 100)} |"
        for r in por_seg.itertuples())
    top = ", ".join(imp["etiqueta"].head(4))
    odds = coefs.set_index("variable")["odds_ratio"]
    or_valor = odds["num__log_valor"]

    def or_neg(prefijo: str) -> float:
        """Odds ratio del negocio de la primera visita que empieza por 'prefijo' (1 si es la base)."""
        filas = odds[odds.index.str.startswith(f"cat__negocio_visita_{prefijo}")]
        return float(filas.iloc[0]) if len(filas) else 1.0
    cf = coefs[~coefs["variable"].str.startswith("cat__mes")].head(6)
    filas_c = "\n".join(f"| `{r.variable}` | {c.num(r.coeficiente, 3)} | {c.num(r.odds_ratio, 2)} |"
                        for r in cf.itertuples())
    d1 = lift.iloc[0]
    d10 = lift.iloc[-1]
    tasa_prueba = m.iloc[0].tasa_real_prueba
    texto = f"""
# m08 - Probabilidad de recompra a 90 días

## Qué hace
Estima, para cada consumidor final, la probabilidad de volver a comprar (otra visita en una
fecha posterior) dentro de los 90 días siguientes a su **primera visita** (visita = cliente +
fecha + punto de venta; si el primer día fue a dos tiendas se toma la de mayor valor).
Compara una regresión logística (interpretable) con LightGBM y explica el mejor con SHAP.

## Datos usados
- `data/transaccional_{c.VERSION}.parquet` ({c.VALOR_TXT}), `data/features_clientes_{c.VERSION}.parquet`
  (solo perfil: generación, género, contactabilidad y la marca `posible_cliente_previo`) y
  `data/segmentos_{c.VERSION}_{c.SEGMENTACION}.parquet` (solo para reportar por segmento).
- Variables del modelo, **todas de la primera visita o del perfil**: valor de la visita, productos,
  categorías, líneas y negocios distintos, negocio principal de la visita, canal (cluster), región,
  mes del año, generación, género y contactabilidad. No se usa el segmento ni frecuencia, monto
  total o recencia, que se calculan con todo el historial y contienen la respuesta.
- Exclusiones: {c.miles(conteo.get('excluido_cliente_previo', 0))} clientes marcados como
  `posible_cliente_previo` y {c.miles(conteo.get('no_observable', 0))} con primera visita en los últimos
  90 días de la base (después de {(fecha_max - pd.Timedelta(days=VENTANA_DIAS)):%Y-%m-%d}).
- Partición temporal con corte {FECHA_CORTE:%Y-%m-%d}: entrenamiento = primeras visitas antes de
  {(FECHA_CORTE - pd.Timedelta(days=VENTANA_DIAS)):%Y-%m-%d} ({c.miles(conteo.get('entrenamiento', 0))} clientes;
  así su ventana de 90 días termina antes del corte), prueba = primeras visitas desde el corte
  ({c.miles(conteo.get('prueba', 0))}). Los {c.miles(conteo.get('brecha', 0))} clientes intermedios no
  se usan para entrenar ni evaluar.

## Métricas (muestra de prueba, tasa real {c.pct(tasa_prueba * 100)})
| Modelo | AUC | PR-AUC | Brier | Tasa predicha | Lift decil 1 |
|---|---|---|---|---|---|
{filas_m}

Mejor modelo por AUC: **{mejor}**. Su decil 1 recompra {c.pct(d1.tasa_real * 100)} frente a
{c.pct(d10.tasa_real * 100)} del decil 10; los tres primeros deciles capturan
{c.pct(lift.iloc[2].captura_acumulada * 100)} de las recompras. La calibración está en
`02_calibracion.png` y `calibracion.csv`.

Coeficientes más grandes de la logística (variables numéricas estandarizadas):

| Variable | Coeficiente | Odds ratio |
|---|---|---|
{filas_c}

## Recompra por segmento (clientes observables)
| Segmento | Clientes | Tasa real | Tasa predicha | Tasa real contactables |
|---|---|---|---|---|
{filas_s}

## Hallazgos principales
- Las variables que más mueven la probabilidad (SHAP) son: {top}.
- Dirección de los efectos (logística, categoría base del negocio = Baños): una primera visita de
  mayor valor aumenta la recompra (odds ratio {c.num(or_valor, 2)} por desviación estándar del log
  del valor); empezar por Revestimientos (OR {c.num(or_neg('REVESTIMIENTOS'), 2)}) o Pinturas
  (OR {c.num(or_neg('PINTURAS'), 2)}) se asocia con volver más que empezar por Baños o Cocinas
  (OR {c.num(or_neg('COCINAS'), 2)}). Las regiones pequeñas y "SIN REGIÓN" recompran menos.
- La señal es moderada: con solo la primera visita el modelo ordena bien a los clientes (lift de
  {c.num(d1.lift, 2)}x en el decil 1) pero no separa con certeza quién vuelve; sirve para priorizar
  contactos, no para decidir caso a caso.
- La diferencia entre la logística y LightGBM es pequeña; la logística es suficiente si se prefiere
  interpretabilidad.
- La tasa predicha por segmento se aleja de la real, sobre todo en "Proyecto recurrente": es
  esperado, porque el segmento se calculó con todo el historial (incluye la recompra) y el modelo,
  a propósito, no lo usa. La tasa **real** por segmento es la línea base del piloto (m14).
- En la prueba el modelo sobreestima un poco la tasa promedio (la recompra baja levemente en los
  últimos meses de la base); el orden de los clientes se mantiene, que es lo que importa para priorizar.

## Limitaciones
- Las fechas vienen desplazadas por Corona; la estacionalidad (mes) se interpreta con cuidado.
- La base empieza en {df['fecha_primera_visita'].min():%Y-%m-%d}: algunos "primeros" clientes pueden
  haber comprado antes; se excluyen los marcados como `posible_cliente_previo`, pero puede quedar
  algún cliente previo sin marcar.
- La recompra solo se observa en tiendas y canales registrados en la base.
- La probabilidad de todos los clientes se calcula con el modelo entrenado en la muestra de
  entrenamiento; para los clientes de entrenamiento es una estimación dentro de muestra.
- Es un modelo predictivo, no causal: no dice qué acción aumenta la recompra (eso lo mide m14).
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
