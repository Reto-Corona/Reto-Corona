"""
m02 - Mezclas gaussianas (GMM) como comparación de la segmentación con_pca_90

Mismo espacio que segmentacion.R con CORONA_PCA=con_pca_90:
    - lee data/features_clientes_<version>.parquet, consumidores finales con monto > 0, ordenados por cliente
    - 8 variables: log1p(ticket_visita), log(frecuencia_visitas), log(lineas_por_visita), log(n_categorias)
      y las participaciones de Revestimientos, Baños, Cocinas y Pinturas
    - estandarización como scale() de R (desviación con n - 1) y PCA sin volver a centrar; se conservan
      las componentes que explican al menos 90% de la varianza (5)

Prueba GMM de 2 a 8 componentes (covarianza completa) y elige con BIC. Con 6 componentes compara
contra K-means (Rand ajustado) y reporta el porcentaje de clientes "ambiguos" (probabilidad máxima de
pertenencia < 0,6) por segmento.

Escribe:
    data/modelos/m02_gmm.parquet               componente GMM, probabilidad máxima y ambigüedad por cliente
    salidas/modelos/m02_gmm/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m02_gmm.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

import comun as c

NOMBRE = "m02_gmm"
VARIABLES = ["log_ticket", "log_visitas", "log_lineas_visita", "log_categorias",
             "share_revestimientos_e_instalacion", "share_banos", "share_cocinas",
             "share_pinturas_e_impermeabilizantes"]
ETIQUETAS = {
    "log_ticket": "Valor de la visita", "log_visitas": "Número de visitas", "log_lineas_visita": "Productos por visita",
    "log_categorias": "Variedad de categorías", "share_revestimientos_e_instalacion": "Gasto en Revestimientos",
    "share_banos": "Gasto en Baños", "share_cocinas": "Gasto en Cocinas",
    "share_pinturas_e_impermeabilizantes": "Gasto en Pinturas",
}
VARIANZA_OBJETIVO = 0.90
K_CANDIDATOS = range(2, 9)
K_COMPARACION = 6
UMBRAL_AMBIGUO = 0.6
INICIOS = 3


def preparar() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Mismas variables, estandarización y PCA que segmentacion.R (con_pca_90)."""
    c.verificar_insumos(c.FEATURES_PARQUET, c.SEGMENTOS_PARQUET)
    f = pd.read_parquet(c.FEATURES_PARQUET).sort_values("cliente")
    f = f[(f["tipo_cliente"] == "Consumidor final") & (f["monto"] > 0)].reset_index(drop=True)
    x = pd.DataFrame({
        "log_ticket": np.log1p(f["ticket_visita"]),
        "log_visitas": np.log(f["frecuencia_visitas"]),
        "log_lineas_visita": np.log(f["lineas_por_visita"]),
        "log_categorias": np.log(f["n_categorias"]),
        **{v: f[v] for v in VARIABLES if v.startswith("share_")},
    })[VARIABLES]
    x_std = ((x - x.mean()) / x.std(ddof=1)).to_numpy()
    # PCA como prcomp(center = FALSE, scale. = FALSE): SVD de la matriz ya estandarizada
    _, sv, vt = np.linalg.svd(x_std, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    n_comp = int(np.searchsorted(np.cumsum(var), VARIANZA_OBJETIVO) + 1)
    scores = x_std @ vt[:n_comp].T
    pca = pd.DataFrame({"componente": np.arange(1, len(var) + 1), "varianza": var, "acumulada": np.cumsum(var)})
    seg = pd.read_parquet(c.SEGMENTOS_PARQUET)
    f = f.merge(seg[["cliente", "segmento_kmeans"]], on="cliente", how="left")
    return f, x_std, scores, pca


def ajustar(scores: np.ndarray, k: int) -> GaussianMixture:
    """GMM de covarianza completa con varios inicios."""
    return GaussianMixture(n_components=k, covariance_type="full", n_init=INICIOS, max_iter=500,
                           random_state=c.SEMILLA).fit(scores)


def graficar_bic(carpeta, t: pd.DataFrame) -> None:
    """BIC y AIC según el número de componentes."""
    fig, ax = c.plt.subplots(figsize=(9, 5))
    ax.plot(t["k"], t["bic"] / 1e6, marker="o", color=c.AZUL, lw=2, label="BIC")
    ax.plot(t["k"], t["aic"] / 1e6, marker="o", color=c.NARANJA, lw=2, label="AIC")
    mejor = t.loc[t["bic"].idxmin()]
    ax.annotate(f"Mínimo BIC: k = {int(mejor.k)}", (mejor.k, mejor.bic / 1e6), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=9, color=c.AZUL)
    ax.axvline(K_COMPARACION, color=c.EJE, ls="--", lw=1)
    ax.set_xticks(list(t["k"]))
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Criterio (millones; menor es mejor)")
    ax.legend(loc="upper right")
    c.guardar_figura(fig, carpeta, "01_bic_componentes", "Mezclas gaussianas: ¿cuántos componentes?",
                     f"Mismo espacio que con_pca_90 (PCA de 8 variables, 5 componentes). Línea punteada: k = {K_COMPARACION}.")


def graficar_cruce(carpeta, cruce: pd.DataFrame) -> None:
    """Porcentaje de cada segmento K-means que cae en cada componente GMM."""
    pct = cruce.div(cruce.sum(axis=1), axis=0) * 100
    fig, ax = c.plt.subplots(figsize=(10, 5.5))
    cmap = c.g.LinearSegmentedColormap.from_list("azul", [c.SUPERFICIE, c.AZUL])
    ax.imshow(pct.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.iat[i, j]
            ax.text(j, i, c.num(v, 0), ha="center", va="center", fontsize=9, color="white" if v > 55 else c.TINTA)
    ax.set_yticks(range(len(pct)), [f"{s}. {c.NOMBRES_SEGMENTOS[s]}" for s in pct.index])
    ax.set_xticks(range(pct.shape[1]), [f"G{k}" for k in pct.columns])
    ax.set_xlabel("Componente GMM (numerado por el segmento K-means con el que más coincide)")
    ax.grid(False)
    c.guardar_figura(fig, carpeta, "02_cruce_kmeans_gmm", "¿Coinciden K-means y GMM?",
                     f"Porcentaje de cada segmento K-means (fila) en cada componente GMM con k = {K_COMPARACION}.")


def graficar_ambiguos(carpeta, amb: pd.DataFrame) -> None:
    """Porcentaje de clientes ambiguos por segmento K-means."""
    t = amb.sort_values("segmento")
    fig, ax = c.plt.subplots(figsize=(10, 5))
    barras = ax.bar([c.NOMBRES_SEGMENTOS[s].replace(" en una", "\nen una") for s in t["segmento"]],
                    t["pct_ambiguos"], color=c.NARANJA, width=0.6)
    c.rotular_barras(ax, barras, [c.pct(v) for v in t["pct_ambiguos"]])
    ax.yaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.pct(v, 0)))
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8.5)
    c.guardar_figura(fig, carpeta, "03_ambiguos_por_segmento", "Clientes ambiguos por segmento",
                     f"Probabilidad máxima de pertenencia a un componente GMM < {c.num(UMBRAL_AMBIGUO, 1)} (k = {K_COMPARACION}).")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    f, x_std, scores, pca = preparar()
    n_comp = scores.shape[1]
    print(f"  Consumidores: {len(f):,} | componentes PCA: {n_comp} ({pca['acumulada'].iloc[n_comp - 1]:.1%} de varianza)")
    pca.to_csv(carpeta / "pca_varianza.csv", index=False, float_format=c.FORMATO_CSV)

    filas, modelos = [], {}
    for k in K_CANDIDATOS:
        m = ajustar(scores, k)
        modelos[k] = m
        filas.append({"k": k, "bic": m.bic(scores), "aic": m.aic(scores), "log_verosimilitud_media": m.score(scores),
                      "convergio": m.converged_})
        print(f"  k = {k}: BIC {filas[-1]['bic']:,.0f}")
    bic = pd.DataFrame(filas)
    bic.to_csv(carpeta / "bic_por_k.csv", index=False, float_format=c.FORMATO_CSV)
    graficar_bic(carpeta, bic)

    # Comparación con K-means con k = 6
    m6 = modelos[K_COMPARACION]
    prob = m6.predict_proba(scores)
    comp = prob.argmax(axis=1)
    # Renumerar cada componente según el segmento K-means con el que más coincide (solo para leerlo)
    cruce0 = pd.crosstab(f["segmento_kmeans"], comp)
    mapa, usados = {}, set()
    for j in cruce0.max(axis=0).sort_values(ascending=False).index:
        candidatos = [s for s in cruce0[j].sort_values(ascending=False).index if s not in usados]
        mapa[j] = candidatos[0] if candidatos else 100 + j
        usados.add(mapa[j])
    f["componente_gmm"] = [mapa[j] for j in comp]
    f["prob_max"] = prob.max(axis=1)
    f["ambiguo"] = f["prob_max"] < UMBRAL_AMBIGUO
    ari = adjusted_rand_score(f["segmento_kmeans"], f["componente_gmm"])
    cruce = pd.crosstab(f["segmento_kmeans"], f["componente_gmm"])
    cruce.to_csv(carpeta / "cruce_kmeans_gmm.csv", float_format=c.FORMATO_CSV)
    graficar_cruce(carpeta, cruce)
    coincide = (f["segmento_kmeans"] == f["componente_gmm"]).mean() * 100
    print(f"  Rand ajustado K-means vs GMM (k = {K_COMPARACION}): {ari:.3f} | mismo grupo: {coincide:.1f}%")

    amb = (f.groupby("segmento_kmeans").agg(clientes=("cliente", "size"), pct_ambiguos=("ambiguo", "mean"),
                                            prob_max_mediana=("prob_max", "median"),
                                            pct_mismo_grupo=("componente_gmm", lambda s: 0.0))
           .reset_index().rename(columns={"segmento_kmeans": "segmento"}))
    amb["pct_mismo_grupo"] = [(f.loc[f["segmento_kmeans"] == s, "componente_gmm"] == s).mean() * 100 for s in amb["segmento"]]
    amb["pct_ambiguos"] *= 100
    amb.insert(1, "nombre_segmento", amb["segmento"].map(c.NOMBRES_SEGMENTOS))
    amb.to_csv(carpeta / "ambiguos_por_segmento.csv", index=False, float_format=c.FORMATO_CSV)
    pd.DataFrame([{"k_minimo_bic": int(bic.loc[bic["bic"].idxmin(), "k"]), "rand_ajustado_k6": ari,
                   "pct_mismo_grupo": coincide, "pct_ambiguos": f["ambiguo"].mean() * 100}]).to_csv(
        carpeta / "metricas.csv", index=False, float_format=c.FORMATO_CSV)
    graficar_ambiguos(carpeta, amb)

    # Perfil de los componentes GMM (variables estandarizadas, como el mapa de calor de segmentacion.R)
    perfil = pd.DataFrame(x_std, columns=VARIABLES).groupby(f["componente_gmm"].values).mean()
    perfil.insert(0, "clientes", f.groupby("componente_gmm").size())
    perfil.insert(1, "pct_ventas", f.groupby("componente_gmm")["monto"].sum() / f["monto"].sum() * 100)
    perfil.index.name = "componente_gmm"
    perfil.to_csv(carpeta / "perfil_componentes_gmm.csv", float_format=c.FORMATO_CSV)

    f[["cliente", "segmento_kmeans", "componente_gmm", "prob_max", "ambiguo"]].to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(f):,} clientes)")
    escribir_resumen(carpeta, f, pca, n_comp, bic, ari, coincide, amb, perfil, cruce)


def escribir_resumen(carpeta, f, pca, n_comp, bic, ari, coincide, amb, perfil, cruce) -> None:
    """resumen.md del modelo."""
    mejor = int(bic.loc[bic["bic"].idxmin(), "k"])
    filas_b = "\n".join(f"| {int(r.k)} | {c.miles(r.bic)} | {c.miles(r.aic)} |" for r in bic.itertuples())
    filas_a = "\n".join(f"| {r.segmento}. {r.nombre_segmento} | {c.miles(r.clientes)} | {c.pct(r.pct_ambiguos)} | "
                        f"{c.num(r.prob_max_mediana, 2)} | {c.pct(r.pct_mismo_grupo)} |" for r in amb.itertuples())
    total_amb = f["ambiguo"].mean() * 100
    mas_amb = amb.sort_values("pct_ambiguos", ascending=False).iloc[0]
    menos_amb = amb.sort_values("pct_ambiguos").iloc[0]
    caida = (bic["bic"].iloc[-1] - bic["bic"].iloc[0]) / abs(bic["bic"].iloc[0]) * 100
    fila2 = cruce.loc[2] / cruce.loc[2].sum() * 100
    destino2 = int(fila2.idxmax())
    pct_destino2 = fila2.max()
    puro = perfil.loc[destino2, "share_revestimientos_e_instalacion"] > 0.5
    nombre_destino = f"G{destino2}" + (" (casi solo Revestimientos)" if puro else "")
    solo = amb.loc[amb["segmento"] >= 3, "pct_mismo_grupo"]
    puros = [k for k in perfil.index
             if (perfil.loc[k, [v for v in VARIABLES if v.startswith("share_")]] > 1.5).any()]
    if ari >= 0.8:
        lectura = "las dos técnicas encuentran prácticamente los mismos grupos"
    elif ari >= 0.5:
        lectura = "las dos técnicas coinciden en la estructura general pero difieren en los bordes"
    else:
        lectura = "las dos técnicas parten los datos de forma bastante distinta"
    texto = f"""
# m02 - Mezclas gaussianas (GMM) como comparación de la segmentación

## Qué hace
Ajusta mezclas gaussianas (covarianza completa, `sklearn`) en el **mismo espacio** que la segmentación
de trabajo `con_pca_90`: las 8 variables de comportamiento de `segmentacion.R`, estandarizadas igual
que `scale()` de R, y su PCA con {n_comp} componentes ({c.pct(pca['acumulada'].iloc[n_comp - 1] * 100)} de la varianza). A
diferencia de K-means, GMM asigna a cada cliente una probabilidad de pertenecer a cada grupo, lo que
permite medir qué tan nítida es la frontera entre segmentos.

## Datos usados
- `data/features_clientes_{c.VERSION}.parquet` (mismo filtro que `segmentacion.R`: consumidores finales con
  monto > 0, {c.miles(len(f))} clientes) y `data/segmentos_{c.VERSION}_{c.SEGMENTACION}.parquet` para comparar.

## Métricas
| Componentes | BIC | AIC |
|---|---|---|
{filas_b}

El BIC es mínimo con **k = {mejor}** (de 2 a 8 baja {c.num(abs(caida), 1)}%): con datos tan grandes el BIC
tiende a preferir más componentes, por lo que no contradice el k = 6 elegido por el codo en K-means.

Con k = {K_COMPARACION}: **Rand ajustado K-means vs GMM = {c.num(ari, 3)}**; {c.pct(coincide)} de los clientes cae en el
componente GMM que corresponde a su segmento K-means. Clientes ambiguos (probabilidad máxima <
{c.num(UMBRAL_AMBIGUO, 1)}): **{c.pct(total_amb)}**.

| Segmento K-means | Clientes | % ambiguos | Prob. máxima mediana | % en su componente GMM |
|---|---|---|---|---|
{filas_a}

## Hallazgos principales
- Con un Rand ajustado de {c.num(ari, 3)}, {lectura}.
- El segmento con más clientes ambiguos es **{mas_amb.nombre_segmento}** ({c.pct(mas_amb.pct_ambiguos)}) y el más
  nítido **{menos_amb.nombre_segmento}** ({c.pct(menos_amb.pct_ambiguos)}). Los ambiguos son clientes entre dos perfiles:
  conviene tratarlos con mensajes menos específicos.
- **GMM organiza a los clientes por la mezcla de negocios, K-means por valor y frecuencia.** Los
  componentes {", ".join(f"G{k}" for k in puros)} son clientes de un solo negocio (participación cercana a 1) y GMM
  los separa con probabilidad casi 1. En cambio, {c.pct(pct_destino2)} del segmento K-means "Proyecto grande en
  una visita" cae en {nombre_destino}: GMM no distingue a quien compró mucho en una visita de quien
  compró poco, si los dos compraron el mismo negocio.
- La ambigüedad es baja ({c.pct(total_amb)}) no porque los segmentos sean nítidos, sino porque las participaciones
  por negocio se concentran en 0 y 1 y los componentes gaussianos quedan casi degenerados: la
  probabilidad de pertenencia de GMM es poco informativa en estos datos.
- Para el negocio conviene mantener K-means como segmentación de trabajo: separa justamente valor y
  recurrencia, que es lo que usan m08, m11 y m14. GMM queda como verificación de que la estructura por
  negocio es robusta (los segmentos "Solo ..." coinciden en {c.pct(solo.min(), 0)} a {c.pct(solo.max(), 0)} con sus componentes).

## Limitaciones
- GMM supone grupos gaussianos en el espacio del PCA; las participaciones por negocio están muy
  concentradas en 0 y 1, así que los componentes no son exactamente gaussianos.
- La numeración de los componentes GMM se alinea con K-means solo para leer la comparación.
- El umbral de ambigüedad ({c.num(UMBRAL_AMBIGUO, 1)}) es una convención.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
