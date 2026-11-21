"""
m03 - Segmentación de empresas

Universo: clientes con tipo_cliente = 'Empresa' y monto > 0 (data/features_clientes_<version>.parquet).

1. Clasificación ABC por monto: A = empresas que suman el 80% de las ventas de empresas (ordenadas de
   mayor a menor), B = el siguiente 15%, C = el resto.
2. K-means sobre las mismas 8 variables de comportamiento de los consumidores (las de segmentacion.R)
   más log(monto), estandarizadas. k entre 2 y 8 elegido por silueta (muestra de 5.000 empresas).
3. Perfil de cada grupo y cruce con el ABC (y con el tamaño de empresa que ya trae la base).

Escribe:
    data/modelos/m03_empresas.parquet          ABC y grupo K-means por empresa
    salidas/modelos/m03_empresas/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m03_empresas.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import comun as c

NOMBRE = "m03_empresas"
VARIABLES = ["log_ticket", "log_visitas", "log_lineas_visita", "log_categorias",
             "share_revestimientos_e_instalacion", "share_banos", "share_cocinas",
             "share_pinturas_e_impermeabilizantes", "log_monto"]
ETIQUETAS = {
    "log_ticket": "Valor de la visita", "log_visitas": "Número de visitas", "log_lineas_visita": "Productos por visita",
    "log_categorias": "Variedad de categorías", "share_revestimientos_e_instalacion": "Gasto en Revestimientos",
    "share_banos": "Gasto en Baños", "share_cocinas": "Gasto en Cocinas",
    "share_pinturas_e_impermeabilizantes": "Gasto en Pinturas", "log_monto": "Monto total",
}
K_CANDIDATOS = range(2, 9)
MUESTRA_SILUETA = 5_000
CORTES_ABC = (0.80, 0.95)


def cargar() -> pd.DataFrame:
    """Empresas con monto positivo y las variables de comportamiento."""
    c.verificar_insumos(c.FEATURES_PARQUET)
    f = pd.read_parquet(c.FEATURES_PARQUET).sort_values("cliente")
    f = f[(f["tipo_cliente"] == "Empresa") & (f["monto"] > 0)].reset_index(drop=True)
    f["log_ticket"] = np.log1p(f["ticket_visita"])
    f["log_visitas"] = np.log(f["frecuencia_visitas"])
    f["log_lineas_visita"] = np.log(f["lineas_por_visita"])
    f["log_categorias"] = np.log(f["n_categorias"])
    f["log_monto"] = np.log1p(f["monto"])
    return f


def clasificar_abc(f: pd.DataFrame) -> pd.Series:
    """A/B/C según la participación acumulada en el monto (ordenado de mayor a menor)."""
    orden = f["monto"].sort_values(ascending=False)
    acumulado = orden.cumsum() / orden.sum()
    # Una empresa es A si antes de ella el acumulado no había llegado al 80%
    previo = acumulado.shift(fill_value=0)
    clase = np.where(previo < CORTES_ABC[0], "A", np.where(previo < CORTES_ABC[1], "B", "C"))
    return pd.Series(clase, index=orden.index).reindex(f.index)


def elegir_k(x: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """K-means para cada k y silueta sobre una muestra fija."""
    rng = np.random.default_rng(c.SEMILLA)
    muestra = rng.choice(len(x), size=min(MUESTRA_SILUETA, len(x)), replace=False)
    filas, modelos = [], {}
    for k in K_CANDIDATOS:
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=c.SEMILLA).fit(x)
        modelos[k] = km
        filas.append({"k": k, "inercia": km.inertia_, "silueta": silhouette_score(x[muestra], km.labels_[muestra])})
        print(f"  k = {k}: silueta {filas[-1]['silueta']:.3f}")
    return pd.DataFrame(filas), modelos


def graficar_abc(carpeta, f: pd.DataFrame) -> None:
    """Curva de concentración de las ventas de empresas con los cortes ABC."""
    orden = f["monto"].sort_values(ascending=False).to_numpy()
    x = np.arange(1, len(orden) + 1) / len(orden) * 100
    y = np.cumsum(orden) / orden.sum() * 100
    fig, ax = c.plt.subplots(figsize=(9, 5.5))
    ax.plot(x, y, color=c.AZUL, lw=2)
    for corte, nombre in zip(CORTES_ABC, ["A", "B"]):
        pos = np.searchsorted(y, corte * 100) / len(orden) * 100
        ax.axvline(pos, color=c.NARANJA, ls="--", lw=1)
        ax.annotate(f"Fin de {nombre}: {c.pct(pos)} de las empresas = {c.pct(corte * 100, 0)} de las ventas",
                    (pos, corte * 100), xytext=(8, -14), textcoords="offset points", fontsize=9, color=c.NARANJA)
    ax.set_xlabel("Empresas (de mayor a menor monto, %)")
    ax.set_ylabel("Ventas acumuladas de empresas (%)")
    ax.xaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.pct(v, 0)))
    ax.yaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.pct(v, 0)))
    c.guardar_figura(fig, carpeta, "01_curva_abc", "Concentración de las ventas a empresas",
                     "Clasificación ABC: A = 80% de las ventas, B = siguiente 15%, C = el resto.")


def graficar_silueta(carpeta, t: pd.DataFrame, k: int) -> None:
    """Silueta e inercia por k."""
    fig, a1 = c.plt.subplots(figsize=(9, 5))
    a1.plot(t["k"], t["silueta"], marker="o", color=c.AZUL, lw=2, label="Silueta")
    a1.set_ylabel("Silueta (mayor es mejor)")
    a1.axvline(k, color=c.NARANJA, ls="--", lw=1)
    a1.set_xticks(list(t["k"]))
    a1.set_xlabel("Número de grupos (k)")
    a2 = a1.twinx()
    a2.plot(t["k"], t["inercia"] / 1e3, marker="s", color=c.TENUE, lw=1.5, label="Inercia")
    a2.set_ylabel("Inercia (miles)", color=c.TENUE)
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    c.guardar_figura(fig, carpeta, "02_eleccion_de_k", "Elección de k para las empresas",
                     f"K-means sobre 9 variables estandarizadas; k = {k} maximiza la silueta.")


def graficar_perfil(carpeta, perfil: pd.DataFrame) -> None:
    """Mapa de calor del perfil estandarizado de cada grupo."""
    m = perfil[VARIABLES]
    fig, ax = c.plt.subplots(figsize=(11, 0.7 * len(m) + 2.6))
    cmap = c.g.LinearSegmentedColormap.from_list("div", [c.ROJO, c.GRIS_MEDIO, c.AZUL])
    ax.imshow(m.values.clip(-2, 2), cmap=cmap, vmin=-2, vmax=2, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, c.num(m.iat[i, j], 1), ha="center", va="center", fontsize=8.5)
    ax.set_xticks(range(m.shape[1]), [ETIQUETAS[v] for v in VARIABLES], rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(m)), [f"Grupo {g} ({c.miles(n)})" for g, n in zip(perfil.index, perfil["empresas"])])
    ax.grid(False)
    c.guardar_figura(fig, carpeta, "03_perfil_grupos", "Perfil de los grupos de empresas",
                     "Media de cada variable estandarizada (azul = por encima del promedio de las empresas).")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    f = cargar()
    f["abc"] = clasificar_abc(f)
    abc = f.groupby("abc").agg(empresas=("cliente", "size"), ventas=("monto", "sum"),
                               monto_mediano=("monto", "median")).reset_index()
    abc["pct_empresas"] = abc["empresas"] / abc["empresas"].sum() * 100
    abc["pct_ventas"] = abc["ventas"] / abc["ventas"].sum() * 100
    abc.to_csv(carpeta / "abc.csv", index=False, float_format=c.FORMATO_CSV)
    print(abc.to_string(index=False))
    graficar_abc(carpeta, f)

    x = f[VARIABLES].to_numpy()
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    sil, modelos = elegir_k(x)
    sil.to_csv(carpeta / "silueta_por_k.csv", index=False, float_format=c.FORMATO_CSV)
    k = int(sil.loc[sil["silueta"].idxmax(), "k"])
    graficar_silueta(carpeta, sil, k)
    a = abc.set_index("abc")
    pd.DataFrame([{"empresas": len(f), "k": k, "silueta": sil["silueta"].max(),
                   "empresas_a": int(a.loc["A", "empresas"]), "pct_empresas_a": a.loc["A", "pct_empresas"]}]).to_csv(
        carpeta / "metricas.csv", index=False, float_format=c.FORMATO_CSV)

    # Grupos numerados de mayor a menor monto mediano (como segmentacion.R)
    etiquetas = modelos[k].labels_
    orden = pd.Series(f["monto"].values).groupby(etiquetas).median().sort_values(ascending=False).index
    f["grupo"] = pd.Series(etiquetas).map({g: i + 1 for i, g in enumerate(orden)}).values

    perfil = pd.DataFrame(x, columns=VARIABLES).groupby(f["grupo"].values).mean()
    resumen = f.groupby("grupo").agg(
        empresas=("cliente", "size"), ventas=("monto", "sum"), monto_mediano=("monto", "median"),
        ticket_mediano=("ticket_visita", "median"), visitas_mediana=("frecuencia_visitas", "median"),
        categorias_mediana=("n_categorias", "median"), pct_revestimientos=("share_revestimientos_e_instalacion", "mean"),
        pct_banos=("share_banos", "mean"), pct_cocinas=("share_cocinas", "mean"),
        pct_pinturas=("share_pinturas_e_impermeabilizantes", "mean"), recencia_mediana=("recencia_dias", "median"))
    resumen["pct_empresas"] = resumen["empresas"] / resumen["empresas"].sum() * 100
    resumen["pct_ventas"] = resumen["ventas"] / resumen["ventas"].sum() * 100
    for col in ("pct_revestimientos", "pct_banos", "pct_cocinas", "pct_pinturas"):
        resumen[col] *= 100
    perfil = perfil.join(resumen[["empresas"]])
    perfil.index.name = "grupo"
    resumen.to_csv(carpeta / "perfil_grupos.csv", float_format=c.FORMATO_CSV)
    perfil.to_csv(carpeta / "perfil_grupos_estandarizado.csv", float_format=c.FORMATO_CSV)
    graficar_perfil(carpeta, perfil)

    cruce = pd.crosstab(f["grupo"], f["abc"])
    cruce_ventas = pd.crosstab(f["grupo"], f["abc"], values=f["monto"], aggfunc="sum", normalize="columns") * 100
    cruce.to_csv(carpeta / "cruce_grupo_abc.csv", float_format=c.FORMATO_CSV)
    cruce_ventas.to_csv(carpeta / "cruce_grupo_abc_pct_ventas.csv", float_format=c.FORMATO_CSV)
    tam = pd.crosstab(f["grupo"], f["segmento_empresa"])
    tam.to_csv(carpeta / "cruce_grupo_tamano.csv", float_format=c.FORMATO_CSV)
    print(resumen[["empresas", "pct_ventas", "monto_mediano", "visitas_mediana"]].to_string())
    print(cruce.to_string())

    f[["cliente", "abc", "grupo", "segmento_empresa", "monto", "frecuencia_visitas"]].to_parquet(
        c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(f):,} empresas)")
    escribir_resumen(carpeta, f, abc, sil, k, resumen, perfil, cruce, cruce_ventas)


def describir(fila: pd.Series, perfil: pd.Series) -> str:
    """Descripción corta de un grupo a partir de su perfil estandarizado."""
    rasgos = []
    nombres = {"log_monto": ("monto alto", "monto bajo"), "log_visitas": ("muchas visitas", "pocas visitas"),
               "log_ticket": ("ticket alto", "ticket bajo"), "log_categorias": ("muchas categorías", "pocas categorías")}
    for v, (alto, bajo) in nombres.items():
        if perfil[v] > 0.5:
            rasgos.append(alto)
        elif perfil[v] < -0.5:
            rasgos.append(bajo)
    negocio = max(["pct_revestimientos", "pct_banos", "pct_cocinas", "pct_pinturas"], key=lambda k: fila[k])
    if fila[negocio] >= 60:
        rasgos.append("casi solo " + {"pct_revestimientos": "Revestimientos", "pct_banos": "Baños",
                                      "pct_cocinas": "Cocinas", "pct_pinturas": "Pinturas"}[negocio])
    return ", ".join(rasgos) or "perfil promedio"


def escribir_resumen(carpeta, f, abc, sil, k, resumen, perfil, cruce, cruce_ventas) -> None:
    """resumen.md del modelo."""
    a = abc.set_index("abc")
    filas_abc = "\n".join(f"| {i} | {c.miles(r.empresas)} | {c.pct(r.pct_empresas)} | {c.pct(r.pct_ventas)} | "
                          f"{c.miles(r.monto_mediano)} |" for i, r in a.iterrows())
    filas_sil = " · ".join(f"k={int(r.k)}: {c.num(r.silueta, 3)}" for r in sil.itertuples())
    filas_g = "\n".join(
        f"| {g} | {c.miles(r.empresas)} | {c.pct(r.pct_ventas)} | {c.miles(r.monto_mediano)} | {c.num(r.visitas_mediana, 0)} | "
        f"{c.miles(r.ticket_mediano)} | {describir(r, perfil.loc[g])} | "
        + " / ".join(c.miles(cruce.loc[g].get(x, 0)) for x in "ABC") + " |"
        for g, r in resumen.iterrows())
    grupo_a = cruce["A"].idxmax() if "A" in cruce else None
    pct_a_en = cruce_ventas.loc[grupo_a, "A"] if grupo_a is not None else np.nan
    texto = f"""
# m03 - Segmentación de empresas

## Qué hace
1. **ABC por monto:** A = empresas que suman el 80% de las ventas a empresas, B = el siguiente 15%,
   C = el resto.
2. **K-means** sobre las mismas 8 variables de comportamiento de los consumidores (las de
   `segmentacion.R`) más `log(monto)`, estandarizadas; k entre 2 y 8 elegido por silueta (muestra de
   {c.miles(MUESTRA_SILUETA)} empresas, semilla {c.SEMILLA}). Los grupos se numeran de mayor a menor monto mediano.
3. Perfil de cada grupo y cruce con el ABC.

## Datos usados
`data/features_clientes_{c.VERSION}.parquet`: {c.miles(len(f))} empresas con monto > 0 ({c.miles(f['monto'].sum())} en ventas,
unidades de la base).

## Métricas
### ABC
| Clase | Empresas | % empresas | % ventas | Monto mediano |
|---|---|---|---|---|
{filas_abc}

### K-means
Silueta por k: {filas_sil}. **k = {k}** (silueta {c.num(sil['silueta'].max(), 3)}).

| Grupo | Empresas | % ventas | Monto mediano | Visitas (mediana) | Ticket mediano | Rasgos | Empresas A / B / C |
|---|---|---|---|---|---|---|---|
{filas_g}

## Hallazgos principales
- Las ventas a empresas están muy concentradas: **{c.miles(a.loc['A', 'empresas'])} empresas ({c.pct(a.loc['A', 'pct_empresas'])})
  suman el 80% de las ventas**; la clase C ({c.pct(a.loc['C', 'pct_empresas'])} de las empresas) aporta solo
  {c.pct(a.loc['C', 'pct_ventas'])}.
- El grupo {grupo_a} concentra la mayoría de las empresas A ({c.pct(pct_a_en)} de las ventas de la clase A):
  es el grupo de cuentas clave, que requiere gestión por asesor y no campañas masivas.
- Los demás grupos se diferencian por el negocio y la recurrencia, igual que en los consumidores; los
  grupos con pocas visitas y monto bajo son candidatos a canales de autoservicio o digitales.

## Limitaciones
- "Empresa" es la marca del maestro de clientes; incluye desde contratistas pequeños hasta
  distribuidores, con comportamientos muy distintos.
- La silueta con muchas variables de participación (concentradas en 0 y 1) favorece pocos grupos;
  el k elegido es el más separable, no necesariamente el más útil comercialmente.
- El monto está en unidades de la base y corresponde a ~31 meses de historia.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
