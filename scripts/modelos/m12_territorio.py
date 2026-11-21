"""
m12 - Potencial por departamento con datos externos del DANE

Cruza los consumidores finales de Corona con dos fuentes del DANE por departamento:
    principal:     proyecciones (hogares y población) del año más reciente en 2023-2025
    verificación:  Censo Nacional de Población y Vivienda (CNPV) 2018 (hogares y población)

Con cada fuente calcula por departamento:
    - clientes por cada 1.000 hogares y por cada 1.000 habitantes
    - ventas por hogar y por habitante
    - gasto mediano por cliente
    - regresión log(clientes) ~ log(hogares): el residuo indica dónde hay menos clientes
      de los esperados para su tamaño
    - matriz de cuatro cuadrantes (penetración frente a gasto mediano)
    - ranking de oportunidad = clientes faltantes frente a lo esperado x gasto mediano
Después compara las dos fuentes (Spearman entre rankings y cambios de cuadrante).

Requiere los archivos del DANE en data/externos/ (ver data/externos/README.md).

Escribe:
    data/modelos/m12_territorio.parquet           una fila por departamento
    salidas/modelos/m12_territorio/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m12_territorio.py
"""
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

import comun as c

NOMBRE = "m12_territorio"
ANIOS_PROYECCION = (2023, 2024, 2025)  # se usa el más reciente disponible

ARCHIVOS = {
    "censo_hogares": c.DATA_EXTERNOS / "HOGARES_Cuadros_CNPV_2018.XLSX",
    "censo_poblacion": c.DATA_EXTERNOS / "CNPV-2018-Poblacion-Ajustada-por-Cobertura.xls",
    "proy_hogares": c.DATA_EXTERNOS / "anexo-proyecciones-hogares-dptal-mpal-2018-2042.xlsx",
    "proy_poblacion": c.DATA_EXTERNOS / "PPED-AreaDep-2018-2050_VP.xlsx",
}

# Nombre de la región en las ventas (sin tildes, como lo deja comun.normalizar) -> código DIVIPOLA
CODIGOS_DIVIPOLA = {
    "ANTIOQUIA": "05", "ATLANTICO": "08", "BOGOTA": "11", "BOLIVAR": "13", "BOYACA": "15",
    "CALDAS": "17", "CAQUETA": "18", "CAUCA": "19", "CESAR": "20", "CORDOBA": "23",
    "CUNDINAMARCA": "25", "CHOCO": "27", "HUILA": "41", "LA GUAJIRA": "44", "MAGDALENA": "47",
    "META": "50", "NARINO": "52", "NORTE SANTANDER": "54", "QUINDIO": "63", "RISARALDA": "66",
    "SANTANDER": "68", "SUCRE": "70", "TOLIMA": "73", "VALLE": "76", "ARAUCA": "81",
    "CASANARE": "85", "PUTUMAYO": "86", "SAN ANDRES": "88", "AMAZONAS": "91", "GUAINIA": "94",
    "GUAVIARE": "95", "VAUPES": "97", "VICHADA": "99",
}
CUADRANTES = {
    (True, True): "Consolidar (alta penetración, alto gasto)",
    (False, True): "Expandir (baja penetración, alto gasto)",
    (True, False): "Subir ticket (alta penetración, bajo gasto)",
    (False, False): "Baja prioridad (baja penetración, bajo gasto)",
}
COLOR_CUADRANTE = dict(zip(CUADRANTES.values(), [c.AZUL, c.NARANJA, c.VERDE, c.TENUE]))


# ---------------------------------------------------------------------------
# Lectura de las fuentes del DANE
# ---------------------------------------------------------------------------
def hoja(xls: pd.ExcelFile, contiene: str) -> str:
    """Nombre de la hoja que contiene un texto (los nombres traen tildes)."""
    for nombre in xls.sheet_names:
        if c.normalizar(contiene) in c.normalizar(nombre):
            return nombre
    raise ValueError(f"No encontré una hoja con '{contiene}' en {xls.io}")


def leer_censo_hogares() -> pd.Series:
    """CNPV 2018, cuadro 1HD: total de hogares por código de departamento."""
    d = pd.read_excel(ARCHIVOS["censo_hogares"], sheet_name="1HD", header=None)
    d = d[d[0].astype(str).str.match(r"^\d{2}_")]
    return pd.Series(d[1].astype(float).values, index=d[0].str[:2], name="hogares")


def leer_censo_poblacion() -> pd.Series:
    """CNPV 2018: población ajustada por cobertura por código de departamento."""
    xls = pd.ExcelFile(ARCHIVOS["censo_poblacion"])
    d = pd.read_excel(xls, sheet_name=hoja(xls, "Ajuste por Cobertura CNPV 2018"), header=None, dtype={0: str})
    d = d[d[0].astype(str).str.fullmatch(r"\d{2}") & (d[0] != "00")]
    return pd.Series(d[2].astype(float).values, index=d[0], name="poblacion")


def anio_proyeccion(disponibles) -> int:
    """El año más reciente de ANIOS_PROYECCION que esté en la fuente."""
    anios = [a for a in ANIOS_PROYECCION if a in set(int(x) for x in disponibles)]
    if not anios:
        raise ValueError(f"La proyección no trae ninguno de los años {ANIOS_PROYECCION}.")
    return max(anios)


def leer_proy_hogares() -> tuple[pd.Series, int]:
    """Proyección de hogares (área Total) del año más reciente en 2023-2025."""
    xls = pd.ExcelFile(ARCHIVOS["proy_hogares"])
    d = pd.read_excel(xls, sheet_name=hoja(xls, "Hogares dpto"), header=None, dtype={0: str})
    fila_anios = d.index[d[3].astype(str).str.fullmatch(r"20\d{2}(\.0)?")][0]
    anios = {int(float(v)): j for j, v in d.loc[fila_anios].items() if re.fullmatch(r"20\d{2}(\.0)?", str(v))}
    anio = anio_proyeccion(anios)
    d = d[d[0].astype(str).str.fullmatch(r"\d{2}") & (d[0] != "00") & (d[2].astype(str).str.strip() == "Total")]
    return pd.Series(d[anios[anio]].astype(float).values, index=d[0], name="hogares"), anio


def leer_proy_poblacion() -> tuple[pd.Series, int]:
    """Proyección de población (área Total) del año más reciente en 2023-2025."""
    xls = pd.ExcelFile(ARCHIVOS["proy_poblacion"])
    d = pd.read_excel(xls, sheet_name=hoja(xls, "PobDepartamental"), header=None, dtype={0: str})
    d = d[d[0].astype(str).str.fullmatch(r"\d{2}")]
    d = d[d[3].astype(str).str.strip() == "Total"]
    d[2] = d[2].astype(int)
    anio = anio_proyeccion(d[2].unique())
    d = d[d[2] == anio]
    return pd.Series(d[4].astype(float).values, index=d[0], name="poblacion"), anio


def cargar_fuentes() -> dict:
    """Las dos fuentes, cada una con hogares y población por código DIVIPOLA."""
    for archivo in ARCHIVOS.values():
        if not archivo.exists():
            raise FileNotFoundError(
                f"Falta {archivo}. Descárgalo según data/externos/README.md y déjalo en data/externos/."
            )
    hog_p, anio_h = leer_proy_hogares()
    pob_p, anio_p = leer_proy_poblacion()
    fuentes = {
        f"Proyección DANE {anio_h}": pd.concat([hog_p, pob_p], axis=1),
        "Censo CNPV 2018": pd.concat([leer_censo_hogares(), leer_censo_poblacion()], axis=1),
    }
    for nombre, f in fuentes.items():
        if len(f) != 33 or f.isna().any().any():
            raise ValueError(f"{nombre}: esperaba 33 departamentos completos y hay {len(f)}.")
        print(f"  {nombre}: {f['hogares'].sum():,.0f} hogares, {f['poblacion'].sum():,.0f} habitantes")
    return fuentes


# ---------------------------------------------------------------------------
# Datos de Corona por departamento
# ---------------------------------------------------------------------------
def datos_corona() -> tuple[pd.DataFrame, dict]:
    """Clientes, ventas y gasto mediano de consumidores finales por departamento."""
    con = c.conexion()
    ventas = con.execute("""
        SELECT t.region, SUM(t.valor_corregido) AS ventas
        FROM trx t JOIN features f USING (cliente)
        WHERE f.tipo_cliente = 'Consumidor final'
        GROUP BY ALL
    """).df()
    total_ventas = ventas["ventas"].sum()
    ventas_sin = ventas.loc[ventas["region"].isna(), "ventas"].sum()
    # Referencia: SIN REGIÓN sobre todas las ventas (incluye empresas)
    todas = con.execute("""
        SELECT SUM(valor_corregido) AS total,
               SUM(CASE WHEN region IS NULL THEN valor_corregido ELSE 0 END) AS sin_region
        FROM trx
    """).fetchone()

    clientes = c.cargar_clientes()
    sin_region = clientes["region_principal"].isna()
    por_dep = (clientes[~sin_region].groupby("region_principal")
               .agg(clientes=("cliente", "size"), gasto_mediano=("monto", "median"),
                    pct_contactable=("es_contactable", "mean"))
               .reset_index().rename(columns={"region_principal": "region"}))
    por_dep = por_dep.merge(ventas.dropna(subset=["region"]), on="region", how="outer")
    por_dep["codigo"] = por_dep["region"].map(lambda r: CODIGOS_DIVIPOLA.get(c.normalizar(r)))
    faltan = por_dep.loc[por_dep["codigo"].isna(), "region"].tolist()
    if faltan:
        raise ValueError(f"Regiones sin código DIVIPOLA: {faltan}. Agrégalas a CODIGOS_DIVIPOLA.")
    info = {
        "ventas_total": total_ventas,
        "ventas_sin_region": ventas_sin,
        "pct_ventas_sin_region": ventas_sin / total_ventas * 100,
        "clientes_total": len(clientes),
        "clientes_sin_region": int(sin_region.sum()),
        "pct_ventas_sin_region_todas": todas[1] / todas[0] * 100,
    }
    print(f"  SIN REGIÓN: {info['pct_ventas_sin_region']:.1f}% de las ventas de consumidores finales, "
          f"{info['clientes_sin_region']:,} clientes sin región principal")
    return por_dep, info


# ---------------------------------------------------------------------------
# Análisis por fuente
# ---------------------------------------------------------------------------
def analizar(corona: pd.DataFrame, fuente: pd.DataFrame, nombre: str) -> tuple[pd.DataFrame, dict]:
    """Indicadores, regresión log-log, cuadrantes y ranking de oportunidad con una fuente."""
    d = corona.merge(fuente, left_on="codigo", right_index=True, how="left")
    d["fuente"] = nombre
    d["clientes_por_mil_hogares"] = d["clientes"] / d["hogares"] * 1000
    d["clientes_por_mil_habitantes"] = d["clientes"] / d["poblacion"] * 1000
    d["ventas_por_hogar"] = d["ventas"] / d["hogares"]
    d["ventas_por_habitante"] = d["ventas"] / d["poblacion"]

    # Regresión log(clientes) ~ log(hogares) y la misma con población
    reg = {}
    for base in ("hogares", "poblacion"):
        X = sm.add_constant(np.log(d[base]))
        modelo = sm.OLS(np.log(d["clientes"]), X).fit()
        reg[base] = modelo
        d[f"residuo_{base}"] = modelo.resid
    m = reg["hogares"]
    d["clientes_esperados"] = np.exp(m.fittedvalues)
    d["clientes_faltantes"] = (d["clientes_esperados"] - d["clientes"]).clip(lower=0)
    d["oportunidad_cop"] = d["clientes_faltantes"] * d["gasto_mediano"]

    # Cuadrantes: penetración y gasto mediano frente a la mediana de los departamentos
    alta_pen = d["clientes_por_mil_hogares"] >= d["clientes_por_mil_hogares"].median()
    alto_gasto = d["gasto_mediano"] >= d["gasto_mediano"].median()
    d["cuadrante"] = [CUADRANTES[(p, g)] for p, g in zip(alta_pen, alto_gasto)]
    d["muestra_pequena"] = d["clientes"] < c.g.MIN_CLIENTES_TERRITORIO
    d = d.sort_values(["oportunidad_cop", "residuo_hogares"], ascending=[False, True])
    d["ranking_oportunidad"] = np.arange(1, len(d) + 1)

    resumen_reg = {
        "fuente": nombre,
        "pendiente_hogares": m.params.iloc[1], "r2_hogares": m.rsquared,
        "pendiente_poblacion": reg["poblacion"].params.iloc[1], "r2_poblacion": reg["poblacion"].rsquared,
        "n_departamentos": len(d),
    }
    return d, resumen_reg


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def etiqueta(region: str) -> str:
    """'NORTE SANTANDER' -> 'Norte Santander'."""
    return str(region).title()


def graficar_regresion(carpeta, d: pd.DataFrame, nombre: str, reg: dict, archivo: str) -> None:
    """Dispersión log-log con la recta ajustada; resalta los 8 mayores residuos negativos."""
    fig, ax = c.plt.subplots(figsize=(10, 6.5))
    x, y = np.log10(d["hogares"]), np.log10(d["clientes"])
    ax.scatter(x, y, s=34, color=c.AZUL, zorder=3)
    xs = np.linspace(x.min(), x.max(), 50)
    b = reg["pendiente_hogares"]
    a = np.log10(d["clientes_esperados"]).mean() - b * x.mean()
    ax.plot(xs, a + b * xs, color=c.NARANJA, lw=1.6)
    bajos = d.nsmallest(8, "residuo_hogares")["region"]
    for _, r in d.iterrows():
        resaltar = r["region"] in set(bajos)
        ax.annotate(etiqueta(r["region"]), (np.log10(r["hogares"]), np.log10(r["clientes"])),
                    xytext=(4, 3), textcoords="offset points", fontsize=7.5,
                    color=c.ROJO if resaltar else c.TINTA_2, fontweight="bold" if resaltar else "normal")
    ax.set_xlabel("Hogares (log10)")
    ax.set_ylabel("Consumidores finales (log10)")
    c.guardar_figura(fig, carpeta, archivo, f"Clientes frente a hogares ({nombre})",
                     f"log(clientes) = a + {c.num(b, 2)}·log(hogares), R² = {c.num(reg['r2_hogares'], 2)}. "
                     "En rojo, los 8 departamentos con menos clientes de los esperados.",
                     nota=c.FUENTE + f" Hogares: DANE, {nombre}.")


def graficar_cuadrantes(carpeta, d: pd.DataFrame, nombre: str, archivo: str) -> None:
    """Matriz penetración frente a gasto mediano."""
    fig, ax = c.plt.subplots(figsize=(10.5, 7))
    for cuad, color in COLOR_CUADRANTE.items():
        s = d[d["cuadrante"] == cuad]
        ax.scatter(s["clientes_por_mil_hogares"], s["gasto_mediano"] / 1000,
                   s=np.sqrt(s["clientes"]) * 1.6, color=color, alpha=0.85, label=cuad, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(etiqueta(r["region"]), (r["clientes_por_mil_hogares"], r["gasto_mediano"] / 1000),
                    xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=c.TINTA_2)
    ax.axvline(d["clientes_por_mil_hogares"].median(), color=c.EJE, ls="--", lw=1)
    ax.axhline(d["gasto_mediano"].median() / 1000, color=c.EJE, ls="--", lw=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.num(v, 0) if v >= 1 else c.num(v, 1)))
    ax.set_xlabel("Clientes por cada 1.000 hogares (escala log)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.miles(v)))
    ax.set_ylabel("Gasto mediano por cliente (miles de COP, escala log)")
    leyenda = ax.legend(loc="upper right", fontsize=8.5)
    for marca in leyenda.legend_handles:  # mismo tamaño de marca en la leyenda
        marca.set_sizes([40])
    c.guardar_figura(fig, carpeta, archivo, f"Penetración frente a gasto ({nombre})",
                     "Líneas punteadas: medianas de los departamentos. El tamaño del punto indica el número de clientes.",
                     nota=c.FUENTE + f" Hogares: DANE, {nombre}.")


def graficar_ranking(carpeta, d: pd.DataFrame, nombre: str) -> None:
    """Top 12 de oportunidad (clientes faltantes x gasto mediano)."""
    top = d.head(12).iloc[::-1]
    fig, ax = c.plt.subplots(figsize=(10, 6))
    colores = [COLOR_CUADRANTE[q] for q in top["cuadrante"]]
    barras = ax.barh([etiqueta(r) for r in top["region"]], top["oportunidad_cop"] / 1e9, color=colores, height=0.65)
    c.rotular_barras(ax, barras, [f"{c.num(v / 1e9, 1)} mil M · {c.miles(f)} clientes"
                                  for v, f in zip(top["oportunidad_cop"], top["clientes_faltantes"])],
                     horizontal=True, size=8)
    ax.set_xlim(0, top["oportunidad_cop"].max() / 1e9 * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Oportunidad (miles de millones de COP)")
    c.guardar_figura(fig, carpeta, "05_ranking_oportunidad", f"Departamentos con oportunidad ({nombre})",
                     "Clientes faltantes frente a lo esperado por hogares x gasto mediano por cliente. Color = cuadrante.",
                     nota=c.FUENTE + f" Hogares: DANE, {nombre}.")


def graficar_comparacion(carpeta, comp: pd.DataFrame, nombres: list, rho: float) -> None:
    """Posición en el ranking con cada fuente."""
    fig, ax = c.plt.subplots(figsize=(8, 7))
    ax.scatter(comp["ranking_principal"], comp["ranking_verificacion"], s=30, color=c.AZUL, zorder=3)
    ax.plot([1, 33], [1, 33], color=c.EJE, ls="--", lw=1)
    for _, r in comp.iterrows():
        ax.annotate(etiqueta(r["region"]), (r["ranking_principal"], r["ranking_verificacion"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=7,
                    color=c.ROJO if r["cambia_cuadrante"] else c.TINTA_2)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(f"Puesto en el ranking ({nombres[0]})")
    ax.set_ylabel(f"Puesto en el ranking ({nombres[1]})")
    c.guardar_figura(fig, carpeta, "06_comparacion_fuentes", "¿Cambia el ranking según la fuente?",
                     f"Spearman = {c.num(rho, 3)}. En rojo, departamentos que cambian de cuadrante.",
                     nota=c.FUENTE + " Hogares: DANE.")


# ---------------------------------------------------------------------------
def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")

    fuentes = cargar_fuentes()
    corona, info = datos_corona()
    nombres = list(fuentes)  # [principal, verificación]

    resultados, regresiones = {}, []
    for nombre, fuente in fuentes.items():
        d, reg = analizar(corona, fuente, nombre)
        resultados[nombre] = d
        regresiones.append(reg)
    regresiones = pd.DataFrame(regresiones)
    regresiones.to_csv(carpeta / "regresiones.csv", index=False, float_format=c.FORMATO_CSV)
    print(regresiones.to_string(index=False))

    columnas = ["ranking_oportunidad", "region", "codigo", "clientes", "ventas", "gasto_mediano", "pct_contactable",
                "hogares", "poblacion", "clientes_por_mil_hogares", "clientes_por_mil_habitantes",
                "ventas_por_hogar", "ventas_por_habitante", "residuo_hogares", "residuo_poblacion",
                "clientes_esperados", "clientes_faltantes", "oportunidad_cop", "cuadrante", "muestra_pequena"]
    principal, verif = resultados[nombres[0]], resultados[nombres[1]]
    principal[columnas].to_csv(carpeta / "departamentos_proyeccion.csv", index=False, float_format=c.FORMATO_CSV)
    verif[columnas].to_csv(carpeta / "departamentos_censo2018.csv", index=False, float_format=c.FORMATO_CSV)
    print(principal[["ranking_oportunidad", "region", "clientes", "clientes_por_mil_hogares",
                     "gasto_mediano", "residuo_hogares", "oportunidad_cop", "cuadrante"]].head(12).to_string(index=False))

    # Comparación de fuentes
    comp = principal[["region", "codigo", "ranking_oportunidad", "cuadrante", "residuo_hogares"]].merge(
        verif[["codigo", "ranking_oportunidad", "cuadrante", "residuo_hogares"]], on="codigo",
        suffixes=("_principal", "_verificacion"))
    comp = comp.rename(columns={"ranking_oportunidad_principal": "ranking_principal",
                                "ranking_oportunidad_verificacion": "ranking_verificacion"})
    comp["cambia_cuadrante"] = comp["cuadrante_principal"] != comp["cuadrante_verificacion"]
    rho_rank = spearmanr(comp["ranking_principal"], comp["ranking_verificacion"]).statistic
    rho_resid = spearmanr(comp["residuo_hogares_principal"], comp["residuo_hogares_verificacion"]).statistic
    comp.sort_values("ranking_principal").to_csv(carpeta / "comparacion_fuentes.csv", index=False, float_format=c.FORMATO_CSV)
    top10_comun = len(set(principal.head(10)["codigo"]) & set(verif.head(10)["codigo"]))
    print(f"  Spearman rankings: {rho_rank:.3f} | residuos: {rho_resid:.3f} | "
          f"cambian de cuadrante: {int(comp['cambia_cuadrante'].sum())} | top-10 en común: {top10_comun}")

    # Gráficas
    graficar_regresion(carpeta, principal, nombres[0], regresiones.iloc[0], "01_regresion_proyeccion")
    graficar_regresion(carpeta, verif, nombres[1], regresiones.iloc[1], "02_regresion_censo2018")
    graficar_cuadrantes(carpeta, principal, nombres[0], "03_cuadrantes_proyeccion")
    graficar_cuadrantes(carpeta, verif, nombres[1], "04_cuadrantes_censo2018")
    graficar_ranking(carpeta, principal, nombres[0])
    graficar_comparacion(carpeta, comp, nombres, rho_rank)

    # Puntajes por departamento (fuente principal + verificación)
    salida = principal[columnas].merge(
        verif[["codigo", "ranking_oportunidad", "cuadrante", "residuo_hogares", "clientes_por_mil_hogares"]]
        .rename(columns=lambda x: x if x == "codigo" else f"{x}_censo2018"), on="codigo")
    salida.insert(0, "fuente_principal", nombres[0])
    salida.to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  {c.ruta_puntajes(NOMBRE).relative_to(c.RAIZ)} ({len(salida)} departamentos)")
    pd.DataFrame([info]).to_csv(carpeta / "cobertura_region.csv", index=False, float_format=c.FORMATO_CSV)

    escribir_resumen(carpeta, nombres, principal, verif, regresiones, comp, info, rho_rank, rho_resid, top10_comun)


def escribir_resumen(carpeta, nombres, principal, verif, regresiones, comp, info, rho_rank, rho_resid, top10) -> None:
    """resumen.md del modelo."""
    r = regresiones.set_index("fuente")
    top = principal.head(10)
    filas_top = "\n".join(
        f"| {x.ranking_oportunidad} | {etiqueta(x.region)} | {c.miles(x.clientes)} | "
        f"{c.num(x.clientes_por_mil_hogares, 1)} | {c.miles(x.gasto_mediano)} | {c.num(x.residuo_hogares, 2)} | "
        f"{c.miles(x.clientes_faltantes)} | {c.num(x.oportunidad_cop / 1e9, 1)} | {x.cuadrante.split(' (')[0]} | "
        f"{int(verif.set_index('codigo').loc[x.codigo, 'ranking_oportunidad'])} |"
        for x in top.itertuples())
    cambian = comp[comp["cambia_cuadrante"]]
    lista_cambian = ("ninguno" if cambian.empty else ", ".join(
        f"{etiqueta(x.region)} ({x.cuadrante_verificacion.split(' (')[0]} → {x.cuadrante_principal.split(' (')[0]})"
        for x in cambian.itertuples()))
    cuad = principal.groupby("cuadrante")["region"].apply(lambda s: ", ".join(etiqueta(v) for v in s))
    filas_cuad = "\n".join(f"- **{k}**: {v}" for k, v in cuad.items())
    pequenos = ", ".join(etiqueta(v) for v in principal.loc[principal["muestra_pequena"], "region"])
    cambian_conclusion = ("Las conclusiones **no cambian** de forma material"
                          if rho_rank >= 0.9 and top10 >= 8 else
                          "Las conclusiones cambian en algunos departamentos (ver tabla y lista)")
    texto = f"""
# m12 - Potencial por departamento con datos del DANE

## Qué hace
Cruza los consumidores finales de Corona por departamento con hogares y población del DANE y
calcula, con cada fuente: clientes por cada 1.000 hogares (y por cada 1.000 habitantes), ventas
por hogar (y por habitante) y gasto mediano por cliente. Ajusta `log(clientes) ~ log(hogares)`
(y la misma con población); el residuo negativo indica menos clientes de los esperados para el
tamaño del departamento. Clasifica los departamentos en cuatro cuadrantes (penetración frente a
gasto mediano, cortes en la mediana) y ordena la oportunidad como
`clientes faltantes frente a lo esperado x gasto mediano por cliente`.

## Datos usados
- Corona: `data/transaccional_{c.VERSION}.parquet` (ventas por región de la línea),
  `data/features_clientes_{c.VERSION}.parquet` (región principal y monto de cada cliente) y la
  segmentación `{c.SEGMENTACION}` (consumidores finales segmentados: {c.miles(info['clientes_total'])}).
- DANE (detalle, URL y fecha de descarga en `data/externos/README.md`):
  - **Fuente principal: {nombres[0]}** (hogares y población proyectados con base en el CNPV 2018).
  - **Verificación: {nombres[1]}** (hogares censados y población ajustada por cobertura).
- El cruce se hace por código DIVIPOLA (Bogotá, Valle, Norte de Santander, San Andrés y las
  tildes se resuelven en `CODIGOS_DIVIPOLA`). Los 33 departamentos cruzan.
- **SIN REGIÓN** se excluye: equivale a {c.pct(info['pct_ventas_sin_region'])} de las ventas de
  consumidores finales ({c.num(info['ventas_sin_region'] / 1e9, 1)} mil millones de COP); además
  {c.miles(info['clientes_sin_region'])} clientes no tienen región principal. Sobre el total de ventas
  (incluidas empresas) SIN REGIÓN pesa {c.pct(info['pct_ventas_sin_region_todas'])}: casi todo lo que no tiene
  región es venta a empresas, que este análisis no usa.

### Por qué la proyección es la fuente principal
El censo 2018 es el último conteo completo de hogares del país, pero los datos de Corona
corresponden a los últimos dos años y medio. La proyección del DANE para el año más cercano
({nombres[0].split()[-1]}) describe mejor el tamaño de cada departamento en ese periodo (crecimiento de
hogares, migración, hogares más pequeños). El censo se usa para verificar que las conclusiones no
dependan de la fuente.

## Métricas
| Fuente | Pendiente log-hogares | R² hogares | Pendiente log-población | R² población |
|---|---|---|---|---|
""" + "\n".join(
        f"| {f} | {c.num(x.pendiente_hogares, 2)} | {c.num(x.r2_hogares, 3)} | "
        f"{c.num(x.pendiente_poblacion, 2)} | {c.num(x.r2_poblacion, 3)} |" for f, x in r.iterrows()) + f"""

Comparación de fuentes: Spearman entre rankings de oportunidad = **{c.num(rho_rank, 3)}**;
Spearman entre residuos = {c.num(rho_resid, 3)}; {top10} de los 10 primeros coinciden; cambian de
cuadrante {int(comp['cambia_cuadrante'].sum())} departamentos: {lista_cambian}.

## Ranking de oportunidad ({nombres[0]})
| # | Departamento | Clientes | Clientes por 1.000 hogares | Gasto mediano (COP) | Residuo | Clientes faltantes | Oportunidad (mil M COP) | Cuadrante | # con censo 2018 |
|---|---|---|---|---|---|---|---|---|---|
{filas_top}

Cuadrantes ({nombres[0]}):
{filas_cuad}

## Hallazgos principales
- El tamaño explica buena parte de la presencia de Corona (R² {c.num(r.iloc[0].r2_hogares, 2)}), pero los
  clientes crecen **más que proporcionalmente** con los hogares (pendiente {c.num(r.iloc[0].pendiente_hogares, 2)}):
  al duplicar los hogares, los clientes se multiplican por {c.num(2 ** r.iloc[0].pendiente_hogares, 1)}. La presencia
  está concentrada en los departamentos grandes y urbanos, y los pequeños tienen penetración muy baja.
- Como la pendiente es mayor que 1, el residuo compara cada departamento con otros de su tamaño:
  un residuo negativo (los más negativos: {", ".join(etiqueta(x) for x in principal.nsmallest(4, "residuo_hogares")["region"])})
  es una brecha frente a departamentos de tamaño parecido, no solo un efecto de tamaño.
- Los departamentos con más oportunidad son {", ".join(etiqueta(x) for x in top.head(5)['region'])}: combinan
  muchos hogares, menos clientes de los esperados y un gasto por cliente relevante.
- {cambian_conclusion} al usar el censo 2018 en lugar de la proyección: el orden relativo de los
  departamentos casi no depende de la fuente, porque el crecimiento de hogares 2018-2025 es parecido
  entre departamentos.

## Limitaciones
- Las fechas de Corona vienen desplazadas; se asume que corresponden a los últimos dos años y medio.
- La región del cliente es la de sus compras (región principal), no su residencia; un cliente que
  vive en Cundinamarca y compra en Bogotá cuenta en Bogotá.
- Los hogares del censo no están ajustados por cobertura (14,2 millones frente a 15,2 millones de
  la proyección para 2018), por lo que las tasas por hogar del censo son algo mayores.
- Departamentos con menos de {c.g.MIN_CLIENTES_TERRITORIO} clientes (muestra pequeña): {pequenos or 'ninguno'};
  sus tasas son ruidosas.
- La regresión tiene 33 puntos y no controla por ingreso, urbanización ni presencia de tiendas; la
  "oportunidad" es una cota indicativa, no una proyección de ventas.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
