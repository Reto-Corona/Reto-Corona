"""
graficas.py - Reto Analytics Corona
Genera las gráficas del análisis descriptivo -> salidas/graficas/*.png

Cubre los análisis que pide el documento guía (evolución de las compras en el
tiempo, patrones generacionales, diferencias por región y género, frecuencia,
grupos de mayor y menor consumo, afinidad, canales) y otras de interés
(tipo de cliente, calidad de la edad, recompra por cohorte).

Requiere haber corrido antes:
    limpieza_transaccional.py  limpieza_clientes.py  variables_clientes.py

Uso (desde cualquier carpeta):
    pip install duckdb pandas matplotlib
    python scripts/limpieza/graficas.py
"""
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # solo guarda archivos, no abre ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent.parent
DATA = RAIZ / "data"
OUT = RAIZ / "salidas" / "graficas"

# ---------------------------------------------------------------------------
# Estilo: una sola paleta para todas las gráficas
# ---------------------------------------------------------------------------
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
AZUL = "#2a78d6"
NARANJA = "#eb6834"
ROJO = "#e34948"
GRIS_MEDIO = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# El color sigue a la categoría: cada negocio y cada generación conserva el suyo
# en todas las gráficas.
NEGOCIOS = {
    "REVESTIMIENTOS E INSTALACIÓN": "Revestimientos",
    "BAÑOS": "Baños",
    "COCINAS": "Cocinas",
    "PINTURAS E IMPERMEABILIZANTES": "Pinturas",
    "SERVICIOS": "Servicios",
}
COLOR_NEGOCIO = dict(zip(NEGOCIOS, SERIES))
GENERACIONES = ["Generación Z", "Millennials", "Generación X", "Baby Boomers"]
COLOR_GENERACION = dict(zip(GENERACIONES, SERIES))
GENEROS = ["Femenino", "Masculino", "No identificado"]

# Nombres oficiales de los canales (correo de Corona). Son canales o formatos
# comerciales, no segmentos de consumidor.
CANALES = {
    "cluster_1": "Centro Corona C2 (precio accesible)",
    "cluster_2": "Centro Corona C1 (ticket alto)",
    "cluster_3": "Sin información",
    "cluster_4": "Tienda Cerámica",
    "cluster_5": "Constructor Corona",
    "cluster_6": "American Standard",
    "cluster_7": "Institucional Corona",
    "cluster_8": "Outlet",
    "cluster_9": "Canal digital",
}
# Un canal con menos líneas que esto no se grafica (se avisa en la nota)
MIN_LINEAS_CANAL = 100
# Un departamento con menos clientes que esto no entra en la gráfica de territorios
MIN_CLIENTES_TERRITORIO = 500

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
# La nota dice si el valor está corregido; main() la ajusta según los datos.
FUENTE = "Fuente: bases de Corona. Las fechas vienen desplazadas por Corona."
FUENTE_CORREGIDA = "Fuente: bases de Corona, con valor corregido. Las fechas vienen desplazadas por Corona."


def configurar_estilo() -> None:
    """Calibri (la fuente del informe) si está instalada; ejes y rejilla discretos."""
    instaladas = {f.name for f in font_manager.fontManager.ttflist}
    fuente = next(f for f in ["Calibri", "Segoe UI", "Arial", "DejaVu Sans"] if f in instaladas)
    plt.rcParams.update({
        "font.family": fuente,
        "font.size": 10.5,
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.7,
        "xtick.color": TENUE,
        "ytick.color": TENUE,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.labelcolor": TINTA_2,
        "text.color": TINTA,
        "legend.frameon": False,
    })


def miles(x: float) -> str:
    """12345.6 -> '12.346' (formato colombiano)."""
    return f"{x:,.0f}".replace(",", ".")


def pct(x: float, dec: int = 1) -> str:
    """12.34 -> '12,3%'."""
    return f"{x:.{dec}f}".replace(".", ",") + "%"


def etiqueta_mes(fecha) -> str:
    return f"{MESES[fecha.month - 1]} {fecha.year % 100:02d}"


def guardar(fig, nombre: str, titulo: str, subtitulo: str, nota: str | None = None) -> None:
    """Pone título, subtítulo y nota con márgenes fijos en pulgadas y guarda el PNG."""
    nota = FUENTE if nota is None else nota
    alto = fig.get_figheight()
    fig.tight_layout(rect=[0, 0.42 / alto, 1, 1 - 0.78 / alto])
    fig.text(0.012, 1 - 0.14 / alto, titulo, fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 1 - 0.44 / alto, subtitulo, fontsize=10.5, color=TINTA_2, va="top")
    fig.text(0.012, 0.10 / alto, nota, fontsize=8.5, color=TENUE, va="bottom")
    fig.savefig(OUT / f"{nombre}.png", dpi=200)
    plt.close(fig)
    print(f"  {nombre}.png")


def rotular_barras(ax, barras, textos, horizontal: bool = False, size: float = 9) -> None:
    """Escribe el valor en el extremo de cada barra."""
    for barra, texto in zip(barras, textos):
        if horizontal:
            ax.annotate(texto, (barra.get_width(), barra.get_y() + barra.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=size, color=TINTA_2)
        else:
            ax.annotate(texto, (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=size, color=TINTA_2)


def barras_agrupadas(ax, categorias, serie_a, serie_b, nombres) -> None:
    """Dos barras por categoría (p. ej. % de clientes frente a % de ventas)."""
    x = np.arange(len(categorias))
    ancho = 0.36
    a = ax.bar(x - ancho / 2 - 0.02, serie_a, ancho, color=AZUL, label=nombres[0])
    b = ax.bar(x + ancho / 2 + 0.02, serie_b, ancho, color=NARANJA, label=nombres[1])
    rotular_barras(ax, a, [pct(v) for v in serie_a])
    rotular_barras(ax, b, [pct(v) for v in serie_b])
    ax.set_xticks(x, categorias)
    ax.set_ylim(0, max(max(serie_a), max(serie_b)) * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_color(EJE)
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.10))


def separar_etiquetas(valores, minimo: float) -> list:
    """Sube las etiquetas que quedarían encimadas al final de una línea."""
    orden = np.argsort(valores)
    pos = np.array(valores, dtype=float)
    for i in range(1, len(orden)):
        anterior, actual = orden[i - 1], orden[i]
        if pos[actual] - pos[anterior] < minimo:
            pos[actual] = pos[anterior] + minimo
    return pos.tolist()


def color_texto_sobre(rgba) -> str:
    """Tinta clara u oscura según lo oscuro que sea el fondo de la celda."""
    r, g, b = rgba[:3]
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else TINTA


# ---------------------------------------------------------------------------
# Gráficas que pide el documento guía
# ---------------------------------------------------------------------------
def evolucion_ventas_negocio(con) -> None:
    """Evolución de las compras en el tiempo: ventas mensuales, total y por negocio."""
    df = con.execute("""
        SELECT date_trunc('month', fecha) AS mes, negocio, SUM(valor_corregido) / 1e6 AS ventas
        FROM trx GROUP BY 1, 2 ORDER BY 1""").df()
    tabla = df.pivot(index="mes", columns="negocio", values="ventas").fillna(0)
    tabla.index = pd.to_datetime(tabla.index)
    paneles = [("Total", tabla.sum(axis=1), TINTA_2)]
    paneles += [(NEGOCIOS[n], tabla[n], COLOR_NEGOCIO[n]) for n in NEGOCIOS if n in tabla]

    fig, ejes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    marcas = [m for m in tabla.index if m.month in (1, 7)]
    for ax, (nombre, serie, color) in zip(ejes.flat, paneles):
        ax.plot(serie.index, serie.values, color=color, linewidth=2)
        ax.set_title(nombre, loc="left", fontsize=11, fontweight="bold", color=TINTA_2)
        ax.set_ylim(0, serie.max() * 1.15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: miles(v)))
        ax.set_xticks(marcas, [etiqueta_mes(m) for m in marcas])
        ax.grid(axis="x", visible=False)
        ax.spines["bottom"].set_color(EJE)
    guardar(fig, "01_evolucion_ventas_negocio",
            "Evolución mensual de las ventas, total y por negocio",
            "Millones de pesos por mes. Cada panel tiene su propia escala: compare la forma, no la altura.")


def generaciones_en_el_tiempo(con) -> None:
    """Patrones generacionales en el tiempo: quiénes son los compradores de cada mes."""
    df = con.execute("""
        WITH c AS (
          SELECT DISTINCT date_trunc('month', t.fecha) AS mes, t.cliente, f.generacion
          FROM trx t JOIN cli f USING (cliente)
          WHERE f.tipo_cliente = 'Consumidor final' AND f.generacion <> 'Sin edad')
        SELECT mes, generacion, COUNT(*) AS compradores FROM c GROUP BY 1, 2 ORDER BY 1""").df()
    tabla = df.pivot(index="mes", columns="generacion", values="compradores").fillna(0)
    tabla.index = pd.to_datetime(tabla.index)
    tabla = tabla.div(tabla.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    for g in GENERACIONES:
        ax.plot(tabla.index, tabla[g], color=COLOR_GENERACION[g], linewidth=2, label=g)
    finales = [tabla[g].iloc[-1] for g in GENERACIONES]
    alturas = separar_etiquetas(finales, minimo=tabla.values.max() * 0.045)
    for g, valor, y in zip(GENERACIONES, finales, alturas):
        ax.annotate(f"{g}  {pct(valor)}", (tabla.index[-1], y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=TINTA_2)
    marcas = [m for m in tabla.index if m.month in (1, 4, 7, 10)]
    ax.set_xticks(marcas, [etiqueta_mes(m) for m in marcas])
    ax.set_ylim(0, tabla.values.max() * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_color(EJE)
    ax.margins(x=0.01)
    ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.09))
    fig.subplots_adjust(right=0.80)
    guardar(fig, "02_generaciones_en_el_tiempo",
            "Participación de cada generación entre los compradores del mes",
            "Porcentaje de los consumidores finales que compraron cada mes. Excluye empresas.")


def _indice_afinidad(con, columna: str, orden: list) -> pd.DataFrame:
    """
    Índice de afinidad = participación del negocio en el gasto del grupo,
    dividida por su participación en el gasto de todos los consumidores, por 100.
    100 es el promedio; por encima, el grupo gasta proporcionalmente más ahí.
    """
    df = con.execute(f"""
        SELECT f.{columna} AS grupo, t.negocio, SUM(t.valor_corregido) AS v
        FROM trx t JOIN cli f USING (cliente)
        WHERE f.tipo_cliente = 'Consumidor final' AND f.generacion <> 'Sin edad'
        GROUP BY 1, 2""").df()
    tabla = df.pivot(index="grupo", columns="negocio", values="v").fillna(0)
    tabla = tabla.reindex(index=[g for g in orden if g in tabla.index], columns=list(NEGOCIOS))
    mezcla_grupo = tabla.div(tabla.sum(axis=1), axis=0)
    mezcla_total = tabla.sum(axis=0) / tabla.values.sum()
    return mezcla_grupo.div(mezcla_total, axis=1) * 100


def afinidad_por_perfil(con) -> None:
    """Afinidad por generación y por género: qué negocio pesa más o menos en cada grupo."""
    tablas = [("Por generación", _indice_afinidad(con, "generacion", GENERACIONES)),
              ("Por género", _indice_afinidad(con, "genero", GENEROS))]
    mapa = LinearSegmentedColormap.from_list("divergente", [ROJO, GRIS_MEDIO, AZUL])
    norma = TwoSlopeNorm(vmin=50, vcenter=100, vmax=150)

    fig, ejes = plt.subplots(2, 1, figsize=(8.5, 6.2), height_ratios=[4, 3])
    for ax, (nombre, tabla) in zip(ejes, tablas):
        imagen = ax.imshow(tabla.values, cmap=mapa, norm=norma, aspect="auto")
        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                valor = tabla.values[i, j]
                ax.text(j, i, f"{valor:.0f}", ha="center", va="center", fontsize=11,
                        color=color_texto_sobre(mapa(norma(valor))))
        ax.set_xticks(range(tabla.shape[1]), [NEGOCIOS[n] for n in tabla.columns])
        ax.set_yticks(range(tabla.shape[0]), tabla.index)
        ax.set_title(nombre, loc="left", fontsize=11, fontweight="bold", color=TINTA_2)
        ax.grid(visible=False)
        # separación entre celdas con el color del fondo, sin bordes
        ax.set_xticks(np.arange(-0.5, tabla.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, tabla.shape[0]), minor=True)
        ax.grid(which="minor", color=SUPERFICIE, linewidth=2.5)
        ax.tick_params(which="minor", length=0)
        ax.spines[:].set_visible(False)
    barra = fig.colorbar(imagen, ax=ejes, orientation="vertical", fraction=0.035, pad=0.03)
    barra.set_label("Índice de afinidad (100 = promedio)")
    barra.outline.set_visible(False)
    alto = fig.get_figheight()
    fig.text(0.012, 1 - 0.14 / alto, "Afinidad de cada perfil con cada negocio",
             fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 1 - 0.44 / alto,
             "Peso del negocio en el gasto del grupo frente al promedio de los consumidores finales. "
             "Azul: gasta más ahí; rojo: menos.", fontsize=10.5, color=TINTA_2, va="top")
    fig.text(0.012, 0.10 / alto, FUENTE, fontsize=8.5, color=TENUE, va="bottom")
    fig.subplots_adjust(top=1 - 1.0 / alto, bottom=0.55 / alto, left=0.16, right=0.86, hspace=0.42)
    fig.savefig(OUT / "03_afinidad_por_perfil.png", dpi=200)
    plt.close(fig)
    print("  03_afinidad_por_perfil.png")


def ventas_por_region(con) -> None:
    """Diferencias por región: participación de cada departamento en las ventas."""
    df = con.execute("""
        SELECT COALESCE(region, 'Sin región') AS region,
               100.0 * SUM(valor_corregido) / SUM(SUM(valor_corregido)) OVER () AS pct_ventas
        FROM trx GROUP BY 1 ORDER BY 2""").df()
    fig, ax = plt.subplots(figsize=(8, 9))
    nombres = df["region"].where(df["region"] == "Sin región", df["region"].str.title())
    barras = ax.barh(nombres, df["pct_ventas"], height=0.66, color=AZUL)
    rotular_barras(ax, barras, [pct(v) if v >= 0.05 else "<0,1%" for v in df["pct_ventas"]],
                   horizontal=True)
    ax.set_xlim(0, df["pct_ventas"].max() * 1.12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.01)
    guardar(fig, "04_ventas_por_region",
            "Participación de cada departamento en las ventas",
            "Porcentaje del valor total vendido, según el departamento registrado en la compra.")


def territorios_tamano_y_gasto(con) -> None:
    """Territorios: cuántos clientes tiene cada departamento y cuánto gasta el cliente típico."""
    df = con.execute("""
        SELECT region_principal AS region, COUNT(*) AS clientes, MEDIAN(monto) / 1e3 AS gasto
        FROM cli WHERE region_principal IS NOT NULL GROUP BY 1 ORDER BY 2""").df()
    # con muy pocos clientes la mediana no es representativa y deforma la escala
    fuera = df[df["clientes"] < MIN_CLIENTES_TERRITORIO]
    df = df[df["clientes"] >= MIN_CLIENTES_TERRITORIO].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["clientes"], df["gasto"], s=70, color=AZUL, edgecolor=SUPERFICIE, linewidth=1.5, zorder=3)
    for i, fila in df.iterrows():  # rótulos alternados arriba y abajo para que no se encimen
        arriba = i % 2 == 0
        ax.annotate(fila["region"].title(), (fila["clientes"], fila["gasto"]),
                    xytext=(0, 8 if arriba else -8), textcoords="offset points", ha="center",
                    va="bottom" if arriba else "top", fontsize=9, color=TINTA_2)
    ax.set_xscale("log")
    ax.set_ylim(0, df["gasto"].max() * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: miles(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: miles(v)))
    ax.set_xlabel("Número de clientes (escala logarítmica)")
    ax.set_ylabel("Gasto mediano por cliente (miles de pesos)")
    ax.spines["bottom"].set_color(EJE)
    guardar(fig, "05_territorios_tamano_y_gasto",
            "Tamaño de la base de clientes y gasto típico por departamento",
            "Cada punto es un departamento, el principal de cada cliente.",
            f"No se grafican {len(fuera)} departamentos con menos de {miles(MIN_CLIENTES_TERRITORIO)} clientes. {FUENTE}")


def frecuencia_de_visitas(con) -> None:
    """Frecuencia: cuántos clientes hay por número de visitas y cuánto pesan en ventas."""
    df = con.execute("""
        SELECT CASE WHEN frecuencia_visitas <= 3 THEN CAST(frecuencia_visitas AS VARCHAR)
                    WHEN frecuencia_visitas <= 5 THEN '4 a 5'
                    WHEN frecuencia_visitas <= 10 THEN '6 a 10' ELSE '11 o más' END AS visitas,
               100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct_clientes,
               100.0 * SUM(monto) / SUM(SUM(monto)) OVER () AS pct_ventas
        FROM cli GROUP BY 1 ORDER BY MIN(frecuencia_visitas)""").df()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    barras_agrupadas(ax, df["visitas"], df["pct_clientes"], df["pct_ventas"],
                     ["% de los clientes", "% de las ventas"])
    ax.set_xlabel("Número de visitas del cliente en el periodo")
    guardar(fig, "06_frecuencia_de_visitas",
            "Clientes y ventas según el número de visitas",
            "Una visita es una compra de un cliente en una tienda en un día.")


def concentracion_de_ventas(con) -> None:
    """Grupos de mayor y menor consumo: qué parte de las ventas hace cada tramo de clientes."""
    monto = np.sort(con.execute("SELECT monto FROM cli").df()["monto"].to_numpy())[::-1]
    acumulado = np.cumsum(monto) / monto.sum() * 100
    clientes = np.arange(1, len(monto) + 1) / len(monto) * 100
    paso = max(len(monto) // 2000, 1)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.plot([0, 100], [0, 100], color=EJE, linewidth=1)
    ax.plot(np.r_[0, clientes[::paso], 100], np.r_[0, acumulado[::paso], 100], color=AZUL, linewidth=2)
    ax.annotate("Si todos compraran igual", (62, 62), xytext=(6, -12), textcoords="offset points",
                fontsize=9, color=TENUE)
    for corte in (1, 10, 20, 50):
        y = acumulado[int(len(monto) * corte / 100) - 1]
        ax.scatter([corte], [y], s=55, color=AZUL, edgecolor=SUPERFICIE, linewidth=1.5, zorder=3)
        ax.annotate(f"El {corte}% que más compra hace el {pct(y)} de las ventas", (corte, y),
                    xytext=(9, -11), textcoords="offset points", fontsize=9.5, color=TINTA_2,
                    bbox=dict(facecolor=SUPERFICIE, edgecolor="none", pad=1.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    ax.set_xlabel("Clientes, ordenados del que más compra al que menos")
    ax.set_ylabel("Ventas acumuladas")
    ax.spines["bottom"].set_color(EJE)
    guardar(fig, "07_concentracion_de_ventas",
            "Concentración de las ventas en pocos clientes",
            "Porcentaje acumulado de las ventas frente al porcentaje acumulado de clientes (incluye empresas).")


def ventas_y_ticket_por_canal(con) -> None:
    """Canales: peso de cada canal en las ventas y tamaño típico de una visita."""
    df = con.execute("""
        WITH visitas AS (
          SELECT cluster, cliente, fecha, punto_de_venta, SUM(valor_corregido) AS v, COUNT(*) AS lineas
          FROM trx GROUP BY ALL)
        SELECT cluster, SUM(lineas) AS lineas,
               100.0 * SUM(v) / SUM(SUM(v)) OVER () AS pct_ventas,
               MEDIAN(v) / 1e3 AS ticket
        FROM visitas GROUP BY 1""").df()
    fuera = df[df["lineas"] < MIN_LINEAS_CANAL]
    df = df[df["lineas"] >= MIN_LINEAS_CANAL].sort_values("pct_ventas")
    nombres = df["cluster"].map(CANALES)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    barras = a.barh(nombres, df["pct_ventas"], height=0.6, color=AZUL)
    rotular_barras(a, barras, [pct(v) for v in df["pct_ventas"]], horizontal=True)
    a.set_xlim(0, df["pct_ventas"].max() * 1.18)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    a.set_title("Participación en las ventas", loc="left", fontsize=11, fontweight="bold", color=TINTA_2)
    barras = b.barh(nombres, df["ticket"], height=0.6, color=AZUL)
    rotular_barras(b, barras, [miles(v) for v in df["ticket"]], horizontal=True)
    b.set_xlim(0, df["ticket"].max() * 1.18)
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: miles(v)))
    b.set_title("Valor mediano de una visita (miles de pesos)", loc="left", fontsize=11,
                fontweight="bold", color=TINTA_2)
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
    nota = FUENTE
    if len(fuera):
        excluidos = ", ".join(f"{CANALES[c]} ({int(n)} {'línea' if n == 1 else 'líneas'})"
                              for c, n in zip(fuera["cluster"], fuera["lineas"]))
        nota = f"No se grafica: {excluidos}. {FUENTE}"
    guardar(fig, "08_ventas_y_ticket_por_canal",
            "Peso de cada canal en las ventas y tamaño de la visita",
            "Los canales son formatos comerciales de Corona, no segmentos de consumidor.", nota)


# ---------------------------------------------------------------------------
# Otras gráficas de interés
# ---------------------------------------------------------------------------
def tipo_de_cliente(con) -> None:
    """Consumidores y empresas: cuántos son y cuánto venden."""
    df = con.execute("""
        SELECT CASE WHEN tipo_cliente = 'Consumidor final' THEN 'Consumidor final'
                    ELSE segmento_empresa END AS grupo,
               100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct_clientes,
               100.0 * SUM(monto) / SUM(SUM(monto)) OVER () AS pct_ventas
        FROM cli WHERE tipo_cliente IS NOT NULL GROUP BY 1""").df()
    orden = ["Consumidor final", "Empresa pequeña", "Empresa mediana", "Empresa grande / distribuidor"]
    df = df.set_index("grupo").reindex(orden)
    etiquetas = ["Consumidor\nfinal", "Empresa\npequeña", "Empresa\nmediana", "Empresa grande\no distribuidor"]
    fig, ax = plt.subplots(figsize=(8, 5))
    barras_agrupadas(ax, etiquetas, df["pct_clientes"], df["pct_ventas"],
                     ["% de los clientes", "% de las ventas"])
    guardar(fig, "09_tipo_de_cliente",
            "Consumidores finales y empresas: clientes frente a ventas",
            "El tamaño de la empresa se define por su monto: hasta el percentil 50, hasta el 90 y el 10% superior.")


def perfil_de_generaciones(con) -> None:
    """Generaciones: tamaño, gasto típico y qué tan contactables son."""
    df = con.execute("""
        SELECT generacion, COUNT(*) / 1e3 AS clientes, MEDIAN(monto) / 1e3 AS monto,
               100.0 * AVG(es_contactable::INT) AS contactable
        FROM cli WHERE tipo_cliente = 'Consumidor final' AND generacion <> 'Sin edad'
        GROUP BY 1""").df().set_index("generacion").reindex(GENERACIONES)
    paneles = [("Clientes (miles)", "clientes", lambda v: miles(v)),
               ("Gasto mediano (miles de pesos)", "monto", lambda v: miles(v)),
               ("Autorizó tratamiento de datos", "contactable", lambda v: pct(v))]
    etiquetas = ["Gen. Z", "Millennials", "Gen. X", "Boomers"]
    fig, ejes = plt.subplots(1, 3, figsize=(11, 4.4))
    for ax, (nombre, columna, formato) in zip(ejes, paneles):
        barras = ax.bar(etiquetas, df[columna], width=0.6, color=AZUL)
        rotular_barras(ax, barras, [formato(v) for v in df[columna]])
        ax.set_title(nombre, loc="left", fontsize=11, fontweight="bold", color=TINTA_2)
        ax.set_ylim(0, df[columna].max() * 1.15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: miles(v)))
        ax.grid(axis="x", visible=False)
        ax.spines["bottom"].set_color(EJE)
    guardar(fig, "10_perfil_de_generaciones",
            "Tamaño, gasto y contactabilidad de cada generación",
            "Solo consumidores finales. Generación según la edad a 2026: Z hasta 29, Millennials 30 a 45, X 46 a 61, Boomers 62 o más.")


def distribucion_de_edad(con) -> None:
    """Calidad del dato: edades con muchos más clientes que sus vecinas."""
    df = con.execute("""
        SELECT edad, COUNT(*) AS clientes, MAX(edad_dudosa) AS dudosa
        FROM cli WHERE edad IS NOT NULL GROUP BY 1 ORDER BY 1""").df()
    colores = np.where(df["dudosa"] == 1, NARANJA, AZUL)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["edad"], df["clientes"], width=0.78, color=colores)
    for _, fila in df[df["dudosa"] == 1].iterrows():
        ax.annotate(f"{int(fila['edad'])} años: {miles(fila['clientes'])}", (fila["edad"], fila["clientes"]),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, color=TINTA_2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: miles(v)))
    ax.set_xlabel("Edad a 2026")
    ax.set_ylabel("Clientes")
    ax.set_ylim(0, df["clientes"].max() * 1.12)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_color(EJE)
    ax.legend(handles=[Patch(color=AZUL, label="Edad registrada"),
                       Patch(color=NARANJA, label="Pico dudoso (posible valor por defecto)")],
              loc="upper left")
    guardar(fig, "11_distribucion_de_edad",
            "Distribución de la edad de los consumidores",
            "Dos edades concentran muchos más clientes que las vecinas; se marcan con la bandera edad_dudosa.")


def recompra_por_cohorte(con) -> None:
    """
    Recompra: de los clientes que compraron por primera vez en un mes, qué
    porcentaje vuelve a comprar en cada uno de los 12 meses siguientes.
    Se omiten las cohortes de enero y febrero de 2000, porque ahí caen clientes
    que ya existían antes del inicio de los datos.
    """
    df = con.execute("""
        WITH meses AS (SELECT DISTINCT cliente, date_trunc('month', fecha) AS mes FROM trx),
             primera AS (SELECT cliente, MIN(mes) AS cohorte FROM meses GROUP BY 1)
        SELECT p.cohorte, datediff('month', p.cohorte, m.mes) AS k, COUNT(*) AS clientes
        FROM meses m JOIN primera p USING (cliente) GROUP BY 1, 2""").df()
    df["cohorte"] = pd.to_datetime(df["cohorte"])
    tabla = df.pivot(index="cohorte", columns="k", values="clientes").fillna(0)
    ultimo = tabla.index.max()
    pct_vuelve = tabla.div(tabla[0], axis=0) * 100
    pct_vuelve = pct_vuelve.loc[pct_vuelve.index >= "2000-03-01", range(1, 13)]
    for cohorte in pct_vuelve.index:  # meses que todavía no han ocurrido
        visibles = (ultimo.year - cohorte.year) * 12 + ultimo.month - cohorte.month
        pct_vuelve.loc[cohorte, [k for k in pct_vuelve.columns if k > visibles]] = np.nan
    pct_vuelve = pct_vuelve.dropna(how="all")

    mapa = LinearSegmentedColormap.from_list("azules", ["#cde2fb", "#3987e5", "#0d366b"])
    mapa.set_bad(SUPERFICIE)
    fig, ax = plt.subplots(figsize=(8, 8.5))
    imagen = ax.imshow(pct_vuelve.values, cmap=mapa, aspect="auto", vmin=0)
    ax.set_xticks(range(12), range(1, 13))
    ax.set_yticks(range(len(pct_vuelve)), [etiqueta_mes(c) for c in pct_vuelve.index])
    ax.set_xlabel("Meses después de la primera compra")
    ax.set_ylabel("Mes de la primera compra")
    ax.grid(visible=False)
    ax.set_xticks(np.arange(-0.5, 12), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pct_vuelve)), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.spines[:].set_visible(False)
    barra = fig.colorbar(imagen, ax=ax, fraction=0.04, pad=0.03)
    barra.set_label("% de la cohorte que compra ese mes")
    barra.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    barra.outline.set_visible(False)
    guardar(fig, "12_recompra_por_cohorte",
            "Recompra: cuántos clientes vuelven después de su primera compra",
            "Cada fila es el grupo de clientes que compró por primera vez ese mes.")


def main() -> None:
    for archivo in ("transaccional_corregido.parquet", "features_clientes_v2.parquet"):
        if not (DATA / archivo).exists():
            raise FileNotFoundError(
                f"No se encontró {DATA / archivo}. Corre primero limpieza_transaccional.py, "
                "limpieza_clientes.py y variables_clientes.py"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    configurar_estilo()
    con = duckdb.connect()
    con.execute(f"CREATE VIEW trx AS SELECT * FROM '{DATA / 'transaccional_corregido.parquet'}'")
    con.execute(f"CREATE VIEW cli AS SELECT * FROM '{DATA / 'features_clientes_v2.parquet'}'")
    global FUENTE
    if con.execute("SELECT SUM(valor_corregido_flag) FROM trx").fetchone()[0] > 0:
        FUENTE = FUENTE_CORREGIDA

    print(f"Guardando gráficas en {OUT} ...")
    evolucion_ventas_negocio(con)
    generaciones_en_el_tiempo(con)
    afinidad_por_perfil(con)
    ventas_por_region(con)
    territorios_tamano_y_gasto(con)
    frecuencia_de_visitas(con)
    concentracion_de_ventas(con)
    ventas_y_ticket_por_canal(con)
    tipo_de_cliente(con)
    perfil_de_generaciones(con)
    distribucion_de_edad(con)
    recompra_por_cohorte(con)
    print("Listo.")


if __name__ == "__main__":
    main()
