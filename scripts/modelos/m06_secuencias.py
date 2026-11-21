"""
m06 - Minería de secuencias de compra

Clientes: consumidores finales segmentados con 2 o más visitas (visita = cliente + fecha + punto de venta).
Secuencia = negocios comprados en cada visita, en orden de fecha. Cada visita es un elemento con la
combinación de negocios de esa visita (p. ej. "Baños+Revestimientos").

1. PrefixSpan (librería prefixspan): patrones frecuentes de 2 y 3 pasos con soporte mínimo
   SOPORTE_MIN (proporción de clientes cuya secuencia contiene el patrón, en orden pero no
   necesariamente seguido). Se usan las primeras MAX_VISITAS visitas de cada cliente.
2. Matriz de transición del negocio principal (el de mayor valor) en la visita n al de la visita
   n + 1, y tiempo mediano (días) entre esos pasos. También solo para el paso de la 1.a a la 2.a visita.

Escribe:
    data/modelos/m06_secuencias.parquet        patrones frecuentes
    salidas/modelos/m06_secuencias/*.csv, *.png, resumen.md

Uso:
    python scripts/modelos/m06_secuencias.py
"""
import numpy as np
import pandas as pd
from prefixspan import PrefixSpan

import comun as c

NOMBRE = "m06_secuencias"
SOPORTE_MIN = 0.005     # 0,5% de los clientes con 2+ visitas
MAX_VISITAS = 10        # PrefixSpan usa las primeras 10 visitas (cubre a casi todos los clientes)
ABREV = {
    "REVESTIMIENTOS E INSTALACIÓN": "Revestimientos", "BAÑOS": "Baños", "COCINAS": "Cocinas",
    "PINTURAS E IMPERMEABILIZANTES": "Pinturas", "SERVICIOS": "Servicios",
}
ORDEN = ["Revestimientos", "Baños", "Cocinas", "Pinturas", "Servicios"]


def cargar_visitas() -> pd.DataFrame:
    """Una fila por visita con su combinación de negocios y su negocio principal."""
    con = c.conexion()
    v = con.execute("""
        WITH lineas AS (
          SELECT t.cliente, t.fecha::DATE AS fecha, t.punto_de_venta, t.negocio, SUM(t.valor_corregido) AS valor
          FROM trx t JOIN features f USING (cliente) JOIN segmentos s USING (cliente)
          WHERE f.tipo_cliente = 'Consumidor final' AND t.negocio IS NOT NULL
          GROUP BY ALL
        ),
        -- Negocio principal: el de mayor valor; si empatan, el primero en orden alfabético
        -- (mismo criterio de desempate que variables_clientes.py, para que el resultado sea fijo)
        rk AS (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY cliente, fecha, punto_de_venta
                                       ORDER BY ROUND(valor, 2) DESC, negocio) AS orden
          FROM lineas
        )
        SELECT cliente, fecha, punto_de_venta,
               LIST(negocio ORDER BY negocio) AS negocios,
               MAX(CASE WHEN orden = 1 THEN negocio END) AS negocio_principal
        FROM rk GROUP BY ALL
        ORDER BY cliente, fecha, punto_de_venta
    """).df()
    v["fecha"] = pd.to_datetime(v["fecha"])
    v["elemento"] = v["negocios"].apply(lambda l: "+".join(sorted({ABREV[x] for x in l}, key=ORDEN.index)))
    v["principal"] = v["negocio_principal"].map(ABREV)
    v["n_visita"] = v.groupby("cliente").cumcount() + 1
    v["visitas_cliente"] = v.groupby("cliente")["cliente"].transform("size")
    return v


def agrega_negocio(patron: str) -> bool:
    """True si en un paso posterior aparece un negocio que no estaba en el primero (venta cruzada en el tiempo)."""
    pasos = [set(p.split("+")) for p in patron.split(" → ")]
    return any(paso - pasos[0] for paso in pasos[1:])


def patrones(v: pd.DataFrame, n_clientes: int) -> pd.DataFrame:
    """PrefixSpan sobre las secuencias de elementos (combinaciones de negocios por visita)."""
    secuencias = v[v["n_visita"] <= MAX_VISITAS].groupby("cliente")["elemento"].apply(list).tolist()
    ps = PrefixSpan(secuencias)
    ps.minlen, ps.maxlen = 2, 3
    minimo = int(np.ceil(SOPORTE_MIN * n_clientes))
    res = ps.frequent(minimo)
    t = pd.DataFrame([{"pasos": len(p), "patron": " → ".join(p), "clientes": s, "soporte": s / n_clientes}
                      for s, p in res])
    t["cambia_negocio"] = t["patron"].apply(agrega_negocio)
    return t.sort_values(["pasos", "clientes"], ascending=[True, False]).reset_index(drop=True)


def transiciones(v: pd.DataFrame, solo_primera: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Matriz de transición del negocio principal n -> n+1, conteos y días medianos."""
    v = v.copy()
    g = v.groupby("cliente")
    v["siguiente"] = g["principal"].shift(-1)
    v["dias"] = (g["fecha"].shift(-1) - v["fecha"]).dt.days
    pares = v.dropna(subset=["siguiente"])
    if solo_primera:
        pares = pares[pares["n_visita"] == 1]
    conteo = pd.crosstab(pares["principal"], pares["siguiente"]).reindex(index=ORDEN, columns=ORDEN, fill_value=0)
    prob = conteo.div(conteo.sum(axis=1).replace(0, np.nan), axis=0)
    dias = pares.pivot_table(index="principal", columns="siguiente", values="dias", aggfunc="median") \
        .reindex(index=ORDEN, columns=ORDEN)
    return prob, conteo, dias


def graficar_matriz(carpeta, prob, conteo, dias, nombre, titulo, subtitulo) -> None:
    """Mapa de calor de la transición con probabilidad y días medianos."""
    fig, ax = c.plt.subplots(figsize=(9.5, 6.5))
    cmap = c.g.LinearSegmentedColormap.from_list("azul", [c.SUPERFICIE, c.AZUL])
    ax.imshow(prob.fillna(0).values * 100, cmap=cmap, vmin=0, vmax=100)
    for i in range(len(ORDEN)):
        for j in range(len(ORDEN)):
            p = prob.iat[i, j]
            if np.isnan(p):
                continue
            d = dias.iat[i, j]
            texto = f"{c.num(p * 100, 1)}%" + ("" if np.isnan(d) else f"\n{c.miles(d)} días")
            ax.text(j, i, texto, ha="center", va="center", fontsize=8.5, color="white" if p > 0.55 else c.TINTA)
    ax.set_xticks(range(len(ORDEN)), ORDEN)
    ax.set_yticks(range(len(ORDEN)), [f"{n} ({c.miles(conteo.loc[n].sum())})" for n in ORDEN])
    ax.set_xlabel("Negocio principal de la visita siguiente")
    ax.grid(False)
    c.guardar_figura(fig, carpeta, nombre, titulo, subtitulo)


def graficar_patrones(carpeta, t: pd.DataFrame) -> None:
    """Patrones de 2 y 3 pasos más frecuentes que cambian de negocio."""
    top = pd.concat([t[(t["pasos"] == 2) & t["cambia_negocio"]].head(8),
                     t[(t["pasos"] == 3) & t["cambia_negocio"]].head(7)]).iloc[::-1]
    fig, ax = c.plt.subplots(figsize=(10.5, 6.5))
    colores = [c.AZUL if p == 2 else c.NARANJA for p in top["pasos"]]
    barras = ax.barh(top["patron"], top["soporte"] * 100, color=colores, height=0.65)
    c.rotular_barras(ax, barras, [f"{c.pct(s * 100)} · {c.miles(n)}" for s, n in zip(top["soporte"], top["clientes"])],
                     horizontal=True, size=8)
    ax.set_xlim(0, top["soporte"].max() * 100 * 1.3)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(c.g.FuncFormatter(lambda v, _: c.pct(v, 0)))
    ax.set_xlabel("Clientes con 2+ visitas cuya secuencia contiene el patrón")
    c.guardar_figura(fig, carpeta, "03_patrones_cambio_negocio", "Secuencias frecuentes que cambian de negocio",
                     "Azul: 2 pasos; naranja: 3 pasos. El patrón aparece en orden, no necesariamente en visitas seguidas.")


def main() -> None:
    c.configurar_estilo()
    carpeta = c.carpeta_salida(NOMBRE)
    print(f"== {NOMBRE} (versión {c.VERSION}) ==")
    v = cargar_visitas()
    total = v["cliente"].nunique()
    v = v[v["visitas_cliente"] >= 2]
    n = v["cliente"].nunique()
    cobertura = (v.drop_duplicates("cliente")["visitas_cliente"] <= MAX_VISITAS).mean()
    print(f"  Consumidores: {total:,} | con 2+ visitas: {n:,} ({n / total:.1%}) | visitas: {len(v):,}")

    t = patrones(v, n)
    t.to_csv(carpeta / "patrones_frecuentes.csv", index=False, float_format=c.FORMATO_CSV)
    t.to_parquet(c.ruta_puntajes(NOMBRE), index=False)
    print(f"  Patrones: {len(t):,} (2 pasos: {(t['pasos'] == 2).sum()}, 3 pasos: {(t['pasos'] == 3).sum()})")
    print(t.groupby("pasos").head(5).to_string(index=False))
    graficar_patrones(carpeta, t)

    prob, conteo, dias = transiciones(v)
    prob1, conteo1, dias1 = transiciones(v, solo_primera=True)
    for nombre, x in [("transicion_todas", prob), ("transicion_conteo_todas", conteo), ("dias_mediana_todas", dias),
                      ("transicion_primera_segunda", prob1), ("transicion_conteo_primera_segunda", conteo1),
                      ("dias_mediana_primera_segunda", dias1)]:
        x.to_csv(carpeta / f"{nombre}.csv", float_format=c.FORMATO_CSV)
    graficar_matriz(carpeta, prob, conteo, dias, "01_transicion_todas", "Del negocio de una visita al de la siguiente",
                    "Todas las visitas consecutivas. Celda: % de la fila y días medianos entre las dos visitas.")
    graficar_matriz(carpeta, prob1, conteo1, dias1, "02_transicion_primera_segunda", "De la primera a la segunda visita",
                    "Solo el paso de la 1.ª a la 2.ª visita. Celda: % de la fila y días medianos.")
    print(prob1.round(3).to_string())
    primeras = v[v["n_visita"] <= 2].sort_values(["cliente", "n_visita"])
    dias12 = primeras.groupby("cliente")["fecha"].agg(lambda f: (f.iloc[-1] - f.iloc[0]).days)
    pd.DataFrame([{"clientes_total": total, "clientes_2_visitas": n, "patrones": len(t),
                   "patrones_negocio_nuevo": int(t["cambia_negocio"].sum()),
                   "dias_mediana_1a_2a": dias12.median(),
                   "pct_repite_negocio_1a_2a": float(conteo1.values.diagonal().sum() / conteo1.values.sum() * 100)}]).to_csv(
        carpeta / "metricas.csv", index=False, float_format=c.FORMATO_CSV)
    escribir_resumen(carpeta, total, n, len(v), cobertura, t, prob, conteo, dias, prob1, conteo1, dias1, v)


def escribir_resumen(carpeta, total, n, n_visitas, cobertura, t, prob, conteo, dias, prob1, conteo1, dias1, v) -> None:
    """resumen.md del modelo."""
    def tabla_patrones(x, k):
        return "\n".join(f"| {r.patron} | {c.miles(r.clientes)} | {c.pct(r.soporte * 100)} |" for r in x.head(k).itertuples())

    def tabla_matriz(p, d):
        filas = []
        for i in ORDEN:
            celdas = [("–" if np.isnan(p.loc[i, j]) else f"{c.pct(p.loc[i, j] * 100)} ({c.miles(d.loc[i, j])} d)")
                      for j in ORDEN]
            filas.append(f"| {i} | " + " | ".join(celdas) + " |")
        return "\n".join(filas)
    diag = np.diag(prob1.fillna(0).values)
    repite = dict(zip(ORDEN, diag))
    mas_fiel = max(repite, key=repite.get)
    menos_fiel = min((k for k in ORDEN if conteo1.loc[k].sum() >= 500), key=repite.get)
    cambio = {i: (prob1.loc[i].drop(i).idxmax(), prob1.loc[i].drop(i).max()) for i in ORDEN if conteo1.loc[i].sum() >= 500}
    texto_cambio = "; ".join(f"desde {i}, {j} ({c.pct(p * 100)})" for i, (j, p) in cambio.items())
    pares = v.assign(sig=v.groupby("cliente")["fecha"].shift(-1))
    pares = pares[pares["n_visita"] == 1].dropna(subset=["sig"])
    dias_12 = (pares["sig"] - pares["fecha"]).dt.days
    cambian = t[t["cambia_negocio"]]
    texto = f"""
# m06 - Minería de secuencias de compra

## Qué hace
Busca el orden en que los clientes compran los negocios de Corona a lo largo de sus visitas:
1. **PrefixSpan** (librería `prefixspan`) sobre la secuencia de visitas de cada cliente, donde cada
   visita es la combinación de negocios comprados (por ejemplo `Baños+Revestimientos`). Patrones de 2 y 3
   pasos con soporte mínimo de {c.pct(SOPORTE_MIN * 100, 1)} de los clientes; el patrón tiene que aparecer en
   ese orden, no necesariamente en visitas seguidas.
2. **Matriz de transición** del negocio principal de la visita n al de la visita n + 1 (todas las
   visitas consecutivas y solo de la 1.ª a la 2.ª), con el tiempo mediano en días entre las dos visitas.

## Datos usados
- `data/transaccional_{c.VERSION}.parquet`, consumidores finales segmentados.
- **{c.miles(n)} clientes con 2 o más visitas** ({c.pct(n / total * 100)} de {c.miles(total)}), {c.miles(n_visitas)} visitas.
- PrefixSpan usa las primeras {MAX_VISITAS} visitas de cada cliente ({c.pct(cobertura * 100)} de los clientes tiene
  {MAX_VISITAS} o menos).

## Métricas
Patrones frecuentes: **{c.miles(len(t))}** ({c.miles((t['pasos'] == 2).sum())} de 2 pasos y {c.miles((t['pasos'] == 3).sum())} de 3);
con un negocio nuevo en un paso posterior (venta cruzada en el tiempo): {c.miles(len(cambian))}.

### Patrones de 2 pasos más frecuentes
| Patrón | Clientes | Soporte |
|---|---|---|
{tabla_patrones(t[t['pasos'] == 2], 10)}

### Patrones de 3 pasos más frecuentes
| Patrón | Clientes | Soporte |
|---|---|---|
{tabla_patrones(t[t['pasos'] == 3], 10)}

### Patrones que agregan un negocio nuevo
| Patrón | Clientes | Soporte |
|---|---|---|
{tabla_patrones(cambian, 10)}

### Transición de la 1.ª a la 2.ª visita (negocio principal; % de la fila y días medianos)
| Desde \\ Hacia | {' | '.join(ORDEN)} |
|---|{'---|' * len(ORDEN)}
{tabla_matriz(prob1, dias1)}

### Transición entre todas las visitas consecutivas
| Desde \\ Hacia | {' | '.join(ORDEN)} |
|---|{'---|' * len(ORDEN)}
{tabla_matriz(prob, dias)}

## Hallazgos principales
- Los patrones más frecuentes son **repetir el mismo negocio**: quien vuelve suele volver por lo mismo.
  De la 1.ª a la 2.ª visita, el negocio que más se repite es {mas_fiel} ({c.pct(repite[mas_fiel] * 100)}) y el que
  menos {menos_fiel} ({c.pct(repite[menos_fiel] * 100)}).
- Cuando cambian de negocio en la segunda visita, el destino más común es: {texto_cambio}.
- Entre la 1.ª y la 2.ª visita pasan **{c.miles(dias_12.median())} días** (mediana; cuartiles {c.miles(dias_12.quantile(0.25))} y
  {c.miles(dias_12.quantile(0.75))}). Es la ventana natural para una recomendación de venta cruzada (ver m09).

## Limitaciones
- Una visita con varios negocios se resume por su negocio principal en la matriz de transición; en
  PrefixSpan se conserva la combinación completa.
- PrefixSpan cuenta un patrón si aparece en orden en la secuencia, aunque haya otras visitas en medio.
- Solo se observan las visitas de los ~31 meses de la base; secuencias más largas quedan cortadas.
"""
    c.escribir_resumen(carpeta, texto)


if __name__ == "__main__":
    main()
