"""
Prepara las tablas del tablero de Power BI

Requiere haber corrido antes el pipeline completo (scripts/run_pipeline.py),
porque lee los parquet de las dos versiones y los seis escenarios de segmentación.

Escribe en data/powerbi/:
    ventas.parquet       una fila por línea de compra, con el valor original y el corregido
    clientes.parquet     la ficha de cada cliente y su segmento en los seis escenarios
    segmentos.parquet    nombres de los segmentos del modelo de trabajo
    regiones.parquet     región con el nombre que entiende el mapa
    calendario.parquet   un día por fila

No se pierde información: fecha_pedido es igual a fecha en todas las líneas, y
transaccional_corregido ya trae 'valor' (original) y 'valor_corregido'.

Uso (desde cualquier carpeta):
    pip install duckdb
    python scripts/powerbi/preparar_tablas.py
"""
from pathlib import Path

import duckdb

RAIZ = Path(__file__).resolve().parent.parent.parent
DATA = RAIZ / "data"
SALIDA = DATA / "powerbi"

TRX = DATA / "transaccional_corregido.parquet"
FEATURES = DATA / "features_clientes_corregido.parquet"
FEATURES_ORIGINAL = DATA / "features_clientes_original.parquet"
ESCENARIOS = [f"{v}_{p}" for v in ("original", "corregido") for p in ("sin_pca", "con_pca", "con_pca_90")]
MODELO_TRABAJO = "corregido_con_pca_90"

NOMBRES_SEGMENTOS = {
    0: "No segmentado",
    1: "Proyecto recurrente",
    2: "Proyecto grande en una visita",
    3: "Solo Baños",
    4: "Solo Pinturas",
    5: "Solo Cocinas",
    6: "Solo Revestimientos",
}

# Nombre de cada departamento como lo reconoce el mapa de Power BI
NOMBRES_MAPA = {
    "BOGOTÁ": "Bogotá", "VALLE": "Valle del Cauca", "NORTE SANTANDER": "Norte de Santander",
    "SAN ANDRÉS": "San Andrés y Providencia", "BOYACA": "Boyacá", "CAQUETA": "Caquetá",
    "CHOCO": "Chocó", "GUAINIA": "Guainía", "QUINDIO": "Quindío", "VAUPES": "Vaupés",
}


def guardar(con: duckdb.DuckDBPyConnection, consulta: str, nombre: str) -> None:
    ruta = SALIDA / f"{nombre}.parquet"
    con.execute(f"COPY ({consulta}) TO '{ruta}' (FORMAT parquet, COMPRESSION snappy)")
    filas = con.execute(f"SELECT count(*) FROM '{ruta}'").fetchone()[0]
    print(f"{nombre:12s} {filas:>10,} filas")


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    con = duckdb.connect()

    # Ventas: se renombra 'valor' para que no se confunda con el corregido.
    # La visita (cliente + fecha + tienda) queda como un número para contarla rápido.
    guardar(con, f"""
        SELECT fecha, id_pedido, cliente, asesor,
               coalesce(region, 'SIN REGIÓN') AS region, municipio, punto_de_venta, cluster,
               negocio, categoria, subcategoria, producto, cantidad, precio,
               valor AS valor_original, valor_corregido, valor_corregido_flag,
               dense_rank() OVER (ORDER BY cliente, fecha, punto_de_venta) AS visita
        FROM '{TRX}'
    """, "ventas")

    # Clientes: ficha corregida + montos originales + segmento en cada escenario
    uniones = "\n".join(
        f"LEFT JOIN '{DATA / f'segmentos_{e}.parquet'}' s_{e} USING (cliente)" for e in ESCENARIOS
    )
    columnas_seg = ",\n".join(
        f"s_{e}.segmento_kmeans AS seg_{e}_kmeans, s_{e}.segmento_kmedoids AS seg_{e}_kmedoids"
        for e in ESCENARIOS
    )
    guardar(con, f"""
        SELECT f.*,
               o.monto AS monto_original, o.ticket_visita AS ticket_visita_original,
               -- Rangos de visitas: la distribución tiene una cola muy larga
               CASE WHEN f.frecuencia_visitas <= 3 THEN f.frecuencia_visitas::VARCHAR
                    WHEN f.frecuencia_visitas <= 5 THEN '4 a 5'
                    WHEN f.frecuencia_visitas <= 10 THEN '6 a 10'
                    ELSE '11 o más' END AS rango_visitas,
               CASE WHEN f.frecuencia_visitas <= 3 THEN f.frecuencia_visitas
                    WHEN f.frecuencia_visitas <= 5 THEN 4
                    WHEN f.frecuencia_visitas <= 10 THEN 5
                    ELSE 6 END AS orden_rango_visitas,
               coalesce(s_{MODELO_TRABAJO}.segmento_kmeans, 0) AS segmento,
               {columnas_seg}
        FROM '{FEATURES}' f
        LEFT JOIN '{FEATURES_ORIGINAL}' o USING (cliente)
        {uniones}
    """, "clientes")

    filas = ", ".join(f"({k}, '{v}')" for k, v in NOMBRES_SEGMENTOS.items())
    guardar(con, f"SELECT * FROM (VALUES {filas}) t(segmento, nombre_segmento)", "segmentos")

    regiones = con.execute(f"SELECT DISTINCT region FROM '{SALIDA / 'ventas.parquet'}'").df()
    regiones["departamento_mapa"] = [
        None if r == "SIN REGIÓN" else NOMBRES_MAPA.get(r, r.title()) + ", Colombia"
        for r in regiones["region"]
    ]
    con.register("regiones", regiones)
    guardar(con, "SELECT * FROM regiones", "regiones")

    guardar(con, f"""
        SELECT d::DATE AS fecha, year(d) AS anio, month(d) AS mes,
               year(d) * 100 + month(d) AS anio_mes, quarter(d) AS trimestre,
               strftime(d, '%Y-%m') AS mes_texto
        FROM range(DATE '2000-01-01', DATE '2003-01-01', INTERVAL 1 DAY) t(d)
    """, "calendario")

    # Verificación: mismas filas y mismos totales que las bases de origen
    origen = con.execute(f"SELECT count(*), sum(valor), sum(valor_corregido) FROM '{TRX}'").fetchone()
    nuevo = con.execute(
        f"SELECT count(*), sum(valor_original), sum(valor_corregido) FROM '{SALIDA / 'ventas.parquet'}'"
    ).fetchone()
    assert origen[0] == nuevo[0] and abs(origen[1] - nuevo[1]) < 1 and abs(origen[2] - nuevo[2]) < 1
    clientes_origen = con.execute(f"SELECT count(*) FROM '{FEATURES}'").fetchone()[0]
    clientes_nuevo = con.execute(f"SELECT count(*) FROM '{SALIDA / 'clientes.parquet'}'").fetchone()[0]
    assert clientes_origen == clientes_nuevo
    print(f"Verificación OK: {nuevo[0]:,} líneas, valor original {nuevo[1]:,.0f}, corregido {nuevo[2]:,.0f}")


if __name__ == "__main__":
    main()
