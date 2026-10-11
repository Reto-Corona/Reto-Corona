"""
variables_clientes.py - Reto Analytics Corona
Prepara la base para la segmentación:
  1. Construye las variables por cliente          -> data/features_clientes_v2.parquet
  2. Imprime 4 descriptivos para el informe (y los guarda en salidas/*.csv)

Requiere haber corrido antes:
    limpieza_transaccional.py  (crea transaccional_corregido.parquet, con valor_corregido)
    limpieza_clientes.py       (crea clientes_limpio.parquet)

Uso (desde cualquier carpeta):
    pip install duckdb pandas
    python scripts/limpieza/variables_clientes.py
"""
from pathlib import Path
import unicodedata
import duckdb

RAIZ = Path(__file__).resolve().parent.parent.parent
DATA = RAIZ / "data"
OUT = RAIZ / "salidas"
OUT.mkdir(exist_ok=True)

for archivo in ("transaccional_corregido.parquet", "clientes_limpio.parquet"):
    if not (DATA / archivo).exists():
        raise FileNotFoundError(
            f"No se encontró {DATA / archivo}. Corre primero limpieza_transaccional.py "
            "y limpieza_clientes.py"
        )

con = duckdb.connect()
con.execute(f"CREATE TABLE trx AS SELECT * FROM '{DATA / 'transaccional_corregido.parquet'}'")

# ---------------------------------------------------------------------------
# 1. Variables por cliente (frecuencia en visitas = cliente + fecha + punto de venta)
# ---------------------------------------------------------------------------
negocios = [n for (n,) in con.execute(
    "SELECT DISTINCT negocio FROM trx WHERE negocio IS NOT NULL ORDER BY 1").fetchall()]
def col(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    return "share_" + "".join(ch if ch.isalnum() else "_" for ch in n).strip("_")
shares = ",\n       ".join(
    f"SUM(CASE WHEN negocio = '{n}' THEN valor_corregido ELSE 0 END) / NULLIF(SUM(valor_corregido), 0) AS {col(n)}"
    for n in negocios)

con.execute(f"""
CREATE TABLE features AS
WITH ref AS (SELECT MAX(fecha) + 1 AS f FROM trx),
visitas AS (
  SELECT cliente, fecha, punto_de_venta, SUM(valor_corregido) AS v
  FROM trx GROUP BY cliente, fecha, punto_de_venta
),
v_cli AS (
  SELECT cliente,
         COUNT(*) AS frecuencia_visitas,
         MIN(fecha) AS primera_compra,
         MAX(fecha) AS ultima_compra,
         COUNT(DISTINCT strftime(fecha, '%Y-%m')) AS meses_con_compra
  FROM visitas GROUP BY cliente
),
mes AS (
  SELECT cliente, strftime(fecha, '%Y-%m') AS m, SUM(valor_corregido) AS v
  FROM trx GROUP BY cliente, m
),
pico AS (SELECT cliente, MAX(v) / NULLIF(SUM(v), 0) AS concentracion_mes_pico FROM mes GROUP BY cliente),
-- Desempates (para que el resultado sea igual en cada corrida): si dos opciones
-- empatan, gana la de mayor valor y, si siguen empatadas, la primera en orden
-- alfabético. El valor se redondea a centavos porque la suma en paralelo puede
-- variar en el último decimal entre corridas.
neg AS (
  SELECT cliente, negocio, SUM(valor_corregido) AS v,
         ROW_NUMBER() OVER (PARTITION BY cliente
                            ORDER BY ROUND(SUM(valor_corregido), 2) DESC, negocio) AS rk
  FROM trx GROUP BY cliente, negocio
),
reg AS (
  SELECT cliente, region,
         ROW_NUMBER() OVER (PARTITION BY cliente
                            ORDER BY COUNT(*) DESC, ROUND(SUM(valor_corregido), 2) DESC, region) AS rk
  FROM trx WHERE region IS NOT NULL GROUP BY cliente, region
),
clu AS (
  SELECT cliente, cluster,
         ROW_NUMBER() OVER (PARTITION BY cliente
                            ORDER BY COUNT(*) DESC, ROUND(SUM(valor_corregido), 2) DESC, cluster) AS rk
  FROM trx WHERE cluster IS NOT NULL GROUP BY cliente, cluster
),
t AS (
  SELECT cliente,
         COUNT(DISTINCT id_pedido) AS n_pedidos,
         COUNT(*) AS n_lineas,
         SUM(valor_corregido) AS monto,
         COUNT(DISTINCT negocio) AS n_negocios,
         COUNT(DISTINCT categoria) AS n_categorias,
         COUNT(DISTINCT punto_de_venta) AS n_pdv,
         MAX(valor_corregido_flag) AS tuvo_valor_corregido,
       {shares}
  FROM trx GROUP BY cliente
)
SELECT t.*,
       v.frecuencia_visitas,
       DATEDIFF('day', v.ultima_compra, (SELECT f FROM ref)) AS recencia_dias,
       DATEDIFF('day', v.primera_compra, v.ultima_compra) AS dias_activo,
       CASE WHEN v.frecuencia_visitas > 1
            THEN DATEDIFF('day', v.primera_compra, v.ultima_compra) / (v.frecuencia_visitas - 1) END
            AS dias_entre_visitas,
       t.monto / v.frecuencia_visitas AS ticket_visita,
       t.n_lineas * 1.0 / v.frecuencia_visitas AS lineas_por_visita,
       v.meses_con_compra,
       p.concentracion_mes_pico,
       n.negocio AS negocio_principal,
       n.v / NULLIF(t.monto, 0) AS share_negocio_principal,
       r.region AS region_principal,
       k.cluster AS cluster_principal,
       c.tipo_cliente, c.edad, c.genero, c.tratamiento_datos, c.es_contactable, c.primera_fecha,
       CASE WHEN c.edad IS NULL THEN 'Sin edad'
            WHEN c.edad <= 29 THEN 'Generación Z'
            WHEN c.edad <= 45 THEN 'Millennials'
            WHEN c.edad <= 61 THEN 'Generación X'
            ELSE 'Baby Boomers' END AS generacion,
       CASE WHEN c.edad IN (49, 67) THEN 1 ELSE 0 END AS edad_dudosa,
       CASE WHEN c.primera_fecha < DATE '2000-03-01' THEN 1 ELSE 0 END AS posible_cliente_previo,
       CASE WHEN c.cliente IS NULL THEN 0 ELSE 1 END AS en_maestro
FROM t
JOIN v_cli v USING (cliente)
JOIN pico p USING (cliente)
LEFT JOIN neg n ON n.cliente = t.cliente AND n.rk = 1
LEFT JOIN reg r ON r.cliente = t.cliente AND r.rk = 1
LEFT JOIN clu k ON k.cliente = t.cliente AND k.rk = 1
LEFT JOIN '{DATA / "clientes_limpio.parquet"}' c ON c.cliente = t.cliente
""")
# Tamaño de empresa: misma idea del script anterior (percentiles 50 y 90 entre empresas),
# pero con el monto corregido en vez de la suma de cantidades, que mezcla unidades distintas.
con.execute("""
CREATE OR REPLACE TABLE features AS
WITH p AS (
  SELECT QUANTILE_CONT(monto, 0.50) AS p50, QUANTILE_CONT(monto, 0.90) AS p90
  FROM features WHERE tipo_cliente = 'Empresa'
)
SELECT f.*,
       CASE WHEN f.tipo_cliente <> 'Empresa' OR f.tipo_cliente IS NULL THEN 'No aplica'
            WHEN f.monto <= p.p50 THEN 'Empresa pequeña'
            WHEN f.monto <= p.p90 THEN 'Empresa mediana'
            ELSE 'Empresa grande / distribuidor' END AS segmento_empresa
FROM features f, p
""")
con.execute(f"COPY features TO '{DATA / 'features_clientes_v2.parquet'}' (FORMAT PARQUET)")

# ---------------------------------------------------------------------------
# 2. Descriptivos (con valor corregido)
# ---------------------------------------------------------------------------
consultas = {
    "ventas_por_negocio": """
        SELECT negocio,
               COUNT(DISTINCT cliente) AS clientes,
               ROUND(SUM(valor_corregido)) AS ventas,
               ROUND(100.0 * SUM(valor_corregido) / SUM(SUM(valor_corregido)) OVER (), 1) AS pct_ventas
        FROM trx GROUP BY negocio ORDER BY ventas DESC""",
    "pareto_clientes": """
        WITH o AS (SELECT monto, ROW_NUMBER() OVER (ORDER BY monto DESC) * 1.0 / COUNT(*) OVER () AS pos
                   FROM features)
        SELECT tramo, ROUND(100.0 * SUM(monto) / (SELECT SUM(monto) FROM features), 1) AS pct_ventas
        FROM (SELECT monto, CASE WHEN pos <= 0.01 THEN '1. Top 1%'  WHEN pos <= 0.10 THEN '2. Top 2-10%'
                                 WHEN pos <= 0.20 THEN '3. Top 11-20%' WHEN pos <= 0.50 THEN '4. 21-50%'
                                 ELSE '5. 51-100%' END AS tramo FROM o)
        GROUP BY tramo ORDER BY tramo""",
    "frecuencia_visitas": """
        SELECT CASE WHEN frecuencia_visitas = 1 THEN '1' WHEN frecuencia_visitas = 2 THEN '2'
                    WHEN frecuencia_visitas = 3 THEN '3' WHEN frecuencia_visitas <= 5 THEN '4-5'
                    WHEN frecuencia_visitas <= 10 THEN '6-10' ELSE '11+' END AS visitas,
               COUNT(*) AS clientes,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_clientes,
               ROUND(100.0 * SUM(monto) / SUM(SUM(monto)) OVER (), 1) AS pct_ventas
        FROM features GROUP BY visitas ORDER BY MIN(frecuencia_visitas)""",
    "consumidores_por_generacion": """
        SELECT generacion,
               COUNT(*) AS clientes,
               ROUND(100.0 * SUM(monto) / SUM(SUM(monto)) OVER (), 1) AS pct_ventas,
               ROUND(MEDIAN(monto)) AS monto_mediano,
               ROUND(MEDIAN(ticket_visita)) AS ticket_mediano,
               ROUND(AVG(frecuencia_visitas), 2) AS visitas_prom,
               ROUND(100.0 * AVG(es_contactable::INT), 1) AS pct_contactable
        FROM features WHERE tipo_cliente = 'Consumidor final'
        GROUP BY generacion ORDER BY MIN(edad)""",
}

print(f"Líneas: {con.execute('SELECT COUNT(*) FROM trx').fetchone()[0]:,} | "
      f"corregidas: {con.execute('SELECT SUM(valor_corregido_flag) FROM trx').fetchone()[0]:,}")
print(f"Clientes en features: {con.execute('SELECT COUNT(*) FROM features').fetchone()[0]:,}\n")
for nombre, sql in consultas.items():
    df = con.execute(sql).df()
    df.to_csv(OUT / f"{nombre}.csv", index=False)
    print(f"== {nombre} ==\n{df.to_string(index=False)}\n")