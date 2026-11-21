"""
Resumen de todos los modelos (m02 a m14) en salidas/modelos/resumen_modelos.md.

Lee los CSV que escribe cada modelo en salidas/modelos/<script>/ y arma:
    - una tabla con técnica, pregunta, métrica principal y hallazgo principal de cada modelo
    - cómo se conectan los modelos entre sí
    - los supuestos clave (efecto de las acciones y costo de contacto) con su sensibilidad

Se corre al final de run_modelos.py; si falta la salida de un modelo, su fila lo indica.

Uso:
    python scripts/modelos/resumen_modelos.py
"""
import pandas as pd

import comun as c


def leer(modelo: str, archivo: str, **kw) -> pd.DataFrame:
    """CSV de un modelo (error claro si no existe)."""
    ruta = c.SALIDAS_MODELOS / modelo / archivo
    if not ruta.exists():
        raise FileNotFoundError(ruta)
    return pd.read_csv(ruta, **kw)


def fila_m02():
    m = leer("m02_gmm", "metricas.csv").iloc[0]
    amb = leer("m02_gmm", "ambiguos_por_segmento.csv").set_index("segmento")
    return ("Mezclas gaussianas (GMM), BIC", "¿La segmentación K-means se sostiene con otra técnica?",
            f"Rand ajustado K-means vs GMM (k = 6): {c.num(m.rand_ajustado_k6, 3)}",
            f"GMM agrupa por mezcla de negocios y K-means por valor y frecuencia: solo "
            f"{c.pct(amb.loc[2, 'pct_mismo_grupo'])} de \"Proyecto grande\" coincide; los \"Solo ...\" sí coinciden.")


def fila_m03():
    m = leer("m03_empresas", "metricas.csv").iloc[0]
    abc = leer("m03_empresas", "abc.csv").set_index("abc")
    cruce = leer("m03_empresas", "cruce_grupo_abc.csv", index_col=0)
    g = cruce["A"].idxmax()
    return ("ABC + K-means (silueta)", "¿Cómo se agrupan las empresas?",
            f"k = {int(m.k)} (silueta {c.num(m.silueta, 3)}); A = {c.pct(abc.loc['A', 'pct_empresas'])} de las empresas",
            f"{c.miles(abc.loc['A', 'empresas'])} empresas suman el 80% de las ventas a empresas; el grupo {g} reúne "
            f"{c.pct(cruce.loc[g, 'A'] / cruce['A'].sum() * 100)} de ellas (cuentas clave).")


def fila_m04():
    m = leer("m04_red_bayesiana", "metricas.csv").set_index("modelo")
    a = m.loc["Estructura aprendida (hill climbing + BIC)"]
    return ("Red bayesiana (pgmpy, hill climbing + BIC)", "¿Qué explica la recompra, en probabilidades interpretables?",
            f"AUC {c.num(a.auc, 3)} (experta {c.num(m.loc['Estructura experta', 'auc'], 3)}; m08 {c.num(m.loc['m08 LightGBM', 'auc'], 3)})",
            f"La recompra depende directamente solo de {a.padres_recompra.replace('valor', 'valor de la visita')}; "
            "canal, región y perfil no agregan información.")


def fila_m05():
    top = leer("m05_canasta", "top20_reglas.csv")
    cruz = leer("m05_canasta", "reglas_venta_cruzada.csv")
    r = cruz.iloc[0]
    return ("Reglas de asociación (FP-Growth)", "¿Qué se compra junto en la misma visita?",
            f"{c.miles(len(cruz))} combinaciones de venta cruzada con lift > 1,2",
            f"{(top['cruza_negocios'] == False).sum()} de las 20 reglas más fuertes son del mismo negocio; la venta cruzada "
            f"más fuerte es {r.regla.split('  →  ')[0]} → {r.regla.split('  →  ')[1]} (lift {c.num(r.lift, 1)}).")


def fila_m06():
    m = leer("m06_secuencias", "metricas.csv").iloc[0]
    p = leer("m06_secuencias", "patrones_frecuentes.csv")
    nuevo = p[p["cambia_negocio"]].iloc[0]
    return ("Minería de secuencias (PrefixSpan) + matriz de transición", "¿En qué orden compran los negocios?",
            f"{c.miles(m.clientes_2_visitas)} clientes con 2+ visitas; {int(m.patrones)} patrones",
            f"{c.pct(m.pct_repite_negocio_1a_2a)} repite el negocio de la 1.ª a la 2.ª visita; la venta cruzada en el tiempo "
            f"más común es {nuevo.patron} ({c.pct(nuevo.soporte * 100)}).")


def fila_m07():
    e = leer("m07_elasticidad", "elasticidades.csv")
    pm = leer("m07_elasticidad", "elasticidad_producto_mes.csv").set_index("grupo")
    tot = e[e["nivel"] == "Total"].iloc[0]
    neg = e[e["nivel"] == "Negocio"].set_index("grupo")
    robustos = [g for g in neg.index if g in pm.index and pm.loc[g, "ic95_sup"] < 0
                and abs(neg.loc[g, "elasticidad"] - pm.loc[g, "elasticidad"]) < 0.3]
    mas = neg.loc[robustos, "elasticidad"].idxmin() if robustos else neg["elasticidad"].idxmin()
    return ("Regresión log-log con efectos fijos de producto y mes", "¿Qué negocios responden más al precio?",
            f"Elasticidad total {c.num(tot.elasticidad, 2)} [{c.num(tot.ic95_inf, 2)}; {c.num(tot.ic95_sup, 2)}]",
            f"{mas} es el más sensible al precio ({c.num(neg.loc[mas, 'elasticidad'], 2)}) y el resultado es robusto; en "
            "Revestimientos la elasticidad por línea es descuento por volumen.")


def fila_m08():
    m = leer("m08_recompra", "metricas.csv").set_index("modelo")
    s = leer("m08_recompra", "recompra_por_segmento.csv").set_index("segmento")
    best = m["auc"].idxmax()
    return (f"Logística vs. LightGBM + SHAP", "¿Quién vuelve a comprar en 90 días?",
            f"AUC {c.num(m.loc[best, 'auc'], 3)} ({best}); lift decil 1 = {c.num(m.loc[best, 'lift_decil_1'], 2)}x",
            f"El valor y el negocio de la primera visita son lo que más pesa; el segmento 2 recompra "
            f"{c.pct(s.loc[2, 'tasa_real'] * 100)} ({c.pct(s.loc[2, 'tasa_real_contactables'] * 100)} entre contactables, "
            "la línea base del piloto).")


def fila_m09():
    m = leer("m09_supervivencia", "metricas.csv").iloc[0]
    k = leer("m09_supervivencia", "concentracion_recompra.csv").iloc[0]
    return ("Kaplan-Meier + Cox (lifelines)", "¿Cuándo se hace la segunda compra?",
            f"Concordancia de Cox (prueba) {c.num(m.concordancia_prueba, 3)}",
            f"La segunda compra se concentra al inicio: mediana día {c.miles(k.mediana)}, {c.pct(k.pct_30)} en 30 días; hay que "
            f"contactar en las primeras dos semanas.")


def fila_m10():
    v = leer("m10_clv", "metricas_validacion.csv").iloc[0]
    s = leer("m10_clv", "resumen_por_segmento.csv").set_index("segmento")
    return ("BG/NBD + Gamma-Gamma (lifetimes)", "¿Cuánto valdrá cada cliente en 12 meses?",
            f"Error en compras totales {c.num(v.error_total_pct, 1)}%; AUC vuelve a comprar {c.num(v.auc_vuelve_a_comprar, 3)}",
            f"\"Proyecto recurrente\" concentra {c.pct(s.loc[1, 'pct_valor'])} del valor esperado a 12 meses.")


def fila_m11():
    s = leer("m11_mdp", "sensibilidad_costo_efecto.csv")
    b = s[(s["pct_ticket"] == 0.02) & (s["factor_efecto"] == 1.0)].iloc[0]
    return ("Cadena de Markov + MDP (iteración de valor)", "¿Qué acción conviene en cada etapa del ciclo de vida?",
            f"{c.pct(b.pct_contactables_con_accion)} de los contactables con acción recomendada (supuestos base)",
            f"La decisión depende de supuestos: según costo y efecto va de {c.pct(s['pct_contactables_con_accion'].min())} a "
            f"{c.pct(s['pct_contactables_con_accion'].max())}; en el segmento 2 es de borde.")


def fila_m12():
    comp = leer("m12_territorio", "comparacion_fuentes.csv")
    from scipy.stats import spearmanr
    rho = spearmanr(comp["ranking_principal"], comp["ranking_verificacion"]).statistic
    top = comp.sort_values("ranking_principal")["region"].head(3).str.title().tolist()
    reg = leer("m12_territorio", "regresiones.csv").iloc[0]
    return ("Regresión log-log con datos del DANE", "¿Dónde hay menos clientes de los esperados?",
            f"R² {c.num(reg.r2_hogares, 2)}; Spearman proyección vs censo {c.num(rho, 3)}",
            f"Mayor oportunidad: {', '.join(top)}; el resultado no depende de la fuente del DANE.")


def fila_m13():
    b = leer("m13_mochila", "comparacion_presupuesto_base.csv").iloc[0]
    return ("Programación entera (OR-Tools, mochila)", "¿A quién contactar con un presupuesto dado?",
            f"Valor neto {c.miles(b.valor_mochila)} vs. {c.miles(b.valor_regla_simple)} de la regla simple (10.000 contactos)",
            f"Contactar por probabilidad sola destruye valor; m11 y m13 coinciden en {c.pct(b.pct_coinciden_m11_misma_accion)} "
            "cuando evalúan la misma acción.")


def fila_m14():
    t = leer("m14_piloto", "tamano_muestra.csv").set_index("aumento_pp")
    r = t.loc[3]
    return ("Diseño experimental (potencia, statsmodels)", "¿Cómo medir el efecto real de contactar?",
            f"{c.miles(r.n_por_grupo)} clientes por grupo para +3 pp (base {c.pct(r.tasa_base * 100)})",
            f"Con clientes nuevos del segmento 2 el reclutamiento toma {c.num(r.meses_de_reclutamiento, 1)} meses más 90 días de "
            "seguimiento.")


MODELOS = [("m02", "m02_gmm", fila_m02), ("m03", "m03_empresas", fila_m03), ("m04", "m04_red_bayesiana", fila_m04),
           ("m05", "m05_canasta", fila_m05), ("m06", "m06_secuencias", fila_m06), ("m07", "m07_elasticidad", fila_m07),
           ("m08", "m08_recompra", fila_m08), ("m09", "m09_supervivencia", fila_m09), ("m10", "m10_clv", fila_m10),
           ("m11", "m11_mdp", fila_m11), ("m12", "m12_territorio", fila_m12), ("m13", "m13_mochila", fila_m13),
           ("m14", "m14_piloto", fila_m14)]


def seccion_supuestos() -> str:
    """Supuestos de m11 y su sensibilidad (tabla 3 x 3)."""
    s = leer("m11_mdp", "sensibilidad_costo_efecto.csv")
    sup = leer("m11_mdp", "supuestos_acciones.csv")
    vc = sup[sup["accion"] == "Venta cruzada"].set_index("estado")["aumento_prob_compra"]
    de = sup[sup["accion"] == "Descuento"].set_index("estado")["aumento_prob_compra"]
    pm = sup.loc[sup["accion"] == "Descuento", "perdida_margen"].iloc[0]
    costo = sup.loc[sup["accion"] == "Venta cruzada", "costo_contacto_unidades_base"].iloc[0]
    efectos = list(dict.fromkeys(s["efecto"]))
    filas = []
    for costo_n in dict.fromkeys(s["costo"]):
        celdas = []
        for e in efectos:
            x = s[(s["costo"] == costo_n) & (s["efecto"] == e)].iloc[0]
            celdas.append(f"{c.pct(x.pct_contactables_con_accion)} · {x.accion_segmento2_nuevo}")
        filas.append(f"| {costo_n} | " + " | ".join(celdas) + " |")
    curva = leer("m13_mochila", "curva_presupuesto.csv")
    mejor = curva.loc[curva["optimo_valor"].idxmax()]
    return f"""
## Supuestos clave y su sensibilidad
Los efectos de las acciones **no están en los datos**; están definidos una sola vez en
`scripts/modelos/m11_mdp.py` (`EFECTOS`, `costo_contacto`, `ganancia_inmediata`) y m13 y m14 los importan.

| Supuesto | Valor base | Dónde se usa |
|---|---|---|
| Aumento de la probabilidad de compra con venta cruzada (Nuevo / Recurrente / Leal / Dormido / Perdido) | {' / '.join(c.num(vc[e] * 100, 1) for e in ['Nuevo', 'Recurrente', 'Leal', 'Dormido', 'Perdido'])} pp | m11, m13 |
| Aumento con descuento | {' / '.join(c.num(de[e] * 100, 1) for e in ['Nuevo', 'Recurrente', 'Leal', 'Dormido', 'Perdido'])} pp | m11 |
| Pérdida de margen del descuento (sobre toda la compra del mes) | {c.pct(pm * 100, 0)} | m11 |
| Costo por contacto | 2% del ticket mediano por visita = {c.miles(costo)} unidades de la base | m11, m13, m14 |
| Factor de descuento mensual del MDP | 0,95 | m11 |

**Sensibilidad (m11):** % de contactables con alguna acción recomendada · acción en el segmento 2, estado Nuevo.

| Costo por contacto | {' | '.join(efectos)} |
|---|{'---|' * len(efectos)}
{chr(10).join(filas)}

**Sensibilidad al presupuesto (m13):** con la agrupación por decil de valor, el valor neto máximo
({c.miles(mejor.optimo_valor)}) se alcanza con unos {c.miles(mejor.contactos)} contactos; más presupuesto no agrega valor con
los supuestos base. El piloto (m14) es el que debe reemplazar estos supuestos por efectos medidos.
"""


def main() -> None:
    filas, faltan = [], []
    for codigo, carpeta, funcion in MODELOS:
        try:
            tecnica, pregunta, metrica, hallazgo = funcion()
        except (FileNotFoundError, KeyError, IndexError) as e:
            faltan.append(codigo)
            tecnica = pregunta = metrica = "—"
            hallazgo = f"Sin salida ({type(e).__name__}); correr `python scripts/modelos/{carpeta}.py`."
        filas.append(f"| [{codigo}]({carpeta}/resumen.md) | {tecnica} | {pregunta} | {metrica} | {hallazgo} |")
    try:
        supuestos = seccion_supuestos()
    except (FileNotFoundError, KeyError, IndexError):
        supuestos = "\n## Supuestos clave y su sensibilidad\nFaltan las salidas de m11 o m13.\n"
    texto = f"""
# Resumen de los modelos (m02 a m14)

Versión del valor: `{c.VERSION}`; segmentación de trabajo: `{c.SEGMENTACION}` (6 segmentos de consumidores
finales). Cada modelo tiene su detalle en `salidas/modelos/<script>/resumen.md`. Valores monetarios en
las unidades (escaladas) de la base.

| Modelo | Técnica | Pregunta | Métrica principal | Hallazgo principal |
|---|---|---|---|---|
{chr(10).join(filas)}

## Cómo se conectan
1. **Quiénes son los clientes.** La segmentación K-means (`con_pca_90`) es la base; **m02** la verifica con
   otra técnica y **m03** segmenta aparte a las empresas, que no entran en los demás modelos.
2. **Qué compran y en qué orden.** **m05** (misma visita) y **m06** (visitas sucesivas) dicen qué
   recomendar en la venta cruzada; **m07** dice dónde el precio mueve la demanda (insumo para el
   supuesto de descuento).
3. **Quién vuelve, cuándo y cuánto vale.** **m08** (probabilidad de recompra a 90 días) y **m04** (la misma
   pregunta con una red bayesiana) identifican a quién contactar; **m09** dice cuándo (la recompra se
   concentra en las primeras semanas); **m10** estima el valor a 12 meses de cada cliente.
4. **Qué hacer.** **m11** (MDP) define el efecto de cada acción y elige la política por etapa del ciclo de
   vida; **m13** usa la misma definición del efecto, el valor de m10 y un presupuesto para elegir
   clientes concretos; **m12** prioriza territorios.
5. **Cómo medirlo.** **m14** diseña el piloto en el segmento 2, con la línea base de m08 y la ventana de
   contacto de m09; sus resultados reemplazan los supuestos de m11 y m13, y en la fase 2 alimentan un
   modelo de uplift.
{supuestos}"""
    destino = c.SALIDAS_MODELOS / "resumen_modelos.md"
    destino.write_text(texto.strip() + "\n", encoding="utf-8")
    print(f"  {destino.relative_to(c.RAIZ)}" + (f" (faltan: {', '.join(faltan)})" if faltan else ""))


if __name__ == "__main__":
    main()
