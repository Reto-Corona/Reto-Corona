# Reto Analytics Corona: una nueva mirada al consumidor

Equipo **Oltre i dati**: Nicolás Fajardo y Felipe Ardila (Universidad de los Andes).

Solución analítica en cinco etapas para segmentar a los clientes de Corona y proponer acciones medibles:

| Etapa | Pregunta | Modelos |
|---|---|---|
| 1. Identificar | ¿Quién es cada cliente? | Segmentación (PCA + K-means / K-medoids), empresas (ABC + K-means) |
| 2. Entender | ¿Qué compra y en qué orden? | Canasta (FP-Growth), secuencias (PrefixSpan), elasticidad precio |
| 3. Predecir | ¿Quién vuelve y cuándo? | Recompra (logística y LightGBM), red bayesiana, supervivencia (Kaplan-Meier y Cox) |
| 4. Decidir | ¿Qué hacer y dónde crecer? | Cadena de Markov y MDP, territorio con datos del DANE |
| 5. Validar | ¿Funciona de verdad? | Diseño del piloto y KPIs |

## Estructura

```
data/            Bases de Corona (no incluidas) y archivos intermedios (.parquet)
data/externos/   Datos del DANE y su README de descarga
scripts/         Limpieza, preparación y pipeline principal (run_pipeline.py)
scripts/segmentacion/  Segmentación en R
scripts/modelos/       Modelos m03–m14 (run_modelos.py)
scripts/powerbi/       Preparación de tablas para el tablero
powerbi/         Tablero de Power BI (.pbip)
salidas/         Resultados: original/ y corregido/ (segmentación), modelos/ y modelos_original/
docs/            Documento guía del reto y manual de usuario
```

## Inicio rápido

```bash
pip install -r requirements.txt          # Python 3.10+
python scripts/run_pipeline.py           # limpieza, corrección del valor y fichas de cliente
Rscript scripts/segmentacion/segmentacion.R   # segmentación (6 escenarios)
python scripts/modelos/run_modelos.py    # todos los modelos con el valor corregido
python scripts/modelos/run_modelos.py --version original   # mismos modelos con el valor original
```

Un solo modelo: `python scripts/modelos/m08_recompra.py --version corregido`.

## Notas

- Los montos están en las unidades escaladas de la base de Corona, no en pesos.
- El valor de 8,5% de las líneas venía dividido por 10, 100 o 1.000; se recalcula como precio × cantidad / 2,65. Todo corre con ambas versiones y las conclusiones no cambian.
- Las bases de Corona son confidenciales y no se incluyen en el repositorio.
- Uso de IA: Claude (Anthropic) como apoyo en programación, revisión y redacción; los resultados son reproducibles con este código.

Ver `docs/Manual_de_usuario.pdf` para el detalle de instalación, ejecución y lectura de resultados.