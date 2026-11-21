# Datos externos

Archivos oficiales del DANE (Departamento Administrativo Nacional de Estadística de Colombia)
que usa `scripts/modelos/m12_territorio.py`. Se guardan tal como se descargaron, sin editar.

Fecha de descarga de los cuatro archivos: **2026-11-18**.

| Archivo | Fuente | Año de referencia | Qué contiene | Uso en m12 |
|---|---|---|---|---|
| `HOGARES_Cuadros_CNPV_2018.XLSX` | DANE, Censo Nacional de Población y Vivienda (CNPV) 2018, cuadros de hogares | 2018 | Hoja `1HD`: número de hogares (total, cabecera y centros poblados/rural disperso) por departamento, con código DIVIPOLA. Otras hojas: hogares por municipio, adultos mayores, jefatura. | Hogares 2018 (fuente de verificación) |
| `CNPV-2018-Poblacion-Ajustada-por-Cobertura.xls` | DANE, CNPV 2018, información técnica | 2018 | Hoja `Ajuste por Cobertura CNPV 2018`: población censal ajustada por cobertura y omisión censal por departamento y área. | Población 2018 (fuente de verificación) |
| `anexo-proyecciones-hogares-dptal-mpal-2018-2042.xlsx` | DANE, Proyecciones de viviendas y hogares basadas en el CNPV 2018 (serie actualizada el 23-dic-2025) | 2018-2042 (se usa 2025) | Hoja `Proyecciones Hogares dpto`: hogares a 30 de junio de cada año por departamento y área. Hoja `Proyecciones Hogares mpio`: lo mismo por municipio. | Hogares 2025 (fuente principal) |
| `PPED-AreaDep-2018-2050_VP.xlsx` | DANE, Proyecciones de población departamentales 2018-2050 basadas en el CNPV 2018 (actualizadas el 30-jul-2025) | 2018-2050 (se usa 2025) | Hoja `PobDepartamentalxÁrea`: población proyectada por departamento, año y área (cabecera, centros poblados y rural disperso, total). | Población 2025 (fuente principal) |

## URL de descarga

- CNPV 2018, hogares: <https://www.dane.gov.co/files/censo2018/informacion-tecnica/HOGARES_Cuadros_CNPV_2018.XLSX>
  (página: <https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/censo-nacional-de-poblacion-y-vivenda-2018/informacion-tecnica>)
- CNPV 2018, población ajustada por cobertura: <https://www.dane.gov.co/files/censo2018/informacion-tecnica/CNPV-2018-Poblacion-Ajustada-por-Cobertura.xls>
  (misma página)
- Proyecciones de hogares: <https://www.dane.gov.co/files/operaciones/PPVH/anexo-proyecciones-hogares-dptal-mpal-2018-2042.xlsx>
  (página: <https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-viviendas-y-hogares>)
- Proyecciones de población: <https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Departamental/PPED-AreaDep-2018-2050_VP.xlsx>
  (página: <https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion>)

## Notas

- El año de proyección es el más reciente disponible dentro de 2023-2025, es decir **2025**
  (el script lo elige automáticamente).
- Los hogares del censo 2018 son conteos censales (14,2 millones) y no están ajustados por
  cobertura; la proyección para 2018 parte de una base ajustada (15,2 millones). Por eso las tasas
  por hogar del censo son algo más altas que las de la proyección.
- El cruce con las ventas se hace con el código DIVIPOLA del departamento (diccionario
  `CODIGOS_DIVIPOLA` en `m12_territorio.py`), no con el nombre, para evitar problemas con tildes y
  nombres abreviados (Bogotá, Valle, Norte de Santander, San Andrés).
- Si el DANE cambia las URL, descargar los mismos archivos desde las páginas indicadas y dejarlos
  en esta carpeta con el mismo nombre.
