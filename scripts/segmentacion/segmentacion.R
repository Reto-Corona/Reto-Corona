#   1. Prepara las variables de comportamiento de compra
#   2. Prueba varios números de grupos (k) y mide la calidad de cada uno
#   3. Ajusta el modelo final y mide su estabilidad
#   4. Describe cada segmento -> data/segmentos_<version>_<pca>.parquet
#                              -> salidas/<version>/<pca>/*
#
# Requiere haber corrido antes (Python):
#     limpieza_transaccional.py  limpieza_clientes.py  variables_clientes.py
#
# El script produce una de las seis variantes del cuadro:
#     CORONA_VERSION ∈ {original, corregido}             (valor por defecto: corregido)
#     CORONA_PCA     ∈ {sin_pca, con_pca, con_pca_90}    (valor por defecto: sin_pca)
#
# - sin_pca:     agrupa sobre las 8 variables estandarizadas.
# - con_pca:     PCA con corte de 80% de varianza (4 componentes); k lo elige el silhouette.
# - con_pca_90:  PCA con corte de 90% de varianza (5 componentes), k = 6 fijo. La
#   componente 5 (~11% de varianza, descartada en con_pca) carga sobre log_visitas vs
#   log_categorias y log_ticket, es decir, separa a los clientes que vuelven muchas
#   veces de los que compran mucho en una sola visita. k = 6 coincide con el codo de
#   la inercia.
#
# Uso (desde cualquier carpeta):
#     install.packages(c("nanoparquet", "cluster", "ggplot2"))
#     CORONA_VERSION=corregido CORONA_PCA=sin_pca     Rscript scripts/segmentacion/segmentacion.R
#     CORONA_VERSION=corregido CORONA_PCA=con_pca     Rscript scripts/segmentacion/segmentacion.R
#     CORONA_VERSION=corregido CORONA_PCA=con_pca_90  Rscript scripts/segmentacion/segmentacion.R
#     ... (idem con CORONA_VERSION=original)


suppressPackageStartupMessages({
  library(nanoparquet)  # leer y escribir .parquet
  library(cluster)      # clara (K-medoids para bases grandes) y silhouette
  library(ggplot2)
})

# Rutas de acceso
argumentos <- commandArgs(trailingOnly = FALSE)
archivo <- sub("^--file=", "", argumentos[grep("^--file=", argumentos)])
RAIZ <- if (length(archivo)) normalizePath(file.path(dirname(archivo), "..", "..")) else getwd()
DATA <- file.path(RAIZ, "data")
OUT_BASE <- file.path(RAIZ, "salidas")

# Variantes: version (original/corregido) y uso de PCA (sin_pca/con_pca/con_pca_90)
leer_version <- function() {
  v <- tolower(Sys.getenv("CORONA_VERSION", "corregido"))
  if (!v %in% c("original", "corregido"))
    stop("CORONA_VERSION debe ser 'original' o 'corregido' (recibí '", v, "').")
  v
}
leer_pca <- function() {
  p <- tolower(Sys.getenv("CORONA_PCA", "sin_pca"))
  if (!p %in% c("sin_pca", "con_pca", "con_pca_90"))
    stop("CORONA_PCA debe ser 'sin_pca', 'con_pca' o 'con_pca_90' (recibí '", p, "').")
  p
}

VERSION <- leer_version()
PCA_FLAG <- leer_pca()
USAR_PCA <- PCA_FLAG %in% c("con_pca", "con_pca_90")

OUT <- file.path(OUT_BASE, VERSION, PCA_FLAG)
GRAFICAS <- file.path(OUT, "graficas")
dir.create(GRAFICAS, recursive = TRUE, showWarnings = FALSE)

RUTA_FEATURES <- file.path(DATA, paste0("features_clientes_", VERSION, ".parquet"))
RUTA_SEGMENTOS <- file.path(DATA, paste0("segmentos_", VERSION, "_", PCA_FLAG, ".parquet"))

# Parámetros
SEMILLA <- 2026            # fija el azar: el resultado es igual en cada corrida
K_CANDIDATOS <- 2:8        # números de grupos que se prueban
K_MINIMO <- 3              # con 2 grupos no se pueden diferenciar acciones comerciales
# k final: NULL = lo elige el silhouette. Se fija en con_pca_90 porque la componente
# que se añade al pasar de 80% a 90% de varianza (log_visitas vs log_categorias y
# log_ticket) separa clientes recurrentes de los de una sola visita, y k = 6 coincide
# con el codo de la inercia.
K_FINAL <- if (PCA_FLAG == "con_pca_90") 6 else NULL
TAMANO_MUESTRA <- 5000     # clientes con los que se calcula el silhouette
INICIOS_KMEANS <- 10       # K-means se repite desde 10 puntos de partida distintos
ITERACIONES_KMEANS <- 300  # suficiente para evitar las advertencias "did not converge"
MUESTRAS_CLARA <- 20       # K-medoids (clara) busca medoides en 20 submuestras...
TAMANO_CLARA <- 1000       # ...de 1.000 clientes cada una
REPETICIONES_ESTABILIDAD <- 10
# Umbral de varianza acumulada para quedarse con los pcs: 0.80 para con_pca (típico),
# 0.90 para con_pca_90 (para no perder la componente que distingue recurrentes).
VARIANZA_OBJETIVO_PCA <- if (PCA_FLAG == "con_pca_90") 0.90 else 0.80

# Variables utiles para comportamiento de compra
NOMBRES_VARIABLES <- c(
  log_ticket = "Valor de la visita",
  log_visitas = "Número de visitas",
  log_lineas_visita = "Productos por visita",
  log_categorias = "Variedad de categorías",
  share_revestimientos_e_instalacion = "Gasto en Revestimientos",
  share_banos = "Gasto en Baños",
  share_cocinas = "Gasto en Cocinas",
  share_pinturas_e_impermeabilizantes = "Gasto en Pinturas"
)

#juegue

cargar_consumidores <- function() {
  # Solo consumidores finales con compras, las empresas se segmentan aparte,
  # porque muy pocas concentran gran parte de las ventas.
  if (!file.exists(RUTA_FEATURES)) {
    stop("No se encontró ", RUTA_FEATURES, ". Corre primero los tres scripts de Python.")
  }
  clientes <- as.data.frame(read_parquet(RUTA_FEATURES))

  #Criterio para ordenar
  clientes <- clientes[order(clientes$cliente), ]
  clientes[which(clientes$tipo_cliente == "Consumidor final" & clientes$monto > 0), ]
}

preparar_variables <- function(clientes) {
  # Logaritmo en las variables muy concentradas
  # estandarización

  x <- data.frame(
    log_ticket = log1p(clientes$ticket_visita),
    log_visitas = log(clientes$frecuencia_visitas),
    log_lineas_visita = log(clientes$lineas_por_visita),
    log_categorias = log(clientes$n_categorias),
    clientes[, grep("^share_", names(NOMBRES_VARIABLES), value = TRUE)]
  )
  scale(as.matrix(x[, names(NOMBRES_VARIABLES)]))
}

aplicar_pca <- function(x_std) {
  # Mantiene el número de componentes necesario para explicar >= VARIANZA_OBJETIVO_PCA.
  # x_std ya viene estandarizado (media 0, sd 1), por eso center = FALSE y scale. = FALSE.
  modelo <- prcomp(x_std, center = FALSE, scale. = FALSE)
  var_individual <- modelo$sdev^2 / sum(modelo$sdev^2)
  var_acumulada <- cumsum(var_individual)
  n_componentes <- which(var_acumulada >= VARIANZA_OBJETIVO_PCA)[1]
  list(
    modelo = modelo,
    n_componentes = n_componentes,
    var_individual = var_individual,
    var_acumulada = var_acumulada,
    scores = modelo$x[, seq_len(n_componentes), drop = FALSE],
    cargas = modelo$rotation[, seq_len(n_componentes), drop = FALSE]
  )
}

ajustar_kmeans <- function(x, k) {
  kmeans(x, centers = k, nstart = INICIOS_KMEANS, iter.max = ITERACIONES_KMEANS,
         algorithm = "MacQueen")
}

ajustar_kmedoids <- function(x, k) {
  # clara = K-medoids para bases grandes: busca los medoides en submuestras y
  # luego asigna a todos los clientes al medoide más cercano.
  clara(x, k, metric = "euclidean", samples = MUESTRAS_CLARA, sampsize = TAMANO_CLARA,
        pamLike = TRUE, rngR = TRUE, keep.data = FALSE)
}

silhouette_promedio <- function(etiquetas, distancias) {
  # Entre -1 y 1. Cerca de 1: cada cliente está mucho más cerca de su grupo
  # que del grupo vecino. Cerca de 0: los grupos se solapan.
  mean(silhouette(etiquetas, distancias)[, "sil_width"])
}

indice_rand_ajustado <- function(a, b) {
  # Mide cuánto coinciden dos agrupaciones de los mismos clientes.
  # 1 = idénticas; 0 = lo que coincidirían por puro azar.
  tabla <- table(a, b)
  juntos <- sum(choose(tabla, 2))
  en_a <- sum(choose(rowSums(tabla), 2))
  en_b <- sum(choose(colSums(tabla), 2))
  esperado <- en_a * en_b / choose(sum(tabla), 2)
  (juntos - esperado) / ((en_a + en_b) / 2 - esperado)
}

asignar_al_centro_mas_cercano <- function(x, centros) {
  distancias <- sapply(seq_len(nrow(centros)), function(j) {
    colSums((t(x) - centros[j, ])^2)
  })
  max.col(-distancias, ties.method = "first")
}

ordenar_por_valor <- function(etiquetas, monto) {
  # Renumera los segmentos: el 1 es el de mayor gasto mediano, y así en orden.
  orden <- order(-tapply(monto, etiquetas, median))
  match(etiquetas, as.integer(names(tapply(monto, etiquetas, median))[orden]))
}

evaluar_candidatos <- function(x, muestra, distancias) {
  # Para cada k: inercia de K-means (para el "codo") y silhouette de ambos métodos.
  filas <- lapply(K_CANDIDATOS, function(k) {
    km <- ajustar_kmeans(x, k)
    kmed <- ajustar_kmedoids(x, k)
    cat(sprintf("  k = %d listo\n", k))
    data.frame(
      k = k,
      inercia_kmeans = km$tot.withinss,
      silhouette_kmeans = silhouette_promedio(km$cluster[muestra], distancias),
      silhouette_kmedoids = silhouette_promedio(kmed$clustering[muestra], distancias)
    )
  })
  do.call(rbind, filas)
}

medir_estabilidad <- function(x, k, etiquetas, muestra) {
  # Repite K-means con la mitad de los clientes, elegida al azar cada vez, y
  # compara contra el modelo final. Si los grupos son reales, deben repetirse.
  sapply(seq_len(REPETICIONES_ESTABILIDAD), function(i) {
    mitad <- sample(nrow(x), nrow(x) %/% 2)
    modelo <- ajustar_kmeans(x[mitad, ], k)
    indice_rand_ajustado(etiquetas[muestra],
                         asignar_al_centro_mas_cercano(x[muestra, ], modelo$centers))
  })
}

perfilar_segmentos <- function(clientes, etiquetas) {
  # Una fila por segmento, en unidades originales, incluyendo las variables
  # que NO entraron al modelo (edad, generación, contactabilidad, recencia).
  partes <- split(clientes, etiquetas)
  filas <- lapply(names(partes), function(s) {
    g <- partes[[s]]
    data.frame(
      segmento = as.integer(s),
      clientes = nrow(g),
      pct_clientes = round(100 * nrow(g) / nrow(clientes), 1),
      pct_ventas = round(100 * sum(g$monto) / sum(clientes$monto), 1),
      monto_mediano = round(median(g$monto)),
      ticket_mediano = round(median(g$ticket_visita)),
      visitas_promedio = round(mean(g$frecuencia_visitas), 2),
      pct_una_visita = round(100 * mean(g$frecuencia_visitas == 1), 1),
      productos_por_visita = round(mean(g$lineas_por_visita), 2),
      categorias_promedio = round(mean(g$n_categorias), 2),
      pct_revestimientos = round(100 * mean(g$share_revestimientos_e_instalacion), 1),
      pct_banos = round(100 * mean(g$share_banos), 1),
      pct_cocinas = round(100 * mean(g$share_cocinas), 1),
      pct_pinturas = round(100 * mean(g$share_pinturas_e_impermeabilizantes), 1),
      pct_servicios = round(100 * mean(g$share_servicios), 1),
      edad_mediana = median(g$edad, na.rm = TRUE),
      pct_gen_z = round(100 * mean(g$generacion == "Generación Z"), 1),
      pct_millennials = round(100 * mean(g$generacion == "Millennials"), 1),
      pct_gen_x = round(100 * mean(g$generacion == "Generación X"), 1),
      pct_boomers = round(100 * mean(g$generacion == "Baby Boomers"), 1),
      pct_contactable = round(100 * mean(g$es_contactable), 1),
      recencia_mediana_dias = median(g$recencia_dias)
    )
  })
  do.call(rbind, filas)
}

# --- Gráficas (mismo estilo que graficas.py) ---------------------------------
AZUL <- "#2a78d6"; NARANJA <- "#eb6834"; ROJO <- "#e34948"
SUPERFICIE <- "#fcfcfb"; TINTA_2 <- "#52514e"; REJILLA <- "#e1e0d9"

tema_corona <- function() {
  if (.Platform$OS.type == "windows") windowsFonts(Calibri = windowsFont("Calibri"))
  familia <- if (.Platform$OS.type == "windows") "Calibri" else "sans"
  theme_minimal(base_size = 11, base_family = familia) +
    theme(
      plot.background = element_rect(fill = SUPERFICIE, colour = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(colour = REJILLA, linewidth = 0.3),
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(colour = TINTA_2, margin = margin(b = 12)),
      plot.caption = element_text(colour = "#898781", hjust = 0, size = 8.5),
      plot.title.position = "plot", plot.caption.position = "plot",
      strip.text = element_text(face = "bold", colour = TINTA_2, hjust = 0, size = 11),
      axis.text = element_text(colour = TINTA_2), axis.title = element_text(colour = TINTA_2),
      legend.position = "top", legend.title = element_blank()
    )
}

graficar_eleccion_de_k <- function(metricas, k_final) {
  largo <- rbind(
    data.frame(k = metricas$k, metodo = "K-means", panel = "Silhouette (más alto es mejor)",
               valor = metricas$silhouette_kmeans),
    data.frame(k = metricas$k, metodo = "K-medoids", panel = "Silhouette (más alto es mejor)",
               valor = metricas$silhouette_kmedoids),
    data.frame(k = metricas$k, metodo = "K-means", panel = "Inercia de K-means (se busca el codo)",
               valor = metricas$inercia_kmeans / 1e6)
  )
  grafica <- ggplot(largo, aes(k, valor, colour = metodo)) +
    geom_vline(xintercept = k_final, colour = REJILLA, linewidth = 6) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.6) +
    facet_wrap(~panel, scales = "free_y") +
    scale_colour_manual(values = c("K-means" = AZUL, "K-medoids" = NARANJA)) +
    scale_x_continuous(breaks = metricas$k) +
    labs(
      title = "Elección del número de segmentos",
      subtitle = sprintf("La franja marca el número elegido: %d segmentos.", k_final),
      x = "Número de segmentos (k)", y = NULL,
      caption = "Silhouette calculado sobre una muestra de consumidores finales. Inercia en millones."
    ) +
    tema_corona()
  ggsave(file.path(GRAFICAS, "13_eleccion_de_k.png"), grafica, width = 10, height = 5, dpi = 200)
}

graficar_perfil <- function(centros, perfil) {
  # Cada celda: cuánto se aleja el segmento del consumidor promedio en esa
  # variable, en desviaciones estándar (0 = igual al promedio). Siempre se
  # calcula sobre las 8 variables originales estandarizadas, no sobre las
  # componentes principales, para que el gráfico se pueda interpretar.
  largo <- data.frame(
    segmento = rep(seq_len(nrow(centros)), times = ncol(centros)),
    variable = rep(NOMBRES_VARIABLES[colnames(centros)], each = nrow(centros)),
    valor = as.vector(centros)
  )
  rotulos <- sprintf("Segmento %d\n%s%% de clientes", perfil$segmento,
                     format(perfil$pct_clientes, decimal.mark = ","))
  largo$segmento <- factor(rotulos[largo$segmento], levels = rotulos)
  largo$variable <- factor(largo$variable, levels = rev(NOMBRES_VARIABLES))
  grafica <- ggplot(largo, aes(segmento, variable, fill = valor)) +
    geom_tile(colour = SUPERFICIE, linewidth = 1.2) +
    geom_text(aes(label = format(round(valor, 1), nsmall = 1, decimal.mark = ","),
                  colour = abs(valor) > 1.4), size = 3.6, show.legend = FALSE) +
    scale_colour_manual(values = c("FALSE" = "#0b0b0b", "TRUE" = "#ffffff")) +
    scale_fill_gradient2(low = ROJO, mid = "#f0efec", high = AZUL, midpoint = 0,
                         limits = c(-2, 2), oob = scales::squish,
                         name = "Frente al consumidor promedio") +
    labs(
      title = "Qué distingue a cada segmento",
      subtitle = "Azul: por encima del consumidor promedio; rojo: por debajo. En desviaciones estándar.",
      x = NULL, y = NULL,
      caption = "Segmentos de K-means sobre consumidores finales, ordenados de mayor a menor gasto mediano."
    ) +
    tema_corona() +
    theme(panel.grid = element_blank(), legend.position = "none")
  ggsave(file.path(GRAFICAS, "14_perfil_de_segmentos.png"), grafica,
         width = 1.6 * nrow(centros) + 3.2, height = 5.4, dpi = 200)
}

# =============================================================================
# Ejecución
# =============================================================================

main <- function() {
  cat(sprintf("== Versión: %s | PCA: %s ==\n", VERSION, PCA_FLAG))
  set.seed(SEMILLA)
  clientes <- cargar_consumidores()
  x_std <- preparar_variables(clientes)  # 8 variables estandarizadas (interpretables)
  cat(sprintf("Consumidores finales: %s | Variables: %d\n",
              format(nrow(x_std), big.mark = ".", decimal.mark = ","), ncol(x_std)))

  # Espacio sobre el que se agrupa: puede ser x_std o los scores del PCA.
  n_componentes_pca <- NA_integer_
  if (USAR_PCA) {
    pca <- aplicar_pca(x_std)
    n_componentes_pca <- pca$n_componentes
    cat(sprintf("PCA: %d componentes explican %.1f%% de la varianza.\n",
                pca$n_componentes, 100 * pca$var_acumulada[pca$n_componentes]))
    x_cluster <- pca$scores
  } else {
    x_cluster <- x_std
  }

  muestra <- sample(nrow(x_cluster), TAMANO_MUESTRA)
  distancias <- dist(x_cluster[muestra, ])

  cat("Probando números de segmentos...\n")
  metricas <- evaluar_candidatos(x_cluster, muestra, distancias)
  validos <- metricas[metricas$k >= K_MINIMO, ]
  k <- if (is.null(K_FINAL)) validos$k[which.max(validos$silhouette_kmeans)] else K_FINAL

  cat(sprintf("Ajustando el modelo final con k = %d...\n", k))
  km <- ajustar_kmeans(x_cluster, k)
  kmed <- ajustar_kmedoids(x_cluster, k)
  clientes$segmento_kmeans <- ordenar_por_valor(km$cluster, clientes$monto)
  clientes$segmento_kmedoids <- ordenar_por_valor(kmed$clustering, clientes$monto)
  estabilidad <- medir_estabilidad(x_cluster, k, clientes$segmento_kmeans, muestra)
  acuerdo <- indice_rand_ajustado(clientes$segmento_kmeans, clientes$segmento_kmedoids)

  # Silhouette del modelo final medido también sobre las 8 variables originales
  # estandarizadas, con la misma muestra. Esto permite comparar las variantes con
  # y sin PCA en la misma escala, porque el espacio de PCs no es el de x_std.
  distancias_std <- dist(x_std[muestra, ])
  silhouette_kmeans_std <- silhouette_promedio(
    clientes$segmento_kmeans[muestra], distancias_std)
  silhouette_kmedoids_std <- silhouette_promedio(
    clientes$segmento_kmedoids[muestra], distancias_std)

  perfil <- perfilar_segmentos(clientes, clientes$segmento_kmeans)
  perfil_kmedoids <- perfilar_segmentos(clientes, clientes$segmento_kmedoids)
  # Centros en las 8 variables originales estandarizadas (sirven para interpretar
  # cada segmento). Si hubo PCA, estos centros NO son los del k-means (que están
  # en el espacio de componentes), sino el promedio por segmento en x_std.
  centros <- apply(x_std, 2, function(columna) tapply(columna, clientes$segmento_kmeans, mean))
  # Los medoides son clientes reales: el "cliente típico" de cada segmento
  medoides <- clientes[kmed$i.med, c("cliente", "segmento_kmedoids", "monto", "ticket_visita",
                                     "frecuencia_visitas", "n_categorias", "negocio_principal",
                                     "edad", "generacion")]
  medoides <- medoides[order(medoides$segmento_kmedoids), ]

  write_parquet(clientes[, c("cliente", "segmento_kmeans", "segmento_kmedoids")],
                RUTA_SEGMENTOS)
  write.csv(metricas, file.path(OUT, "segmentacion_metricas.csv"), row.names = FALSE)
  write.csv(perfil, file.path(OUT, "segmentacion_perfil_kmeans.csv"), row.names = FALSE)
  write.csv(perfil_kmedoids, file.path(OUT, "segmentacion_perfil_kmedoids.csv"), row.names = FALSE)
  write.csv(medoides, file.path(OUT, "segmentacion_medoides.csv"), row.names = FALSE)

  # Resumen de una fila para la comparación de versiones (lo lee comparacion.py).
  # silhouette_kmeans/kmedoids se miden en el espacio donde se agrupó (x_std en
  # sin_pca, los PCs en las variantes de PCA). silhouette_kmeans_std/kmedoids_std
  # se miden sobre las 8 variables originales estandarizadas, para comparar
  # directamente las tres variantes.
  resumen <- data.frame(
    version = VERSION,
    pca = PCA_FLAG,
    k = k,
    silhouette_kmeans = metricas$silhouette_kmeans[metricas$k == k],
    silhouette_kmedoids = metricas$silhouette_kmedoids[metricas$k == k],
    silhouette_kmeans_std = silhouette_kmeans_std,
    silhouette_kmedoids_std = silhouette_kmedoids_std,
    estabilidad_media = mean(estabilidad),
    estabilidad_min = min(estabilidad),
    acuerdo_kmeans_kmedoids = acuerdo,
    n_componentes_pca = n_componentes_pca,
    n_clientes = nrow(clientes)
  )
  write.csv(resumen, file.path(OUT, "resumen_version.csv"), row.names = FALSE)

  if (USAR_PCA) {
    varianza <- data.frame(
      componente = seq_along(pca$var_individual),
      var_individual = round(pca$var_individual, 6),
      var_acumulada = round(pca$var_acumulada, 6),
      conservada = seq_along(pca$var_individual) <= pca$n_componentes
    )
    write.csv(varianza, file.path(OUT, "pca_varianza.csv"), row.names = FALSE)
    cargas <- as.data.frame(pca$cargas)
    cargas <- cbind(variable = rownames(cargas), cargas)
    write.csv(cargas, file.path(OUT, "pca_cargas.csv"), row.names = FALSE)
  }

  graficar_eleccion_de_k(metricas, k)
  graficar_perfil(centros, perfil)

  # --- Reporte ---------------------------------------------------------------
  cat("\n== Métricas por número de segmentos ==\n")
  print(format(metricas, digits = 4), row.names = FALSE)
  cat(sprintf("\nNúmero de segmentos elegido: %d\n", k))
  cat(sprintf("Silhouette final (espacio de agrupación): K-means %.3f | K-medoids %.3f\n",
              metricas$silhouette_kmeans[metricas$k == k],
              metricas$silhouette_kmedoids[metricas$k == k]))
  cat(sprintf("Silhouette final sobre las 8 variables estandarizadas (comparable entre variantes): K-means %.3f | K-medoids %.3f\n",
              silhouette_kmeans_std, silhouette_kmedoids_std))
  cat(sprintf("Estabilidad de K-means (Rand ajustado, %d repeticiones): media %.3f, mínimo %.3f\n",
              REPETICIONES_ESTABILIDAD, mean(estabilidad), min(estabilidad)))
  cat(sprintf("Acuerdo entre K-means y K-medoids (Rand ajustado): %.3f\n", acuerdo))
  cat("\n== Perfil de los segmentos (K-means) ==\n")
  print(t(perfil), quote = FALSE)
  cat("\n== Cliente típico de cada segmento (medoides) ==\n")
  print(medoides, row.names = FALSE)
  cat(sprintf("\nGuardado: %s, %s, graficas 13_ y 14_\nListo.\n",
              RUTA_SEGMENTOS, OUT))
}

main()
