#   1. Prepara las variables de comportamiento de compra
#   2. Prueba varios números de grupos (k) y mide la calidad de cada uno
#   3. Ajusta el modelo final y mide su estabilidad
#   4. Describe cada segmento -> data/segmentos_clientes.parquet y salidas/
#
# Requiere haber corrido antes (Python):
#     limpieza_transaccional.py  limpieza_clientes.py  variables_clientes.py
#
# Uso (desde cualquier carpeta):
#     install.packages(c("nanoparquet", "cluster", "ggplot2"))
#     Rscript scripts/segmentacion/segmentacion.R


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
OUT <- file.path(RAIZ, "salidas")
GRAFICAS <- file.path(OUT, "graficas")
dir.create(GRAFICAS, recursive = TRUE, showWarnings = FALSE)

# Parámetros
SEMILLA <- 2026            # fija el azar: el resultado es igual en cada corrida
K_CANDIDATOS <- 2:8        # números de grupos que se prueban
K_MINIMO <- 3              # con 2 grupos no se pueden diferenciar acciones comerciales
K_FINAL <- NULL            # NULL = lo elige el silhouette; o fijar un número a mano
TAMANO_MUESTRA <- 5000     # clientes con los que se calcula el silhouette
INICIOS_KMEANS <- 10       # K-means se repite desde 10 puntos de partida distintos
MUESTRAS_CLARA <- 20       # K-medoids (clara) busca medoides en 20 submuestras...
TAMANO_CLARA <- 1000       # ...de 1.000 clientes cada una
REPETICIONES_ESTABILIDAD <- 10

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
  ruta <- file.path(DATA, "features_clientes_v2.parquet")
  if (!file.exists(ruta)) {
    stop("No se encontró ", ruta, ". Corre primero los tres scripts de Python.")
  }
  clientes <- as.data.frame(read_parquet(ruta))
  
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

ajustar_kmeans <- function(x, k) {
  kmeans(x, centers = k, nstart = INICIOS_KMEANS, iter.max = 100, algorithm = "MacQueen")
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
  # variable, en desviaciones estándar (0 = igual al promedio).
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
  set.seed(SEMILLA)
  clientes <- cargar_consumidores()
  x <- preparar_variables(clientes)
  cat(sprintf("Consumidores finales: %s | Variables: %d\n",
              format(nrow(x), big.mark = ".", decimal.mark = ","), ncol(x)))

  muestra <- sample(nrow(x), TAMANO_MUESTRA)
  distancias <- dist(x[muestra, ])

  cat("Probando números de segmentos...\n")
  metricas <- evaluar_candidatos(x, muestra, distancias)
  validos <- metricas[metricas$k >= K_MINIMO, ]
  k <- if (is.null(K_FINAL)) validos$k[which.max(validos$silhouette_kmeans)] else K_FINAL

  cat(sprintf("Ajustando el modelo final con k = %d...\n", k))
  km <- ajustar_kmeans(x, k)
  kmed <- ajustar_kmedoids(x, k)
  clientes$segmento_kmeans <- ordenar_por_valor(km$cluster, clientes$monto)
  clientes$segmento_kmedoids <- ordenar_por_valor(kmed$clustering, clientes$monto)
  estabilidad <- medir_estabilidad(x, k, clientes$segmento_kmeans, muestra)
  acuerdo <- indice_rand_ajustado(clientes$segmento_kmeans, clientes$segmento_kmedoids)

  perfil <- perfilar_segmentos(clientes, clientes$segmento_kmeans)
  perfil_kmedoids <- perfilar_segmentos(clientes, clientes$segmento_kmedoids)
  # centros en el mismo orden que los segmentos renumerados
  centros <- apply(x, 2, function(columna) tapply(columna, clientes$segmento_kmeans, mean))
  # Los medoides son clientes reales: el "cliente típico" de cada segmento
  medoides <- clientes[kmed$i.med, c("cliente", "segmento_kmedoids", "monto", "ticket_visita",
                                     "frecuencia_visitas", "n_categorias", "negocio_principal",
                                     "edad", "generacion")]
  medoides <- medoides[order(medoides$segmento_kmedoids), ]

  write_parquet(clientes[, c("cliente", "segmento_kmeans", "segmento_kmedoids")],
                file.path(DATA, "segmentos_clientes.parquet"))
  write.csv(metricas, file.path(OUT, "segmentacion_metricas.csv"), row.names = FALSE)
  write.csv(perfil, file.path(OUT, "segmentacion_perfil_kmeans.csv"), row.names = FALSE)
  write.csv(perfil_kmedoids, file.path(OUT, "segmentacion_perfil_kmedoids.csv"), row.names = FALSE)
  write.csv(medoides, file.path(OUT, "segmentacion_medoides.csv"), row.names = FALSE)
  graficar_eleccion_de_k(metricas, k)
  graficar_perfil(centros, perfil)

  # --- Reporte ---------------------------------------------------------------
  cat("\n== Métricas por número de segmentos ==\n")
  print(format(metricas, digits = 4), row.names = FALSE)
  cat(sprintf("\nNúmero de segmentos elegido: %d\n", k))
  cat(sprintf("Silhouette final: K-means %.3f | K-medoids %.3f\n",
              metricas$silhouette_kmeans[metricas$k == k],
              metricas$silhouette_kmedoids[metricas$k == k]))
  cat(sprintf("Estabilidad de K-means (Rand ajustado, %d repeticiones): media %.3f, mínimo %.3f\n",
              REPETICIONES_ESTABILIDAD, mean(estabilidad), min(estabilidad)))
  cat(sprintf("Acuerdo entre K-means y K-medoids (Rand ajustado): %.3f\n", acuerdo))
  cat("\n== Perfil de los segmentos (K-means) ==\n")
  print(t(perfil), quote = FALSE)
  cat("\n== Cliente típico de cada segmento (medoides) ==\n")
  print(medoides, row.names = FALSE)
  cat("\nGuardado: data/segmentos_clientes.parquet, salidas/segmentacion_*.csv,",
      "salidas/graficas/13_ y 14_\nListo.\n")
}

main()
