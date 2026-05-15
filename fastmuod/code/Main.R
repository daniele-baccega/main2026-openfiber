library(dplyr)
library(tidyr)
library(purrr)
library(lubridate)
library(ggplot2)
library(readr)
library(zoo)
library(connector)
library(cluster)
library(tibble)
library(stringr)
library(dtw)
library(dbscan)
library(pheatmap)

# Create figures and csv directories if they don't exist
dir.create("figures", showWarnings = FALSE)
dir.create("csv", showWarnings = FALSE)

source("muod_cpp.R")

df_raw <- read_csv("../../python/data/Max_Max_Rate_Gbps/data_Max_Max_Rate_Gbps.csv")

print(df_raw)

df_raw <- df_raw %>%
  mutate(
    Date      = as.Date(Time),
    day_index = as.integer(difftime(Time, Date, units = "mins")) %/% 15L + 1L
  )

tunnel_patterns <- df_raw$Tunnel %>% unique()

# Filter tunnel_patterns to only include those that exist in the data
available_tunnels <- unique(df_raw$Tunnel)
tunnel_patterns <- tunnel_patterns[sapply(tunnel_patterns, function(pattern) {
  any(grepl(paste0("^", pattern), available_tunnels))
})]

# Check if any valid patterns remain
if (length(tunnel_patterns) == 0) {
  stop("ERROR: None of the specified tunnel patterns exist in the data!")
}

cat("Patterns to analyze:", paste(tunnel_patterns, collapse = ", "), "\n\n")

# Function to get tunnel-specific output paths
get_output_path <- function(tunnel, filename, type = "figures") {
  paste0(type, "/", tunnel, "/", filename)
}

# Create tunnel-specific subdirectories
for (tn_pattern in tunnel_patterns) {
  dir.create(paste0("figures/", tn_pattern), showWarnings = FALSE, recursive = TRUE)
  dir.create(paste0("csv/", tn_pattern), showWarnings = FALSE, recursive = TRUE)
}

# =====================================================
# MAIN LOOP: Analyze each tunnel separately
# =====================================================

for (tunnel_pattern in tunnel_patterns) {
  cat("\n", strrep("=", 70), "\n")
  cat("ANALYZING TUNNEL PATTERN:", tunnel_pattern, "\n")
  cat(strrep("=", 70), "\n\n")
  
  # Filter data for this tunnel pattern
  df <- df_raw %>%
    filter(sapply(Tunnel, function(t) grepl(paste0("^", tunnel_pattern), t)))
  
  selected_tunnels <- unique(df$Tunnel)
  cat("Processing tunnels:", paste(selected_tunnels, collapse = ", "), "\n\n")
  
  # For single tunnel per iteration (if multiple matches, take first)
  if (length(selected_tunnels) > 1) {
    cat("WARNING: Multiple tunnels match pattern. Processing first match only.\n\n")
  }
  # Use pattern name for output directories
  current_tunnel <- tunnel_pattern

  n_per_day <- df %>%
    count(Tunnel, Date) %>%
    pull(n) %>%
    unique()

  # Use data as-is without gap filling or NA removal
  total_df_clean <- df %>%
    filter(!is.na(Original))

  cat("Processing tunnel pattern:", tunnel_pattern, "\n")
  cat("Actual tunnel names:", paste(selected_tunnels, collapse = ", "), "\n\n")

  # Create consistent day_id mapping for ALL dates in the tunnel data
  date_mapping <- total_df_clean %>%
    distinct(Date) %>%
    arrange(Date) %>%
    mutate(day_id = row_number())

  # Matrix creation (use original data without filling)
  total_mat_residuals <- total_df_clean %>%
    left_join(date_mapping, by = "Date") %>%
    arrange(Tunnel, day_id, day_index) %>%
    pivot_wider(id_cols = c(Tunnel, day_id), names_from = day_index, values_from = Residual) %>%
    select(-day_id, -Tunnel) %>%
    as.matrix()

  # Also create matrix with original data
  total_mat_original <- total_df_clean %>%
    left_join(date_mapping, by = "Date") %>%
    arrange(day_id, day_index) %>%
    pivot_wider(id_cols = c(Tunnel, day_id), names_from = day_index, values_from = Original) %>%
    select(-day_id, -Tunnel) %>%
    as.matrix()

  # Outlier detection on filled data
  out <- get_outliers(t(total_mat_residuals), method = "fast", cut_method = "tangent")
  shape_days     <- out$shape
  amplitude_days <- out$amplitude
  magnitude_days <- out$magnitude

  # Plotting
  mat_residuals_df <- as.data.frame(total_mat_residuals)
  mat_residuals_df$day <- seq_len(nrow(total_mat_residuals))

  df_long_residuals <- mat_residuals_df %>%
    pivot_longer(cols = -day, names_to = "time_idx", values_to = "value") %>%
    mutate(
      time_idx = as.integer(time_idx),
      outlier_type = case_when(
        day %in% shape_days     ~ "shape",
        day %in% amplitude_days ~ "amplitude",
        day %in% magnitude_days ~ "magnitude",
        TRUE                    ~ "none"
      )
    )

  # Create df_long_original with original values
  mat_original_df <- as.data.frame(total_mat_original)
  mat_original_df$day <- seq_len(nrow(total_mat_original))

  df_long_original <- mat_original_df %>%
    pivot_longer(cols = -day, names_to = "time_idx", values_to = "value_original") %>%
    mutate(time_idx = as.integer(time_idx)) %>%
    select(day, time_idx, value_original)

  p1 <- ggplot(df_long_residuals, aes(x = time_idx, y = value, group = day)) +
    geom_line(color = "black", alpha = 0.3) +
    geom_line(
      data = df_long_residuals %>% filter(outlier_type != "none"),
      aes(color = outlier_type), linewidth = 0.6, alpha = 0.3
    ) +
    scale_color_manual(values = c(shape = "red", amplitude = "blue", magnitude = "green")) +
    scale_x_continuous(breaks = c(1,24,48,72,96), labels = c("00:00","06:00","12:00","18:00","23:45"), expand = expansion(add = c(1, 1))) +
    labs(
      x = "",
      y = "Traffic (% of capacity)",
      color = "Outlier type",
      title = ""
    ) +
    theme_minimal()
  print(p1)
  ggsave(get_output_path(current_tunnel, "network_traffic_all.png"), p1, width = 10, height = 6, dpi = 300)


  p2 <- ggplot(df_long_residuals, aes(x = time_idx, y = value, group = day)) +
    geom_line(
      data = df_long_residuals %>% filter(outlier_type != "none"),
      aes(color = outlier_type), linewidth = 0.6, alpha = 0.3
    ) +
    scale_color_manual(values = c(shape = "red", amplitude = "blue", magnitude = "green")) +
    scale_x_continuous(breaks = c(1,24,48,72,96), labels = c("00:00","06:00","12:00","18:00","23:45"), expand = expansion(add = c(1, 1))) +
    labs(
      x = "",
      y = "Traffic (% of capacity)",
      color = "Outlier type",
      title = ""
    ) +
    theme_minimal()
  print(p2)
  ggsave(get_output_path(current_tunnel, "network_traffic_outliers.png"), p2, width = 10, height = 6, dpi = 300)

  outlier_days <- sort(unique(c(shape_days, amplitude_days, magnitude_days)))
  cat("Number of outlier days for tunnel", current_tunnel, ":", length(outlier_days), "\n\n")
  id_mapping <- setNames(seq_along(outlier_days), outlier_days)

  data_local <- df_long_residuals %>%
    filter(day %in% outlier_days) %>%
    mutate(
      ID = id_mapping[as.character(day)],
      Observation = value,
      Time = time_idx - min(time_idx) + 1L
    ) %>%
    mutate(
      ID = as.numeric(ID),
      Time = as.integer(Time)
    ) %>%
    arrange(ID, Time) %>%
    select(ID, Observation, Time)


  cluster_cols <- c(
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#b2df8a", "#fb9a99", "#A25CCF"
  )

  normal_days <- df_long_residuals %>% 
    filter(outlier_type == "none") %>%
    mutate(Time = time_idx + 1) %>%
    filter(Time >= 1 & Time <= 96) %>%
    group_by(Time) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
    mutate(group = "Normal")

  # =====================================================
  # Distribution of non-outlier (normal) values
  # =====================================================

  non_outlier_data <- df_long_residuals %>%
    filter(outlier_type == "none")

  p11 <- ggplot(non_outlier_data, aes(x = value)) +
    geom_histogram(binwidth = 0.5, fill = "#1b9e77", color = "black", alpha = 0.7) +
    geom_density(aes(y = after_stat(density) * nrow(non_outlier_data) * 0.5), 
                color = "red", linewidth = 1.2) +
    labs(
      title = "",
      x = "Traffic (% of capacity)",
      y = "Frequency"
    ) +
    theme_bw(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  print(p11)
  ggsave(get_output_path(current_tunnel, "distribution_non_outliers.png"), p11, width = 10, height = 6, dpi = 300)

  # Summary statistics for non-outlier values
  summary_stats <- non_outlier_data %>%
    summarise(
      Mean = mean(value, na.rm = TRUE),
      Median = median(value, na.rm = TRUE),
      SD = sd(value, na.rm = TRUE),
      Min = min(value, na.rm = TRUE),
      Max = max(value, na.rm = TRUE),
      Q1 = quantile(value, 0.25, na.rm = TRUE),
      Q3 = quantile(value, 0.75, na.rm = TRUE),
      N = n()
    )

  cat("Summary statistics for non-outlier trajectories:\n")
  print(summary_stats)

  normal_intraday_profile <- df_long_residuals %>%
    filter(outlier_type == "none") %>%
    group_by(time_idx) %>%
    summarise(
      mean_value = mean(value, na.rm = TRUE),
      median_value = median(value, na.rm = TRUE),
      mad_value = median(abs(value - median(value, na.rm = TRUE)), na.rm = TRUE),
      q25_value = quantile(value, 0.25, na.rm = TRUE),
      q75_value = quantile(value, 0.75, na.rm = TRUE),
      min_value = min(value, na.rm = TRUE),
      max_value = max(value, na.rm = TRUE),
      .groups = "drop"
    )

  # =====================================================
  # Extract Anomalous Time Segments from Outlier Days
  # Using Non-Outlier Intraday Profile as Reference
  # =====================================================

  # Map day indices to dates
  date_by_day <- date_mapping %>%
    rename(day = day_id)

  # Step 2: For each outlier day, identify anomalous time segments
  # A time point is anomalous if it deviates significantly from normal

  # Z-score threshold for anomalies
  z_threshold <- 3  # Adjust as needed

  outlier_segments_list <- list()

  for (outlier_day in sort(unique(df_long_residuals$day[df_long_residuals$outlier_type != "none"]))) {
    outlier_day_data <- df_long_residuals %>%
      filter(day == outlier_day) %>%
      left_join(normal_intraday_profile, by = "time_idx") %>%
      mutate(
        z_score = 0.6745 * (value - median_value) / mad_value,
        is_anomalous = abs(z_score) > z_threshold
      )
    
    # Find continuous anomalous segments
    outlier_day_data <- outlier_day_data %>%
      mutate(
        segment_id = cumsum(is_anomalous != lag(is_anomalous, default = FALSE))
      )
    
    # Extract anomalous segments only
    anomalous_segs <- outlier_day_data %>%
      filter(is_anomalous) %>%
      group_by(segment_id) %>%
      summarise(
        day = first(day),
        outlier_type = first(outlier_type),
        time_start = min(time_idx),
        time_end = max(time_idx),
        n_timepoints = n(),
        mean_z_score = mean(z_score, na.rm = TRUE),
        max_z_score = max(abs(z_score), na.rm = TRUE),
        mean_value_segment = mean(value, na.rm = TRUE),
        mean_expected = mean(mean_value, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        duration_hours = n_timepoints * 0.25,  # Assuming 15-min intervals
        time_start_str = paste0(sprintf("%02d", floor((time_start)/4)), ":", 
                                sprintf("%02d", ((time_start) %% 4) * 15)),
        time_end_str = paste0(sprintf("%02d", floor((time_end)/4)), ":", 
                              sprintf("%02d", ((time_end) %% 4) * 15)),
        # Add indicator for whether anomaly is positive or negative
        anomaly_sign = ifelse(mean_z_score > 0, "positive", "negative")
      )
    
    if (nrow(anomalous_segs) > 0) {
      outlier_segments_list[[as.character(outlier_day)]] <- anomalous_segs
    }
  }

  # Combine all anomalous segments
  all_anomalous_segments <- bind_rows(outlier_segments_list, .id = "day_chr") %>%
    mutate(day = as.integer(day_chr)) %>%
    select(-day_chr) %>%
    left_join(
      date_by_day %>% rename(absolute_day = day) %>% rename(day = absolute_day),
      by = "day"
    ) %>%
    filter(time_start != time_end)

  cat("\n=== Anomalous Time Segments Extracted ===\n")
  cat("Total anomalous segments found:", nrow(all_anomalous_segments), "\n\n")
  print(all_anomalous_segments %>% 
        select(Date, time_start_str, time_end_str, duration_hours, 
              mean_z_score, max_z_score, outlier_type))

  # Visualization: Anomalous segments overlay on normal profile
  p15 <- df_long_residuals %>%
    filter(outlier_type != "none") %>%
    left_join(normal_intraday_profile, by = "time_idx") %>%
    mutate(z_score = 0.6745 * (value - median_value) / mad_value) %>%
    ggplot(aes(x = time_idx, y = value, group = day)) +
    geom_line(color = "gray", alpha = 0.3, linewidth = 0.4) +
    # Normal profile
    geom_line(data = normal_intraday_profile, aes(x = time_idx, y = mean_value, group = NA),
              color = "#2d5016", linewidth = 2.5, alpha = 1) +
    # Threshold lines
    geom_line(data = normal_intraday_profile, aes(x = time_idx, y = mean_value + z_threshold * mad_value / 0.6745, group = NA),
              color = "orange", linewidth = 1.2, alpha = 1, linetype = "dashed") +
    geom_line(data = normal_intraday_profile, aes(x = time_idx, y = mean_value - z_threshold * mad_value / 0.6745, group = NA),
              color = "orange", linewidth = 1.2, alpha = 1, linetype = "dashed") +
    # Highlight anomalous points
    geom_point(
      data = df_long_residuals %>%
        filter(outlier_type != "none") %>%
        left_join(normal_intraday_profile, by = "time_idx") %>%
        mutate(z_score = 0.6745 * (value - median_value) / mad_value) %>%
        filter(abs(z_score) > z_threshold),
      aes(color = "Anomalous Points"),
      size = 2, alpha = 0.3
    ) +
    scale_color_manual(values = c("Anomalous Points" = "red")) +
    scale_x_continuous(breaks = c(1,24,48,72,96), labels = c("00:00","06:00","12:00","18:00","23:45"), expand = c(0, 0)) +
    labs(
      title = "",
      x = "",
      y = "Traffic (% of capacity)"
    ) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text.x = element_text(angle = 0, hjust = 0.5, size = 20),
          axis.text.y = element_text(size = 20),
          axis.title.y = element_text(size = 24, face="bold"),
          legend.position = "none",
          plot.margin = margin(t = 5, r = 30, b = 5, l = 5, unit = "pt"))
  print(p15)
  ggsave(get_output_path(current_tunnel, "anomalous_segments_overlay.png"), p15, width = 16, height = 7, dpi = 300)

  # Summary statistics for anomalous segments
  cat("\n=== Summary of Anomalous Segments ===\n")
  cat("Segments by Outlier Type:\n")
  segment_summary <- all_anomalous_segments %>%
    group_by(outlier_type) %>%
    summarise(
      n_segments = n(),
      n_days = n_distinct(day),
      mean_duration_hours = mean(duration_hours, na.rm = TRUE),
      mean_z_score = mean(mean_z_score, na.rm = TRUE),
      .groups = "drop"
    )
  print(segment_summary)

  cat("\n\nTop 15 Most Severe Anomalous Segments (by max Z-score):\n")
  top_segments <- all_anomalous_segments %>%
    arrange(desc(max_z_score)) %>%
    head(15) %>%
    select(Date, time_start_str, time_end_str, duration_hours, 
          mean_z_score, max_z_score, outlier_type)
  print(top_segments)

  # =====================================================
  # Compute DTW Distance Matrix Between Anomalous Segments
  # =====================================================

  cat("\n=== Computing DTW Distances Between Anomalous Segments ===\n")

  # Extract segment time series from outlier days
  segment_data_list <- list()

  for (i in seq_len(nrow(all_anomalous_segments))) {
    segment <- all_anomalous_segments[i, ]
    segment_day <- segment$day
    segment_start <- segment$time_start
    segment_end <- segment$time_end
    
    # Extract the time series for this segment from the original data
    segment_ts <- df_long_residuals %>%
      filter(day == segment_day & time_idx >= segment_start & time_idx <= segment_end) %>%
      arrange(time_idx) %>%
      pull(value)
    
    # Store the segment time series
    segment_data_list[[i]] <- segment_ts
  }

  # Compute pairwise DTW distances sequentially
  n_segments <- length(segment_data_list)

  cat("Computing DTW distances for", n_segments, "segments...\n")

  # Load required library
  library(dtw)

  # Compute all pairwise distances sequentially
  distance_pairs <- lapply(1:n_segments, function(i) {
    distances <- numeric(n_segments)
    for (j in i:n_segments) {
      distance <- dtw(segment_data_list[[i]], segment_data_list[[j]], 
                      keep.internals = FALSE)$distance
      distances[j] <- distance
    }
    list(i = i, distances = distances)
  })

  # Build symmetric distance matrix
  dtw_distance_matrix <- matrix(0, nrow = n_segments, ncol = n_segments)
  for (pair in distance_pairs) {
    i <- pair$i
    for (j in i:n_segments) {
      dtw_distance_matrix[i, j] <- pair$distances[j]
      dtw_distance_matrix[j, i] <- pair$distances[j]
    }
    if (i %% 10 == 0) {
      cat("  Processed", i, "segments\n")
    }
  }

  # Create a tibble with segment information and add row names to matrix
  segment_info <- all_anomalous_segments %>%
    mutate(segment_id = row_number()) %>%
    select(segment_id, Date, time_start_str, time_end_str, duration_hours, max_z_score, outlier_type, anomaly_sign)

  rownames(dtw_distance_matrix) <- segment_info$segment_id
  colnames(dtw_distance_matrix) <- segment_info$segment_id

  cat("\nDTW Distance Matrix dimensions:", dim(dtw_distance_matrix), "\n\n")

  # Print summary statistics
  cat("DTW Distance Summary Statistics:\n")
  cat("Min distance:", min(dtw_distance_matrix[upper.tri(dtw_distance_matrix)]), "\n")
  cat("Max distance:", max(dtw_distance_matrix[upper.tri(dtw_distance_matrix)]), "\n")
  cat("Mean distance:", mean(dtw_distance_matrix[upper.tri(dtw_distance_matrix)]), "\n")
  cat("Median distance:", median(dtw_distance_matrix[upper.tri(dtw_distance_matrix)]), "\n\n")

  # Show first few rows/cols of DTW distance matrix
  cat("First 5x5 block of DTW Distance Matrix:\n")
  print(dtw_distance_matrix[1:min(5, nrow(dtw_distance_matrix)), 
                            1:min(5, ncol(dtw_distance_matrix))])

  # =====================================================
  # Hierarchical Clustering with Dendrogram
  # =====================================================

  cat("\n=== Hierarchical Clustering on DTW Distance Matrix ===\n")

  # Perform hierarchical clustering with different linkage methods
  hc_results <- list()
  hc_results[["complete"]] <- hclust(as.dist(dtw_distance_matrix), method = "complete")

  # Use ward linkage for main analysis
  hc_main <- hc_results[["complete"]]

  cat("\n=== Elbow Method Analysis for Optimal k ===\n")

  max_k <- min(20, nrow(dtw_distance_matrix) - 1)  # Don't exceed number of segments
  wcss <- numeric(max_k)  # Within-cluster sum of squares for Elbow Method

  for (k in 2:max_k) {
    clusters <- cutree(hc_main, k = k)
    
    # Calculate WCSS for Elbow Method
    dist_matrix <- as.matrix(as.dist(dtw_distance_matrix))
    wcss_k <- 0
    for (i in unique(clusters)) {
      cluster_members <- which(clusters == i)
      if (length(cluster_members) > 1) {
        cluster_dist <- dist_matrix[cluster_members, cluster_members]
        wcss_k <- wcss_k + sum(cluster_dist) / 2
      }
    }
    wcss[k] <- wcss_k
    cat("k =", k, "- WCSS:", round(wcss[k], 2), "\n")
  }

  # =====================================================
  # Find Elbow Point using Perpendicular Distance Method
  # =====================================================

  # Use the perpendicular distance from the line connecting first and last points
  # This is more robust than rate of change method
  k_values <- 2:max_k
  wcss_values <- wcss[2:max_k]

  # Normalize coordinates to [0,1] range for better distance calculation
  k_norm <- (k_values - min(k_values)) / (max(k_values) - min(k_values))
  wcss_norm <- (wcss_values - min(wcss_values)) / (max(wcss_values) - min(wcss_values))

  # Line from first to last point
  x1 <- k_norm[1]
  y1 <- wcss_norm[1]
  x2 <- k_norm[length(k_norm)]
  y2 <- wcss_norm[length(wcss_norm)]

  # Calculate perpendicular distance for each point
  distances <- numeric(length(k_values))
  for (i in seq_along(k_values)) {
    x0 <- k_norm[i]
    y0 <- wcss_norm[i]
    
    # Perpendicular distance from point (x0, y0) to line through (x1, y1) and (x2, y2)
    numerator <- abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator <- sqrt((y2 - y1)^2 + (x2 - x1)^2)
    distances[i] <- numerator / denominator
  }

  # The elbow is the point with maximum perpendicular distance
  elbow_idx <- which.max(distances)
  best_k_elbow <- k_values[elbow_idx]

  cat("Optimal number of clusters (by Elbow - Perpendicular Distance):", best_k_elbow, "\n")
  cat("WCSS at elbow:", round(wcss[best_k_elbow], 2), "\n")
  cat("Max perpendicular distance:", round(distances[elbow_idx], 4), "\n\n")

  # Plot Elbow Method
  p_elbow <- ggplot(data.frame(k = 2:max_k, wcss = wcss[2:max_k]), 
                    aes(x = k, y = wcss)) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    geom_point(data = data.frame(k = best_k_elbow, wcss = wcss[best_k_elbow]), 
              aes(x = k, y = wcss), color = "red", size = 5, shape = 21, stroke = 2) +
    labs(x = "Number of Clusters (k)", 
        y = "Within-Cluster Sum of Squares (WCSS)",
        title = "Elbow Method for Optimal Clustering",
        subtitle = "Red = Optimal Elbow (Perpendicular Distance)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, size = 10))

  print(p_elbow)
  ggsave(get_output_path(current_tunnel, "elbow_method.png"), p_elbow, width = 8, height = 6, dpi = 300)

  # Plot dendrograms for different linkage methods
  linkage_methods <- c("complete")
  for (method in linkage_methods) {
    hc <- hc_results[[method]]
    
    p_dend <- ggplot(data = data.frame(x = 1:length(hc$order)), 
                    aes(x = hc$order[x])) +
      theme_void()
    
    # Create dendrogram using base R and convert to ggplot-compatible format
    png(get_output_path(current_tunnel, paste0("dendrogram_", method, ".png")), width = 1200, height = 600, res = 100)
    plot(hc, main = paste("Hierarchical Clustering Dendrogram -", toupper(method), "Linkage"),
        xlab = "Segment ID", ylab = "DTW Distance", cex = 0.8)
    dev.off()
    
    cat("Saved dendrogram for", method, "linkage\n")
  }

  # Cut dendrogram at optimal k (using elbow method)
  cat("\n=== Cutting Dendrogram at k =", best_k_elbow, "Clusters ===\n")

  n_hc_clusters <- best_k_elbow
  hc_clusters <- cutree(hc_main, k = n_hc_clusters)

  cat("Number of clusters:", n_hc_clusters, "\n\n")

  # Create hierarchical cluster assignments
  hc_cluster_assignments <- tibble(
    segment_id = seq_len(n_segments),
    cluster = factor(hc_clusters)
  ) %>%
    left_join(segment_info, by = "segment_id") %>%
    arrange(cluster, segment_id)

  cat("=== Hierarchical Cluster Assignments ===\n\n")
  print(hc_cluster_assignments %>% 
        select(segment_id, cluster, Date, time_start_str, time_end_str, duration_hours, max_z_score))

  # Summary by hierarchical cluster
  cat("\n=== Summary by Hierarchical Cluster ===\n")
  hc_cluster_summary <- hc_cluster_assignments %>%
    group_by(cluster) %>%
    summarise(
      n_segments = n(),
      mean_duration = mean(duration_hours, na.rm = TRUE),
      mean_z_score = mean(max_z_score, na.rm = TRUE),
      date_range = paste(min(Date), "to", max(Date)),
      .groups = "drop"
    ) %>%
    arrange(cluster)

  print(hc_cluster_summary)

  # Save all clusters to separate CSV files
  for (clust_id in sort(unique(hc_cluster_assignments$cluster))) {
    cluster_data_csv <- hc_cluster_assignments %>%
      filter(cluster == clust_id) %>%
      select(segment_id, cluster, Date, time_start_str, time_end_str, duration_hours, max_z_score, outlier_type, anomaly_sign) %>%
      arrange(desc(max_z_score))
    
    filename <- paste0("cluster_", clust_id, ".csv")
    filepath <- get_output_path(current_tunnel, filename, type = "csv")
    write_csv(cluster_data_csv, filepath)
    cat("Saved cluster", clust_id, "to", filepath, "\n")
  }

  # =====================================================
  # Plot Time Series for Each Cluster
  # =====================================================

  cat("\n=== Plotting Time Series for Each Cluster ===\n\n")

  for (clust_id in sort(unique(hc_cluster_assignments$cluster))) {
    # Get segments in this cluster
    cluster_segment_ids <- hc_cluster_assignments %>%
      filter(cluster == clust_id) %>%
      pull(segment_id)
    
    # Get all data points for segments in this cluster with padding to full day
    cluster_data <- tibble()
    
    # Create full day time index (0-96 representing 00:00 to 23:45)
    full_day_idx <- 1:96
    full_day_labels <- paste0(
      sprintf("%02d", floor((full_day_idx - 1) / 4)),
      ":",
      sprintf("%02d", ((full_day_idx - 1) %% 4) * 15)
    )
    
    for (seg_id in cluster_segment_ids) {
      segment <- all_anomalous_segments %>%
        filter(row_number() == seg_id)
      
      segment_day <- segment$day
      segment_start <- as.integer(round(segment$time_start))
      segment_end <- as.integer(round(segment$time_end))
      
      # Extract time series for this segment
      seg_data <- df_long_residuals %>%
        filter(day == segment_day & time_idx >= segment_start & time_idx <= segment_end) %>%
        select(day, time_idx, value)
      
      # Create padded data with NAs for missing time points
      padded_data <- tibble(
        time_idx = full_day_idx
      ) %>%
        left_join(seg_data, by = "time_idx") %>%
        mutate(
          segment_id = seg_id,
          date_label = segment$Date,
          time_label = full_day_labels
        ) %>%
        select(segment_id, date_label, time_idx, time_label, value)
      
      cluster_data <- bind_rows(cluster_data, padded_data)
    }
    
    # Create plot for this cluster with all segments on same plot
    p_cluster <- ggplot(cluster_data, aes(x = factor(time_label, levels = full_day_labels), y = value, group = segment_id)) +
      geom_line(linewidth = 0.6, alpha = 0.5, color = "black") +
      geom_point(size = 1, alpha = 0.4, color = "black") +
      scale_x_discrete(
        breaks = c("00:00", "06:00", "12:00", "18:00", "23:45"),
        labels = c("00:00", "06:00", "12:00", "18:00", "23:45"),
        expand = expansion(add = c(0.5, 0.5))
      ) +
      labs(
        title = "",
        x = "",
        y = "Traffic (% of capacity)"
      ) +
      theme_bw(base_size = 12) +
      theme(plot.title = element_text(hjust = 0.5, face = "bold"),
            axis.text.x = element_text(angle = 0, hjust = 0.5, size = 8))
    
    print(p_cluster)
    ggsave(get_output_path(current_tunnel, paste0("cluster_", clust_id, "_segments.png")), p_cluster, width = 12, height = 7, dpi = 300)
    
    cat("Saved plot for cluster", clust_id, "\n")
  }

  # =====================================================
  # Plot Time Series for Each Cluster Using Original Values
  # =====================================================

  cat("\n=== Plotting Time Series for Each Cluster (Original Values) ===\n\n")

  for (clust_id in sort(unique(hc_cluster_assignments$cluster))) {
    # Get segments in this cluster
    cluster_segment_ids <- hc_cluster_assignments %>%
      filter(cluster == clust_id) %>%
      pull(segment_id)
    
    # Get all data points for segments in this cluster with padding to full day
    cluster_data_original <- tibble()
    
    # Create full day time index (0-96 representing 00:00 to 23:45)
    full_day_idx <- 1:96
    full_day_labels <- paste0(
      sprintf("%02d", floor((full_day_idx - 1) / 4)),
      ":",
      sprintf("%02d", ((full_day_idx - 1) %% 4) * 15)
    )
    
    for (seg_id in cluster_segment_ids) {
      segment <- all_anomalous_segments %>%
        filter(row_number() == seg_id)
      
      segment_day <- segment$day
      segment_start <- as.integer(round(segment$time_start))
      segment_end <- as.integer(round(segment$time_end))
      
      # Extract time series for this segment from ORIGINAL data
      seg_data_original <- df_long_original %>%
        filter(day == segment_day & time_idx >= segment_start & time_idx <= segment_end) %>%
        select(day, time_idx, value_original)
      
      # Create padded data with NAs for missing time points
      padded_data_original <- tibble(
        time_idx = full_day_idx
      ) %>%
        left_join(seg_data_original, by = "time_idx") %>%
        mutate(
          segment_id = seg_id,
          date_label = segment$Date,
          time_label = full_day_labels
        ) %>%
        select(segment_id, date_label, time_idx, time_label, value_original)
      
      cluster_data_original <- bind_rows(cluster_data_original, padded_data_original)
    }
    
    # Create plot for this cluster with all segments on same plot (Original values)
    p_cluster_original <- ggplot(cluster_data_original, aes(x = factor(time_label, levels = full_day_labels), y = value_original, group = segment_id)) +
      geom_line(linewidth = 0.6, alpha = 0.5, color = "black") +
      geom_point(size = 1, alpha = 0.4, color = "black") +
      scale_x_discrete(
        breaks = c("00:00", "06:00", "12:00", "18:00", "23:45"),
        labels = c("00:00", "06:00", "12:00", "18:00", "23:45"),
        expand = expansion(add = c(0.5, 0.5))
      ) +
      labs(
        title = "",
        x = "",
        y = "Traffic (% of capacity)"
      ) +
      theme_bw(base_size = 12) +
      theme(plot.title = element_text(hjust = 0.5, face = "bold"),
            axis.text.x = element_text(angle = 0, hjust = 0.5, size = 8))
    
    print(p_cluster_original)
    ggsave(get_output_path(current_tunnel, paste0("cluster_", clust_id, "_segments_original.png")), p_cluster_original, width = 12, height = 7, dpi = 300)
    
    cat("Saved original plot for cluster", clust_id, "\n")
  }

  # =====================================================
  # Combined Plot: All Clusters with Different Colors
  # =====================================================

  cat("\n=== Creating Combined Plot with All Clusters ===\n\n")

  # Collect data from all clusters
  all_clusters_combined <- tibble()
  full_day_idx <- 1:96
  full_day_labels <- paste0(
    sprintf("%02d", floor((full_day_idx - 1) / 4)),
    ":",
    sprintf("%02d", ((full_day_idx - 1) %% 4) * 15)
  )

  for (seg_id in seq_len(nrow(all_anomalous_segments))) {
    segment <- all_anomalous_segments[seg_id, ]
    segment_day <- segment$day
    segment_start <- as.integer(round(segment$time_start))
    segment_end <- as.integer(round(segment$time_end))
    
    # Get cluster assignment for this segment
    cluster_assignment <- hc_cluster_assignments %>%
      filter(segment_id == seg_id) %>%
      pull(cluster)
    
    # Extract time series from ORIGINAL data
    seg_data_original <- df_long_original %>%
      filter(day == segment_day & time_idx >= segment_start & time_idx <= segment_end) %>%
      select(day, time_idx, value_original)
    
    # Create padded data
    padded_data <- tibble(
      time_idx = full_day_idx
    ) %>%
      left_join(seg_data_original, by = "time_idx") %>%
      mutate(
        segment_id = seg_id,
        cluster_id = cluster_assignment,
        date_label = segment$Date,
        time_label = full_day_labels
      ) %>%
      select(segment_id, cluster_id, date_label, time_idx, time_label, value_original)
    
    all_clusters_combined <- bind_rows(all_clusters_combined, padded_data)
  }

  # Define color palette for clusters (fallback)
  n_clusters_total <- n_distinct(hc_cluster_assignments$cluster)
  cluster_colors <- c(
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
    "#a6761d", "#666666", "#1f78b4", "#b2df8a", "#fb9a99", "#A25CCF",
    "#FF00FF", "#00FF00", "#0000FF", "#FF6600", "#00CCFF", "#FFCC00",
    "#CC00FF", "#00FF99"
  )

  # Ensure cluster_id is a factor with consistent levels and create labels for legend
  cluster_levels <- if (exists("hc_cluster_assignments")) {
    levels(hc_cluster_assignments$cluster)
  } else {
    sort(unique(all_clusters_combined$cluster_id))
  }
  all_clusters_combined <- all_clusters_combined %>%
    mutate(cluster_id = factor(cluster_id, levels = cluster_levels))
  cluster_labels <- paste0("Cluster ", cluster_levels)

  # Calculate average intraday profile for non-outlier days using original values
  normal_intraday_profile_original <- df_long_residuals %>%
    filter(outlier_type == "none") %>%
    select(day, time_idx) %>%
    left_join(
      df_long_original %>% select(day, time_idx, value_original),
      by = c("day", "time_idx")
    ) %>%
    group_by(time_idx) %>%
    summarise(
      mean_value_original = mean(value_original, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      time_label = paste0(
        sprintf("%02d", floor((time_idx - 1) / 4)),
        ":",
        sprintf("%02d", ((time_idx - 1) %% 4) * 15)
      ),
      time_label = factor(time_label, levels = full_day_labels)
    )

  # Create combined plot
  p_all_clusters <- ggplot(all_clusters_combined, aes(x = time_idx, 
                                                      y = value_original, 
                                                      group = segment_id,
                                                      color = cluster_id)) +
    geom_line(linewidth = 1, alpha = 0.6) +
    geom_point(size = 1, alpha = 0.4) +
    # Add average profile line for non-outlier days
    geom_line(
      data = normal_intraday_profile_original,
      aes(x = time_idx, y = mean_value_original, group = NA, color = NA),
      linewidth = 2.5,
      color = "#2d5016",
      alpha = 1,
      inherit.aes = FALSE
    ) +
    geom_vline(xintercept = c(46, 56, 73, 81, 96), color = "gray", linetype = "dashed") +
    scale_color_brewer(
      name = "Cluster",
      palette = "Dark2",
      labels = cluster_labels
    ) +
    guides(color = guide_legend(ncol = 3, byrow = TRUE,
                                override.aes = list(size = 1.6, alpha = 1))) +
    scale_x_continuous(
      breaks = c(1, 24, 48, 72, 96),
      labels = c("00:00", "06:00", "12:00", "18:00", "23:45"),
      expand = expansion(add = c(0.5, 0.5))
    ) +
    scale_y_continuous(
      limits = c(0, max(all_clusters_combined$value_original, na.rm = TRUE) * 1.05)
    ) +
    labs(
      title = "",
      x = "",
      y = "Traffic (% of capacity)"
    ) +
    theme_bw(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 22),
      axis.text.y = element_text(size = 22),
      axis.title.y = element_text(size = 26, face = "bold"),
      legend.position = c(0.02, 0.98),
      legend.justification = c(0, 1),
      legend.background = element_rect(fill = alpha("white", 0.85), colour = NA),
      legend.key = element_rect(fill = NA, colour = NA),
      legend.key.width = grid::unit(3, "lines"),
      legend.key.height = grid::unit(1.2, "lines"),
      legend.title = element_text(size = 22),
      legend.text = element_text(size = 18),
      plot.margin = margin(t = 5, r = 25, b = 5, l = 5, unit = "pt") 
    )

  print(p_all_clusters)
  ggsave(get_output_path(current_tunnel, "all_clusters_combined_original.png"), p_all_clusters, width = 16, height = 7, dpi = 300)

  cat("Saved combined plot to: ", get_output_path(current_tunnel, "all_clusters_combined_original.png"), "\n")

  cat("\n=== Analysis Complete for tunnel:", current_tunnel, "===\n")
}

cat("\n", strrep("=", 70), "\n")
cat("ALL TUNNEL ANALYSES COMPLETE\n")
cat(strrep("=", 70), "\n\n")

# =====================================================
# Match Overlap Analysis
# =====================================================
cat("\n", strrep("=", 70), "\n")
cat("ANALYZING CLUSTER OVERLAP WITH FOOTBALL MATCHES\n")
cat(strrep("=", 70), "\n\n")

# Load match data
matches_serie_a_file <- "../../data/cleaned/matches.csv"
matches_champions_file <- "../../data/cleaned/champions_matches.csv"

if (file.exists(matches_serie_a_file)) {
  cat("Loading Serie A matches from:", matches_serie_a_file, "\n")
  matches_serie_a <- read_csv(matches_serie_a_file, col_names = c("timestamp", "match_count"), col_types = "cd")
  matches_serie_a <- matches_serie_a %>%
    mutate(timestamp = as.POSIXct(timestamp, format = "%d-%b-%Y %H:%M:%S", tz = ""))
  cat("  Loaded", nrow(matches_serie_a), "time slots\n")
  cat("  Slots with matches:", sum(matches_serie_a$match_count > 0), "\n")
} else {
  cat("WARNING: Serie A matches file not found:", matches_serie_a_file, "\n")
  matches_serie_a <- tibble(timestamp = as.POSIXct(character()), match_count = numeric())
}

if (file.exists(matches_champions_file)) {
  cat("Loading Champions League matches from:", matches_champions_file, "\n")
  matches_champions <- read_csv(matches_champions_file, col_names = c("timestamp", "match_count"), col_types = "cd")
  matches_champions <- matches_champions %>%
    mutate(timestamp = as.POSIXct(timestamp, format = "%d-%b-%Y %H:%M:%S", tz = ""))
  cat("  Loaded", nrow(matches_champions), "time slots\n")
  cat("  Slots with matches:", sum(matches_champions$match_count > 0), "\n")
} else {
  cat("WARNING: Champions League matches file not found:", matches_champions_file, "\n")
  matches_champions <- tibble(timestamp = as.POSIXct(character()), match_count = numeric())
}

# Function to check overlap between a time range and matches
check_match_overlap <- function(start_time, end_time, match_data) {
  if (nrow(match_data) == 0) return(FALSE)
  
  # Filter matches that fall within or overlap with the segment time range
  # A match overlaps if its timestamp is >= start_time and < end_time
  # Or if the match extends into the segment (match starts before but extends into segment)
  overlapping_matches <- match_data %>%
    filter(
      timestamp >= start_time & timestamp <= end_time &
      match_count > 0
    )
  
  return(nrow(overlapping_matches) > 0)
}

# Analyze overlap for each tunnel's clusters
all_overlap_stats <- list()

for (tunnel_pattern in tunnel_patterns) {
  cat("\nAnalyzing overlaps for tunnel:", tunnel_pattern, "\n")
  
  # Load all cluster CSV files for this tunnel
  cluster_dir <- paste0("csv/", tunnel_pattern)
  
  if (!dir.exists(cluster_dir)) {
    cat("  WARNING: Cluster directory not found:", cluster_dir, "\n")
    next
  }
  
  # Find all cluster_*.csv files
  cluster_files <- list.files(cluster_dir, pattern = "^cluster_\\d+\\.csv$", full.names = TRUE)
  
  if (length(cluster_files) == 0) {
    cat("  WARNING: No cluster files found in:", cluster_dir, "\n")
    next
  }
  
  cat("  Found", length(cluster_files), "cluster files\n")
  
  # Read and combine all cluster files
  clusters <- tibble()
  for (file in cluster_files) {
    cluster_data <- read_csv(file, show_col_types = FALSE)
    # Extract cluster ID from filename
    cluster_id <- as.integer(gsub(".*cluster_(\\d+)\\.csv", "\\1", file))
    cluster_data$cluster_id <- cluster_id
    clusters <- bind_rows(clusters, cluster_data)
  }
  
  cat("  Total segments:", nrow(clusters), "\n")
  
  # Parse time information
  clusters <- clusters %>%
    mutate(
      Date = as.Date(Date),
      time_start = as.POSIXct(paste(Date, time_start_str), format = "%Y-%m-%d %H:%M", tz = ""),
      time_end = as.POSIXct(paste(Date, time_end_str), format = "%Y-%m-%d %H:%M", tz = "")
    )
  
  # Get unique clusters
  unique_clusters <- sort(unique(clusters$cluster_id))
  
  cluster_stats <- tibble()
  
  for (clust_id in unique_clusters) {
    cluster_segments <- clusters %>% filter(cluster_id == clust_id)
    
    total_segments <- nrow(cluster_segments)
    serie_a_overlaps <- 0
    champions_overlaps <- 0
    no_match_overlaps <- 0
    
    # Check each segment
    for (i in 1:nrow(cluster_segments)) {
      seg <- cluster_segments[i, ]
      
      seg_start <- seg$time_start
      seg_end <- seg$time_end
      
      # Check overlaps
      has_serie_a <- check_match_overlap(seg_start, seg_end, matches_serie_a)
      has_champions <- check_match_overlap(seg_start, seg_end, matches_champions)
      
      # Count first match type found (prioritized order)
      if (has_serie_a) {
        serie_a_overlaps <- serie_a_overlaps + 1
      } else if (has_champions) {
        champions_overlaps <- champions_overlaps + 1
      } else {
        no_match_overlaps <- no_match_overlaps + 1
      }
    }
    
    # Store statistics for this cluster
    cluster_stats <- bind_rows(cluster_stats, tibble(
      tunnel = tunnel_pattern,
      cluster_id = clust_id,
      total_segments = total_segments,
      serie_a_overlaps = serie_a_overlaps,
      champions_overlaps = champions_overlaps,
      no_match_overlaps = no_match_overlaps,
      serie_a_percent = round(100 * serie_a_overlaps / total_segments, 2),
      champions_percent = round(100 * champions_overlaps / total_segments, 2),
      no_match_percent = round(100 * no_match_overlaps / total_segments, 2)
    ))
  }
  
  # Print summary
  cat("\nOverlap statistics for", tunnel_pattern, ":\n")
  print(cluster_stats, n = Inf)
  
  # Save results
  output_file <- paste0("csv/", tunnel_pattern, "/cluster_match_overlap.csv")
  write_csv(cluster_stats, output_file)
  cat("\nSaved overlap statistics to:", output_file, "\n")
  
  all_overlap_stats[[tunnel_pattern]] <- cluster_stats
  
  # Extract segments without match overlaps and sort by max_z_score
  cat("\nExtracting non-overlapping segments...\n")
  
  non_overlapping_segments <- tibble()
  
  for (i in 1:nrow(clusters)) {
    seg <- clusters[i, ]
    
    seg_start <- seg$time_start
    seg_end <- seg$time_end
    
    # Check overlaps
    has_serie_a <- check_match_overlap(seg_start, seg_end, matches_serie_a)
    has_champions <- check_match_overlap(seg_start, seg_end, matches_champions)
    
    # If no overlap with any match, add to list
    if (!has_serie_a && !has_champions) {
      non_overlapping_segments <- bind_rows(non_overlapping_segments, seg)
    }
  }
  
  # Sort by max_z_score in descending order
  if (nrow(non_overlapping_segments) > 0) {
    non_overlapping_segments <- non_overlapping_segments %>%
      arrange(desc(max_z_score))
    
    # Save to CSV
    no_match_file <- paste0("csv/", tunnel_pattern, "/segments_no_match_by_zscore.csv")
    write_csv(non_overlapping_segments, no_match_file)
    
    cat("  Found", nrow(non_overlapping_segments), "segments without match overlap\n")
    cat("  Saved to:", no_match_file, "\n")
    
    # Show top 10
    cat("\n  Top 10 non-overlapping segments by z-score:\n")
    print(non_overlapping_segments %>% 
            select(segment_id, cluster_id, Date, time_start_str, time_end_str, max_z_score, outlier_type) %>%
            head(10), n = 10)
  } else {
    cat("  No segments without match overlap found\n")
  }
}

# Create summary plot if we have data
if (length(all_overlap_stats) > 0) {
  combined_stats <- bind_rows(all_overlap_stats)
  
  # Create stacked bar plot using counts (number of segments)
  plot_data <- combined_stats %>%
    select(tunnel, cluster_id, serie_a_overlaps, champions_overlaps, no_match_overlaps) %>%
    pivot_longer(
      cols = ends_with("_overlaps"),
      names_to = "match_type",
      values_to = "count"
    ) %>%
    mutate(
      match_type = factor(match_type,
                         levels = c("no_match_overlaps", "serie_a_overlaps", "champions_overlaps"),
                         labels = c("No Match", "Serie A", "Champions"))
    )

  p_overlap <- ggplot(plot_data, aes(x = factor(cluster_id), y = count, fill = match_type)) +
    geom_bar(stat = "identity", position = "stack") +
    facet_wrap(~tunnel, scales = "free_x") +
    scale_fill_manual(
      values = c("No Match" = "#CCCCCC", "Serie A" = "#0066CC", 
                 "Champions" = "#FF9900")
    ) +
    labs(
      title = "Cluster Overlap with Football Matches",
      x = "Cluster ID",
      y = "Number of Segments",
      fill = "Match Type"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "bottom",
      axis.text.x = element_text(angle = 0, hjust = 0.5)
    )

  print(p_overlap)
  ggsave("figures/cluster_match_overlap_summary.png", p_overlap, width = 14, height = 8, dpi = 300)
  cat("\nSaved summary plot to: figures/cluster_match_overlap_summary.png\n")
}

cat("\n", strrep("=", 70), "\n")
cat("MATCH OVERLAP ANALYSIS COMPLETE\n")
cat(strrep("=", 70), "\n\n")