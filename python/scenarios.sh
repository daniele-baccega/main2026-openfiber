#!/bin/bash

./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 7 --percentile 99 --use_single_kde --plot_pdfcdf --plot_tunnels
./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 7 --percentile 99 --use_saved_data --plot_pdfcdf --plot_tunnels

./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 14 --percentile 99 --use_single_kde
./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 14 --percentile 99 --use_saved_data

./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 21 --percentile 99 --use_single_kde
./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 21 --percentile 99 --use_saved_data

./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 28 --percentile 99 --use_single_kde
./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 28 --percentile 99 --use_saved_data

./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 7 --percentile 99 --use_single_kde --plot_pdfcdf --plot_tunnels
./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 7 --percentile 99 --use_saved_data --plot_pdfcdf --plot_tunnels

./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 14 --percentile 99 --use_single_kde
./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 14 --percentile 99 --use_saved_data

./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 21 --percentile 99 --use_single_kde
./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 21 --percentile 99 --use_saved_data

./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 28 --percentile 99 --use_single_kde
./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 28 --percentile 99 --use_saved_data