#!/bin/bash

./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 7 --prediction_interval 98 --use_single_kde --plot_pdfcdf --plot_tunnels
./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 7 --prediction_interval 98 --use_saved_data --plot_pdfcdf --plot_tunnels

./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 14 --prediction_interval 98 --use_single_kde
./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 14 --prediction_interval 98 --use_saved_data

./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 21 --prediction_interval 98 --use_single_kde
./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 21 --prediction_interval 98 --use_saved_data

./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 28 --prediction_interval 98 --use_single_kde
./run.sh --cutoff_date 2025-04-30 --forecast_date 2025-04-30 --forecast_days 28 --prediction_interval 98 --use_saved_data

./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 7 --prediction_interval 98 --use_single_kde --plot_pdfcdf --plot_tunnels
./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 7 --prediction_interval 98 --use_saved_data --plot_pdfcdf --plot_tunnels

./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 14 --prediction_interval 98 --use_single_kde
./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 14 --prediction_interval 98 --use_saved_data

./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 21 --prediction_interval 98 --use_single_kde
./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 21 --prediction_interval 98 --use_saved_data

./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 28 --prediction_interval 98 --use_single_kde
./run.sh --cutoff_date 2024-09-30 --forecast_date 2024-09-30 --forecast_days 28 --prediction_interval 98 --use_saved_data