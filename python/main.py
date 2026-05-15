'''
  Tunnel data prediction

  main.py

  This script preprocesses tunnel data, generates plots, and creates forecasts based on specified dates.

  Command line parameters:
     --seed:                        random seed
     --use_saved_data:              use previously preprocessed data, if exists
     --plot_tunnels:                plot tunnel data
     --plot_start_date:             start date for plotting (format: YYYY-MM-DD)
     --plot_end_date:               end date for plotting (format: YYYY-MM-DD)
     --forecast_date:               date to forecast (format: YYYY-MM-DD)
     --focus_tunnels:               list of tunnel names or prefixes to focus on for forecasting
                                    (e.g., --focus_tunnels TunnelA TunnelB or --focus_tunnels TunnelA_
                                    to include all tunnels starting with TunnelA_)
     --forecast_days:               number of days to forecast from each date
     --inject_indicator_values:     inject indicator values during forecasting
     --inject_indicator_periods:    inject indicator periods during forecasting
     --inject_indicator_types:      inject indicator types during forecasting
     --traffic_threshold:           traffic threshold for indicator injection
     --percentile:                  percentile for thresholding
     --plot_pdfcdf:                 whether to plot PDF and CDF of forecasts
     --use_single_kde:              whether to use a single KDE for all tunnels when plotting PDF/CDF
     --print_aggregate_statistics:  whether to compute and print aggregate statistics on all tunnels

  Authors: Daniele Baccega
'''

import os
import sys
import pandas as pd

## Import custom model, utils, plots and data generator
from utils import *
from plots import *


sys.excepthook = log_uncaught_exceptions

## Parse command line arguments
args = parse_arguments()
print(args)

## Set the seed to reproduce the results
setup_seed(args.seed)

## Load data
#  Select the measure to analyze ('Max_Max_Rate_Gbps' or 'Max_Avg_Rate_Gbps')
info = {}
info["interested_measure"] = interested_measure = "Max_Max_Rate_Gbps"

#  Determine decompositions directory based on cutoff_date
decompositions_dir = get_decompositions_dir(args.cutoff_date)
print(f"Using decompositions directory: {decompositions_dir}")

#  Get tunnel names from CSV header
trends_file = os.path.join(decompositions_dir, interested_measure + "_Trends.csv")
if not os.path.exists(trends_file):
    print(f"Error: File not found: {trends_file}")
    sys.exit(1)

#  Read header to get tunnel names (excluding 'Time' column)
with open(trends_file, 'r') as f:
    header = f.readline().strip().split(',')
    interested_tunnels = [col for col in header if col != 'Time']

#  Filter to focus tunnels if specified
if args.focus_tunnels:
    focus_patterns = set(args.focus_tunnels)
    interested_tunnels = [t for t in interested_tunnels 
                         if any(t == pattern or t.startswith(pattern) for pattern in focus_patterns)]
    if not interested_tunnels:
        print(f"Error: None of the specified focus_tunnels/prefixes {args.focus_tunnels} matched available tunnels")
        sys.exit(1)
    print(f"Focusing on {len(interested_tunnels)} tunnel(s): {interested_tunnels}")
else:
    print(f"Analyzing all {len(interested_tunnels)} available tunnels")

#  Store interested tunnels in info dictionary for later use
info["interested_tunnels"] = interested_tunnels

#  Create data, plots, and results directories with measure-specific subdirectories
data_dir = os.path.join('data', interested_measure)
plots_dir = os.path.join('plots', interested_measure)
results_dir = os.path.join('results', interested_measure)

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

#  Load preprocessed data if specified, otherwise preprocess from raw data
data_file = os.path.join(data_dir, 'data_' + interested_measure + ('_' + args.forecast_date if args.forecast_date else '') + '.csv')
aggregated_file = os.path.join(data_dir, 'aggregated_' + interested_measure + ('_' + args.forecast_date if args.forecast_date else '') + '.csv')
matches_path = os.path.join(data_dir, 'matches.csv')
matches_champions_path = os.path.join(data_dir, 'matches_champions.csv')
original_full_file = os.path.join(data_dir, 'original_full_' + interested_measure + ('_' + args.forecast_date if args.forecast_date else '') + '.csv')
original_full_agg_file = os.path.join(data_dir, 'original_full_agg_' + interested_measure + ('_' + args.forecast_date if args.forecast_date else '') + '.csv')
if not args.print_aggregate_statistics and args.use_saved_data and os.path.exists(data_file) and os.path.exists(aggregated_file) and \
   os.path.exists(matches_path) and os.path.exists(matches_champions_path) and os.path.exists(original_full_file) and os.path.exists(original_full_agg_file):
    data = pd.read_csv(data_file, parse_dates=['Time'])
    aggregated_data = pd.read_csv(aggregated_file, parse_dates=['Time'])

    matches = pd.read_csv(matches_path, parse_dates=['Time'])
    matches_champions = pd.read_csv(matches_champions_path, parse_dates=['Time'])
    
    original_full = pd.read_csv(original_full_file, parse_dates=['Time'])
    original_full_agg = pd.read_csv(original_full_agg_file, parse_dates=['Time'])
else:
    capacity = pd.read_csv('../data/cleaned/Capacity_Gbps.csv')
    original = pd.read_csv('../data/cleaned/' + interested_measure + '.csv')
    trends = pd.read_csv(os.path.join(decompositions_dir, interested_measure + '_Trends.csv'))
    seasonals = pd.read_csv(os.path.join(decompositions_dir, interested_measure + '_Seasonals.csv'))
    residuals = pd.read_csv(os.path.join(decompositions_dir, interested_measure + '_Residuals.csv'))
    matches = pd.read_csv('../data/cleaned/matches.csv')
    matches_champions = pd.read_csv('../data/cleaned/champions_matches.csv')
    
    trends_sum = pd.read_csv(os.path.join(decompositions_dir, interested_measure + '_Sum_Trends.csv'))
    seasonals_sum = pd.read_csv(os.path.join(decompositions_dir, interested_measure + '_Sum_Seasonals.csv'))
    residuals_sum = pd.read_csv(os.path.join(decompositions_dir, interested_measure + '_Sum_Residuals.csv'))

    capacity_raw = capacity.copy()
    capacity_raw = pd.melt(capacity_raw, id_vars=['Time'], var_name='Tunnel', value_name='Capacity')
    capacity_raw['Time'] = pd.to_datetime(capacity_raw['Time'], dayfirst=True, errors='coerce')
    
    original_raw = original.copy()
    original_raw = pd.melt(original_raw, id_vars=['Time'], var_name='Tunnel', value_name='Original')
    original_raw['Time'] = pd.to_datetime(original_raw['Time'], dayfirst=True, errors='coerce')
    
    aggregated_capacity_raw = capacity_raw.groupby('Time', as_index=False)['Capacity'].sum()
    aggregated_original_raw = original_raw.groupby('Time', as_index=False)['Original'].sum()

    capacity_filtered = capacity.loc[:, ["Time"] + interested_tunnels]
    original_filtered = original.loc[:, ["Time"] + interested_tunnels]

    # Preprocess data for selected tunnels
    data, original_full = preprocess_data(capacity_filtered, original_filtered, trends, seasonals, residuals, forecast_date=args.forecast_date, forecast_days=args.forecast_days)

    # Process aggregated data using the raw summed capacities, summed decompositions, and raw summed original for ground truth
    aggregated_data, original_full_agg = preprocess_aggregated_data(aggregated_capacity_raw, aggregated_original_raw, trends_sum, seasonals_sum, residuals_sum, forecast_date=args.forecast_date, forecast_days=args.forecast_days)

    # Save preprocessed data
    data.to_csv(data_file, index=False)
    aggregated_data.to_csv(aggregated_file, index=False)
    matches.to_csv(matches_path, index=False)
    matches_champions.to_csv(matches_champions_path, index=False)
    original_full.to_csv(original_full_file, index=False)
    original_full_agg.to_csv(original_full_agg_file, index=False)

    #  Compute aggregate statistics on all tunnels
    if args.print_aggregate_statistics:
        capacity_all = capacity.copy()
        original_all = original.copy()

        # Use all tunnels from the header (excluding 'Time')
        all_tunnels = [col for col in capacity_all.columns if col != 'Time']
        capacity_all = capacity_all.loc[:, ["Time"] + all_tunnels]
        original_all = original_all.loc[:, ["Time"] + all_tunnels]

        # Run preprocess_data on all tunnels
        _, original_full_all = preprocess_data(capacity_all, original_all, trends, seasonals, residuals, forecast_date=args.forecast_date, forecast_days=args.forecast_days)
        print_aggregate_statistics(original_full_all, output_dir=results_dir)


#  Print some information about the data
print("Preprocessed data head:\n", data.head())
print("Preprocessed data columns\n:", data.columns)

#  Plot some information for each tunnel
if args.plot_tunnels:
    plot_tunnels(data, output_dir=plots_dir, start_date=args.plot_start_date, end_date=args.plot_end_date)
    plot_tunnels(aggregated_data, output_dir=plots_dir, start_date=args.plot_start_date, end_date=args.plot_end_date)

print("Tunnel names\n:", data['Tunnel'].unique())

if args.forecast_date:
    generate_forecasts_from_dates(data,
                                  aggregated_data,
                                  matches,
                                  matches_champions,
                                  args.forecast_date,
                                  args.forecast_days,
                                  original_full,
                                  original_full_agg,
                                  args.plot_tunnels,
                                  output_dir=plots_dir, 
                                  inject_indicator_values=args.inject_indicator_values,
                                  inject_indicator_periods=args.inject_indicator_periods,
                                  inject_indicator_types=args.inject_indicator_types,
                                  traffic_threshold=args.traffic_threshold,
                                  percentile=args.percentile,
                                  plot_pdfcdf=args.plot_pdfcdf,
                                  results_dir=results_dir,
                                  use_single_kde=args.use_single_kde)
