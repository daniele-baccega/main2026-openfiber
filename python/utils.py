'''
  Tunnel data prediction

  utils.py

  This script provides utility functions for preprocessing tunnel data,
  generating plots, and creating forecasts based on specified dates.

  Authors: Daniele Baccega
'''

import os
import time
import random
import argparse
import traceback
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

## Import custom plots
from plots import *

def log_uncaught_exceptions(ex_cls, ex, tb):
    """
        Log uncaught exceptions to the console.
        
        Args:
            ex_cls: exception class
            ex: exception instance
            tb: traceback object
    """

    print(''.join(traceback.format_tb(tb)))
    print(f"{ex_cls.__name__}: {ex}")

def parse_arguments():
    """
        Parse command line arguments.

        Returns:
            args: parsed arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--use_saved_data', action='store_true', help='Use previosly preprocessed data, if exists')
    parser.add_argument('--plot_tunnels', action='store_true', help='Plot tunnels data')
    parser.add_argument('--plot_start_date', type=str, default=None, help='Start date for plotting period (YYYY-MM-DD). If not provided, plots from beginning of data')
    parser.add_argument('--plot_end_date', type=str, default=None, help='End date for plotting period (YYYY-MM-DD). If not provided, plots until end of data')
    parser.add_argument('--focus_tunnels', nargs='+', default=None, help='Tunnel names or prefixes to focus on. Matches exact names and prefixes. If not provided, all tunnels are analyzed')
    parser.add_argument('--forecast_date', type=str, default=None, help='Date (YYYY-MM-DD) to forecast from (midnight by default)')
    parser.add_argument('--cutoff_date', type=str, default=None, help='Cutoff date for decompositions (YYYY-MM-DD). If not provided, uses whole data')
    parser.add_argument('--forecast_days', type=int, default=7, help='Number of days to forecast ahead (default: 7)')
    parser.add_argument('--inject_indicator_values', nargs='+', type=int, default=None, help='List of indicator values to inject (e.g., 5 3 7)')
    parser.add_argument('--inject_indicator_periods', nargs='+', type=str, default=None, help='List of period boundaries in format: start1 end1 start2 end2 start3 end3 (each as YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--inject_indicator_types', nargs='+', type=str, default=None, help='List of injection types for each period (e.g., serie_a champions serie_a)')
    parser.add_argument('--traffic_threshold', type=float, default=None, help='Traffic threshold in percentage of capacity (e.g., 80 for 80%)')
    parser.add_argument('--prediction_interval', type=int, default=98, help='Prediction interval for forecast (default: 98)')
    parser.add_argument('--plot_pdfcdf', action='store_true', help='Plot PDF/CDF distributions (default: False for faster computation)')
    parser.add_argument('--use_single_kde', action='store_true', help='Use single KDE distribution from all training residuals instead of indicator-based distributions')
    parser.add_argument('--print_aggregate_statistics', action='store_true', help='Print aggregate statistics and save to CSV')

    args = parser.parse_args()
    return args

def setup_seed(seed):
    """
        Set the seeds (for np.random, tf.random, and random) to reproduce the results.

        Args:
            seed: seed value
    """

    random.seed(seed)
    np.random.seed(seed)

def get_decompositions_dir(cutoff_date):
    """
        Determine the decompositions directory path based on cutoff_date.
        If cutoff_date is provided, use data/decompositions/cutoff_<date>/csv/
        Otherwise, use data/decompositions/csv/
    """
    if cutoff_date:
        return os.path.join('../data/decompositions/', f'cutoff_{cutoff_date}', 'csv')
    else:
        return '../data/decompositions/csv'

def fill_gaps(data, tunnel_col='Tunnel', time_col='Time', max_gap_size=10, columns_to_fill=None):
    """
        Fill short gaps in time series data for each tunnel separately (or all data if tunnel_col is None).
        
        Args:
            data: DataFrame with time series data
            tunnel_col: column name for tunnel identifier (None for aggregated data)
            time_col: column name for time
            max_gap_size: maximum number of consecutive NaN values to fill (default: 10)
            columns_to_fill: list of column names to fill. If None, fills all numeric columns
                            except tunnel and time identifiers.
        
        Returns:
            DataFrame with gaps filled
    """

    data = data.copy()
    
    # Define columns to fill if not specified
    if columns_to_fill is None:
        # Fill all numeric columns except tunnel/time identifiers
        exclude_cols = {tunnel_col, time_col, 'TunnelID', 'Indicator'}
        columns_to_fill = [col for col in data.columns 
                          if data[col].dtype in ['float64', 'float32', 'int64', 'int32']
                          and col not in exclude_cols]
    
    # If no tunnel column, process all data as one
    if tunnel_col is None:
        tunnel_indices = data.index
    else:
        # Process each tunnel separately
        tunnels_to_process = [(tunnel, data[data[tunnel_col] == tunnel].index) 
                             for tunnel in data[tunnel_col].unique()]
    
    # Handle aggregated data (no tunnel_col)
    if tunnel_col is None:
        tunnels_to_process = [(None, data.index)]
    
    for _, tunnel_indices in tunnels_to_process:
        # Sort by time within each tunnel
        tunnel_data = data.loc[tunnel_indices].sort_values(time_col)
        
        for col in columns_to_fill:
            if col not in tunnel_data.columns:
                continue
            
            # Get values for this tunnel and column
            values = tunnel_data[col].values.copy()
            
            # Identify gaps (consecutive NaN values)
            nan_mask = np.isnan(values)
            
            # Find starts and lengths of gap sequences
            if nan_mask.any():
                # Use diff to identify start/end of gaps
                nan_padded = np.concatenate(([False], nan_mask, [False]))
                transitions = np.diff(nan_padded.astype(int))
                gap_starts = np.where(transitions == 1)[0]
                gap_ends = np.where(transitions == -1)[0]
                
                # Fill gaps that are short enough
                for start, end in zip(gap_starts, gap_ends):
                    gap_length = end - start
                    
                    if gap_length <= max_gap_size and start > 0 and end < len(values):
                        # Linear interpolation between valid values
                        before_val = values[start - 1]
                        after_val = values[end]
                        
                        # Only interpolate if both boundaries are valid numbers
                        if not np.isnan(before_val) and not np.isnan(after_val):
                            interpolated = np.linspace(before_val, after_val, 
                                                      gap_length + 2)[1:-1]
                            values[start:end] = interpolated
            
            # Update the data
            data.loc[tunnel_indices[tunnel_data.index.get_indexer(tunnel_data.index)], col] = values
    
    return data

def preprocess_data(capacity, original, trends, seasonals, residuals, forecast_date=None, forecast_days=7):
    """
        Preprocess the data by merging capacity, original, trends, seasonals, and residuals.

        Args:
            capacity: DataFrame with capacity data
            original: DataFrame with original measure data
            trends: DataFrame with trend component
            seasonals: DataFrame with seasonal component
            residuals: DataFrame with residual component
            forecast_date: date to forecast (format: YYYY-MM-DD)
            forecast_days: number of days to forecast
        Returns:
            data: preprocessed DataFrame
            original_full: original data merged with capacity (for ground truth extraction)
    """

    capacity['Time'] = pd.to_datetime(capacity['Time'], dayfirst=True, errors='coerce')
    original['Time'] = pd.to_datetime(original['Time'], dayfirst=True, errors='coerce')
    trends['Time'] = pd.to_datetime(trends['Time'], dayfirst=True, errors='coerce')
    seasonals['Time'] = pd.to_datetime(seasonals['Time'], dayfirst=True, errors='coerce')
    residuals['Time'] = pd.to_datetime(residuals['Time'], dayfirst=True, errors='coerce')
    
    original = pd.melt(original, id_vars=['Time'], var_name='Tunnel', value_name='Original')
    capacity = pd.melt(capacity, id_vars=['Time'], var_name='Tunnel', value_name='Capacity')

    # Create original_full for ground truth extraction and statistics (merge with capacity for normalization)
    original_full = original.copy()
    if forecast_date is not None:
        forecast_date_midnight = pd.to_datetime(forecast_date + " 00:00:00", format='%Y-%m-%d %H:%M:%S', errors='coerce')
        if forecast_date_midnight + pd.Timedelta(days=forecast_days) > original_full['Time'].max():
            tunnels = original_full['Tunnel'].unique()

            date_range = pd.date_range(start=original_full['Time'].max() + pd.Timedelta(minutes=15), end=forecast_date_midnight + pd.Timedelta(days=forecast_days), freq='15min')

            # Create a new DataFrame with all combinations of Tunnel and Time
            full_index = pd.MultiIndex.from_product([tunnels, date_range], names=['Tunnel', 'Time'])
            df_full = pd.DataFrame(index=full_index).reset_index()
            print(df_full)

            # Merge with original DataFrame to bring Original values where available
            original_full = pd.concat([original_full, df_full], ignore_index=True)

    # Normalize original_full: merge with capacity for normalization
    capacity_for_norm = capacity.copy()
    original_full = original_full.merge(capacity_for_norm, on=['Time', 'Tunnel'], how='left')
    original_full['Original'] = np.where(
        (original_full['Capacity'] > 0) & ~np.isnan(original_full['Original']),
        (original_full['Original'] / original_full['Capacity']) * 100,
        np.nan
    )
    original_full['Capacity'] = 100  # Capacity is now represented as 100%

    if forecast_date is not None:
        original_full = original_full[(original_full['Time'] > forecast_date_midnight) & (original_full['Time'] < forecast_date_midnight + pd.Timedelta(days=forecast_days))]

    trends = pd.melt(trends, id_vars=['Time'], var_name='Tunnel', value_name='Trend')
    seasonals = pd.melt(seasonals, id_vars=['Time'], var_name='Tunnel', value_name='Seasonal')
    residuals = pd.melt(residuals, id_vars=['Time'], var_name='Tunnel', value_name='Residual')

    seasonals['Tunnel'] = seasonals['Tunnel'].str.replace(r'_\d+$', '', regex=True)
    seasonals = seasonals.groupby(['Time', 'Tunnel'], as_index=False)['Seasonal'].sum()

    data = capacity.merge(trends, on=['Time', 'Tunnel']).merge(original, on=['Time', 'Tunnel']).merge(seasonals, on=['Time', 'Tunnel']).merge(residuals, on=['Time', 'Tunnel'])

    # Fill short gaps (up to 10 consecutive points) in the time series
    print("Filling gaps in time series (max gap size: 10 points)...", flush=True)
    data = fill_gaps(data, tunnel_col='Tunnel', time_col='Time', max_gap_size=10,
                    columns_to_fill=['Capacity', 'Original', 'Trend', 'Seasonal', 'Residual'])
    print("Gap filling completed.", flush=True)

    # Normalize all traffic values to percentage of capacity
    print("Normalizing traffic values to percentage of capacity...", flush=True)
    capacity = data['Capacity'].values
    data['Original'] = np.where(
        (capacity > 0) & ~np.isnan(data['Original'].values),
        (data['Original'].values / capacity) * 100,
        np.nan
    )
    data['Trend'] = np.where(
        (capacity > 0) & ~np.isnan(data['Trend'].values),
        (data['Trend'].values / capacity) * 100,
        np.nan
    )
    data['Seasonal'] = np.where(
        (capacity > 0) & ~np.isnan(data['Seasonal'].values),
        (data['Seasonal'].values / capacity) * 100,
        np.nan
    )
    data['Residual'] = np.where(
        (capacity > 0) & ~np.isnan(data['Residual'].values),
        (data['Residual'].values / capacity) * 100,
        np.nan
    )
    data['Capacity'] = 100  # Capacity is now represented as 100%
    print("Normalization completed.", flush=True)

    # Encode Tunnel names to numeric IDs
    unique_tunnels = data['Tunnel'].unique()
    tunnel_to_id = {tunnel: idx for idx, tunnel in enumerate(unique_tunnels)}
    data['TunnelID'] = data['Tunnel'].map(tunnel_to_id)

    return data, original_full

def preprocess_aggregated_data(aggregated_capacity, aggregated_original, trends_sum, seasonals_sum, residuals_sum, forecast_date=None, forecast_days=7):
    """
        Preprocess aggregated (summed) data from pre-computed sum files.
        Importantly: aggregated_capacity should contain RAW (non-normalized) summed capacity values,
        computed by summing all tunnel capacities BEFORE normalization.

        Args:
            aggregated_capacity: DataFrame with aggregated capacity data (raw sum of all tunnel capacities, NOT normalized)
            aggregated_original: DataFrame with aggregated measured traffic (raw sum of all tunnel traffic for ground truth)
            trends_sum: DataFrame with aggregated trend component
            seasonals_sum: DataFrame with aggregated seasonal component
            residuals_sum: DataFrame with aggregated residual component
            forecast_date: date to forecast (format: YYYY-MM-DD)
            forecast_days: number of days to forecast

        Returns:
            aggregated_data: preprocessed aggregated DataFrame (cut at training period)
            original_full_agg: ground truth data including forecast period (for plotting)
    """

    # Parse sum data
    trends_sum['Time'] = pd.to_datetime(trends_sum['Time'], dayfirst=True, errors='coerce')
    seasonals_sum['Time'] = pd.to_datetime(seasonals_sum['Time'], dayfirst=True, errors='coerce')
    residuals_sum['Time'] = pd.to_datetime(residuals_sum['Time'], dayfirst=True, errors='coerce')

    trends_sum = trends_sum.rename(columns={trends_sum.columns[1]: 'Trend'})
    seasonals_sum = seasonals_sum.rename(columns={col: 'Seasonal_Partial' for col in seasonals_sum.columns if col != 'Time'})
    seasonals_sum = seasonals_sum.assign(Seasonal=seasonals_sum.drop(columns='Time').sum(axis=1))[['Time', 'Seasonal']]
    residuals_sum = residuals_sum.rename(columns={residuals_sum.columns[1]: 'Residual'})

    # Start merging with trends
    aggregated_data = trends_sum.merge(seasonals_sum, on='Time', how='outer').merge(residuals_sum, on='Time', how='outer')
    
    # Sum raw capacity from all tunnels grouped by time
    aggregated_capacity['Time'] = pd.to_datetime(aggregated_capacity['Time'], dayfirst=True, errors='coerce')
    aggregated_data = aggregated_data.merge(aggregated_capacity, on='Time', how='outer')
    
    # Compute Original by summing components
    aggregated_data['Original'] = aggregated_data['Trend'] + aggregated_data['Seasonal'] + aggregated_data['Residual']
    
    # Create original_full_agg from actual measured aggregated traffic (not reconstructed)
    aggregated_original['Time'] = pd.to_datetime(aggregated_original['Time'], dayfirst=True, errors='coerce')
    original_full_agg = aggregated_original[['Time', 'Original']].copy()
    original_full_agg = original_full_agg.merge(aggregated_capacity[['Time', 'Capacity']], on='Time', how='left')
    
    # Extend data if forecast_date is beyond available data
    if forecast_date is not None:
        forecast_date_midnight = pd.to_datetime(forecast_date + " 00:00:00", format='%Y-%m-%d %H:%M:%S', errors='coerce')
        if forecast_date_midnight + pd.Timedelta(days=forecast_days) > original_full_agg['Time'].max():
            date_range = pd.date_range(start=original_full_agg['Time'].max() + pd.Timedelta(minutes=15), end=forecast_date_midnight + pd.Timedelta(days=forecast_days), freq='15min')
            df_ext = pd.DataFrame({'Time': date_range})
            original_full_agg = pd.concat([original_full_agg, df_ext], ignore_index=True)

    # Fill gaps
    print("Filling gaps in aggregated time series (max gap size: 10 points)...", flush=True)
    aggregated_data = fill_gaps(aggregated_data, tunnel_col=None, time_col='Time', max_gap_size=10,
                               columns_to_fill=['Capacity', 'Original', 'Trend', 'Seasonal', 'Residual'])
    print("Gap filling completed for aggregated data.", flush=True)

    # Normalize all traffic values to percentage of capacity
    print("Normalizing aggregated traffic values to percentage of total capacity (sum of all tunnels)...", flush=True)
    print(f"  Total Aggregated Capacity stats: mean={aggregated_data['Capacity'].mean():.2f} Gbps, min={aggregated_data['Capacity'].min():.2f} Gbps, max={aggregated_data['Capacity'].max():.2f} Gbps", flush=True)
    print(f"  Number of tunnels represented: sum of all tunnel capacities", flush=True)
    
    capacity_agg = aggregated_data['Capacity'].values
    aggregated_data['Original'] = np.where(
        (capacity_agg > 0) & ~np.isnan(aggregated_data['Original'].values),
        (aggregated_data['Original'].values / capacity_agg) * 100,
        np.nan
    )
    aggregated_data['Trend'] = np.where(
        (capacity_agg > 0) & ~np.isnan(aggregated_data['Trend'].values),
        (aggregated_data['Trend'].values / capacity_agg) * 100,
        np.nan
    )
    aggregated_data['Seasonal'] = np.where(
        (capacity_agg > 0) & ~np.isnan(aggregated_data['Seasonal'].values),
        (aggregated_data['Seasonal'].values / capacity_agg) * 100,
        np.nan
    )
    aggregated_data['Residual'] = np.where(
        (capacity_agg > 0) & ~np.isnan(aggregated_data['Residual'].values),
        (aggregated_data['Residual'].values / capacity_agg) * 100,
        np.nan
    )
    aggregated_data['Capacity'] = 100  # Capacity is now represented as 100%
    
    # Normalize original_full_agg as well
    capacity_orig = original_full_agg['Capacity'].values
    original_full_agg['Original'] = np.where(
        (capacity_orig > 0) & ~np.isnan(original_full_agg['Original'].values),
        (original_full_agg['Original'].values / capacity_orig) * 100,
        np.nan
    )
    original_full_agg['Capacity'] = 100
    
    print("Normalization completed for aggregated data using summed tunnel capacities.", flush=True)
    
    # Cut aggregated_data to training period (before forecast_date)
    if forecast_date is not None:
        forecast_date_midnight = pd.to_datetime(forecast_date + " 00:00:00", format='%Y-%m-%d %H:%M:%S', errors='coerce')
        aggregated_data = aggregated_data[aggregated_data['Time'] < forecast_date_midnight]
        # Cut original_full_agg to extended forecast period (includes forecast data for ground truth)
        original_full_agg = original_full_agg[(original_full_agg['Time'] > forecast_date_midnight) & 
                                              (original_full_agg['Time'] < forecast_date_midnight + pd.Timedelta(days=forecast_days))]

    # Add tunnel identifier
    aggregated_data['Tunnel'] = 'AGGREGATED_ALL_TUNNELS'
    aggregated_data['TunnelID'] = 0

    # Time in seconds since midnight
    seconds_in_day = 24 * 60 * 60
    time_seconds = (aggregated_data['Time'].dt.hour * 3600 +
                    aggregated_data['Time'].dt.minute * 60 +
                    aggregated_data['Time'].dt.second)

    # Daily cyclic features
    aggregated_data['DailySin'] = np.sin(2 * np.pi * time_seconds / seconds_in_day)
    aggregated_data['DailyCos'] = np.cos(2 * np.pi * time_seconds / seconds_in_day)

    # Weekly cycle
    day_of_week = aggregated_data['Time'].dt.dayofweek
    seconds_in_week = 7 * seconds_in_day
    time_seconds_week = day_of_week * seconds_in_day + time_seconds

    aggregated_data['WeeklySin'] = np.sin(2 * np.pi * time_seconds_week / seconds_in_week)
    aggregated_data['WeeklyCos'] = np.cos(2 * np.pi * time_seconds_week / seconds_in_week)

    return aggregated_data, original_full_agg

def print_aggregate_statistics(original_full, output_dir='./results/'):
    """
        Compute and print all aggregate statistics on the dataset, and save to CSV.
        
        Args:
            original_full (pd.DataFrame): DataFrame with tunnel data and capacity information
            output_dir (str): Directory to save statistics CSV (default: './results/')
        
        Returns:
            dict: Dictionary with all computed statistics
    """
    
    # Fixed thresholds: 70%, 80%, 90%
    thresholds = [70, 80, 90]
    if original_full.empty:
        print("Warning: original_full is empty, skipping statistics")
        return None
    
    print("\n" + "="*70)
    print("PRELIMINARY DATA ANALYSIS — AGGREGATE STATISTICS")
    print("="*70 + "\n")
    
    # Initialize stats dictionary
    stats_dict = {}

    # Mean and median utilization
    util_dict = compute_utilization_stats(original_full)
    if util_dict:
        print("1. MEDIAN UTILIZATION")
        print(f"   Median utilization: {util_dict['median']:.2f}%\n")

        # 2. 95th percentile utilization
        print("2. 95TH PERCENTILE UTILIZATION")
        print(f"   P95 utilization: {util_dict['p95']:.2f}%\n")
        stats_dict.update({
            'Utilization_Median_Percent': util_dict['median'],
            'Utilization_P95_Percent': util_dict['p95']
        })

    # Peak-to-Average Ratio
    par_dict = compute_par_stats()
    if par_dict:
        print("3. PEAK-TO-AVERAGE RATIO (PAR)")
        print(f"   PAR median:  {par_dict['median']:.2f}")
        print(f"   PAR range:   [{par_dict['min']:.2f}, {par_dict['max']:.2f}]\n")
        stats_dict.update({
            'PAR_Median': par_dict['median'],
            'PAR_Min': par_dict['min'],
            'PAR_Max': par_dict['max']
        })

    # Time slots over thresholds and at 100%
    if 'Original' in original_full.columns:
        n_total = len(original_full)
        
        # Compute stats for 100% first (shared across all thresholds)
        at_100 = (original_full['Original'] >= 100.0)
        n_100 = at_100.sum()
        frac_100 = n_100 / n_total if n_total > 0 else float('nan')
        
        # Print header for threshold analysis
        print("4. TIME SLOTS OVER DIFFERENT THRESHOLDS")
        
        # Process each threshold
        for threshold in sorted(thresholds):
            over_thresh = (original_full['Original'] > threshold)
            n_over = over_thresh.sum()
            frac_over = n_over / n_total if n_total > 0 else float('nan')
            
            print(f"   Over {threshold:.0f}%: {n_over:,} slots ({frac_over:.4f} of all 15-min slots)")
            
            # Add to stats dict
            stats_dict.update({
                f'Num_Slots_Over_{int(threshold)}': n_over,
                f'Frac_Slots_Over_{int(threshold)}': frac_over,
            })
        
        # Add 100% stats (same for all thresholds)
        print(f"   At 100%: {n_100:,} slots ({frac_100:.4f} of all 15-min slots)\n")
        stats_dict.update({
            'Num_Slots_At_100': n_100,
            'Frac_Slots_At_100': frac_100
        })

    print("="*70 + "\n")

    # Save statistics to CSV
    os.makedirs(output_dir, exist_ok=True)
    stats_df = pd.DataFrame([stats_dict])
    csv_path = os.path.join(output_dir, 'aggregate_statistics.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved aggregate statistics to {csv_path}\n")

    return stats_dict

def compute_utilization_stats(original_full):
    """
        Compute utilization statistics (median/P95) as percentage of capacity.
        
        Args:
            original_full (pd.DataFrame): DataFrame with 'Original' and 'Capacity' columns
            
        Returns:
            dict: Dictionary with 'median', 'p95' utilization percentages
    """
    if original_full.empty or 'Original' not in original_full.columns or 'Capacity' not in original_full.columns:
        return None
    
    utilization = (original_full['Original'] / original_full['Capacity']) * 100
    return {
        'median': utilization.median(),
        'p95': utilization.quantile(0.95)
    }

def compute_par_stats():
    """
        Compute Peak-to-Average Ratio (PAR) statistics for each tunnel using Max_Avg_Rate_Gbps.csv as avg and Max_Max_Rate_Gbps.csv for max values.
        For each 15-min time slot, compute PAR = max/avg, then average per link and overall.
        
        Args:
            original_full (pd.DataFrame): DataFrame with 'Tunnel' and 'Original' columns (not used for PAR here)
        Returns:
            dict: Dictionary with 'mean', 'median', 'min', 'max' PAR values across tunnels, or None if empty
    """
    # Load max and avg rate files
    max_file = os.path.join('..', 'data', 'cleaned', 'Max_Max_Rate_Gbps.csv')
    avg_file = os.path.join('..', 'data', 'cleaned', 'Max_Avg_Rate_Gbps.csv')
    if not (os.path.exists(max_file) and os.path.exists(avg_file)):
        print(f"PAR computation: Required files not found: {max_file}, {avg_file}")
        return None
    max_df = pd.read_csv(max_file)
    avg_df = pd.read_csv(avg_file)

    # Melt to long format
    max_long = max_df.melt(id_vars=['Time'], var_name='Tunnel', value_name='Max')
    avg_long = avg_df.melt(id_vars=['Time'], var_name='Tunnel', value_name='Avg')
    # Merge on Time and Tunnel
    merged = pd.merge(max_long, avg_long, on=['Time', 'Tunnel'], how='inner')
    # Compute PAR for each 15-min slot
    merged['PAR'] = merged.apply(lambda row: row['Max'] / row['Avg'] if row['Avg'] > 0 else float('nan'), axis=1)
    # Drop NaN
    merged = merged.dropna(subset=['PAR'])
    # Compute mean PAR per link
    par_by_tunnel = merged.groupby('Tunnel')['PAR'].mean()
    # Compute stats across links
    if not par_by_tunnel.empty:
        return {
            'mean': par_by_tunnel.mean(),
            'median': par_by_tunnel.median(),
            'min': par_by_tunnel.min(),
            'max': par_by_tunnel.max()
        }
    return None

def compute_single_kde(df, value_column='Residual', tunnel_column='Tunnel'):
    """
        Compute a single KDE distribution from all values in the training set for each tunnel.
        
        Args:
            df: dataframe with data
            value_column: column name for the value to compute KDE for
            tunnel_column: column name for tunnel names
            
        Returns:
            kde_distributions: dict of {tunnel_name: kde_object}
    """
    
    kde_distributions = {}
    
    for tunnel in df[tunnel_column].unique():
        tunnel_df = df[df[tunnel_column] == tunnel]
        all_values = tunnel_df[value_column].dropna().values
        
        try:
            if len(all_values) < 2:
                print(f"Insufficient data for tunnel '{tunnel}' to compute KDE")
                kde_distributions[tunnel] = None
            else:
                kde = gaussian_kde(all_values)
                kde_distributions[tunnel] = kde
        except Exception as e:
            print(f"KDE failed for tunnel '{tunnel}': {e}")
            kde_distributions[tunnel] = None
    
    return kde_distributions

def compute_kde_no_match(df, value_column='Residual', serie_a_col='Indicator_SerieA',
                         champions_col='Indicator_Champions', tunnel_column='Tunnel'):
    """
        Compute a KDE distribution for each tunnel using only timestamps with no matches.

        Args:
            df: dataframe with data
            value_column: column name for the value to compute KDE for
            serie_a_col: column name for Serie A indicator values
            champions_col: column name for Champions League indicator values
            tunnel_column: column name for tunnel names

        Returns:
            kde_distributions: dict of {tunnel_name: kde_object}
    """

    kde_distributions = {}
    no_match_df = df[(df[serie_a_col] == 0) & (df[champions_col] == 0)]

    for tunnel in df[tunnel_column].unique():
        tunnel_df = no_match_df[no_match_df[tunnel_column] == tunnel]
        all_values = tunnel_df[value_column].dropna().values

        try:
            kde = gaussian_kde(all_values)
            kde_distributions[tunnel] = kde
        except Exception as e:
            print(f"No-match KDE failed for tunnel '{tunnel}': {e}")
            kde_distributions[tunnel] = None

    return kde_distributions

def compute_kde_by_indicator(df, value_column='Residual', indicator_column='Indicator', tunnel_column='Tunnel'):
    """
        Compute KDE distributions for each tunnel and indicator value.
        
        Args:
            df: dataframe with data
            value_column: column name for the value to compute KDE for
            indicator_column: column name for indicator values
            tunnel_column: column name for tunnel names
            
        Returns:
            kde_distributions: dict of {tunnel_name: {indicator_value: kde_object}}
    """    

    # Filter indicators 1..10 for consistency
    df = df[df[indicator_column].isin(range(1, 11))].copy()

    kde_distributions = {}

    for tunnel in df[tunnel_column].unique():
        tunnel_df = df[df[tunnel_column] == tunnel]

        kde_distributions[tunnel] = {}
        for ind in sorted(tunnel_df[indicator_column].unique(), key=int):
            group_values = tunnel_df[tunnel_df[indicator_column] == ind][value_column].dropna().values

            try:
                kde = gaussian_kde(group_values)
                kde_distributions[tunnel][int(ind)] = kde
            except Exception as e:
                print(f"KDE failed for tunnel '{tunnel}', indicator {ind}: {e}")
                kde_distributions[tunnel][int(ind)] = None

    return kde_distributions

def get_closest_kde(kde_dict_tunnel, indicator_val):
    """
        Get KDE for the closest available indicator value.
        
        Args:
            kde_dict_tunnel: dict of {indicator_value: kde_object} for a single tunnel
            indicator_val: desired indicator value
        
        Returns:
            tuple: (kde_object, actual_indicator_used) or (None, None) if no KDEs available
    """
    
    # If exact match exists, return it
    if indicator_val in kde_dict_tunnel and kde_dict_tunnel[indicator_val] is not None:
        return kde_dict_tunnel[indicator_val], indicator_val
    
    # Find closest available indicator
    available_indicators = [ind for ind, kde in kde_dict_tunnel.items() if kde is not None]
    
    if not available_indicators:
        return None, None
    
    closest_ind = min(available_indicators, key=lambda x: abs(x - indicator_val))
    return kde_dict_tunnel[closest_ind], closest_ind

def sample_from_kde(kde, num_samples=1000, prediction_interval=98):
    """
        Sample from a KDE distribution and compute statistics.
        
        Args:
            kde: KDE object (from scipy.stats.gaussian_kde) or None
            num_samples: number of samples to draw (default: 100)
            prediction_interval: prediction interval for interval computation (default: 98 for 1st and 99th percentiles)
        
        Returns:
            tuple: (mean, q_low, q_high) computed from samples, or (0.0, 0.0, 0.0) if kde is None
    """
    
    samples = kde.resample(num_samples)[0]
    mean_val = np.mean(samples)
    q_low = np.percentile(samples, (100 - prediction_interval) / 2)
    q_high = np.percentile(samples, 100 - (100 - prediction_interval) / 2)
    return mean_val, q_low, q_high

def forecast_trend_carry_forward(train_values, forecast_times):
    """
        Forecast trend by repeating the last observed value across all forecast periods.
        Simple carry-forward approach (no extrapolation).
        
        Args:
            train_values: array of training values
            forecast_times: pandas Timestamps to forecast for
            
        Returns:
            tuple: (forecast_values, forecast_lower, forecast_upper) where all three are identical (flat projection)
    """
    
    train_values = np.asarray(train_values)
    
    # Remove NaNs and get last value
    mask = ~np.isnan(train_values)
    train_values_clean = train_values[mask]
    
    if len(train_values_clean) == 0:
        last_value = 0.0
    else:
        last_value = train_values_clean[-1]
    
    # Project last value forward
    forecast = np.full(len(forecast_times), last_value)
    forecast_lower = np.full(len(forecast_times), last_value)
    forecast_upper = np.full(len(forecast_times), last_value)
    
    print(f"Trend forecast: using last value carry-forward = {last_value:.4f}", flush=True)
    
    return forecast, forecast_lower, forecast_upper

def forecast_seasonal_naive(seasonal_values, seasonal_times, forecast_times, period=None):
    """
        Forecast seasonal component using Seasonal Naive method.
        Takes the last week of training data and repeats it cyclically for the entire forecast period.
        
        Args:
            seasonal_values: array of seasonal component values
            seasonal_times: corresponding timestamps for seasonal_values 
            forecast_times: timestamps to forecast for
            period: optional period in points (96 = 24 hours for 15-min data)
            recent_cycles: number of recent cycles (weeks) to search backward (default: 14 days, unused now)
            
        Returns:
            tuple: (forecast_values, forecast_lower, forecast_upper) where lower=upper=forecast (deterministic copy, no CI)
    """
    
    seasonal_values = np.asarray(seasonal_values)
    seasonal_times = np.asarray(seasonal_times)
    forecast_times = np.asarray(forecast_times)
    
    if len(seasonal_values) < 2:
        return np.zeros(len(forecast_times)), np.zeros(len(forecast_times)), np.zeros(len(forecast_times))
    
    if period is None:
        period = 96  # Default to daily cycle (96 points × 15 min = 24 hours)
    
    # Extract last week of seasonal values (7 days × 96 points/day = 672 points)
    week_length = 7 * period  # 672 points for 7 days
    last_week_seasonal = seasonal_values[-week_length:] if len(seasonal_values) >= week_length else seasonal_values
    
    print(f"SSA seasonal forecasting: Repeating last week pattern (period={period}, week_length={len(last_week_seasonal)} points)", flush=True)
    
    # Repeat the last week pattern cyclically to match forecast length
    forecast = np.tile(last_week_seasonal, int(np.ceil(len(forecast_times) / len(last_week_seasonal))))[:len(forecast_times)]
    
    # Return same value for lower/upper (no CI bands)
    return forecast

def generate_tunnel_forecast(train_data, past_day_data, forecast_date_midnight, forecast_end,
                            tunnel_name, kde_dict_serie_a, kde_dict_champions, single_kde, no_match_kde_dict,
                            use_single_kde, matches_serie_a_dict, matches_champions_dict, injection_periods,
                            forecast_days, output_dir, plot_tunels, plot_pdfcdf, inject_indicator_values, 
                            traffic_threshold, prediction_interval, original_full):
    """
        Generate forecast for a single tunnel.
        
        Args:
            train_data, past_day_data, future_data: training/past/future data for tunnel
            forecast_date_midnight, forecast_end: forecast date and end time
            tunnel_name: name of the tunnel
            kde_dict_serie_a, kde_dict_champions: KDE dicts for indicators
            single_kde: single KDE object if use_single_kde is True
            no_match_kde_dict: dict of {tunnel_name: kde_object} for no-match timestamps
            use_single_kde: whether to use single KDE or indicator-based KDEs
            matches_serie_a_dict, matches_champions_dict, matches_dict: indicator mappings
            injection_periods: list of (value, start, end) tuples for indicator injection
            forecast_days, output_dir, plot_tunels, plot_pdfcdf: parameters
            inject_indicator_values, traffic_threshold, prediction_interval: forecast parameters
            original_full: full original dataset for ground truth extraction in forecast period
        
        Returns:
            threshold_df: threshold exceeds dataframe or None
    """
    
    # Optionally plot PDF and CDF distributions
    if plot_pdfcdf:
        if use_single_kde:
            single_kde_dict = {tunnel_name: {0: single_kde}}
            pdf_output_dir = os.path.join(output_dir, 'PDF_SingleKDE')
            plot_pdf_cdf_by_indicator(single_kde_dict, train_data, value_column="Residual", output_dir=pdf_output_dir, 
                                     plot_tunnels=plot_tunels, forecast_date=forecast_date_midnight, 
                                     forecast_days=forecast_days, single_kde_mode=True)
        else:
            pdf_output_dir = os.path.join(output_dir, 'PDF_Combined')
            plot_pdf_cdf_by_indicator(kde_dict_serie_a, train_data, value_column="Residual", 
                                     indicator_column="Indicator_SerieA",
                                     second_indicator_column="Indicator_Champions",
                                     second_kde_distributions=kde_dict_champions,
                                     second_label="Champions League",
                                     output_dir=pdf_output_dir, 
                                     plot_tunnels=plot_tunels, forecast_date=forecast_date_midnight, forecast_days=forecast_days,
                                     no_match_cols=("Indicator_SerieA", "Indicator_Champions"))
    
    # Prepare past day data for plotting
    past_day_vals = past_day_data['Original'].values
    past_day_times = pd.to_datetime(past_day_data['Time'], dayfirst=True, errors='coerce')
    
    past_day_indicators_serie_a = np.array([matches_serie_a_dict.get(t, 0) for t in past_day_times])
    past_day_indicators_champions = np.array([matches_champions_dict.get(t, 0) for t in past_day_times])
    
    original_full = original_full[original_full['Tunnel'] == tunnel_name].copy()

    # Build forecast time range and ensure missing timestamps are present (so plotting shows full requested range)
    forecast_end = forecast_date_midnight + pd.Timedelta(days=forecast_days)

    freq = '15min'

    # Create full future times from forecast_date_midnight (inclusive) up to forecast_end (exclusive)
    rng = pd.date_range(start=forecast_date_midnight, end=forecast_end, freq=freq)
    full_future_times = rng[rng < forecast_end]

    # Prepare a dataframe with all future timestamps and merge available original values
    future_df_full = pd.DataFrame({'Time': full_future_times})
    tunnel_future = original_full[['Time', 'Original']].drop_duplicates()
    merged_future = future_df_full.merge(tunnel_future, on='Time', how='left')

    # Export arrays used by plotting code
    future_times = merged_future['Time'].values
    future_truth = merged_future['Original'].values

    # Build series-specific indicators using the matches dicts (fall back to 0)
    future_indicators_serie_a = np.array([matches_serie_a_dict.get(pd.Timestamp(t), 0) for t in future_times])
    future_indicators_champions = np.array([matches_champions_dict.get(pd.Timestamp(t), 0) for t in future_times])
    
    # Compute trend forecast using carry-forward (last value)
    try:
        train_trend_array = train_data['Trend'].values
        future_trend_forecast, future_trend_lower, future_trend_upper = forecast_trend_carry_forward(
            train_trend_array, future_times
        )
        # Ensure non-negative
        future_trend_forecast = np.maximum(future_trend_forecast, 0.0)
        future_trend_lower = np.maximum(future_trend_lower, 0.0)
        future_trend_upper = np.maximum(future_trend_upper, 0.0)
    except Exception as e:
        print(f"Warning: Trend forecasting failed ({e}), using last value carry forward")
        last_trend_value = max(0.0, float(train_data.iloc[-1].get('Trend', 0.0)))
        future_trend_forecast = np.full(len(future_times), last_trend_value)
        future_trend_lower = np.full(len(future_times), last_trend_value)
        future_trend_upper = np.full(len(future_times), last_trend_value)
    
    # Prepare plot training data (only last 30 days for visualization)
    plot_train_cutoff = forecast_date_midnight - pd.Timedelta(days=30)
    plot_train_mask = (train_data['Time'] >= plot_train_cutoff) & (train_data['Time'] < forecast_date_midnight)
    plot_train_data = train_data[plot_train_mask]
    
    # Compute seasonal forecast using Seasonal Naive (copy matching past pattern)
    past_seasonals = train_data['Seasonal'].values
    past_times = train_data['Time'].values
    future_seasonal_forecast = forecast_seasonal_naive(
        past_seasonals,
        past_times,
        future_times,
        period=96  # 96 15-min intervals = 24 hours (daily cycle)
    )
    
    # Plot trend and seasonality components with ground truth
    try:
        # Extract ground truth trend and seasonal from original_full if available
        tunnel_original_full = original_full[original_full['Tunnel'] == tunnel_name].copy()
        tunnel_original_full = tunnel_original_full.sort_values('Time')
        future_trend_gt = np.full(len(future_times), np.nan)
        future_seasonal_gt = np.full(len(future_times), np.nan)
        
        # Only extract if columns exist
        has_trend = 'Trend' in tunnel_original_full.columns
        has_seasonal = 'Seasonal' in tunnel_original_full.columns
        
        if has_trend or has_seasonal:
            for idx, time_t in enumerate(future_times):
                time_t_ts = pd.Timestamp(time_t)
                matching = tunnel_original_full[tunnel_original_full['Time'] == time_t_ts]
                if len(matching) > 0:
                    if has_trend:
                        future_trend_gt[idx] = matching.iloc[0]['Trend']
                    if has_seasonal:
                        future_seasonal_gt[idx] = matching.iloc[0]['Seasonal']
    except Exception as e:
        print(f"Warning: Failed to plot trend/seasonality components for {tunnel_name}: {e}", flush=True)
    
    # Apply indicator injection directly into the appropriate indicator array
    if injection_periods:
        for t, time_t in enumerate(future_times):
            time_t_ts = pd.Timestamp(time_t)
            for period_tuple in injection_periods:
                # Unpack 4-tuple (value, start, end, type)
                inject_value, inject_start, inject_end, inject_type = period_tuple

                if inject_start <= time_t_ts <= inject_end:
                    # Inject directly into the appropriate indicator array
                    if inject_type == 'serie_a':
                        future_indicators_serie_a[t] = int(inject_value)
                        print(f"Injecting Serie A indicator value {inject_value} at {time_t_ts} for {tunnel_name}", flush=True)
                    elif inject_type == 'champions':
                        future_indicators_champions[t] = int(inject_value)
                        print(f"Injecting Champions League indicator value {inject_value} at {time_t_ts} for {tunnel_name}", flush=True)
    
    # Generate forecasts
    forecast_mean = np.zeros(len(future_times))
    forecast_q_low = np.zeros(len(future_times))
    forecast_q_high = np.zeros(len(future_times))
    
    for t, _ in enumerate(future_times):
        time_t = future_times.iloc[t] if hasattr(future_times, 'iloc') else future_times[t]

        indicator_val_serie_a = int(future_indicators_serie_a[t])
        indicator_val_champions = int(future_indicators_champions[t])
        seasonal_val = float(future_seasonal_forecast[t])
        trend_val = future_trend_forecast[t]
        
        base_pred = trend_val + seasonal_val
        
        if use_single_kde:
            residual_mean, residual_q_low, residual_q_high = sample_from_kde(single_kde, num_samples=1000, prediction_interval=prediction_interval)
        else:
            no_match_kde = no_match_kde_dict.get(tunnel_name) if no_match_kde_dict else None

            if indicator_val_serie_a == 0 and indicator_val_champions == 0 and no_match_kde is not None:
                residual_mean, residual_q_low, residual_q_high = sample_from_kde(no_match_kde, num_samples=1000, prediction_interval=prediction_interval)
            elif indicator_val_champions > 0:
                kde_champions, _ = get_closest_kde(kde_dict_champions[tunnel_name], indicator_val_champions)
                residual_mean, residual_q_low, residual_q_high = sample_from_kde(kde_champions, num_samples=1000, prediction_interval=prediction_interval)
            else:
                kde_serie_a, _ = get_closest_kde(kde_dict_serie_a[tunnel_name], indicator_val_serie_a)
                residual_mean, residual_q_low, residual_q_high = sample_from_kde(kde_serie_a, num_samples=1000, prediction_interval=prediction_interval)
        
        forecast_mean[t] = np.clip(base_pred + residual_mean, 0, 100)
        forecast_q_low[t] = np.clip(base_pred + residual_q_low, 0, 100)
        forecast_q_high[t] = np.clip(base_pred + residual_q_high, 0, 100)
    
    # Check if this tunnel has large gaps in future data
    has_large_gap = detect_large_gaps_in_future_data(future_truth, max_consecutive_missing_days=3)
    
    # Plot forecast
    threshold_df, metrics_dict = plot_date_forecast(forecast_days, past_day_times, past_day_vals, future_times, future_truth, 
                                      forecast_mean, forecast_q_low, forecast_q_high,
                                      tunnel_name, forecast_date_midnight, output_dir,
                                      inject_indicator_values=inject_indicator_values,
                                      traffic_threshold=traffic_threshold, prediction_interval=prediction_interval,
                                      future_indicators_serie_a=future_indicators_serie_a, 
                                      future_indicators_champions=future_indicators_champions,
                                      past_indicators_serie_a=past_day_indicators_serie_a, 
                                      past_indicators_champions=past_day_indicators_champions,
                                      use_single_kde=use_single_kde,
                                      injection_periods=injection_periods)
    
    # Add large gap flag to metrics
    metrics_dict['Has_Large_Gap'] = has_large_gap
    
    return threshold_df, metrics_dict

def detect_large_gaps_in_future_data(future_truth, max_consecutive_missing_days=3):
    """
        Detect if future ground truth data has large gaps (more than max_consecutive_missing_days).
        A gap is defined as consecutive NaN values representing more than N days of missing data.
        
        Args:
            future_truth: array of future ground truth values (may contain NaNs)
            max_consecutive_missing_days: threshold for "large" gap (default: 3 days)
            
        Returns:
            bool: True if tunnel has a large gap (more than max_consecutive_missing_days with no data), False otherwise
    """
    
    # 15-minute data: 96 points per day
    points_per_day = 96
    max_consecutive_points = max_consecutive_missing_days * points_per_day
    
    # Find NaN mask
    nan_mask = np.isnan(future_truth)
    
    # If no NaNs, no large gaps
    if not np.any(nan_mask):
        return False
    
    # Find consecutive NaN sequences
    # Pad with False to detect first and last sequences
    nan_padded = np.concatenate(([False], nan_mask, [False]))
    transitions = np.diff(nan_padded.astype(int))
    gap_starts = np.where(transitions == 1)[0]  # Where NaN sequences start
    gap_ends = np.where(transitions == -1)[0]    # Where NaN sequences end
    
    # Check if any gap exceeds the threshold
    for start, end in zip(gap_starts, gap_ends):
        gap_length = end - start
        if gap_length > max_consecutive_points:
            return True  # Found a large gap
    
    return False  # No large gaps

def generate_aggregated_forecast(agg_train_data, agg_past_day_data, agg_future_data, forecast_date_midnight, forecast_end,
                                 kde_dict_serie_a_agg, kde_dict_champions_agg, single_kde_agg, no_match_kde_agg,
                                 use_single_kde,
                                 matches_serie_a_dict, matches_champions_dict, injection_periods,
                                 forecast_days, output_dir, inject_indicator_values, traffic_threshold, prediction_interval, original_full_agg):
    """
        Generate aggregated forecast across all tunnels.
        
        Args:
            agg_train_data, agg_past_day_data, agg_future_data: aggregated training/past/future data
            forecast_date_midnight, forecast_end: forecast date and end time
            kde_dict_serie_a_agg, kde_dict_champions_agg: KDE dicts for indicators
            single_kde_agg: single KDE object if use_single_kde is True
            no_match_kde_agg: KDE object computed on no-match timestamps
            use_single_kde: whether to use single KDE or indicator-based KDEs
            matches_serie_a_dict, matches_champions_dict, matches_dict: indicator mappings
            injection_periods: list of (value, start, end) tuples for indicator injection
            forecast_days: number of days to forecast
            output_dir: directory to save output files
            inject_indicator_values: list of indicator values to inject
            traffic_threshold: traffic threshold for exceedance
            prediction_interval: prediction interval for forecast
            original_full_agg: full aggregated data including forecast period (for ground truth extraction)
        
        Returns:
            threshold_df: threshold exceeds dataframe or None
    """
    
    # Prepare past day data for plotting
    past_day_vals_agg = agg_past_day_data['Original'].values
    past_day_times_agg = pd.to_datetime(agg_past_day_data['Time'], dayfirst=True, errors='coerce')
    
    past_day_indicators_serie_a_agg = np.array([matches_serie_a_dict.get(t, 0) for t in past_day_times_agg])
    past_day_indicators_champions_agg = np.array([matches_champions_dict.get(t, 0) for t in past_day_times_agg])
    
    # Build future timestamps and extract ground truth from original_full_agg
    freq = '15min'
    full_future_times = pd.date_range(start=forecast_date_midnight, end=forecast_end, freq=freq)
    full_future_times = full_future_times[full_future_times < forecast_end]
    
    # Create future dataframe with all timestamps and merge available original values from original_full_agg
    future_df_full = pd.DataFrame({'Time': full_future_times})
    agg_future = original_full_agg[['Time', 'Original']].drop_duplicates()
    merged_future = future_df_full.merge(agg_future, on='Time', how='left')
    
    # Export arrays used by plotting code
    future_times_agg = merged_future['Time'].values
    future_truth_agg = merged_future['Original'].values
    
    future_indicators_serie_a_agg = np.array([matches_serie_a_dict.get(pd.Timestamp(t), 0) for t in future_times_agg])
    future_indicators_champions_agg = np.array([matches_champions_dict.get(pd.Timestamp(t), 0) for t in future_times_agg])

    # Compute trend forecast using carry-forward (last value)
    try:
        train_trend_array_agg = agg_train_data['Trend'].values
        future_trend_forecast_agg, future_trend_lower_agg, future_trend_upper_agg = forecast_trend_carry_forward(
            train_trend_array_agg, future_times_agg
        )
        # Ensure non-negative
        future_trend_forecast_agg = np.maximum(future_trend_forecast_agg, 0.0)
        future_trend_lower_agg = np.maximum(future_trend_lower_agg, 0.0)
        future_trend_upper_agg = np.maximum(future_trend_upper_agg, 0.0)
    except Exception as e:
        print(f"Warning: Trend forecasting failed for aggregated ({e}), using last value carry forward")
        last_trend_value_agg = max(0.0, float(agg_train_data.iloc[-1].get('Trend', 0.0)))
        future_trend_forecast_agg = np.full(len(future_times_agg), last_trend_value_agg)
        future_trend_lower_agg = np.full(len(future_times_agg), last_trend_value_agg)
        future_trend_upper_agg = np.full(len(future_times_agg), last_trend_value_agg)
    
    # Compute seasonal forecast using Seasonal Naive (copy matching past pattern)
    past_seasonals_agg = agg_train_data['Seasonal'].values
    past_times_agg = agg_train_data['Time'].values
    future_seasonal_forecast_agg = forecast_seasonal_naive(
        past_seasonals_agg,
        past_times_agg,
        future_times_agg,
        period=96  # 96 15-min intervals = 24 hours (daily cycle)
    )
    
    # Prepare plot training data (only last 30 days for visualization)
    forecast_date_midnight_agg = forecast_date_midnight if 'forecast_date_midnight' in locals() else pd.Timestamp.now().normalize()
    plot_train_cutoff_agg = forecast_date_midnight_agg - pd.Timedelta(days=30)
    plot_train_mask_agg = (agg_train_data['Time'] >= plot_train_cutoff_agg) & (agg_train_data['Time'] < forecast_date_midnight_agg)
    
    # Plot trend and seasonality components with ground truth for aggregated data
    try:
        # Extract ground truth trend and seasonal from original_full_agg if available
        original_full_agg_sorted = original_full_agg.sort_values('Time')
        future_trend_gt_agg = np.full(len(future_times_agg), np.nan)
        future_seasonal_gt_agg = np.full(len(future_times_agg), np.nan)
        
        # Only extract if columns exist
        has_trend = 'Trend' in original_full_agg.columns
        has_seasonal = 'Seasonal' in original_full_agg.columns
        
        if has_trend or has_seasonal:
            for idx, time_t in enumerate(future_times_agg):
                time_t_ts = pd.Timestamp(time_t)
                matching = original_full_agg_sorted[original_full_agg_sorted['Time'] == time_t_ts]
                if len(matching) > 0:
                    if has_trend:
                        future_trend_gt_agg[idx] = matching.iloc[0]['Trend']
                    if has_seasonal:
                        future_seasonal_gt_agg[idx] = matching.iloc[0]['Seasonal']
    except Exception as e:
        print(f"Warning: Failed to plot trend/seasonality components for AGGREGATED_ALL_TUNNELS: {e}", flush=True)
    
    # Apply indicator injection directly into the appropriate indicator array
    if injection_periods:
        for t, time_t in enumerate(future_times_agg):
            time_t_ts = pd.Timestamp(time_t)
            for period_tuple in injection_periods:
                # Unpack 4-tuple (value, start, end, type)
                if len(period_tuple) == 4:
                    inject_value, inject_start, inject_end, inject_type = period_tuple
                else:
                    inject_value, inject_start, inject_end = period_tuple
                    inject_type = 'combined'
                
                if inject_start <= time_t_ts <= inject_end:
                    # Inject directly into the appropriate indicator array
                    if inject_type == 'serie_a':
                        future_indicators_serie_a_agg[t] = int(inject_value)
                        print(f"Injecting Serie A indicator value {inject_value} at {time_t_ts} for AGGREGATED_ALL_TUNNELS", flush=True)
                    elif inject_type == 'champions':
                        future_indicators_champions_agg[t] = int(inject_value)
                        print(f"Injecting Champions League indicator value {inject_value} at {time_t_ts} for AGGREGATED_ALL_TUNNELS", flush=True)
                    break
    
    # Generate forecasts
    forecast_mean_agg = np.zeros(len(future_times_agg))
    forecast_q_low_agg = np.zeros(len(future_times_agg))
    forecast_q_high_agg = np.zeros(len(future_times_agg))
    
    for t, _ in enumerate(future_times_agg):
        time_t = future_times_agg.iloc[t] if hasattr(future_times_agg, 'iloc') else future_times_agg[t]
        has_data_at_t = len(agg_future_data) > 0 and (agg_future_data['Time'] == time_t).any()
        
        if has_data_at_t:
            row = agg_future_data[agg_future_data['Time'] == time_t].iloc[0]
            seasonal_val = float(row.get('Seasonal', 0))
            trend_val = float(row.get('Trend', 0))
        else:
            seasonal_val = float(future_seasonal_forecast_agg[t])
            trend_val = future_trend_forecast_agg[t]
        
        base_pred = trend_val + seasonal_val
        
        if use_single_kde:
            residual_mean, residual_q_low, residual_q_high = sample_from_kde(single_kde_agg, num_samples=1000, prediction_interval=prediction_interval)
        else:
            indicator_val_serie_a = int(future_indicators_serie_a_agg[t]) if has_data_at_t else int(future_indicators_serie_a_agg[t])
            indicator_val_champions = int(future_indicators_champions_agg[t]) if has_data_at_t else int(future_indicators_champions_agg[t])
            
            if has_data_at_t:
                row = agg_future_data[agg_future_data['Time'] == time_t].iloc[0]
                indicator_val_serie_a = int(np.round(float(row.get('Indicator_SerieA', 0))))
                indicator_val_champions = int(np.round(float(row.get('Indicator_Champions', 0))))

            if indicator_val_serie_a == 0 and indicator_val_champions == 0 and no_match_kde_agg is not None:
                residual_mean, residual_q_low, residual_q_high = sample_from_kde(no_match_kde_agg, num_samples=1000, prediction_interval=prediction_interval)
            elif indicator_val_champions > 0:
                kde_champions, _ = get_closest_kde(kde_dict_champions_agg['AGGREGATED_ALL_TUNNELS'], indicator_val_champions)
                residual_mean, residual_q_low, residual_q_high = sample_from_kde(kde_champions, num_samples=1000, prediction_interval=prediction_interval)
            else:
                kde_serie_a, _ = get_closest_kde(kde_dict_serie_a_agg['AGGREGATED_ALL_TUNNELS'], indicator_val_serie_a)
                residual_mean, residual_q_low, residual_q_high = sample_from_kde(kde_serie_a, num_samples=1000, prediction_interval=prediction_interval)
        
        forecast_mean_agg[t] = np.clip(base_pred + residual_mean, 0, 100)
        forecast_q_low_agg[t] = np.clip(base_pred + residual_q_low, 0, 100)
        forecast_q_high_agg[t] = np.clip(base_pred + residual_q_high, 0, 100)
    
    # Plot aggregated forecast
    threshold_df_agg, metrics_dict_agg = plot_date_forecast(forecast_days, past_day_times_agg, past_day_vals_agg, future_times_agg, future_truth_agg, 
                                           forecast_mean_agg, forecast_q_low_agg, forecast_q_high_agg,
                                           'AGGREGATED_ALL_TUNNELS', forecast_date_midnight, output_dir,
                                           inject_indicator_values=inject_indicator_values,
                                           traffic_threshold=traffic_threshold, prediction_interval=prediction_interval,
                                           future_indicators_serie_a=future_indicators_serie_a_agg, future_indicators_champions=future_indicators_champions_agg,
                                           past_indicators_serie_a=past_day_indicators_serie_a_agg, past_indicators_champions=past_day_indicators_champions_agg,
                                           use_single_kde=use_single_kde,
                                           injection_periods=injection_periods)
    
    return threshold_df_agg, metrics_dict_agg

def perform_aggregated_analysis(aggregated_data, forecast_date_midnight, forecast_end, matches_serie_a_dict, matches_champions_dict, 
                                matches_dict, use_single_kde, plot_pdfcdf, plot_tunels, output_dir, inject_indicator_values, 
                                injection_periods, traffic_threshold, prediction_interval, results_dir, forecast_days, original_full_agg):
    """
        Perform aggregated analysis across all tunnels and save results.
        
        Args:
            aggregated_data: aggregated data for all tunnels
            forecast_date_midnight: midnight of forecast date
            forecast_end: end time of forecast period
            matches_serie_a_dict: dictionary mapping time to Serie A indicator values
            matches_champions_dict: dictionary mapping time to Champions League indicator values
            matches_dict: combined matches dictionary
            use_single_kde: whether to use single KDE or indicator-based KDEs
            plot_pdfcdf: whether to plot PDF/CDF distributions
            plot_tunels: whether to plot tunnels
            output_dir: output directory for plots
            inject_indicator_values: list of indicator values to inject
            injection_periods: list of injection periods
            traffic_threshold: traffic threshold for exceedance
            prediction_interval: prediction interval for forecast
            results_dir: directory to save results
            forecast_days: number of days to forecast
            original_full_agg: full aggregated data including forecast period (for ground truth extraction)
        
        Returns:
            tuple: (list of threshold dataframes, list of metrics dicts from aggregated analysis)
    """
    
    all_threshold_exceeds_agg = []
    all_metrics_agg = []
    
    print("\n" + "="*80)
    print(f"Starting aggregated analysis (all tunnels summed) for {forecast_date_midnight.strftime('%Y-%m-%d')}")
    print("="*80)
    
    agg_start_time = time.time()
    
    # Add indicator mappings to aggregated data
    aggregated_data['Indicator_SerieA'] = aggregated_data['Time'].map(matches_serie_a_dict).fillna(0).astype(int)
    aggregated_data['Indicator_Champions'] = aggregated_data['Time'].map(matches_champions_dict).fillna(0).astype(int)
    
    train_mask = aggregated_data['Time'] < forecast_date_midnight
    future_mask = (aggregated_data['Time'] >= forecast_date_midnight) & (aggregated_data['Time'] < forecast_end)
    past_day_mask = (aggregated_data['Time'] >= forecast_date_midnight - pd.Timedelta(days=1)) & (aggregated_data['Time'] < forecast_date_midnight)
    
    agg_train_data = aggregated_data[train_mask]
    agg_future_data = aggregated_data[future_mask]
    agg_past_day_data = aggregated_data[past_day_mask]
    
    if len(agg_train_data) >= 96:
        threshold_df_agg = None
        
        if use_single_kde:
            print("Using single KDE from all residuals for AGGREGATED_ALL_TUNNELS")
            kde_dict = compute_single_kde(agg_train_data, value_column="Residual")
            if 'AGGREGATED_ALL_TUNNELS' in kde_dict and kde_dict['AGGREGATED_ALL_TUNNELS'] is not None:
                single_kde_agg = kde_dict['AGGREGATED_ALL_TUNNELS']
                no_match_kde_agg = None
                
                # Optionally plot PDF and CDF distributions for aggregated data
                if plot_pdfcdf:
                    single_kde_dict = {'AGGREGATED_ALL_TUNNELS': {0: single_kde_agg}}
                    pdf_output_dir = os.path.join(output_dir, 'PDF_SingleKDE')
                    plot_pdf_cdf_by_indicator(single_kde_dict, agg_train_data, value_column="Residual", output_dir=pdf_output_dir, 
                                             plot_tunnels=plot_tunels, forecast_date=forecast_date_midnight, 
                                             forecast_days=forecast_days, single_kde_mode=True)
                
                threshold_df_agg, metrics_dict_agg = generate_aggregated_forecast(
                    agg_train_data, agg_past_day_data, agg_future_data, forecast_date_midnight, forecast_end,
                    None, None, single_kde_agg, no_match_kde_agg, use_single_kde,
                    matches_serie_a_dict, matches_champions_dict, matches_dict, injection_periods,
                    forecast_days, output_dir, inject_indicator_values, traffic_threshold, prediction_interval, original_full_agg
                )
                
                if threshold_df_agg is not None:
                    all_threshold_exceeds_agg.append(threshold_df_agg)
                if metrics_dict_agg is not None:
                    all_metrics_agg.append(metrics_dict_agg)
            else:
                print(f"Skipping aggregated on {forecast_date_midnight}: KDE generation failed")
        else:
            kde_dict_serie_a_agg = compute_kde_by_indicator(agg_train_data, value_column="Residual", indicator_column="Indicator_SerieA")
            kde_dict_champions_agg = compute_kde_by_indicator(agg_train_data, value_column="Residual", indicator_column="Indicator_Champions")
            no_match_kde_agg = compute_kde_no_match(
                agg_train_data,
                value_column="Residual",
                serie_a_col="Indicator_SerieA",
                champions_col="Indicator_Champions"
            ).get('AGGREGATED_ALL_TUNNELS')
            
            if 'AGGREGATED_ALL_TUNNELS' in kde_dict_serie_a_agg and 'AGGREGATED_ALL_TUNNELS' in kde_dict_champions_agg:
                # Optionally plot PDF and CDF distributions for aggregated data (combined like single tunnels)
                if plot_pdfcdf:
                    pdf_output_dir = os.path.join(output_dir, 'PDF_Combined')
                    plot_pdf_cdf_by_indicator(kde_dict_serie_a_agg, agg_train_data, value_column="Residual", 
                                             indicator_column="Indicator_SerieA",
                                             second_indicator_column="Indicator_Champions",
                                             second_kde_distributions=kde_dict_champions_agg,
                                             second_label="Champions League",
                                             output_dir=pdf_output_dir, 
                                             plot_tunnels=plot_tunels, forecast_date=forecast_date_midnight, forecast_days=forecast_days,
                                             no_match_cols=("Indicator_SerieA", "Indicator_Champions"))
                
                threshold_df_agg, metrics_dict_agg = generate_aggregated_forecast(
                    agg_train_data, agg_past_day_data, agg_future_data, forecast_date_midnight, forecast_end,
                    kde_dict_serie_a_agg, kde_dict_champions_agg, None, no_match_kde_agg, use_single_kde,
                    matches_serie_a_dict, matches_champions_dict, matches_dict, injection_periods,
                    forecast_days, output_dir, inject_indicator_values, traffic_threshold, prediction_interval, original_full_agg
                )
                
                if threshold_df_agg is not None:
                    all_threshold_exceeds_agg.append(threshold_df_agg)
                if metrics_dict_agg is not None:
                    all_metrics_agg.append(metrics_dict_agg)
        
        agg_elapsed_time = time.time() - agg_start_time
        print(f"Completed aggregated forecast on {forecast_date_midnight.strftime('%Y-%m-%d')} in {agg_elapsed_time:.2f} seconds", flush=True)
    else:
        print(f"Skipping aggregated on {forecast_date_midnight}: insufficient training data")
    
    # Save combined threshold exceeds CSV for all tunnels
    if all_threshold_exceeds_agg:
        df_combined = pd.concat(all_threshold_exceeds_agg, ignore_index=True)
        
        # Create results directory and threshold subdirectory
        os.makedirs(results_dir, exist_ok=True)
        
        if traffic_threshold is not None:
            thresh_dir = os.path.join(results_dir, f"Thresh_{traffic_threshold:.0f}_Perc{prediction_interval}")
            os.makedirs(thresh_dir, exist_ok=True)
        else:
            thresh_dir = results_dir
        
        # Build inject suffix for filename
        inject_suffix = f"_inject_{'_'.join(map(str, inject_indicator_values))}" if inject_indicator_values else ""
        threshold_exceeds_path = os.path.join(thresh_dir, f"threshold_exceeds_{forecast_date_midnight.strftime('%Y-%m-%d')}_{forecast_days}d{inject_suffix}.csv")
        df_combined.to_csv(threshold_exceeds_path, index=False)
        print(f"Saved combined threshold exceeds to {threshold_exceeds_path}", flush=True)
    
    return all_threshold_exceeds_agg, all_metrics_agg

def generate_forecasts_from_dates(data, aggregated_data, matches, matches_champions, forecast_date, forecast_days, original_full, original_full_agg, plot_tunels=False, output_dir='./plots/', 
                                   inject_indicator_values=None, inject_indicator_periods=None, inject_indicator_types=None, traffic_threshold=None, prediction_interval=98, plot_pdfcdf=False, results_dir='./results/', use_single_kde=False):
    """
        Generate forecasts from specific dates.
        
        Args:
            data: preprocessed data
            matches: matches dataframe with indicator values (Serie A)
            matches_champions: matches dataframe with indicator values (Champions League)
            forecast_date: list of dates to forecast from
            forecast_days: number of days to forecast
            output_dir: directory to save plots
            inject_indicator_values: list of values to inject (e.g., [5, 3, 7])
            inject_indicator_periods: list of period boundaries (e.g., [start1, end1, start2, end2, ...])
            inject_indicator_types: list of injection types (optional, deprecated, types now in injection_periods)
            traffic_threshold: traffic threshold for plotting (optional)
            prediction_interval: prediction interval for forecast (default: 98)
            use_single_kde: if True, use single KDE from all residuals; if False, use indicator-based KDEs from combined matches
            original_full: full original time series data for ground truth in forecast period
    """

    # Parse multiple injection periods if provided
    injection_periods = []
    if inject_indicator_values is not None and inject_indicator_periods is not None:
        # Ensure we have matching pairs of values and periods
        num_values = len(inject_indicator_values)
        num_period_parts = len(inject_indicator_periods)
        
        if num_period_parts != num_values * 2:
            print(f"Warning: Number of period boundaries ({num_period_parts}) must be 2x the number of values ({num_values})")
        else:
            # Parse periods into (start, end) tuples paired with values and types
            for i in range(num_values):
                start_str = inject_indicator_periods[i * 2]
                end_str = inject_indicator_periods[i * 2 + 1]
                value = inject_indicator_values[i]
                injection_type = inject_indicator_types[i] if inject_indicator_types and i < len(inject_indicator_types) else 'combined'
                
                start_ts = pd.Timestamp(start_str)
                end_ts = pd.Timestamp(end_str)
                injection_periods.append((value, start_ts, end_ts, injection_type))
                print(f"Parsed injection period {i+1}: value={value}, type={injection_type}, period=[{start_ts}, {end_ts}]", flush=True)
    
    data['Time'] = pd.to_datetime(data['Time'], dayfirst=True, errors='coerce')
    
    # Load matches separately for future indicator values
    matches['Time'] = pd.to_datetime(matches['Time'], dayfirst=True, errors='coerce')
    matches['Indicator'] = pd.to_numeric(matches['Indicator'])
    matches_champions['Time'] = pd.to_datetime(matches_champions['Time'], dayfirst=True, errors='coerce')
    matches_champions['Indicator'] = pd.to_numeric(matches_champions['Indicator'])
    
    # Create separate dictionaries for Serie A and Champions League
    matches_serie_a_dict = dict(zip(matches['Time'], matches['Indicator']))
    matches_champions_dict = dict(zip(matches_champions['Time'], matches_champions['Indicator']))
    
    # When use_single_kde is False, combine both Serie A and Champions League indicators
    # by taking the maximum indicator value at each timestamp (both active = higher indicator)
    if not use_single_kde:
        # Merge both matches on Time, taking max indicator value
        combined_matches = pd.merge(matches, matches_champions, on='Time', how='outer', suffixes=('_serie_a', '_champions'))
        combined_matches['Indicator'] = combined_matches[['Indicator_serie_a', 'Indicator_champions']].max(axis=1)
        matches_dict = dict(zip(combined_matches['Time'], combined_matches['Indicator']))
        print("Using combined Serie A and Champions League indicators for forecasting")
    else:
        # For single KDE, use Serie A indicators only (they will be ignored anyway)
        matches_dict = dict(zip(matches['Time'], matches['Indicator']))
    
    forecast_date = pd.Timestamp(forecast_date)
    
    forecast_date_midnight = forecast_date.normalize()
    forecast_end = forecast_date_midnight + pd.Timedelta(days=forecast_days)

    if forecast_date_midnight < data['Time'].min() + pd.Timedelta(days=1) or forecast_date_midnight > data['Time'].max():
        print(f"Skipping date {forecast_date_midnight}: out of data range")
        return
    
    # Accumulate threshold exceeds from all tunnels
    all_threshold_exceeds = []
    all_metrics = []
    
    for tunnel_name in data['Tunnel'].unique():
        tunnel_start_time = time.time()
        
        tunnel_data = data[data['Tunnel'] == tunnel_name].sort_values('Time').reset_index(drop=True)
        
        # Add separate Serie A and Champions League indicator columns
        tunnel_data['Indicator_SerieA'] = tunnel_data['Time'].map(matches_serie_a_dict).fillna(0).astype(int)
        tunnel_data['Indicator_Champions'] = tunnel_data['Time'].map(matches_champions_dict).fillna(0).astype(int)

        train_mask = tunnel_data['Time'] < forecast_date_midnight
        past_day_mask = (tunnel_data['Time'] >= forecast_date_midnight - pd.Timedelta(days=1)) & (tunnel_data['Time'] < forecast_date_midnight)
        
        train_data = tunnel_data[train_mask]
        past_day_data = tunnel_data[past_day_mask]

        if len(train_data) < 96:
            print(f"Skipping {tunnel_name} on {forecast_date_midnight}: insufficient training or past data")
            continue
        
        # Compute KDE based on selected mode
        if use_single_kde:
            print(f"Using single KDE from all residuals for {tunnel_name}")
            kde_dict = compute_single_kde(train_data, value_column="Residual")
            single_kde = kde_dict[tunnel_name]
            kde_dict_serie_a = None
            kde_dict_champions = None
            no_match_kde_dict = None
        else:
            # Compute KDE for the "no-match" case first (both indicators == 0).
            # Only compute per-indicator KDEs if there are non-zero indicator values
            # in the training data for this tunnel to avoid unnecessary work.
            no_match_kde_dict = compute_kde_no_match(
                train_data,
                value_column="Residual",
                serie_a_col="Indicator_SerieA",
                champions_col="Indicator_Champions"
            )

            # Default empty per-indicator dicts for this tunnel
            kde_dict_serie_a = {tunnel_name: {}}
            kde_dict_champions = {tunnel_name: {}}

            has_serie_a_events = train_data['Indicator_SerieA'].dropna().gt(0).any()
            has_champions_events = train_data['Indicator_Champions'].dropna().gt(0).any()

            if has_serie_a_events:
                kde_dict_serie_a = compute_kde_by_indicator(train_data, value_column="Residual", indicator_column="Indicator_SerieA")

            if has_champions_events:
                kde_dict_champions = compute_kde_by_indicator(train_data, value_column="Residual", indicator_column="Indicator_Champions")

            # Validate that we have at least one usable KDE: either a no-match KDE or
            # some per-indicator KDEs for this tunnel. If none are available, skip.
            no_match_available = (tunnel_name in no_match_kde_dict and no_match_kde_dict[tunnel_name] is not None)
            serie_a_available = tunnel_name in kde_dict_serie_a and any(k is not None for k in kde_dict_serie_a[tunnel_name].values())
            champions_available = tunnel_name in kde_dict_champions and any(k is not None for k in kde_dict_champions[tunnel_name].values())

            if not (no_match_available or serie_a_available or champions_available):
                print(f"Skipping {tunnel_name} on {forecast_date_midnight}: KDE generation failed or insufficient data")
                continue

            single_kde = None
        
        # Generate tunnel forecast
        threshold_df, metrics_dict = generate_tunnel_forecast(
            train_data, past_day_data, forecast_date_midnight, forecast_end,
            tunnel_name, kde_dict_serie_a, kde_dict_champions, single_kde, no_match_kde_dict,
            use_single_kde,
            matches_serie_a_dict, matches_champions_dict, injection_periods,
            forecast_days, output_dir, plot_tunels, plot_pdfcdf, inject_indicator_values,
            traffic_threshold, prediction_interval, original_full
        )
        
        # Accumulate threshold exceeds and metrics from each tunnel
        if threshold_df is not None:
            all_threshold_exceeds.append(threshold_df)
        
        if metrics_dict is not None:
            all_metrics.append(metrics_dict)
        
        tunnel_elapsed_time = time.time() - tunnel_start_time
        print(f"Completed forecast for {tunnel_name} on {forecast_date_midnight.strftime('%Y-%m-%d')} in {tunnel_elapsed_time:.2f} seconds", flush=True)
    
    # Perform aggregated analysis across all tunnels
    agg_threshold_exceeds, agg_metrics = perform_aggregated_analysis(
        aggregated_data, forecast_date_midnight, forecast_end,
        matches_serie_a_dict, matches_champions_dict, matches_dict,
        use_single_kde, plot_pdfcdf, plot_tunels, output_dir,
        inject_indicator_values, injection_periods, traffic_threshold,
        prediction_interval, results_dir, forecast_days, original_full_agg
    )
    
    # Append aggregated results to main threshold exceeds and metrics lists
    all_threshold_exceeds.extend(agg_threshold_exceeds)
    all_metrics.extend(agg_metrics)
    
    # Aggregate and save metrics
    if all_metrics:
        print("\n" + "="*80)
        print("Aggregating and saving PICP/PINAW metrics...")
        print("="*80)
        aggregate_forecast_metrics(all_metrics, output_dir=results_dir, prediction_interval=prediction_interval, 
                                   use_single_kde=use_single_kde, forecast_date=forecast_date, 
                                   forecast_days=forecast_days)
    else:
        print("No metrics collected for this forecast")