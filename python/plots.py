'''
  Tunnel data prediction

  plots.py

  This script provides functions for plotting tunnel data, including
  original, trend, seasonal, and residual components, as well as date
  forecasts with prediction intervals and indicators.

  Authors: Daniele Baccega
'''

import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from plotnine import ggplot, aes, geom_step, labs, theme_bw, scale_fill_manual, guides, scale_color_manual, geom_line, ylim, theme, element_text, geom_violin, guide_legend, facet_wrap, scale_x_discrete, geom_boxplot, scale_x_datetime
from scipy.stats import ks_2samp, mannwhitneyu, normaltest, f_oneway


# Custom theme with larger fonts for better readability
theme_large_fonts = theme_bw() + theme(
    plot_title=element_text(size=20, face='bold'),
    axis_title=element_text(size=18),
    axis_text=element_text(size=16),
    legend_title=element_text(size=18),
    legend_text=element_text(size=16)
)

def plot_tunnels(df, tunnel_column='Tunnel', output_dir='./plots/', start_date=None, end_date=None):
    """
        Plot tunnel data for a specified period - Original, Trend, Seasonal, Residual components.

        Args:
            df: DataFrame with tunnel data
            tunnel_column: column name for tunnel identifiers
            output_dir: directory to save plots
            start_date: start date for filtering (string or pd.Timestamp). If None, uses all data from beginning.
            end_date: end date for filtering (string or pd.Timestamp). If None, uses all data until the end.
    """

    # Filter data based on provided date range
    df_filtered = df.copy()
    
    # Convert start_date and end_date to pd.Timestamp if provided as strings
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        df_filtered = df_filtered[df_filtered['Time'] >= start_date]
    
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        df_filtered = df_filtered[df_filtered['Time'] <= end_date]
    
    if df_filtered.empty:
        print(f"No data found for the specified period")
        return
    
    # Determine if data spans a single year and set x-axis date format accordingly
    df_filtered['Time'] = pd.to_datetime(df_filtered['Time'])
    min_year = df_filtered['Time'].dt.year.min()
    max_year = df_filtered['Time'].dt.year.max()
    single_year = (min_year == max_year)
    date_format = '%m-%d' if single_year else '%Y-%m-%d'
    
    # Create a description of the period for logging
    if start_date is not None and end_date is not None:
        period_desc = f"from {start_date.date()} to {end_date.date()}"
    elif start_date is not None:
        period_desc = f"from {start_date.date()} onwards"
    elif end_date is not None:
        period_desc = f"until {end_date.date()}"
    else:
        period_desc = "for all available data"
    
    print(f"Generating plots for {df_filtered[tunnel_column].nunique()} tunnels {period_desc}...")
    
    # Plot information for each tunnel for the specified period
    for tunnel in df_filtered[tunnel_column].unique():
        tunnel_df = df_filtered[df_filtered[tunnel_column] == tunnel].copy()

        if tunnel_df.empty:
            continue

        plot_original_signal = (
            ggplot(tunnel_df)
            + geom_line(aes(x='Time', y='Original'), color="black")
            + labs(x='', y='Traffic (% of capacity)')
            + ylim(0, 101)
            + scale_x_datetime(date_labels=date_format)
            + theme_large_fonts
        )

        plot_trend = (
            ggplot(tunnel_df)
            + geom_line(aes(x='Time', y='Trend'), color="black")
            + labs(x='', y='Traffic (% of capacity)')
            + ylim(0, 101)
            + scale_x_datetime(date_labels=date_format)
            + theme_large_fonts
        )

        plot_harmonic = (
            ggplot(tunnel_df)
            + geom_line(aes(x='Time', y='Seasonal'), color="black")
            + labs(x='', y='Traffic (% of capacity)')
            + scale_x_datetime(date_labels=date_format)
            + theme_large_fonts
        )

        plot_residual = (
            ggplot()
            + geom_line(tunnel_df, aes(x='Time', y='Residual'))
            + labs(x='', y='Traffic (% of capacity)')
            + scale_x_datetime(date_labels=date_format)
            + theme_large_fonts
        )

        # Create a combined plot with Original, Trend, Seasonal, and Residual
        # Prepare data for combined plot
        plot_data = []
        
        # Add Original data
        plot_data.append(tunnel_df[['Time', 'Original']].copy())
        plot_data[-1]['Component'] = 'Original'
        plot_data[-1].rename(columns={'Original': 'Value'}, inplace=True)
        
        # Add Trend data
        plot_data.append(tunnel_df[['Time', 'Trend']].copy())
        plot_data[-1]['Component'] = 'Trend'
        plot_data[-1].rename(columns={'Trend': 'Value'}, inplace=True)
        
        # Add Seasonal data
        plot_data.append(tunnel_df[['Time', 'Seasonal']].copy())
        plot_data[-1]['Component'] = 'Seasonal'
        plot_data[-1].rename(columns={'Seasonal': 'Value'}, inplace=True)
        
        # Add Residual data
        plot_data.append(tunnel_df[['Time', 'Residual']].copy())
        plot_data[-1]['Component'] = 'Residual'
        plot_data[-1].rename(columns={'Residual': 'Value'}, inplace=True)
        
        combined_df = pd.concat(plot_data, ignore_index=True)

        # Normalize column types to avoid plotnine/pandas sanitization errors
        # Ensure Time is datetime, Value numeric, Component string
        combined_df['Time'] = pd.to_datetime(combined_df['Time'], errors='coerce')
        combined_df['Value'] = pd.to_numeric(combined_df['Value'], errors='coerce')
        combined_df['Component'] = combined_df['Component'].astype(str)
        
        plot_combined = (
            ggplot()
            + geom_line(combined_df, aes(x='Time', y='Value', color='Component'), size=0.8, alpha=0.6)
            + scale_color_manual(values={'Original': 'black', 'Trend': 'blue', 'Seasonal': 'green', 'Residual': '#FF4B33'})
            + labs(x='', y='Traffic (% of capacity)', color='Component')
            + scale_x_datetime(date_labels=date_format)
            + theme_large_fonts
            + theme(legend_position='bottom')
            + ylim(min(combined_df['Value'].min(), 0), 101)
            + guides(color=guide_legend(nrow=1, override_aes={'size': 2.5, 'alpha': 1.0}))
        )

        if output_dir:
            os.makedirs(os.path.join(output_dir, "Original"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "Trend"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "Seasonal"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "Residual"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "Combined"), exist_ok=True)
        
        # Build date range string for filename
        date_range = ""
        if start_date is not None and end_date is not None:
            date_range = f"_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
        elif start_date is not None:
            date_range = f"_from_{start_date.strftime('%Y-%m-%d')}"
        elif end_date is not None:
            date_range = f"_until_{end_date.strftime('%Y-%m-%d')}"
        
        plot_original_signal.save(
            os.path.join(output_dir, "Original", f"{tunnel}{date_range}.png"),
            dpi=150, width=30, height=10, units='cm'
        )

        plot_trend.save(
            os.path.join(output_dir, "Trend", f"{tunnel}{date_range}.png"),
            dpi=150, width=30, height=10, units='cm'
        )

        plot_harmonic.save(
            os.path.join(output_dir, "Seasonal", f"{tunnel}{date_range}.png"),
            dpi=150, width=30, height=10, units='cm'
        )

        plot_residual.save(
            os.path.join(output_dir, "Residual", f"{tunnel}{date_range}.png"),
            dpi=150, width=30, height=10, units='cm'
        )

        if plot_combined:
            plot_combined.save(
                os.path.join(output_dir, "Combined", f"{tunnel}{date_range}.png"),
                dpi=150, width=30, height=10, units='cm'
            )
    
    print(f"Saved plots to: {output_dir}")

def picp(y_true, lower_bounds, upper_bounds):
    """
        Compute Prediction Interval Coverage Probability (PICP).
        Args:
            y_true: true values
            lower_bounds: lower bounds of prediction intervals
            upper_bounds: upper bounds of prediction intervals
        Returns:
            PICP value
    """

    covered = np.logical_and(y_true >= lower_bounds, y_true <= upper_bounds)
    return np.mean(covered)

def pinaw(y_true, lower_bounds, upper_bounds):
    """
        Compute Prediction Interval Normalized Average Width (PINAW).
        Normalized by the range of the true values (max - min).
        Args:
            y_true: true values
            lower_bounds: lower bounds of prediction intervals
            upper_bounds: upper bounds of prediction intervals
        Returns:
            PINAW value
    """
    
    interval_width = upper_bounds - lower_bounds
    range_y_true = np.max(y_true) - np.min(y_true)
    if range_y_true == 0:
        return 0
    return np.mean(interval_width) / range_y_true

def aggregate_forecast_metrics(metrics_list, output_dir='./results/', prediction_interval=98, use_single_kde=False, forecast_date=None, forecast_days=7):
    """
        Aggregate PICP and PINAW metrics across all tunnels and save to CSV.
        
        Args:
            metrics_list: list of metrics dictionaries returned by plot_date_forecast()
            output_dir: directory to save the aggregated metrics CSV
            prediction_interval: prediction interval used in forecasting (default: 98)
            use_single_kde: whether single KDE mode was used (default: False)
            forecast_date: date of the forecast (for directory naming)
            forecast_days: number of days forecasted (for directory naming)
        
        Returns:
            Aggregated metrics dataframe
    """
    
    if not metrics_list:
        print("No metrics to aggregate")
        return None
    
    # Create dataframe from metrics list
    df_metrics = pd.DataFrame(metrics_list)
    
    # Filter out tunnels with large gaps for computing global statistics
    # These tunnels are still saved in individual metrics, but excluded from aggregates
    df_metrics_for_aggregate = df_metrics[df_metrics.get('Has_Large_Gap', False) == False].copy()
    
    # Report excluded tunnels
    tunnels_with_gaps = df_metrics[df_metrics.get('Has_Large_Gap', False) == True]['Tunnel'].unique()
    if len(tunnels_with_gaps) > 0:
        print(f"Excluding {len(tunnels_with_gaps)} tunnel(s) with large data gaps from aggregate statistics: {', '.join(tunnels_with_gaps)}")
    
    # Build experiment-specific directory name
    kde_suffix = 'SingleKDE' if use_single_kde else 'IndicatorKDE'
    if forecast_date is not None:
        forecast_date_str = pd.Timestamp(forecast_date).strftime('%Y%m%d')
    else:
        forecast_date_str = 'NoDate'
    
    exp_dir_name = f'Perc{prediction_interval}{kde_suffix}_D{forecast_date_str}_F{forecast_days}d'
    exp_output_dir = os.path.join(output_dir, exp_dir_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Build filenames with experiment details
    filename_suffix = f'_Perc{prediction_interval}_{kde_suffix}_D{forecast_date_str}_F{forecast_days}d'
    
    # Round numeric columns to 3 decimal places
    df_metrics_rounded = df_metrics.copy()
    numeric_cols = df_metrics_rounded.select_dtypes(include=['float64', 'float32']).columns
    df_metrics_rounded[numeric_cols] = df_metrics_rounded[numeric_cols].round(3)
    
    # Exclude AGGREGATED_ALL_TUNNELS from individual metrics (save only actual tunnels)
    df_metrics_rounded = df_metrics_rounded[df_metrics_rounded['Tunnel'] != 'AGGREGATED_ALL_TUNNELS']
    
    # Save individual metrics
    individual_path = os.path.join(exp_output_dir, f'forecast_metrics_individual{filename_suffix}.csv')
    df_metrics_rounded.to_csv(individual_path, index=False)
    print(f"Saved individual metrics to {individual_path}")
    
    # Also compute global statistics (using only tunnels without large gaps)
    global_stats = {
        'Metric': ['PICP', 'PINAW', 'PICP_Matches', 'Threshold_Count_Truth', 'Threshold_Count_Interval'],
        'Mean': [
            df_metrics_for_aggregate['PICP'].mean(),
            df_metrics_for_aggregate['PINAW'].mean(),
            df_metrics_for_aggregate['PICP_Matches'].mean(),
            df_metrics_for_aggregate['Threshold_Count_Truth'].mean(),
            df_metrics_for_aggregate['Threshold_Count_Interval'].mean()
        ],
        'Std': [
            df_metrics_for_aggregate['PICP'].std(),
            df_metrics_for_aggregate['PINAW'].std(),
            df_metrics_for_aggregate['PICP_Matches'].std(),
            df_metrics_for_aggregate['Threshold_Count_Truth'].std(),
            df_metrics_for_aggregate['Threshold_Count_Interval'].std()
        ],
        'Min': [
            df_metrics_for_aggregate['PICP'].min(),
            df_metrics_for_aggregate['PINAW'].min(),
            df_metrics_for_aggregate['PICP_Matches'].min(),
            df_metrics_for_aggregate['Threshold_Count_Truth'].min(),
            df_metrics_for_aggregate['Threshold_Count_Interval'].min()
        ],
        'Max': [
            df_metrics_for_aggregate['PICP'].max(),
            df_metrics_for_aggregate['PINAW'].max(),
            df_metrics_for_aggregate['PICP_Matches'].max(),
            df_metrics_for_aggregate['Threshold_Count_Truth'].max(),
            df_metrics_for_aggregate['Threshold_Count_Interval'].max()
        ],
        'Count': [
            df_metrics_for_aggregate['PICP'].notna().sum(),
            df_metrics_for_aggregate['PINAW'].notna().sum(),
            df_metrics_for_aggregate['PICP_Matches'].notna().sum(),
            df_metrics_for_aggregate['Threshold_Count_Truth'].notna().sum(),
            df_metrics_for_aggregate['Threshold_Count_Interval'].notna().sum()
        ]
    }
    
    df_global = pd.DataFrame(global_stats).round(3)
    global_path = os.path.join(exp_output_dir, f'forecast_metrics_global{filename_suffix}.csv')
    df_global.to_csv(global_path, index=False)
    print(f"Saved global metrics to {global_path}")
    
    print("\n=== Global Statistics ===")
    print(df_global.to_string(index=False))
    
    return df_metrics, df_global

def plot_date_forecast(forecast_days, past_times, past_vals, future_times, future_truth, forecast_mean, forecast_q_low, forecast_q_high,
                       tunnel_name, forecast_date, output_dir='./plots/', inject_indicator_values=None,
                       traffic_threshold=None, prediction_interval=98, future_indicators_serie_a=None, future_indicators_champions=None,
                       past_indicators_serie_a=None, past_indicators_champions=None, use_single_kde=False, injection_periods=None):
    """
        Plot date forecast for a tunnel, including ground truth, prediction intervals, and indicators. Compute PICP if ground truth is available

        Args:
            forecast_days: number of days forecasted
            past_times: timestamps for past data
            past_vals: values for past data
            future_times: timestamps for future data
            future_truth: ground truth values for future data
            forecast_mean: forecasted mean values for future data
            forecast_q_low: forecasted low percentile values for future data (e.g., 1st percentile for prediction_interval=98)
            forecast_q_high: forecasted high percentile values for future data (e.g., 99th percentile)
            tunnel_name: name of the tunnel
            forecast_date: date of the forecast
            output_dir: directory to save plots
            inject_indicator_values: indicator values injected during forecasting (optional)
            prediction_interval: prediction interval for forecast (default: 98)
            future_indicators_serie_a: Serie A indicator values for future data (optional)
            future_indicators_champions: Champions League indicator values for future data (optional)
            past_indicators_serie_a: Serie A indicator values for past data (optional)
            past_indicators_champions: Champions League indicator values for past data (optional)
            use_single_kde: whether single KDE mode is enabled (optional)
            injection_periods: list of (value, start_ts, end_ts, type) to mark injected indicator ranges (optional)
    """
    
    # Build DateForecasts directory name with threshold, percentile, and KDE mode
    kde_suffix = '_SingleKDE' if use_single_kde else ''
    if traffic_threshold is not None:
        forecast_dir_name = f'DateForecasts_Thresh{int(traffic_threshold)}_Perc{prediction_interval}{kde_suffix}'
    else:
        forecast_dir_name = f'DateForecasts_Perc{prediction_interval}{kde_suffix}'
    
    os.makedirs(os.path.join(output_dir, forecast_dir_name, tunnel_name), exist_ok=True)

    past_mask = pd.to_datetime(past_times) < forecast_date
    future_mask = pd.to_datetime(future_times) >= forecast_date
    
    # Compute PICP and PINAW for signal and trend in the future period (if ground truth exists)
    signal_picp = None
    signal_pinaw = None
    signal_picp_matches = None
    threshold_count_truth = None
    threshold_count_interval = None
    
    if len(future_times[future_mask]) > 0:
        # Future signal PICP and PINAW
        future_truth_vals = future_truth[future_mask]
        future_forecast_vals = forecast_mean[future_mask]
        
        # Only compute metrics if we have valid data (not NaN)
        valid_signal = ~(np.isnan(future_truth_vals) | np.isnan(future_forecast_vals))
        if np.sum(valid_signal) > 0:
            signal_picp = picp(future_truth_vals[valid_signal], forecast_q_low[future_mask][valid_signal], forecast_q_high[future_mask][valid_signal])
            signal_pinaw = pinaw(future_truth_vals[valid_signal], forecast_q_low[future_mask][valid_signal], forecast_q_high[future_mask][valid_signal])
        
        # Compute PICP and PINAW only for samples with matches (Serie A or Champions indicator > 0)
        if future_indicators_serie_a is not None and future_indicators_champions is not None:
            future_ind_serie_a_vals = future_indicators_serie_a[future_mask]
            future_ind_champions_vals = future_indicators_champions[future_mask]
            # Samples where either Serie A or Champions League indicator > 0
            matches_mask = (future_ind_serie_a_vals > 0) | (future_ind_champions_vals > 0)
            valid_matches = valid_signal & matches_mask
            if np.sum(valid_matches) > 0:
                signal_picp_matches = picp(
                    future_truth_vals[valid_matches], 
                    forecast_q_low[future_mask][valid_matches], 
                    forecast_q_high[future_mask][valid_matches]
                )
        
        # Compute threshold counts if traffic_threshold is provided
        if traffic_threshold is not None:
            future_q_high_vals = forecast_q_high[future_mask]
            
            # Helper function to count continuous periods above threshold
            def count_continuous_periods(exceeds_mask, valid_mask):
                """Count continuous periods where condition is true"""
                valid_exceeds = np.logical_and(exceeds_mask, valid_mask)
                if not np.any(valid_exceeds):
                    return 0
                
                # Find transitions from False to True (period starts)
                # Add False at the beginning to detect first period start
                padded = np.concatenate([[False], valid_exceeds, [False]])
                transitions = np.diff(padded.astype(int))
                # Count rising edges (transitions from False to True)
                period_count = np.sum(transitions == 1)
                return period_count
            
            # Count 1: Ground truth exceeds threshold (continuous periods)
            ground_truth_exceeds = future_truth_vals > traffic_threshold
            valid_truth_data = ~np.isnan(future_truth_vals)
            threshold_count_truth = count_continuous_periods(ground_truth_exceeds, valid_truth_data)
            
            # Count 2: Interval (Q_high) exceeds threshold (continuous periods)
            interval_exceeds = future_q_high_vals > traffic_threshold
            valid_interval_data = ~np.isnan(future_q_high_vals)
            threshold_count_interval = count_continuous_periods(interval_exceeds, valid_interval_data)

    # Build ground truth: past + future (with NaN for future periods without data)
    # This handles three cases:
    # 1. Full ground truth in future period
    # 2. No ground truth in future period
    # 3. Partial ground truth in future period (some timestamps have data, some don't)
    ground_truth_times = np.concatenate([
        past_times[past_mask],
        future_times[future_mask]
    ])
    
    # For future ground truth, use actual values where available, NaN otherwise
    future_truth_to_plot = future_truth[future_mask].copy()
    
    ground_truth_vals = np.concatenate([
        past_vals[past_mask],
        future_truth_to_plot
    ])
    
    # Build data arrays with separate Serie A and Champions League indicators
    arrays_to_concat = [ground_truth_times, future_times, past_times, past_times, future_times, future_times]
    values_to_concat = [ground_truth_vals, forecast_mean, past_indicators_serie_a, past_indicators_champions, future_indicators_serie_a, future_indicators_champions]
    series_to_concat = ['Ground Truth'] * len(ground_truth_times) + \
                       ['Forecast Mean'] * len(future_times) + \
                       ['Serie A Matches'] * len(past_times) + \
                       ['Champions League'] * len(past_times) + \
                       ['Serie A Matches'] * len(future_times) + \
                       ['Champions League'] * len(future_times)
    
    df_times = np.concatenate(arrays_to_concat)
    df_values = np.concatenate(values_to_concat)
    df_series = series_to_concat
    
    df_plot = pd.DataFrame({
        'Time': df_times,
        'Value': df_values,
        'Series': df_series
    })
    
    df_band = pd.DataFrame({
        'Time': future_times,
        'Q_low': forecast_q_low,
        'Q_high': forecast_q_high
    })
    
    df_plot['Value'] = pd.to_numeric(df_plot['Value'], errors='coerce')
    df_band['Q_low'] = pd.to_numeric(df_band['Q_low'], errors='coerce')
    df_band['Q_high'] = pd.to_numeric(df_band['Q_high'], errors='coerce')
    
    # Compute threshold line if provided
    df_threshold = None
    df_gt_exceeds = None
    df_interval_exceeds = None
    injection_segments = []
    
    if traffic_threshold is not None:        
        # Create threshold line for future period (always, regardless of ground truth)
        # Create a threshold dataframe for all future times
        df_threshold = pd.DataFrame({
            'Time': future_times[future_mask],
            'Value': traffic_threshold,
            'Series': 'Threshold'
        })
        
        # Also add threshold line for the last day before forecasting (past data)
        past_mask_threshold = (pd.to_datetime(past_times) >= forecast_date - pd.Timedelta(days=1)) & (pd.to_datetime(past_times) < forecast_date)
        if np.any(past_mask_threshold):
            threshold_vals_past = traffic_threshold
            df_threshold_past = pd.DataFrame({
                'Time': past_times[past_mask_threshold],
                'Value': threshold_vals_past,
                'Series': 'Threshold'
            })
            df_threshold = pd.concat([df_threshold_past, df_threshold], ignore_index=True)
        
        # Create a dataframe for ground truth (for threshold detection)
        ground_truth_df = df_plot[df_plot['Series'] == 'Ground Truth'].copy()
        # Filter to only rows where Value is not NaN (actual ground truth exists)
        ground_truth_with_data = ground_truth_df[ground_truth_df['Value'].notna()].reset_index(drop=True)
        
        if len(ground_truth_with_data) > 0:
            # Create dataframes for points that exceed threshold
            # Ground truth exceeds threshold - mark only the start of each continuous period
            # Only consider ground truth data from the forecast period onwards
            future_gt_mask = ground_truth_with_data['Time'] >= forecast_date
            ground_truth_future = ground_truth_with_data[future_gt_mask].reset_index(drop=True)
            
            if len(ground_truth_future) > 0:
                # Get threshold values for future ground truth points
                future_gt_thresholds = []
                for time_val in ground_truth_future['Time']:
                    if time_val in future_times[future_mask].values:
                        idx = np.where(future_times[future_mask].values == time_val)[0][0]
                        future_gt_thresholds.append(traffic_threshold[idx])
                
                gt_exceeds_mask = ground_truth_future['Value'].values > np.array(future_gt_thresholds)
                if gt_exceeds_mask.any():
                    # Find transitions from False to True (period starts)
                    padded = np.concatenate([[False], gt_exceeds_mask, [False]])
                    transitions = np.diff(padded.astype(int))
                    period_starts = np.where(transitions == 1)[0]  # Indices where periods start
                    
                    if len(period_starts) > 0:
                        df_gt_exceeds = ground_truth_future.iloc[period_starts].copy()
                        df_gt_exceeds['Type'] = 'GT Exceeds'
        
        # Interval (Q_high) exceeds threshold - mark only the start of each continuous period
        interval_exceeds_mask = []
        future_times_masked = np.array(future_times[future_mask])
        forecast_q_high_masked = np.array(forecast_q_high[future_mask])
        
        for t, time_val in enumerate(future_times_masked):
            if not np.isnan(forecast_q_high_masked[t]) and forecast_q_high_masked[t] > traffic_threshold[t]:
                interval_exceeds_mask.append(True)
            else:
                interval_exceeds_mask.append(False)
        
        if any(interval_exceeds_mask):
            # Find transitions from False to True (period starts)
            padded = np.concatenate([[False], interval_exceeds_mask, [False]])
            transitions = np.diff(padded.astype(int))
            period_starts = np.where(transitions == 1)[0]  # Indices where periods start
            
            if len(period_starts) > 0:
                # Get only the starting points of each period
                interval_start_times = [future_times_masked[idx] for idx in period_starts]
                interval_start_values = [forecast_q_high_masked[idx] for idx in period_starts]
                
                df_interval_exceeds = pd.DataFrame({
                    'Time': interval_start_times,
                    'Value': interval_start_values,
                    'Type': 'Interval Exceeds'
                })
        
    # Build title with PICP, PINAW and threshold counts if available
    title_str = ''
    if signal_picp is not None:
        title_str += f' PICP: {signal_picp:.3f}'
    if signal_picp_matches is not None:
        title_str += f' | PICP (matches): {signal_picp_matches:.3f}'
    if signal_pinaw is not None:
        title_str += f' | PINAW: {signal_pinaw:.3f}'
    if threshold_count_truth is not None:
        title_str += f' | GT>{traffic_threshold:.0f}%: {threshold_count_truth}'
    if threshold_count_interval is not None:
        title_str += f' | Int>{traffic_threshold:.0f}%: {threshold_count_interval}'
    
    # Use plotnine (ggplot) for plotting with a secondary axis that maps primary -> primary/10
    # Prepare series dataframes
    ground_truth_data = df_plot[df_plot['Series'] == 'Ground Truth'].copy()
    df_band['Time'] = pd.to_datetime(df_band['Time'])

    # Prepare indicators combined and scale them to primary axis (multiply by 10)
    serie_a_data = df_plot[df_plot['Series'] == 'Serie A Matches'].copy()
    champions_data = df_plot[df_plot['Series'] == 'Champions League'].copy()
    if not serie_a_data.empty:
        serie_a_data['Scaled'] = pd.to_numeric(serie_a_data['Value'], errors='coerce') * 10
        serie_a_data['Time'] = pd.to_datetime(serie_a_data['Time'])
    if not champions_data.empty:
        champions_data['Scaled'] = pd.to_numeric(champions_data['Value'], errors='coerce') * 10
        champions_data['Time'] = pd.to_datetime(champions_data['Time'])

    ground_truth_data['Time'] = pd.to_datetime(ground_truth_data['Time'])
    ground_truth_data['Value'] = pd.to_numeric(ground_truth_data['Value'], errors='coerce')

    # Build ggplot
    _, ax1 = plt.subplots(figsize=(25/2.54, 6/2.54))  # dimensioni cm

    # Asse SX: Traffic (band + ground truth)
    ax1 = sns.lineplot(data=ground_truth_data, x='Time', y='Value',
                    color='black', linewidth=1.5, ax=ax1)
    ax1.fill_between(df_band['Time'], df_band['Q_low'], df_band['Q_high'],
                 color='blue', alpha=0.3)
    ax1.set_xlabel('', color='black')
    ax1.set_ylabel('Traffic (% of capacity)', color='black')
    
    # Compute max y-axis from both high band and indicator values
    max_band = df_band['Q_high'].max()
    max_ind_seria = serie_a_data['Value'].max()
    max_ind_champions = champions_data['Value'].max()
    max_indicators = max(max_ind_seria, max_ind_champions)
    max_ground_truth = ground_truth_data['Value'].max() if not ground_truth_data.empty else float('-inf')
    max_y_base = max(max_band, max_indicators * 10, max_ground_truth)
    max_y = max_y_base + 25

    ax1.set_ylim(0, max_y)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)  # simile a theme_bw

    # Format x-axis: adaptive label density based on forecast_days
    # 7 days: every 1 day, 14 days: every 2 days, 28 days: every 4 days
    if forecast_days <= 7:
        interval = 1
    elif forecast_days <= 14:
        interval = 2
    else:
        interval = 4
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Asse DX: Serie A + Champions
    ax2 = ax1.twinx()
    
    # Always plot all indicator types from historical data
    if not serie_a_data.empty:
        sns.lineplot(data=serie_a_data, x='Time', y='Value',
                    color='darkgreen', linestyle='--', linewidth=1.5, ax=ax2, alpha=0.6)
    if not champions_data.empty:
        sns.lineplot(data=champions_data, x='Time', y='Value',
                    color='maroon', linestyle='--', linewidth=1.5, ax=ax2, alpha=0.6)
    
    ax2.set_ylabel('Number of matches', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max_y / 10)  # Match primary axis scaling (primary / 10)

    if injection_periods:
        future_times_ts = pd.to_datetime(future_times)
        if len(future_times_ts) > 0:
            future_start = future_times_ts[0]
            future_end = future_times_ts[-1]
            for period_tuple in injection_periods:
                # Handle both old (value, start, end) and new (value, start, end, type) formats
                if len(period_tuple) == 4:
                    value, start_ts, end_ts, injection_type = period_tuple
                else:
                    value, start_ts, end_ts = period_tuple
                    injection_type = 'combined'
                
                # Choose color based on type
                if injection_type == 'serie_a':
                    color = 'darkgreen'
                else:  # 'champions' or 'combined'
                    color = 'maroon'
                
                start_plot = max(pd.Timestamp(start_ts), future_start)
                end_plot = min(pd.Timestamp(end_ts), future_end)
                if end_plot <= start_plot:
                    continue
                ax2.plot([start_plot, end_plot], [value, value], color=color, linestyle=':', linewidth=2)
                injection_segments.append((value, start_plot, end_plot))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    inject_suffix = f"_inject_{'_'.join(map(str, inject_indicator_values))}" if inject_indicator_values else ""
    output_path = os.path.join(output_dir, forecast_dir_name, tunnel_name, f"forecast_{forecast_date.strftime('%Y-%m-%d')}_{forecast_days}d{inject_suffix}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Use the prediction_interval parameter directly in the forecast interval label
    ci_label = f'Forecast {prediction_interval}% PI'

    handles = [
        Line2D([0], [0], color='black', lw=1.5, label='Ground Truth'),
        Line2D([0], [0], color='blue', lw=10, alpha=0.2, label=ci_label),
        Line2D([0], [0], color='darkgreen', ls='--', lw=1.5, label='Serie A'),
        Line2D([0], [0], color='maroon', ls='--', lw=1.5, label='Champions League')
    ]
    
    ax1.legend(handles=handles, loc='upper left', ncol=4, framealpha=0.95, fontsize=9)

    plt.title(title_str)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV files
    df_plot.to_csv(output_path.replace('.png', '.csv'), index=False)
    df_band.to_csv(output_path.replace('.png', '.csv').replace('forecast_', 'forecast_band_'), index=False)

    print(f"Saved date forecast for {tunnel_name} from {forecast_date.strftime('%Y-%m-%d')} for {forecast_days} days")
    
    # Create metrics dictionary
    metrics_dict = {
        'Tunnel': tunnel_name,
        'Forecast_Date': forecast_date.strftime('%Y-%m-%d') if isinstance(forecast_date, pd.Timestamp) else str(forecast_date),
        'Forecast_Days': forecast_days,
        'PICP': signal_picp,
        'PINAW': signal_pinaw,
        'PICP_Matches': signal_picp_matches,
        'Threshold_Count_Truth': threshold_count_truth,
        'Threshold_Count_Interval': threshold_count_interval
    }
    
    # Prepare and return threshold exceeds dataframe and metrics
    threshold_df = None
    if traffic_threshold is not None:
        threshold_exceeds_data = []
        
        # Add ground truth exceeds
        if df_gt_exceeds is not None:
            gt_data = df_gt_exceeds[['Time', 'Value']].copy()
            gt_data['Type'] = 'Ground Truth'
            gt_data['Tunnel'] = tunnel_name
            gt_data['Threshold_Percent'] = traffic_threshold
            threshold_exceeds_data.append(gt_data)
        
        # Add interval exceeds
        if df_interval_exceeds is not None:
            int_data = df_interval_exceeds[['Time', 'Value']].copy()
            int_data['Type'] = 'Prediction Interval (Q98)'
            int_data['Tunnel'] = tunnel_name
            int_data['Threshold_Percent'] = traffic_threshold
            threshold_exceeds_data.append(int_data)
        
        # Create combined dataframe
        if threshold_exceeds_data:
            threshold_df = pd.concat(threshold_exceeds_data, ignore_index=True)
    
    return threshold_df, metrics_dict

def plot_pdf_cdf_by_indicator(kde_distributions, df, value_column='Traffic (% of capacity)', indicator_column='Indicator', 
                              second_indicator_column=None, second_kde_distributions=None, second_label=None,
                              tunnel_column='Tunnel', output_dir='./plots/', plot_tunnels=False, forecast_date=None, forecast_days=None, 
                              single_kde_mode=False, no_match_cols=None):
    """
        Plot PDF and CDF distributions by indicator value for each tunnel.
        
        Args:
            kde_distributions: dict from compute_kde_by_indicator() with KDE objects
            df: dataframe with original data (for computing statistics)
            value_column: column name for the values
            indicator_column: column name for indicator values
            second_indicator_column: optional column name for second indicator (e.g., Champions League if first is Serie A)
            second_kde_distributions: optional KDE dict for second indicator
            second_label: optional label for second indicator (e.g., "Champions League")
            tunnel_column: column name for tunnel names
            output_dir: directory to save plots
            plot_tunnels: whether to create visualization plots
            single_kde_mode: if True, plot a single KDE without facets (for single KDE mode)
            no_match_cols: tuple of (serie_a_col, champions_col) to define the unique "0" group
    """

    # Keep a copy of the original dataframe for KS comparisons against the
    # true "no-match" (both indicators == 0) distribution.
    df_orig = df.copy()

    # Filter to indicators 0..10
    # Ensure second indicator column exists if combining
    if second_indicator_column is not None and second_indicator_column not in df.columns:
        df[second_indicator_column] = 0
    
    # Skip indicator filtering in single KDE mode (no indicator column exists)
    if not single_kde_mode:
        # If combining two indicators, filter both columns; otherwise just filter the main column
        if second_indicator_column is not None and second_kde_distributions is not None:
            # Fill NaN with 0 for filtering purposes
            df[indicator_column] = df[indicator_column].fillna(-1)
            df[second_indicator_column] = df[second_indicator_column].fillna(-1)
            df = df[df[indicator_column].isin(range(11)) | df[second_indicator_column].isin(range(11))].copy()
        else:
            if indicator_column in df.columns:
                df = df[df[indicator_column].isin(range(11))].copy()

        # For multi-indicator mode, ensure "0" only includes timestamps with no matches
        # Only apply this filter if NOT combining two indicators (when combining, we keep all data)
        if no_match_cols and not (second_indicator_column is not None and second_kde_distributions is not None):
            serie_a_col, champions_col = no_match_cols
            if serie_a_col in df.columns and champions_col in df.columns:
                no_match_mask = (df[serie_a_col] == 0) & (df[champions_col] == 0)
                non_no_match_zero = (df[indicator_column] == 0) & (~no_match_mask)
                df.loc[non_no_match_zero, indicator_column] = np.nan
    
    # If second indicator is provided, create faceted data for separate plots
    if second_indicator_column is not None and second_kde_distributions is not None and not single_kde_mode:
        # Determine labels
        first_label = indicator_column.replace('Indicator_', '').replace('_', ' ') if 'Indicator_' in indicator_column else 'First'
        
        # Create separate data frames for each league
        df[second_indicator_column] = df[second_indicator_column].fillna(-1)
        
        # Create faceted data: one row per indicator value per league
        facet_data = []
        
        # Add Serie A data
        df_serie_a = df[[value_column, indicator_column, tunnel_column]].dropna(subset=[value_column, indicator_column]).copy()
        df_serie_a['League'] = first_label
        df_serie_a['IndicatorValue'] = df_serie_a[indicator_column].astype(int)
        facet_data.append(df_serie_a[['League', 'IndicatorValue', value_column, tunnel_column]])
        
        # Add Champions data
        df_champions = df[[value_column, second_indicator_column, tunnel_column]].copy()
        df_champions = df_champions[df_champions[second_indicator_column] >= 0].dropna(subset=[value_column]).copy()
        df_champions['League'] = second_label
        df_champions['IndicatorValue'] = df_champions[second_indicator_column].astype(int)
        facet_data.append(df_champions[['League', 'IndicatorValue', value_column, tunnel_column]])
        
        # Combine into single dataframe for faceted plotting
        if facet_data and all(len(d) > 0 for d in facet_data):
            df = pd.concat(facet_data, ignore_index=True)
            indicator_column_to_use = 'IndicatorValue'
            facet_column = 'League'
            use_facets = True
        else:
            indicator_column_to_use = indicator_column
            facet_column = None
            use_facets = False
        
        # Use original KDE distributions (indices are still numeric)
    else:
        indicator_column_to_use = indicator_column
        facet_column = None
        use_facets = False
        if second_indicator_column is not None:
            df[second_indicator_column] = df[second_indicator_column].fillna(-1)
    
    # Output directories for PDF and CDF plots
    pdf_dir = os.path.join(output_dir, "PDF" + value_column)
    cdf_dir = os.path.join(output_dir, "CDF" + value_column)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(cdf_dir, exist_ok=True)

    palette = ['dodgerblue', 'orange', 'green', 'red', 'purple', 'brown',
               'gray', 'pink', 'cyan', 'magenta', 'yellow']
    colors = {str(i): palette[i] for i in range(11)}
    labels = {str(i): f'Indicator = {i}' for i in range(11)}
    
    # If using faceted indicators, create simple numeric color mapping
    if use_facets:
        # Extract unique numeric indicators
        unique_indicators = sorted(df[indicator_column_to_use].dropna().unique())
        palette = ['dodgerblue', 'orange', 'green', 'red', 'purple', 'brown',
                   'gray', 'pink', 'cyan', 'magenta', 'yellow']
        colors = {str(int(ind)): palette[int(ind) % len(palette)] for ind in unique_indicators}
        labels = {str(int(ind)): str(int(ind)) for ind in unique_indicators}
    elif indicator_column_to_use == 'combined_indicator':
        # Old combined logic (if still used)
        unique_combined = df[indicator_column_to_use].dropna().unique()
        colors_combined = {}
        palette = ['dodgerblue', 'orange', 'green', 'red', 'purple', 'brown',
                   'gray', 'pink', 'cyan', 'magenta', 'yellow']
        palette_extended = palette * 3
        
        for combined_label in sorted(unique_combined, key=lambda x: (
            0 if x == "0" else int(x.split()[0]),
            0 if "Serie A" in str(x) else (1 if "Champions" in str(x) else 2)
        )):
            if combined_label not in colors_combined:
                colors_combined[combined_label] = palette_extended[len(colors_combined) % len(palette_extended)]
        
        colors = colors_combined
        labels = {k: k for k in unique_combined}
    else:
        # Default for single indicator
        palette = ['dodgerblue', 'orange', 'green', 'red', 'purple', 'brown',
                   'gray', 'pink', 'cyan', 'magenta', 'yellow']
        colors = {str(i): palette[i] for i in range(11)}
        labels = {str(i): f'Indicator = {i}' for i in range(11)}

    for tunnel in kde_distributions.keys():
        tunnel_df = df[df[tunnel_column] == tunnel]

        if tunnel_df.empty:
            continue

        val_min, val_max = tunnel_df[value_column].min(), tunnel_df[value_column].max()

        if single_kde_mode:
            # Single KDE mode - just one distribution from all residuals
            kde = kde_distributions[tunnel].get(0)

            
            # Boxplot for single KDE mode (all residuals)
            boxplot_data = tunnel_df[[value_column]].dropna().copy()
            boxplot_data['group'] = 'All Residuals'
            
            if plot_tunnels:
                # Violin plot with boxplot overlay
                violin_fig = (
                    ggplot(boxplot_data, aes(x='group', y=value_column))
                    + geom_violin(fill='dodgerblue', alpha=0.7)
                    + geom_boxplot(width=0.15, alpha=0.8, fill='white', outlier_alpha=0.5, outlier_size=1)
                    + labs(x='', y='Traffic (% of capacity)', title='')
                    + theme_bw()
                    + theme(
                        plot_title=element_text(size=26, face='bold'),
                        axis_title=element_text(size=22),
                        axis_text=element_text(size=20)
                    )
                )
                violin_fig.save(os.path.join(pdf_dir, f"violin_{tunnel}_{forecast_date.strftime('%Y-%m-%d')}_{forecast_days}d.png"), dpi=150, width=30, height=15, units='cm')
            
            # Single CDF from all residuals
            all_values = tunnel_df[value_column].dropna().sort_values()
            if len(all_values) > 0:
                cdf_vals = np.arange(1, len(all_values) + 1) / len(all_values)
                x_full = np.concatenate([[val_min], all_values.values, [val_max]])
                cdf_full = np.concatenate([[0], cdf_vals, [1]])
            else:
                x_full = np.array([val_min, val_max])
                cdf_full = np.array([0, 1])
            
            cdf_plot_df = pd.DataFrame({
                value_column: x_full,
                'cdf': cdf_full
            })
            
            if plot_tunnels:
                cdf_plot = (
                    ggplot(cdf_plot_df, aes(x=value_column, y='cdf'))
                    + geom_step(alpha=0.8, color='dodgerblue')
                    + labs(x=value_column, y='Cumulative Probability', title='')
                    + theme_large_fonts
                )
                cdf_plot.save(os.path.join(cdf_dir, f"cdf_{tunnel}_{forecast_date.strftime('%Y-%m-%d')}_{forecast_days}d.png"), dpi=150)
        else:
            # Multi-indicator mode - boxplot for each indicator value
            boxplot_data = tunnel_df[[value_column, indicator_column_to_use]].dropna().copy()
            boxplot_data[indicator_column_to_use] = boxplot_data[indicator_column_to_use].astype(int).astype(str)

            # Add 'All residuals' for each league if faceted
            if use_facets and facet_column in tunnel_df.columns:
                # Exclude nan facet
                unique_leagues = [l for l in tunnel_df[facet_column].unique() if pd.notna(l)]
                all_residuals_list = []
                for league in unique_leagues:
                    league_df = tunnel_df[tunnel_df[facet_column] == league]
                    # Indicator violins for this league
                    league_boxplot = league_df[[value_column, indicator_column_to_use]].dropna().copy()
                    league_boxplot[indicator_column_to_use] = league_boxplot[indicator_column_to_use].astype(int).astype(str)
                    league_boxplot[facet_column] = league
                    # All residuals for this league
                    all_residuals = league_df[[value_column]].dropna().copy()
                    all_residuals[indicator_column_to_use] = 'All'
                    all_residuals[facet_column] = league
                    # Combine
                    all_residuals_list.append(pd.concat([league_boxplot, all_residuals], ignore_index=True))
                boxplot_data = pd.concat(all_residuals_list, ignore_index=True)
            else:
                # Not faceted, add single all residuals
                all_residuals = tunnel_df[[value_column]].dropna().copy()
                all_residuals[indicator_column_to_use] = 'All'
                boxplot_data = pd.concat([boxplot_data, all_residuals], ignore_index=True)

            # ---- Distribution equality check: 'All residuals' vs '0' ----
            def _ks_compare_and_report(a_vals, b_vals, label_suffix=''):
                a = np.asarray(a_vals.dropna())
                b = np.asarray(b_vals.dropna())
                if a.size == 0 or b.size == 0:
                    print(f"Statistical tests skipped (empty sample){label_suffix}: sizes a={a.size}, b={b.size}")
                    return

                # Compare full samples (no trimming of extremes)
                a_f = a
                b_f = b

                # KS test for distribution equality
                ks_res = ks_2samp(a_f, b_f)
                ks_stat = getattr(ks_res, 'statistic', None)
                ks_pval = getattr(ks_res, 'pvalue', None)
                print(f"KS test{label_suffix}: statistic={ks_stat:.4f}, pvalue={ks_pval:.15e}, n_a={len(a_f)}, n_b={len(b_f)}")

                # Test for normality using Shapiro-Wilk test
                print(f"Testing normality{label_suffix}:")
                
                try:
                    # D'Agostino-Pearson test for normality (H0: data is normal)
                    # More appropriate for large samples (N > 5000) than Shapiro-Wilk
                    # p-value > 0.05 suggests data is normally distributed
                    # Based on skewness and kurtosis
                    norm_a_stat, norm_a_p = normaltest(a_f)
                    norm_b_stat, norm_b_p = normaltest(b_f)
                    
                    print(f"  D'Agostino-Pearson test (a): statistic={norm_a_stat:.4f}, pvalue={norm_a_p:.15e}, normal={norm_a_p > 0.05}")
                    print(f"  D'Agostino-Pearson test (b): statistic={norm_b_stat:.4f}, pvalue={norm_b_p:.15e}, normal={norm_b_p > 0.05}")
                    
                    # Determine if both distributions are normal
                    both_normal = (norm_a_p > 0.05) and (norm_b_p > 0.05)
                    
                    if both_normal:
                        # Both distributions are normal - apply ANOVA
                        print(f"  Both distributions are normal - applying ANOVA")
                        f_stat, f_pval = f_oneway(a_f, b_f)
                        mean_a = np.mean(a_f)
                        mean_b = np.mean(b_f)
                        print(f"ANOVA test{label_suffix}: statistic={f_stat:.4f}, pvalue={f_pval:.15e} | means: a={mean_a:.4f}, b={mean_b:.4f}, diff={mean_a-mean_b:.4f}")
                    else:
                        # At least one distribution is not normal - use Mann-Whitney U test
                        print(f"  At least one distribution is not normal - using Mann-Whitney U test instead")
                        mwu_res = mannwhitneyu(a_f, b_f, alternative='two-sided')
                        mwu_stat = getattr(mwu_res, 'statistic', None)
                        mwu_pval = getattr(mwu_res, 'pvalue', None)
                        median_a = np.median(a_f)
                        median_b = np.median(b_f)
                        print(f"Mann-Whitney U test{label_suffix}: statistic={mwu_stat:.4f}, pvalue={mwu_pval:.15e} | medians: a={median_a:.4f}, b={median_b:.4f}, diff={median_a-median_b:.4f}")
                except Exception as e:
                    print(f"Error performing normality tests{label_suffix}: {e}")
                    # Fallback to Mann-Whitney U if tests fail
                    mwu_res = mannwhitneyu(a_f, b_f, alternative='two-sided')
                    mwu_stat = getattr(mwu_res, 'statistic', None)
                    mwu_pval = getattr(mwu_res, 'pvalue', None)
                    median_a = np.median(a_f)
                    median_b = np.median(b_f)
                    print(f"Mann-Whitney U test{label_suffix}: statistic={mwu_stat:.4f}, pvalue={mwu_pval:.15e} | medians: a={median_a:.4f}, b={median_b:.4f}, diff={median_a-median_b:.4f}")
                
            # Determine columns used for no-match identification
            if no_match_cols and len(no_match_cols) == 2:
                serie_a_col, champions_col = no_match_cols
            else:
                serie_a_col = indicator_column
                champions_col = second_indicator_column if second_indicator_column is not None else indicator_column

            all_vals_all = tunnel_df[value_column].dropna()
            no_match_vals_all = df_orig[(df_orig[tunnel_column] == tunnel) & (df_orig.get(serie_a_col, 0) == 0) & (df_orig.get(champions_col, 0) == 0)][value_column].dropna()
            _ks_compare_and_report(all_vals_all, no_match_vals_all, label_suffix=f" for tunnel {tunnel} ALL_vs_NoMatch")

            # If faceted and facet_column not present, fill with default value
            if use_facets and facet_column not in boxplot_data.columns:
                boxplot_data[facet_column] = 'All residuals'

            if plot_tunnels:
                # Violin plot by indicator value (with facets if applicable)
                x_limits = ['All', '0', '1', '2', '3', '4', '5']
                if use_facets and facet_column in boxplot_data.columns:
                    violin_fig = (
                        ggplot(boxplot_data, aes(x=indicator_column_to_use, y=value_column, fill=indicator_column_to_use))
                        + geom_violin(alpha=0.7, show_legend=False)
                        + geom_boxplot(width=0.15, alpha=0.8, fill='white', show_legend=False, outlier_alpha=0.5, outlier_size=1)
                        + scale_fill_manual(values={**colors, 'All': 'deepskyblue'})
                        + scale_x_discrete(limits=x_limits)
                        + facet_wrap(f'~{facet_column}', nrow=1)
                        + labs(x='Number of matches', y='Traffic (% of capacity)', title='')
                        + theme_bw()
                        + theme(
                            plot_title=element_text(size=26, face='bold'),
                            axis_title=element_text(size=22),
                            axis_text=element_text(size=20),
                            strip_text=element_text(size=22, face='bold')
                        )
                    )
                else:
                    violin_fig = (
                        ggplot(boxplot_data, aes(x=indicator_column_to_use, y=value_column, fill=indicator_column_to_use))
                        + geom_violin(alpha=0.7, show_legend=False)
                        + geom_boxplot(width=0.15, alpha=0.8, fill='white', show_legend=False, outlier_alpha=0.5, outlier_size=1)
                        + scale_fill_manual(values={**colors, 'All': 'deepskyblue'})
                        + scale_x_discrete(limits=x_limits)
                        + labs(x='Number of matches', y='Traffic (% of capacity)', title='')
                        + theme_bw()
                        + theme(
                            plot_title=element_text(size=26, face='bold'),
                            axis_title=element_text(size=22),
                            axis_text=element_text(size=20)
                        )
                    )
                violin_fig.save(os.path.join(pdf_dir, f"violin_{tunnel}_{forecast_date.strftime('%Y-%m-%d')}_{forecast_days}d.png"), dpi=150, width=30, height=15, units='cm')

            # CDF
            cdf_data = []
            indicator_values = tunnel_df[indicator_column_to_use].dropna().unique()
            
            for ind in sorted(indicator_values, key=lambda x: int(x) if isinstance(x, (int, float)) else int(x.split()[0]) if isinstance(x, str) else x):
                group_values = tunnel_df[tunnel_df[indicator_column_to_use] == ind][value_column].dropna().sort_values()
                if len(group_values) > 0:
                    cdf_vals = np.arange(1, len(group_values) + 1) / len(group_values)
                    x_full = np.concatenate([[val_min], group_values.values, [val_max]])
                    cdf_full = np.concatenate([[0], cdf_vals, [1]])
                else:
                    x_full = np.array([val_min, val_max])
                    cdf_full = np.array([0, 1])

                row_dict = {
                    value_column: x_full,
                    'cdf': cdf_full,
                    indicator_column_to_use: [str(int(ind)) if isinstance(ind, (int, float)) else str(ind)] * len(x_full)
                }
                
                # Add facet column if applicable
                if use_facets and facet_column in tunnel_df.columns:
                    facet_vals = tunnel_df[tunnel_df[indicator_column_to_use] == ind][facet_column].iloc[0] if len(tunnel_df[tunnel_df[indicator_column_to_use] == ind]) > 0 else ''
                    row_dict[facet_column] = [facet_vals] * len(x_full)
                
                # Create DataFrame with arrays as columns (one row per point)
                cdf_data.append(pd.DataFrame(row_dict))

            cdf_plot_df = pd.concat(cdf_data, ignore_index=True)

            if plot_tunnels:
                if use_facets and facet_column in cdf_plot_df.columns:
                    cdf_plot = (
                        ggplot(cdf_plot_df, aes(x=value_column, y='cdf', color=indicator_column_to_use))
                        + geom_step(alpha=0.8)
                        + scale_color_manual(values={str(k): v for k, v in colors.items()}, labels={str(k): v for k, v in labels.items()}, name='Matches')
                        + facet_wrap(f'~{facet_column}', nrow=1)
                        + labs(x=value_column, y='Cumulative Probability')
                        + theme_large_fonts
                    )
                else:
                    cdf_plot = (
                        ggplot(cdf_plot_df, aes(x=value_column, y='cdf', color=indicator_column_to_use))
                        + geom_step(alpha=0.8)
                        + scale_color_manual(values=colors, labels=labels, name='Matches')
                        + labs(x=value_column, y='Cumulative Probability')
                        + theme_large_fonts
                    )
                cdf_plot.save(os.path.join(cdf_dir, f"cdf_{tunnel}_{forecast_date.strftime('%Y-%m-%d')}_{forecast_days}d.png"), dpi=150)

            # Additional comparison: empirical CDF of all residuals vs no-match residuals
            try:
                all_vals = tunnel_df[value_column].dropna().sort_values().values
                # determine no-match columns
                if no_match_cols and len(no_match_cols) == 2:
                    serie_a_col, champions_col = no_match_cols
                else:
                    serie_a_col = indicator_column
                    champions_col = second_indicator_column if second_indicator_column is not None else indicator_column

                no_match_vals = df_orig[(df_orig[tunnel_column] == tunnel) & (df_orig.get(serie_a_col, 0) == 0) & (df_orig.get(champions_col, 0) == 0)][value_column].dropna().sort_values().values

                if len(all_vals) > 0 and len(no_match_vals) > 0:
                    # Create a common x-grid and empirical CDFs via searchsorted
                    x_min = float(min(all_vals.min(), no_match_vals.min()))
                    x_max = float(max(all_vals.max(), no_match_vals.max()))
                    x_grid = np.linspace(x_min, x_max, 300)
                    cdf_all_grid = np.searchsorted(all_vals, x_grid, side='right') / len(all_vals)
                    cdf_no_grid = np.searchsorted(no_match_vals, x_grid, side='right') / len(no_match_vals)

                    # Test for normality
                    print(f"\nCDF comparison for {tunnel}:")
                    
                    # KS test for distribution equality
                    ks_res = ks_2samp(all_vals, no_match_vals)
                    print(f"KS test: statistic={ks_res.statistic:.4f}, pvalue={ks_res.pvalue:.15e}")
                    
                    print(f"Testing normality:")
                    try:
                        # D'Agostino-Pearson test for normality (more appropriate for large samples)
                        norm_all_stat, norm_all_p = normaltest(all_vals)
                        norm_no_stat, norm_no_p = normaltest(no_match_vals)
                        
                        print(f"  D'Agostino-Pearson test (All): statistic={norm_all_stat:.4f}, pvalue={norm_all_p:.15e}, normal={norm_all_p > 0.05}")
                        print(f"  D'Agostino-Pearson test (No-match): statistic={norm_no_stat:.4f}, pvalue={norm_no_p:.15e}, normal={norm_no_p > 0.05}")
                        
                        # Determine if both distributions are normal
                        both_normal = (norm_all_p > 0.05) and (norm_no_p > 0.05)
                        
                        if both_normal:
                            # Both distributions are normal - apply ANOVA
                            print(f"  Both distributions are normal - applying ANOVA")
                            f_stat, f_pval = f_oneway(all_vals, no_match_vals)
                            test_result = f"KS D={ks_res.statistic:.4f} p={ks_res.pvalue:.15e} | ANOVA: F={f_stat:.4f} p={f_pval:.15e}"
                            print(test_result)
                        else:
                            # At least one distribution is not normal - use Mann-Whitney U test
                            print(f"  At least one distribution is not normal - using Mann-Whitney U test")
                            mwu_res = mannwhitneyu(all_vals, no_match_vals, alternative='two-sided')
                            test_result = f"KS D={ks_res.statistic:.4f} p={ks_res.pvalue:.15e} | MWU: U={mwu_res.statistic:.4f} p={mwu_res.pvalue:.15e}"
                            print(test_result)
                    except Exception as e:
                        print(f"Error performing normality tests: {e}")
                        # Fallback to Mann-Whitney U
                        mwu_res = mannwhitneyu(all_vals, no_match_vals, alternative='two-sided')
                        test_result = f"KS D={ks_res.statistic:.4f} p={ks_res.pvalue:.15e} | MWU: U={mwu_res.statistic:.4f} p={mwu_res.pvalue:.15e}"
                        print(test_result)

                    plt.figure(figsize=(6, 4))
                    plt.step(x_grid, cdf_all_grid, where='post', label='All residuals', color='blue')
                    plt.step(x_grid, cdf_no_grid, where='post', label='No-match (0)', color='red')
                    plt.xlabel(value_column)
                    plt.ylabel('Empirical CDF')
                    plt.title(f'CDF compare {tunnel}: {test_result}')
                    plt.legend()
                    out_path = os.path.join(cdf_dir, f"cdf_compare_{tunnel}_{forecast_date.strftime('%Y-%m-%d')}_{forecast_days}d.png") if forecast_date is not None else os.path.join(cdf_dir, f"cdf_compare_{tunnel}.png")
                    plt.savefig(out_path, dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    print(f"CDF compare skipped for {tunnel}: insufficient samples (all={len(all_vals)}, no_match={len(no_match_vals)})")
            except Exception as e:
                print(f"CDF compare failed for {tunnel}: {e}")

    print(f"PDF and CDF plots saved to {output_dir}")

def compare_kde_modes(single_kde_metrics_path, indicator_kde_metrics_path, output_dir=None):
    """
        Compare PICP (matches) between SingleKDE and IndicatorKDE modes.
        Creates scatter plot with line showing equal performance.
        
        Args:
            single_kde_metrics_path: path to forecast_metrics_individual_..._SingleKDE_*.csv
            indicator_kde_metrics_path: path to forecast_metrics_individual_..._IndicatorKDE_*.csv
            output_dir: directory to save plot (default: parent of indicator_kde file)
    """
    
    if not os.path.exists(single_kde_metrics_path) or not os.path.exists(indicator_kde_metrics_path):
        return None
    
    single_kde_df = pd.read_csv(single_kde_metrics_path)
    indicator_kde_df = pd.read_csv(indicator_kde_metrics_path)
    
    # Exclude AGGREGATED_ALL_TUNNELS row
    single_kde_df = single_kde_df[single_kde_df['Tunnel'] != 'AGGREGATED_ALL_TUNNELS']
    indicator_kde_df = indicator_kde_df[indicator_kde_df['Tunnel'] != 'AGGREGATED_ALL_TUNNELS']
    
    if output_dir is None:
        output_dir = os.path.dirname(indicator_kde_metrics_path)
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_df = single_kde_df[['Tunnel', 'PICP_Matches']].copy()
    comparison_df.rename(columns={'PICP_Matches': 'PICP_Matches_SingleKDE'}, inplace=True)
    indicator_cols = indicator_kde_df[['Tunnel', 'PICP_Matches']].copy()
    indicator_cols.rename(columns={'PICP_Matches': 'PICP_Matches_IndicatorKDE'}, inplace=True)
    comparison_df = comparison_df.merge(indicator_cols, on='Tunnel', how='inner')
    comparison_df['ΔPICP_mode'] = comparison_df['PICP_Matches_IndicatorKDE'] - comparison_df['PICP_Matches_SingleKDE']
    
    # Exclude NaN values and cases with zero/very low PICP_Matches (indicates no GT data for matches)
    valid_mask = ~(comparison_df['PICP_Matches_SingleKDE'].isna() | comparison_df['PICP_Matches_IndicatorKDE'].isna())
    valid_mask &= (comparison_df['PICP_Matches_SingleKDE'] > 0) & (comparison_df['PICP_Matches_IndicatorKDE'] > 0)
    valid_comparison = comparison_df[valid_mask].copy()
    delta_valid = valid_comparison['ΔPICP_mode'].dropna()
    
    stats = {
        'num_tunnels_compared': len(valid_comparison),
        'mean_improvement': delta_valid.mean(),
        'tunnels_indicator_better': (delta_valid > 0).sum(),
        'fraction_indicator_better': (delta_valid > 0).sum() / len(delta_valid) if len(delta_valid) > 0 else 0
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_vals = valid_comparison['PICP_Matches_SingleKDE']
    y_vals = valid_comparison['PICP_Matches_IndicatorKDE']
    diff_vals = y_vals - x_vals
    eps = 1e-12
    point_colors = np.where(diff_vals > eps, 'green', np.where(diff_vals < -eps, 'red', 'yellow'))

    scatter = ax.scatter(x_vals,
                        y_vals,
                        c=point_colors,
                        s=100,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.7)
    
    min_val = min(valid_comparison['PICP_Matches_SingleKDE'].min(), 
                 valid_comparison['PICP_Matches_IndicatorKDE'].min())
    max_val = max(valid_comparison['PICP_Matches_SingleKDE'].max(), 
                 valid_comparison['PICP_Matches_IndicatorKDE'].max())
    line_handle, = ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2.5, alpha=0.6, label='Equal performance', zorder=1)
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)
    
    ax.set_xlabel('PICP (matches, no schedule)', fontsize=16, fontweight='bold')
    ax.set_ylabel('PICP (matches, schedule)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    legend_handles = [
        line_handle,
        plt.Line2D([0], [0], marker='o', color='w', label='Improved', markerfacecolor='green', markeredgecolor='black', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Worsened', markerfacecolor='red', markeredgecolor='black', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='No change', markerfacecolor='yellow', markeredgecolor='black', markersize=8)
    ]
    
    textstr = f'Tunnels improved: {stats["tunnels_indicator_better"]}/{stats["num_tunnels_compared"]} ({stats["fraction_indicator_better"]*100:.1f}%)\nMean improvement: {stats["mean_improvement"]:+.4f}'
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='left', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(handles=legend_handles, fontsize=15, loc='lower right', framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'kde_mode_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats


def compare_kde_modes_multi(results_dir, prediction_interval, scenarios, output_dir=None):
    """
        Compare PICP (matches) between SingleKDE and IndicatorKDE across scenarios.
        Scenarios are a list of (forecast_date, forecast_days) pairs.
    """

    if output_dir is None:
        output_dir = os.path.join(results_dir, f"KDE_Comparison_Multi_Perc{prediction_interval}")
    os.makedirs(output_dir, exist_ok=True)

    scenario_rows = []
    for forecast_date, forecast_days in scenarios:
        single_path = os.path.join(
            results_dir,
            f"Perc{prediction_interval}SingleKDE_D{forecast_date}_F{forecast_days}d",
            f"forecast_metrics_individual_Perc{prediction_interval}_SingleKDE_D{forecast_date}_F{forecast_days}d.csv",
        )
        indicator_path = os.path.join(
            results_dir,
            f"Perc{prediction_interval}IndicatorKDE_D{forecast_date}_F{forecast_days}d",
            f"forecast_metrics_individual_Perc{prediction_interval}_IndicatorKDE_D{forecast_date}_F{forecast_days}d.csv",
        )

        if not os.path.exists(single_path) or not os.path.exists(indicator_path):
            print(f"Warning: missing files for scenario D{forecast_date} F{forecast_days}d")
            continue

        single_df = pd.read_csv(single_path)
        indicator_df = pd.read_csv(indicator_path)
        
        # Exclude AGGREGATED_ALL_TUNNELS row
        single_df = single_df[single_df['Tunnel'] != 'AGGREGATED_ALL_TUNNELS']
        indicator_df = indicator_df[indicator_df['Tunnel'] != 'AGGREGATED_ALL_TUNNELS']

        comparison_df = single_df[['Tunnel', 'PICP_Matches']].copy()
        comparison_df.rename(columns={'PICP_Matches': 'PICP_Matches_SingleKDE'}, inplace=True)

        indicator_cols = indicator_df[['Tunnel', 'PICP_Matches']].copy()
        indicator_cols.rename(columns={'PICP_Matches': 'PICP_Matches_IndicatorKDE'}, inplace=True)

        comparison_df = comparison_df.merge(indicator_cols, on='Tunnel', how='inner')
        comparison_df['ΔPICP_mode'] = comparison_df['PICP_Matches_IndicatorKDE'] - comparison_df['PICP_Matches_SingleKDE']
        comparison_df['Scenario'] = f"D{forecast_date}_F{forecast_days}d"
        comparison_df['Forecast_Date'] = str(forecast_date)
        comparison_df['Forecast_Days'] = int(forecast_days)

        # Exclude NaN values and cases with zero/very low PICP_Matches (indicates no GT data for matches)
        valid_mask = ~(comparison_df['PICP_Matches_SingleKDE'].isna() | comparison_df['PICP_Matches_IndicatorKDE'].isna())
        valid_mask &= (comparison_df['PICP_Matches_SingleKDE'] > 0.2) & (comparison_df['PICP_Matches_IndicatorKDE'] > 0.2)
        scenario_rows.append(comparison_df[valid_mask].copy())

    if not scenario_rows:
        print("Error: no valid scenarios found for comparison.")
        return None

    combined_df = pd.concat(scenario_rows, ignore_index=True)
    delta_valid = combined_df['ΔPICP_mode'].dropna()

    stats = {
        'num_tunnels_compared': len(combined_df),
        'mean_improvement': delta_valid.mean(),
        'tunnels_indicator_better': (delta_valid > 0).sum(),
        'fraction_indicator_better': (delta_valid > 0).sum() / len(delta_valid) if len(delta_valid) > 0 else 0,
    }

    # Create color palette for forecast days
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create facets by date; within each date, use different colors for different forecast days
    date_labels = sorted(combined_df['Forecast_Date'].unique())
    days_labels = sorted(combined_df['Forecast_Days'].unique())
    
    color_map_days = {days: color_palette[i % len(color_palette)] for i, days in enumerate(days_labels)}

    # Calculate global min/max for uniform axes across all subplots
    global_min = min(combined_df['PICP_Matches_SingleKDE'].min(), combined_df['PICP_Matches_IndicatorKDE'].min())
    global_max = max(combined_df['PICP_Matches_SingleKDE'].max(), combined_df['PICP_Matches_IndicatorKDE'].max())
    axis_margin = (global_max - global_min) * 0.05
    global_min -= axis_margin
    global_max += axis_margin

    # Create faceted subplots
    num_dates = len(date_labels)
    num_cols = max(2, int(np.ceil(np.sqrt(num_dates))))
    num_rows = int(np.ceil(num_dates / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows))
    axes_flat = axes.flatten() if num_dates > 1 else [axes]

    for idx, date_label in enumerate(date_labels):
        ax = axes_flat[idx]
        date_df = combined_df[combined_df['Forecast_Date'] == date_label]
        
        ax.plot([global_min, global_max], [global_min, global_max], 'k--', linewidth=2.5, alpha=0.6, zorder=1)
        ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)
        
        # Set uniform axis limits
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

        # Plot all forecasting days for this date with different colors
        for days in days_labels:
            days_df = date_df[date_df['Forecast_Days'] == days]
            if len(days_df) == 0:
                continue
                
            x_vals = days_df['PICP_Matches_SingleKDE']
            y_vals = days_df['PICP_Matches_IndicatorKDE']

            ax.scatter(
                x_vals,
                y_vals,
                c=color_map_days[days],
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.7,
                marker='o',
            )

        ax.set_xlabel('PICP (matches, no schedule)', fontsize=14, fontweight='bold')
        ax.set_ylabel('PICP (matches, schedule)', fontsize=14, fontweight='bold')
        
        # Convert date format from YYYYMMDD to readable format
        date_obj = pd.to_datetime(date_label, format='%Y%m%d')
        readable_date = date_obj.strftime('%B %d, %Y')
        ax.set_title(f'Training until {readable_date}', fontsize=15, fontweight='bold')
        
        ax.grid(True, alpha=0.3)

        # Stats box for this date
        date_delta = date_df['PICP_Matches_IndicatorKDE'] - date_df['PICP_Matches_SingleKDE']
        date_delta = date_delta.dropna()
        date_stats = {
            'improved': (date_delta > 0).sum(),
            'total': len(date_delta),
            'fraction': (date_delta > 0).sum() / len(date_delta) if len(date_delta) > 0 else 0,
            'mean': date_delta.mean(),
        }
        
        # Print tunnels that worsened
        worsened_mask = (date_df['PICP_Matches_IndicatorKDE'] - date_df['PICP_Matches_SingleKDE']) < 0
        worsened_tunnels = date_df[worsened_mask]['Tunnel'].unique()
        if len(worsened_tunnels) > 0:
            print(f"\nTraining until {readable_date}:")
            print(f"  Tunnels that worsened: {', '.join(sorted(worsened_tunnels))}")
        
        textstr = (
            f'Improved: {date_stats["improved"]}/{date_stats["total"]} '
            f'({date_stats["fraction"]*100:.1f}%)\n'
            f'Mean: {date_stats["mean"]:+.4f}'
        )
        ax.text(
            0.87,
            0.22,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        )

    # Hide extra subplots if num_dates is not a perfect square
    for idx in range(num_dates, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Create shared legend
    legend_handles = [
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2.5, label='Equal performance'),
    ]
    
    # Add legend entries for forecast days (colors)
    for days in days_labels:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker='o',
                color=color_map_days[days],
                label=f'{days} days forecast',
                markerfacecolor=color_map_days[days],
                markeredgecolor='black',
                markersize=8,
                linestyle='None'
            )
        )

    # Add legend to first subplot
    axes_flat[0].legend(handles=legend_handles, fontsize=10, loc='lower right', ncol=2, framealpha=0.95)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'kde_mode_comparison_faceted.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved faceted comparison plot: {plot_path}")
    return stats