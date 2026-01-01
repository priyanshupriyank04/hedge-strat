"""
Script to add basic OHLC-derived features to NIFTY50 and India VIX daily data.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Input file paths (edit these if your files have different names)
INPUT_DIR = "data"
NIFTY_INPUT = "NIFTY_50.csv"
VIX_INPUT = "INDIA_VIX.csv"

# Output directory
OUTPUT_DIR = "features_output"
NIFTY_OUTPUT = "nifty_features.csv"
VIX_OUTPUT = "vix_features.csv"


def compute_custom_ema(close_series, N):
    """
    Compute custom EMA with special initialization.
    
    For rows where index < N-1:
        ema[t] = mean(close[0:t])   (simple average of available rows)
    For rows where index >= N-1:
        alpha = 2 / (N + 1)
        ema[t] = alpha * close[t] + (1 - alpha) * ema[t-1]
    
    Parameters:
    -----------
    close_series : pandas Series
        Series of close prices
    N : int
        EMA window size
    
    Returns:
    --------
    pandas Series : EMA values
    """
    ema = pd.Series(index=close_series.index, dtype=float)
    alpha = 2.0 / (N + 1)
    
    for t in range(len(close_series)):
        if pd.isna(close_series.iloc[t]):
            ema.iloc[t] = np.nan
        elif t < N - 1:
            # For early rows: simple average of available rows
            available = close_series.iloc[:t+1].dropna()
            if len(available) > 0:
                ema.iloc[t] = available.mean()
            else:
                ema.iloc[t] = np.nan
        else:
            # For rows >= N-1: use EMA formula
            if pd.isna(ema.iloc[t-1]):
                # If previous EMA is NaN, fall back to mean
                available = close_series.iloc[:t+1].dropna()
                if len(available) > 0:
                    ema.iloc[t] = available.mean()
                else:
                    ema.iloc[t] = np.nan
            else:
                ema.iloc[t] = alpha * close_series.iloc[t] + (1 - alpha) * ema.iloc[t-1]
    
    return ema


def compute_wilder_rsi_14(close_series):
    """
    Compute Wilder RSI-14 on close prices.
    
    For first periods where we have < 14 values, compute avg_gain and avg_loss 
    as simple mean of available gain/loss so far.
    Once index >= 13, use Wilder smoothing.
    
    Parameters:
    -----------
    close_series : pandas Series
        Series of close prices
    
    Returns:
    --------
    pandas Series : RSI values
    """
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    rsi = pd.Series(index=close_series.index, dtype=float)
    avg_gain = pd.Series(index=close_series.index, dtype=float)
    avg_loss = pd.Series(index=close_series.index, dtype=float)
    
    for t in range(len(close_series)):
        if t == 0:
            # First row: delta is NaN, so gain and loss are NaN
            avg_gain.iloc[t] = np.nan
            avg_loss.iloc[t] = np.nan
            rsi.iloc[t] = np.nan
        elif t < 14:
            # For early rows: simple mean of available gain/loss
            available_gain = gain.iloc[1:t+1].dropna()
            available_loss = loss.iloc[1:t+1].dropna()
            
            if len(available_gain) > 0:
                avg_gain.iloc[t] = available_gain.mean()
            else:
                avg_gain.iloc[t] = 0.0
            
            if len(available_loss) > 0:
                avg_loss.iloc[t] = available_loss.mean()
            else:
                avg_loss.iloc[t] = 0.0
        else:
            # For rows >= 14: use Wilder smoothing
            if pd.isna(avg_gain.iloc[t-1]) or pd.isna(avg_loss.iloc[t-1]):
                # Fallback to simple mean if previous values are NaN
                available_gain = gain.iloc[1:t+1].dropna()
                available_loss = loss.iloc[1:t+1].dropna()
                
                if len(available_gain) > 0:
                    avg_gain.iloc[t] = available_gain.mean()
                else:
                    avg_gain.iloc[t] = 0.0
                
                if len(available_loss) > 0:
                    avg_loss.iloc[t] = available_loss.mean()
                else:
                    avg_loss.iloc[t] = 0.0
            else:
                # Wilder smoothing
                gain_val = gain.iloc[t] if pd.notna(gain.iloc[t]) else 0.0
                loss_val = loss.iloc[t] if pd.notna(loss.iloc[t]) else 0.0
                
                avg_gain.iloc[t] = (avg_gain.iloc[t-1] * 13 + gain_val) / 14
                avg_loss.iloc[t] = (avg_loss.iloc[t-1] * 13 + loss_val) / 14
        
        # Compute RSI
        if t > 0:
            ag = avg_gain.iloc[t]
            al = avg_loss.iloc[t]
            
            if pd.isna(ag) or pd.isna(al):
                rsi.iloc[t] = np.nan
            elif al == 0 and ag > 0:
                rsi.iloc[t] = 100.0
            elif ag == 0 and al > 0:
                rsi.iloc[t] = 0.0
            elif ag == 0 and al == 0:
                rsi.iloc[t] = 50.0
            else:
                rs = ag / al
                rsi.iloc[t] = 100 - (100 / (1 + rs))
    
    return rsi


def compute_wilder_atr_14(high_series, low_series, close_series):
    """
    Compute Wilder ATR-14 using True Range.
    
    True Range = max(
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    )
    
    For first rows (<14): ATR = mean(TR over available rows)
    After that: ATR[t] = ((ATR[t-1] * 13) + TR[t]) / 14
    
    Parameters:
    -----------
    high_series : pandas Series
        Series of high prices
    low_series : pandas Series
        Series of low prices
    close_series : pandas Series
        Series of close prices
    
    Returns:
    --------
    pandas Series : ATR values
    """
    prev_close = close_series.shift(1)
    
    # Compute True Range
    tr1 = high_series - low_series
    tr2 = (high_series - prev_close).abs()
    tr3 = (low_series - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compute ATR using Wilder smoothing
    atr = pd.Series(index=high_series.index, dtype=float)
    
    for t in range(len(high_series)):
        if pd.isna(tr.iloc[t]):
            atr.iloc[t] = np.nan
        elif t < 14:
            # For early rows: simple mean of available TR
            available_tr = tr.iloc[:t+1].dropna()
            if len(available_tr) > 0:
                atr.iloc[t] = available_tr.mean()
            else:
                atr.iloc[t] = np.nan
        else:
            # For rows >= 14: use Wilder smoothing
            if pd.isna(atr.iloc[t-1]):
                # Fallback to simple mean if previous ATR is NaN
                available_tr = tr.iloc[:t+1].dropna()
                if len(available_tr) > 0:
                    atr.iloc[t] = available_tr.mean()
                else:
                    atr.iloc[t] = np.nan
            else:
                atr.iloc[t] = ((atr.iloc[t-1] * 13) + tr.iloc[t]) / 14
    
    return atr


def normalize_column_names(df):
    """
    Normalize column names to lowercase and handle common variations.
    Returns a copy of the dataframe with normalized column names.
    """
    df = df.copy()
    
    # Create a mapping for column name normalization
    column_mapping = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        
        # Map date variations
        if col_lower in ['date', 'datetime', 'time', 'timestamp']:
            column_mapping[col] = 'date'
        # Map OHLC variations
        elif col_lower in ['open', 'o']:
            column_mapping[col] = 'open'
        elif col_lower in ['high', 'h']:
            column_mapping[col] = 'high'
        elif col_lower in ['low', 'l']:
            column_mapping[col] = 'low'
        elif col_lower in ['close', 'c', 'closing']:
            column_mapping[col] = 'close'
        # Keep other columns as lowercase
        else:
            column_mapping[col] = col_lower
    
    df = df.rename(columns=column_mapping)
    return df


def process_ohlc_file(input_path, output_path):
    """
    Process a single OHLC CSV file to add derived features.
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV file
    output_path : str
        Path to output CSV file
    
    Returns:
    --------
    dict : Summary statistics
    """
    # Read the CSV file
    print(f"\nProcessing: {input_path}")
    df = pd.read_csv(input_path)
    
    # Normalize column names
    df = normalize_column_names(df)
    
    # Check required columns
    required_cols = ['date', 'open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Convert OHLC columns to float, coercing errors to NaN
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date ascending and reset index
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create derived features
    # 1. close_open = Close / Open
    df['close_open'] = np.where(
        (df['open'] > 0) & (df['open'].notna()),
        df['close'] / df['open'],
        np.nan
    )
    
    # 2. high_open = High / Open
    df['high_open'] = np.where(
        (df['open'] > 0) & (df['open'].notna()),
        df['high'] / df['open'],
        np.nan
    )
    
    # 3. low_open = Low / Open
    df['low_open'] = np.where(
        (df['open'] > 0) & (df['open'].notna()),
        df['low'] / df['open'],
        np.nan
    )
    
    # 4. high_low = High - Low
    df['high_low'] = df['high'] - df['low']
    
    # Ensure high_low is numeric (should already be, but double-check)
    df['high_low'] = pd.to_numeric(df['high_low'], errors='coerce')
    
    # Compute normalized range features (hl_norm_*)
    # Data is already sorted by date ascending at this point
    windows = [5, 15, 28, 60, 200]
    for window in windows:
        col_name = f'hl_norm_{window}'
        rolling_mean = df['high_low'].rolling(window=window, min_periods=1).mean()
        
        # Division guard: if rolling_mean <= 0 or NaN, set hl_norm = NaN
        df[col_name] = np.where(
            (rolling_mean > 0) & (rolling_mean.notna()),
            df['high_low'] / rolling_mean,
            np.nan
        )
    
    # Replace +/-inf with NaN for all hl_norm columns
    hl_norm_cols = [f'hl_norm_{w}' for w in windows]
    for col in hl_norm_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Compute EMA features on close price
    # Data is already sorted by date ascending at this point
    ema_windows = [5, 9, 15, 21, 28, 60, 200]
    for window in ema_windows:
        col_name = f'ema_{window}'
        df[col_name] = compute_custom_ema(df['close'], window)
    
    # Compute EMA ratio features
    ema_ratios = [
        ('ema_5', 'ema_15', 'ema_5_over_ema_15'),
        ('ema_9', 'ema_21', 'ema_9_over_ema_21'),
        ('ema_15', 'ema_28', 'ema_15_over_ema_28'),
        ('ema_28', 'ema_60', 'ema_28_over_ema_60'),
        ('ema_60', 'ema_200', 'ema_60_over_ema_200'),
        ('ema_15', 'ema_200', 'ema_15_over_ema_200'),
        ('ema_28', 'ema_200', 'ema_28_over_ema_200'),
        ('ema_5', 'ema_60', 'ema_5_over_ema_60'),
        ('ema_15', 'ema_60', 'ema_15_over_ema_60')
    ]
    
    for num_col, den_col, ratio_col in ema_ratios:
        df[ratio_col] = np.where(
            (df[den_col] > 0) & (df[den_col].notna()),
            df[num_col] / df[den_col],
            np.nan
        )
    
    # Compute close_minus_ema200_over_ema200
    df['close_minus_ema200_over_ema200'] = np.where(
        (df['ema_200'] > 0) & (df['ema_200'].notna()),
        (df['close'] - df['ema_200']) / df['ema_200'],
        np.nan
    )
    
    # Replace +/-inf with NaN for all EMA-related columns
    ema_cols = [f'ema_{w}' for w in ema_windows]
    ema_ratio_cols = [ratio_col for _, _, ratio_col in ema_ratios]
    all_ema_cols = ema_cols + ema_ratio_cols + ['close_minus_ema200_over_ema200']
    
    for col in all_ema_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Detect if processing NIFTY or VIX based on output path
    is_nifty = 'nifty' in output_path.lower()
    is_vix = 'vix' in output_path.lower()
    
    # ==================================================
    # A) RETURNS (CLOSE BASED)
    # ==================================================
    if is_nifty:
        # NIFTY returns
        for period in [5, 28, 60, 200]:
            shifted_close = df['close'].shift(period)
            df[f'ret_{period}'] = np.where(
                (shifted_close > 0) & (shifted_close.notna()),
                df['close'] / shifted_close - 1,
                np.nan
            )
    elif is_vix:
        # VIX returns
        for period in [5, 28]:
            shifted_close = df['close'].shift(period)
            df[f'ret_{period}'] = np.where(
                (shifted_close > 0) & (shifted_close.notna()),
                df['close'] / shifted_close - 1,
                np.nan
            )
    
    # ==================================================
    # B) NIFTY GAP RETURN
    # ==================================================
    if is_nifty:
        prev_close = df['close'].shift(1)
        df['gap_ret'] = np.where(
            (prev_close > 0) & (prev_close.notna()),
            (df['open'] - prev_close) / prev_close,
            np.nan
        )
    
    # ==================================================
    # C) CLOSE / HIGHEST(N)
    # ==================================================
    for window in [20, 60]:
        rolling_max = df['close'].rolling(window=window, min_periods=1).max()
        df[f'close_over_highest_{window}'] = np.where(
            (rolling_max > 0) & (rolling_max.notna()),
            df['close'] / rolling_max,
            np.nan
        )
    
    # ==================================================
    # D) RSI for NIFTY (Wilder RSI-14)
    # ==================================================
    if is_nifty:
        df['rsi_14'] = compute_wilder_rsi_14(df['close'])
    
    # ==================================================
    # 1) NIFTY: count of down days in last 10
    # ==================================================
    if is_nifty:
        # down_day = 1 if close < close.shift(1) else 0
        # For first row, down_day = 0 (no previous close)
        prev_close = df['close'].shift(1)
        df['down_day'] = np.where(
            (prev_close.notna()) & (df['close'] < prev_close),
            1,
            0
        )
        # First row should be 0 (no previous close)
        df.loc[df.index[0], 'down_day'] = 0
        
        # down_days_10 = rolling_sum(down_day, window=10, min_periods=1)
        df['down_days_last_10'] = df['down_day'].rolling(window=10, min_periods=1).sum()
        
        # Drop the temporary down_day column
        df = df.drop(columns=['down_day'])
    
    # ==================================================
    # A) REALIZED VOLATILITY (ROLLING STD OF RETURNS) - NIFTY ONLY
    # ==================================================
    if is_nifty:
        # Compute simple returns: ret_1 = close / close.shift(1) - 1
        ret_1 = df['close'] / df['close'].shift(1) - 1
        
        # Compute rolling standard deviation with partial windows
        for window in [5, 10, 20]:
            df[f'rv_{window}'] = ret_1.rolling(window=window, min_periods=1).std(ddof=1)
    
    # ==================================================
    # B) PARKINSON VOLATILITY (DAILY) - NIFTY ONLY
    # ==================================================
    if is_nifty:
        # parkinson_vol = sqrt( (1 / (4 * ln(2))) * ( ln(high / low) ** 2 ) )
        constant = 1.0 / (4.0 * np.log(2.0))
        
        # Compute ln(high / low) with guards
        high_low_ratio = np.where(
            (df['high'] > 0) & (df['low'] > 0) & (df['high'] >= df['low']) & 
            (df['high'].notna()) & (df['low'].notna()),
            df['high'] / df['low'],
            np.nan
        )
        
        log_ratio = np.log(high_low_ratio)
        df['parkinson_vol'] = np.sqrt(constant * (log_ratio ** 2))
    
    # ==================================================
    # C) ATR (WILDER, PERIOD = 14) - NIFTY ONLY
    # ==================================================
    if is_nifty:
        df['atr_14'] = compute_wilder_atr_14(df['high'], df['low'], df['close'])
    
    # ==================================================
    # D) NORMALIZED ATR - NIFTY ONLY
    # ==================================================
    if is_nifty:
        df['natr_14'] = np.where(
            (df['close'] > 0) & (df['close'].notna()),
            df['atr_14'] / df['close'],
            np.nan
        )
    
    # ==================================================
    # E) TREND EFFICIENCY (WINDOW = 10) - NIFTY ONLY
    # ==================================================
    if is_nifty:
        # net_move = abs(close - close.shift(10))
        net_move = (df['close'] - df['close'].shift(10)).abs()
        
        # total_move = rolling_sum(abs(close - close.shift(1)), window=10, min_periods=1)
        daily_move = (df['close'] - df['close'].shift(1)).abs()
        total_move = daily_move.rolling(window=10, min_periods=1).sum()
        
        # trend_efficiency_10 = net_move / total_move
        df['trend_efficiency_10'] = np.where(
            (total_move > 0) & (total_move.notna()),
            net_move / total_move,
            np.nan
        )
    
    # ==================================================
    # F) INDIA VIX: CLOSE/EMA ratios
    # ==================================================
    if is_vix:
        for window in [5, 15, 28, 60, 200]:
            ema_col = f'ema_{window}'
            df[f'close_over_ema_{window}'] = np.where(
                (df[ema_col] > 0) & (df[ema_col].notna()),
                df['close'] / df[ema_col],
                np.nan
            )
    
    # ==================================================
    # G) INDIA VIX: EMA SPREADS
    # ==================================================
    if is_vix:
        ema_spreads = [
            ('ema_5', 'ema_15', 'ema_5_minus_ema_15'),
            ('ema_5', 'ema_28', 'ema_5_minus_ema_28'),
            ('ema_5', 'ema_60', 'ema_5_minus_ema_60'),
            ('ema_5', 'ema_200', 'ema_5_minus_ema_200'),
            ('ema_15', 'ema_28', 'ema_15_minus_ema_28'),
            ('ema_15', 'ema_60', 'ema_15_minus_ema_60'),
            ('ema_15', 'ema_200', 'ema_15_minus_ema_200'),
            ('ema_28', 'ema_60', 'ema_28_minus_ema_60'),
            ('ema_28', 'ema_200', 'ema_28_minus_ema_200'),
            ('ema_60', 'ema_200', 'ema_60_minus_ema_200')
        ]
        
        for ema1_col, ema2_col, spread_col in ema_spreads:
            df[spread_col] = df[ema1_col] - df[ema2_col]
    
    # ==================================================
    # H) INDIA VIX: Vol-regime & Memory Features
    # ==================================================
    if is_vix:
        # 1) ΔVIX (1-day change)
        df['dvix_1d'] = df['close'] - df['close'].shift(1)
        
        # 2) Vol-of-Vol (rolling std of 1D VIX returns)
        prev_close = df['close'].shift(1)
        vix_ret_1d = np.where(
            (prev_close > 0) & (prev_close.notna()),
            df['close'] / prev_close - 1,
            np.nan
        )
        df['vix_vol_of_vol_20'] = pd.Series(vix_ret_1d, index=df.index).rolling(window=20, min_periods=1).std(ddof=1)
        
        # 3) Rolling VIX percentile ranks
        def compute_percentile_rank(window_values):
            """Compute percentile rank of last value in window."""
            window_values = window_values.dropna()
            if len(window_values) == 0:
                return np.nan
            last_val = window_values.iloc[-1]
            if pd.isna(last_val):
                return np.nan
            return (window_values <= last_val).mean()
        
        for window in [20, 60, 252]:
            df[f'vix_pct_{window}'] = df['close'].rolling(window=window, min_periods=1).apply(
                compute_percentile_rank, raw=False
            )
        
        # 4) Consecutive days VIX above rolling quantile thresholds
        # Compute rolling thresholds
        thr_p60 = df['close'].rolling(window=60, min_periods=1).quantile(0.60)
        thr_p80 = df['close'].rolling(window=60, min_periods=1).quantile(0.80)
        
        # Boolean flags
        above_p60 = df['close'] > thr_p60
        above_p80 = df['close'] > thr_p80
        
        # Compute consecutive run-length counters with loop
        df['vix_cons_days_above_p60'] = 0
        df['vix_cons_days_above_p80'] = 0
        
        for t in range(len(df)):
            if t == 0:
                df.loc[df.index[t], 'vix_cons_days_above_p60'] = 1 if above_p60.iloc[t] else 0
                df.loc[df.index[t], 'vix_cons_days_above_p80'] = 1 if above_p80.iloc[t] else 0
            else:
                if above_p60.iloc[t]:
                    prev_val = df.loc[df.index[t-1], 'vix_cons_days_above_p60']
                    df.loc[df.index[t], 'vix_cons_days_above_p60'] = prev_val + 1 if pd.notna(prev_val) else 1
                else:
                    df.loc[df.index[t], 'vix_cons_days_above_p60'] = 0
                
                if above_p80.iloc[t]:
                    prev_val = df.loc[df.index[t-1], 'vix_cons_days_above_p80']
                    df.loc[df.index[t], 'vix_cons_days_above_p80'] = prev_val + 1 if pd.notna(prev_val) else 1
                else:
                    df.loc[df.index[t], 'vix_cons_days_above_p80'] = 0
    
    # Replace +/-inf with NaN for all new columns
    new_cols = []
    if is_nifty:
        new_cols.extend([f'ret_{p}' for p in [5, 28, 60, 200]])
        new_cols.append('gap_ret')
        new_cols.append('rsi_14')
        new_cols.extend(['rv_5', 'rv_10', 'rv_20'])
        new_cols.append('parkinson_vol')
        new_cols.append('atr_14')
        new_cols.append('natr_14')
        new_cols.append('trend_efficiency_10')
    elif is_vix:
        new_cols.extend([f'ret_{p}' for p in [5, 28]])
        new_cols.extend([f'close_over_ema_{w}' for w in [5, 15, 28, 60, 200]])
        new_cols.extend([
            'ema_5_minus_ema_15', 'ema_5_minus_ema_28', 'ema_5_minus_ema_60', 'ema_5_minus_ema_200',
            'ema_15_minus_ema_28', 'ema_15_minus_ema_60', 'ema_15_minus_ema_200',
            'ema_28_minus_ema_60', 'ema_28_minus_ema_200',
            'ema_60_minus_ema_200'
        ])
        new_cols.extend([
            'dvix_1d', 'vix_vol_of_vol_20',
            'vix_pct_20', 'vix_pct_60', 'vix_pct_252',
            'vix_cons_days_above_p60', 'vix_cons_days_above_p80'
        ])
    
    new_cols.extend(['close_over_highest_20', 'close_over_highest_60'])
    
    for col in new_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Create OHLC sanity check flag
    # True when:
    # - high < max(open, close, low) OR
    # - low > min(open, close, high) OR
    # - high_low < 0
    max_oc = df[['open', 'close']].max(axis=1)
    max_ocl = df[['open', 'close', 'low']].max(axis=1)
    min_oc = df[['open', 'close']].min(axis=1)
    min_och = df[['open', 'close', 'high']].min(axis=1)
    
    condition1 = df['high'] < max_ocl
    condition2 = df['low'] > min_och
    condition3 = df['high_low'] < 0
    
    df['ohlc_sanity_fail'] = condition1 | condition2 | condition3
    
    # Note: We don't save here anymore - main() will handle saving after cross-features
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate summary statistics
    hl_norm_cols = [f'hl_norm_{w}' for w in [5, 15, 28, 60, 200]]
    ema_windows = [5, 9, 15, 21, 28, 60, 200]
    ema_cols = [f'ema_{w}' for w in ema_windows]
    ema_ratios = [
        ('ema_5', 'ema_15', 'ema_5_over_ema_15'),
        ('ema_9', 'ema_21', 'ema_9_over_ema_21'),
        ('ema_15', 'ema_28', 'ema_15_over_ema_28'),
        ('ema_28', 'ema_60', 'ema_28_over_ema_60'),
        ('ema_60', 'ema_200', 'ema_60_over_ema_200'),
        ('ema_15', 'ema_200', 'ema_15_over_ema_200'),
        ('ema_28', 'ema_200', 'ema_28_over_ema_200'),
        ('ema_5', 'ema_60', 'ema_5_over_ema_60'),
        ('ema_15', 'ema_60', 'ema_15_over_ema_60')
    ]
    ema_ratio_cols = [ratio_col for _, _, ratio_col in ema_ratios]
    
    # NaN counts for all feature columns
    nan_counts = {
        'close_open': df['close_open'].isna().sum(),
        'high_open': df['high_open'].isna().sum(),
        'low_open': df['low_open'].isna().sum(),
        'high_low': df['high_low'].isna().sum()
    }
    
    # Add NaN counts for hl_norm columns
    for col in hl_norm_cols:
        nan_counts[col] = df[col].isna().sum()
    
    # Add NaN counts for EMA columns
    for col in ema_cols:
        nan_counts[col] = df[col].isna().sum()
    
    # Add NaN counts for EMA ratio columns
    for col in ema_ratio_cols:
        nan_counts[col] = df[col].isna().sum()
    
    # Add NaN count for close_minus_ema200_over_ema200
    nan_counts['close_minus_ema200_over_ema200'] = df['close_minus_ema200_over_ema200'].isna().sum()
    
    # Add NaN counts for new return columns
    if is_nifty:
        for period in [5, 28, 60, 200]:
            nan_counts[f'ret_{period}'] = df[f'ret_{period}'].isna().sum()
    elif is_vix:
        for period in [5, 28]:
            nan_counts[f'ret_{period}'] = df[f'ret_{period}'].isna().sum()
    
    # Add NaN counts for close_over_highest columns
    for window in [20, 60]:
        nan_counts[f'close_over_highest_{window}'] = df[f'close_over_highest_{window}'].isna().sum()
    
    # Add NaN count for gap_ret (NIFTY only)
    if is_nifty:
        nan_counts['gap_ret'] = df['gap_ret'].isna().sum()
    
    # Add NaN count for rsi_14 (NIFTY only)
    if is_nifty:
        nan_counts['rsi_14'] = df['rsi_14'].isna().sum()
    
    # Add NaN count for down_days_last_10 (NIFTY only)
    if is_nifty:
        nan_counts['down_days_last_10'] = df['down_days_last_10'].isna().sum()
    
    # Add NaN counts for new volatility and trend features (NIFTY only)
    if is_nifty:
        for window in [5, 10, 20]:
            nan_counts[f'rv_{window}'] = df[f'rv_{window}'].isna().sum()
        nan_counts['parkinson_vol'] = df['parkinson_vol'].isna().sum()
        nan_counts['atr_14'] = df['atr_14'].isna().sum()
        nan_counts['natr_14'] = df['natr_14'].isna().sum()
        nan_counts['trend_efficiency_10'] = df['trend_efficiency_10'].isna().sum()
    
    # Add NaN counts for VIX close_over_ema_* columns
    if is_vix:
        for window in [5, 15, 28, 60, 200]:
            nan_counts[f'close_over_ema_{window}'] = df[f'close_over_ema_{window}'].isna().sum()
    
    # Add NaN counts for VIX ema_*_minus_* columns
    if is_vix:
        ema_spread_cols = [
            'ema_5_minus_ema_15', 'ema_5_minus_ema_28', 'ema_5_minus_ema_60', 'ema_5_minus_ema_200',
            'ema_15_minus_ema_28', 'ema_15_minus_ema_60', 'ema_15_minus_ema_200',
            'ema_28_minus_ema_60', 'ema_28_minus_ema_200',
            'ema_60_minus_ema_200'
        ]
        for col in ema_spread_cols:
            nan_counts[col] = df[col].isna().sum()
    
    # Add NaN counts for new VIX vol-regime & memory features
    if is_vix:
        nan_counts['dvix_1d'] = df['dvix_1d'].isna().sum()
        nan_counts['vix_vol_of_vol_20'] = df['vix_vol_of_vol_20'].isna().sum()
        nan_counts['vix_pct_20'] = df['vix_pct_20'].isna().sum()
        nan_counts['vix_pct_60'] = df['vix_pct_60'].isna().sum()
        nan_counts['vix_pct_252'] = df['vix_pct_252'].isna().sum()
    
    # Min/max for hl_norm columns (excluding NaNs)
    hl_norm_stats = {}
    for col in hl_norm_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            hl_norm_stats[col] = {
                'min': col_data.min(),
                'max': col_data.max()
            }
        else:
            hl_norm_stats[col] = {
                'min': np.nan,
                'max': np.nan
            }
    
    # Min/max for trend_efficiency_10 (NIFTY only, excluding NaNs)
    trend_efficiency_stats = None
    if is_nifty:
        col_data = df['trend_efficiency_10'].dropna()
        if len(col_data) > 0:
            trend_efficiency_stats = {
                'min': col_data.min(),
                'max': col_data.max()
            }
        else:
            trend_efficiency_stats = {
                'min': np.nan,
                'max': np.nan
            }
    
    # Min/max for vix_pct columns and max for consecutive days (VIX only)
    vix_pct_stats = None
    vix_cons_days_stats = None
    if is_vix:
        # Min/max for vix_pct columns
        vix_pct_stats = {}
        for window in [20, 60, 252]:
            col_data = df[f'vix_pct_{window}'].dropna()
            if len(col_data) > 0:
                vix_pct_stats[f'vix_pct_{window}'] = {
                    'min': col_data.min(),
                    'max': col_data.max()
                }
            else:
                vix_pct_stats[f'vix_pct_{window}'] = {
                    'min': np.nan,
                    'max': np.nan
                }
        
        # Max values for consecutive days counters
        cons_p60_data = df['vix_cons_days_above_p60'].dropna()
        cons_p80_data = df['vix_cons_days_above_p80'].dropna()
        vix_cons_days_stats = {
            'vix_cons_days_above_p60_max': cons_p60_data.max() if len(cons_p60_data) > 0 else np.nan,
            'vix_cons_days_above_p80_max': cons_p80_data.max() if len(cons_p80_data) > 0 else np.nan
        }
    
    summary = {
        'date_range': (df['date'].min(), df['date'].max()),
        'row_count': len(df),
        'nan_counts': nan_counts,
        'ohlc_sanity_fail_count': df['ohlc_sanity_fail'].sum(),
        'hl_norm_stats': hl_norm_stats,
        'trend_efficiency_stats': trend_efficiency_stats,
        'vix_pct_stats': vix_pct_stats,
        'vix_cons_days_stats': vix_cons_days_stats
    }
    
    return summary, df


def print_summary(file_name, summary):
    """Print a formatted summary for a processed file."""
    print(f"\n{'='*60}")
    print(f"Summary for {file_name}")
    print(f"{'='*60}")
    print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Row count: {summary['row_count']}")
    print(f"\nNaN counts for new columns:")
    for col, count in summary['nan_counts'].items():
        print(f"  {col}: {count}")
    print(f"\nMin/Max for hl_norm columns (excluding NaNs):")
    for col, stats in summary['hl_norm_stats'].items():
        min_val = stats['min']
        max_val = stats['max']
        if pd.notna(min_val) and pd.notna(max_val):
            print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}")
        else:
            print(f"  {col}: min=NaN, max=NaN")
    
    # Print NaN counts for new volatility and trend features (NIFTY only)
    if file_name.upper() == 'NIFTY' and summary.get('trend_efficiency_stats') is not None:
        print(f"\nNaN counts for new volatility and trend features:")
        for col in ['rv_5', 'rv_10', 'rv_20', 'parkinson_vol', 'atr_14', 'natr_14', 'trend_efficiency_10']:
            if col in summary['nan_counts']:
                print(f"  {col}: {summary['nan_counts'][col]}")
        
        # Print min/max for trend_efficiency_10
        print(f"\nMin/Max for trend_efficiency_10 (excluding NaNs):")
        trend_stats = summary['trend_efficiency_stats']
        min_val = trend_stats['min']
        max_val = trend_stats['max']
        if pd.notna(min_val) and pd.notna(max_val):
            print(f"  trend_efficiency_10: min={min_val:.4f}, max={max_val:.4f}")
        else:
            print(f"  trend_efficiency_10: min=NaN, max=NaN")
    
    # Print NaN counts and statistics for new VIX vol-regime & memory features (VIX only)
    if file_name.upper() == 'VIX' and summary.get('vix_pct_stats') is not None:
        print(f"\nNaN counts for new VIX vol-regime & memory features:")
        for col in ['dvix_1d', 'vix_vol_of_vol_20', 'vix_pct_20', 'vix_pct_60', 'vix_pct_252']:
            if col in summary['nan_counts']:
                print(f"  {col}: {summary['nan_counts'][col]}")
        
        # Print min/max for vix_pct columns
        print(f"\nMin/Max for vix_pct columns (excluding NaNs):")
        vix_pct_stats = summary['vix_pct_stats']
        for col in ['vix_pct_20', 'vix_pct_60', 'vix_pct_252']:
            if col in vix_pct_stats:
                stats = vix_pct_stats[col]
                min_val = stats['min']
                max_val = stats['max']
                if pd.notna(min_val) and pd.notna(max_val):
                    print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}")
                else:
                    print(f"  {col}: min=NaN, max=NaN")
        
        # Print max values for consecutive days counters
        print(f"\nMax values for consecutive days counters (excluding NaNs):")
        cons_stats = summary['vix_cons_days_stats']
        max_p60 = cons_stats['vix_cons_days_above_p60_max']
        max_p80 = cons_stats['vix_cons_days_above_p80_max']
        if pd.notna(max_p60):
            print(f"  vix_cons_days_above_p60: max={max_p60:.0f}")
        else:
            print(f"  vix_cons_days_above_p60: max=NaN")
        if pd.notna(max_p80):
            print(f"  vix_cons_days_above_p80: max={max_p80:.0f}")
        else:
            print(f"  vix_cons_days_above_p80: max=NaN")
    
    print(f"\nOHLC sanity failures: {summary['ohlc_sanity_fail_count']}")
    print(f"{'='*60}")


def main():
    """Main function to process both NIFTY and VIX files."""
    # Construct full paths
    nifty_input_path = os.path.join(INPUT_DIR, NIFTY_INPUT)
    vix_input_path = os.path.join(INPUT_DIR, VIX_INPUT)
    
    # Check if input files exist
    if not os.path.exists(nifty_input_path):
        raise FileNotFoundError(f"Could not find NIFTY input file: {nifty_input_path}")
    
    if not os.path.exists(vix_input_path):
        raise FileNotFoundError(f"Could not find VIX input file: {vix_input_path}")
    
    # Process NIFTY file
    nifty_output_path = os.path.join(OUTPUT_DIR, NIFTY_OUTPUT)
    nifty_summary, nifty_df = process_ohlc_file(nifty_input_path, nifty_output_path)
    print_summary("NIFTY", nifty_summary)
    
    # Process VIX file
    vix_output_path = os.path.join(OUTPUT_DIR, VIX_OUTPUT)
    vix_summary, vix_df = process_ohlc_file(vix_input_path, vix_output_path)
    print_summary("VIX", vix_summary)
    
    # ==================================================
    # 2) Cross feature: India VIX * NIFTY returns
    # ==================================================
    print("\n" + "="*60)
    print("Computing cross-features (VIX * NIFTY returns)")
    print("="*60)
    
    # Ensure both dataframes have date parsed and sorted
    nifty_df['date'] = pd.to_datetime(nifty_df['date'])
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    
    # Sort by date
    nifty_df = nifty_df.sort_values('date').reset_index(drop=True)
    vix_df = vix_df.sort_values('date').reset_index(drop=True)
    
    # Handle duplicate dates (keep last)
    nifty_df = nifty_df.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
    vix_df = vix_df.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
    
    # Merge on date (inner join)
    merged_df = pd.merge(
        nifty_df[['date', 'ret_5', 'ret_28', 'ret_60', 'ret_200']],
        vix_df[['date', 'close']],
        on='date',
        how='inner',
        suffixes=('_nifty', '_vix')
    )
    
    # Rename VIX close column for clarity
    merged_df = merged_df.rename(columns={'close': 'vix_close'})
    
    # Compute cross-features
    cross_features = {}
    for period in [5, 28, 60, 200]:
        col_name = f'vix_close_x_nifty_ret_{period}'
        merged_df[col_name] = merged_df['vix_close'] * merged_df[f'ret_{period}']
        cross_features[col_name] = merged_df[col_name]
    
    # Replace +/-inf with NaN
    for col in cross_features.keys():
        merged_df[col] = merged_df[col].replace([np.inf, -np.inf], np.nan)
    
    # Map cross-features back to both dataframes by date
    cross_feature_cols = list(cross_features.keys())
    
    # Add cross-features to NIFTY dataframe
    for col in cross_feature_cols:
        nifty_df[col] = nifty_df['date'].map(dict(zip(merged_df['date'], merged_df[col])))
    
    # Add cross-features to VIX dataframe
    for col in cross_feature_cols:
        vix_df[col] = vix_df['date'].map(dict(zip(merged_df['date'], merged_df[col])))
    
    # Replace +/-inf with NaN in both dataframes
    for col in cross_feature_cols:
        nifty_df[col] = nifty_df[col].replace([np.inf, -np.inf], np.nan)
        vix_df[col] = vix_df[col].replace([np.inf, -np.inf], np.nan)
    
    # Print merge statistics
    print(f"\nMerge statistics:")
    print(f"  Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"  Row count after merge: {len(merged_df)}")
    print(f"  NIFTY rows: {len(nifty_df)}, VIX rows: {len(vix_df)}")
    
    # Print NaN counts for cross-features
    print(f"\nNaN counts for cross-features:")
    for col in cross_feature_cols:
        nifty_nan = nifty_df[col].isna().sum()
        vix_nan = vix_df[col].isna().sum()
        print(f"  {col}: NIFTY={nifty_nan}, VIX={vix_nan}")
    
    # Save both dataframes
    nifty_df.to_csv(nifty_output_path, index=False)
    print(f"\nSaved NIFTY features to: {nifty_output_path}")
    
    vix_df.to_csv(vix_output_path, index=False)
    print(f"Saved VIX features to: {vix_output_path}")
    
    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()

