"""
NIFTY Options Data - IV, Delta, and Surface Features Calculator

This script computes Black-Scholes implied volatility, delta, and surface features
for the tradable universe datasets (30d, 45d, 60d).

Usage:
    python compute_option_iv_delta_and_surface_features.py

Input:
    - options_output/options_tradable_universe_30d.parquet (or .csv)
    - options_output/options_tradable_universe_45d.parquet (or .csv)
    - options_output/options_tradable_universe_60d.parquet (or .csv)

Output:
    - Overwrites the same input files with added columns:
      - iv, delta (per-row)
      - atm_strike, atm_straddle_price, atm_pair_missing (group-level)
      - iv_put_25d, iv_call_25d, skew_25d, iv_ratio_25d (group-level)
      - atm_iv, smile_curvature_25d (group-level)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("options_output")

# Input files (prefer parquet, fallback to csv)
INPUT_FILES = [
    OUTPUT_DIR / "options_tradable_universe_30d",
    OUTPUT_DIR / "options_tradable_universe_45d",
    OUTPUT_DIR / "options_tradable_universe_60d",
]

# Black-Scholes parameters
R = 0.06  # Risk-free annual rate
Q = 0.00  # Dividend yield annual (0 for NIFTY)

# IV solver parameters
SIGMA_LOW = 1e-6
SIGMA_HIGH = 5.0
MAX_ITER = 100
TOL = 1e-6
SMALL_EPS = 1e-4  # For bounds checking


# ============================================================================
# BLACK-SCHOLES FUNCTIONS
# ============================================================================

def N(x):
    """Standard normal CDF"""
    return norm.cdf(x)


def black_scholes_price(S, K, T, r, q, sigma, option_type):
    """
    Compute Black-Scholes option price.
    
    Args:
        S: Underlying price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: "CE" or "PE"
    
    Returns:
        Option price
    """
    if T <= 0 or sigma <= 0:
        return np.nan
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if option_type == "CE":
        price = S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
    elif option_type == "PE":
        price = K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)
    else:
        return np.nan
    
    return price


def black_scholes_delta(S, K, T, r, q, sigma, option_type):
    """
    Compute Black-Scholes delta.
    
    Args:
        S: Underlying price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: "CE" or "PE"
    
    Returns:
        Delta value
    """
    if T <= 0 or sigma <= 0:
        return np.nan
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    
    if option_type == "CE":
        delta = np.exp(-q * T) * N(d1)
    elif option_type == "PE":
        delta = np.exp(-q * T) * (N(d1) - 1)
    else:
        return np.nan
    
    return delta


def check_price_bounds(S, K, T, r, q, market_price, option_type):
    """
    Check if market price is within theoretical bounds.
    
    Returns:
        True if within bounds, False otherwise
    """
    if option_type == "CE":
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        upper = S * np.exp(-q * T)
    elif option_type == "PE":
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        upper = K * np.exp(-r * T)
    else:
        return False
    
    if market_price < intrinsic - SMALL_EPS or market_price > upper + SMALL_EPS:
        return False
    
    return True


def solve_iv_bisection(S, K, T, r, q, market_price, option_type):
    """
    Solve for implied volatility using bisection method.
    
    Args:
        S: Underlying price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        q: Dividend yield
        market_price: Market option price
        option_type: "CE" or "PE"
    
    Returns:
        Implied volatility (sigma) or NaN
    """
    # Guards
    if S <= 0 or K <= 0 or T <= 0:
        return np.nan
    
    if pd.isna(market_price) or market_price <= 0:
        return np.nan
    
    if option_type not in ["CE", "PE"]:
        return np.nan
    
    # Check price bounds
    if not check_price_bounds(S, K, T, r, q, market_price, option_type):
        return np.nan
    
    # Bisection
    sigma_low = SIGMA_LOW
    sigma_high = SIGMA_HIGH
    
    # Check if solution exists
    price_low = black_scholes_price(S, K, T, r, q, sigma_low, option_type)
    price_high = black_scholes_price(S, K, T, r, q, sigma_high, option_type)
    
    if pd.isna(price_low) or pd.isna(price_high):
        return np.nan
    
    # Market price should be between low and high
    if market_price < price_low or market_price > price_high:
        return np.nan
    
    # Bisection loop
    for _ in range(MAX_ITER):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = black_scholes_price(S, K, T, r, q, sigma_mid, option_type)
        
        if pd.isna(price_mid):
            return np.nan
        
        if abs(price_mid - market_price) < TOL:
            return sigma_mid
        
        if price_mid < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
        
        if abs(sigma_high - sigma_low) < TOL:
            return sigma_mid
    
    # Return best estimate
    return (sigma_low + sigma_high) / 2


# ============================================================================
# COMPUTE IV AND DELTA FOR DATAFRAME
# ============================================================================

def compute_iv_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IV and delta for each row in the dataframe.
    
    Args:
        df: DataFrame with columns: underlying_value, strike, dte_days, close, option_type
    
    Returns:
        DataFrame with added columns: iv, delta
    """
    print("\nComputing IV and Delta for each row...")
    
    df = df.copy()
    
    # Initialize columns
    df['iv'] = np.nan
    df['delta'] = np.nan
    
    # Convert to numpy arrays for efficiency
    S = df['underlying_value'].values
    K = df['strike'].values
    T = df['dte_days'].values / 365.0
    market_price = df['close'].values
    option_type = df['option_type'].values
    
    # Compute IV and delta row by row
    n_rows = len(df)
    print(f"  Processing {n_rows:,} rows...")
    
    for i in range(n_rows):
        if i % 10000 == 0 and i > 0:
            print(f"    Processed {i:,} / {n_rows:,} rows ({i/n_rows*100:.1f}%)")
        
        iv = solve_iv_bisection(
            S[i], K[i], T[i], R, Q,
            market_price[i], option_type[i]
        )
        
        df.iloc[i, df.columns.get_loc('iv')] = iv
        
        if not pd.isna(iv):
            delta = black_scholes_delta(
                S[i], K[i], T[i], R, Q, iv, option_type[i]
            )
            df.iloc[i, df.columns.get_loc('delta')] = delta
    
    print(f"  ✓ Completed IV and delta computation")
    
    return df


# ============================================================================
# COMPUTE GROUP-LEVEL SURFACE FEATURES
# ============================================================================

def compute_surface_features_group(group: pd.DataFrame) -> pd.Series:
    """
    Compute surface features for a single (date, expiry) group.
    
    Args:
        group: DataFrame subset for one (date, expiry) group
    
    Returns:
        Series with feature values
    """
    features = pd.Series(dtype=float)
    
    # Get underlying value (should be same, use median if not)
    S = group['underlying_value'].median()
    if pd.isna(S) or S <= 0:
        # Return all NaN features
        return pd.Series({
            'atm_strike': np.nan,
            'atm_straddle_price': np.nan,
            'atm_pair_missing': True,
            'iv_put_25d': np.nan,
            'iv_call_25d': np.nan,
            'skew_25d': np.nan,
            'iv_ratio_25d': np.nan,
            'atm_iv': np.nan,
            'smile_curvature_25d': np.nan,
        })
    
    # A) ATM strike selection
    strikes = group['strike'].unique()
    if len(strikes) == 0:
        features['atm_strike'] = np.nan
        return features
    
    # Find strike closest to S
    distances = np.abs(strikes - S)
    min_distance = np.min(distances)
    candidates = strikes[distances == min_distance]
    
    # Tie-break: choose lower strike if multiple equally close
    atm_strike = np.min(candidates)
    
    features['atm_strike'] = atm_strike
    
    # B) ATM straddle price
    atm_call = group[(group['option_type'] == 'CE') & (group['strike'] == atm_strike)]
    atm_put = group[(group['option_type'] == 'PE') & (group['strike'] == atm_strike)]
    
    if len(atm_call) > 0 and len(atm_put) > 0:
        # If multiple rows, take the first one (should be unique but handle edge case)
        atm_call_close = atm_call['close'].iloc[0] if len(atm_call) > 0 else np.nan
        atm_put_close = atm_put['close'].iloc[0] if len(atm_put) > 0 else np.nan
        if pd.notna(atm_call_close) and pd.notna(atm_put_close) and atm_call_close > 0 and atm_put_close > 0:
            features['atm_straddle_price'] = atm_call_close + atm_put_close
            features['atm_pair_missing'] = False
        else:
            features['atm_straddle_price'] = np.nan
            features['atm_pair_missing'] = True
    else:
        features['atm_straddle_price'] = np.nan
        features['atm_pair_missing'] = True
    
    # C) 25Δ selection
    # Filter for liquid contracts with valid IV and delta
    liquid_mask = (
        (group['is_liquid'] == True) &
        group['iv'].notna() &
        group['delta'].notna() &
        (group['close'] > 0)
    )
    liquid_group = group[liquid_mask].copy()
    
    # Find 25Δ put (delta closest to -0.25)
    puts_25d = liquid_group[liquid_group['option_type'] == 'PE'].copy()
    if len(puts_25d) > 0:
        puts_25d['delta_diff'] = np.abs(puts_25d['delta'] + 0.25)
        put_25d_row = puts_25d.loc[puts_25d['delta_diff'].idxmin()]
        features['iv_put_25d'] = put_25d_row['iv']
    else:
        features['iv_put_25d'] = np.nan
    
    # Find 25Δ call (delta closest to +0.25)
    calls_25d = liquid_group[liquid_group['option_type'] == 'CE'].copy()
    if len(calls_25d) > 0:
        calls_25d['delta_diff'] = np.abs(calls_25d['delta'] - 0.25)
        call_25d_row = calls_25d.loc[calls_25d['delta_diff'].idxmin()]
        features['iv_call_25d'] = call_25d_row['iv']
    else:
        features['iv_call_25d'] = np.nan
    
    # Compute skew and ratio
    if pd.notna(features['iv_put_25d']) and pd.notna(features['iv_call_25d']):
        features['skew_25d'] = features['iv_put_25d'] - features['iv_call_25d']
        if features['iv_call_25d'] > 0:
            features['iv_ratio_25d'] = features['iv_put_25d'] / features['iv_call_25d']
        else:
            features['iv_ratio_25d'] = np.nan
    else:
        features['skew_25d'] = np.nan
        features['iv_ratio_25d'] = np.nan
    
    # D) ATM IV
    atm_call_iv = atm_call['iv'].iloc[0] if len(atm_call) > 0 else np.nan
    atm_put_iv = atm_put['iv'].iloc[0] if len(atm_put) > 0 else np.nan
    
    if pd.notna(atm_call_iv) and pd.notna(atm_put_iv):
        features['atm_iv'] = (atm_call_iv + atm_put_iv) / 2
    elif pd.notna(atm_call_iv):
        features['atm_iv'] = atm_call_iv
    elif pd.notna(atm_put_iv):
        features['atm_iv'] = atm_put_iv
    else:
        features['atm_iv'] = np.nan
    
    # Smile curvature proxy
    if (pd.notna(features['iv_put_25d']) and 
        pd.notna(features['iv_call_25d']) and 
        pd.notna(features['atm_iv'])):
        features['smile_curvature_25d'] = (
            features['iv_put_25d'] + features['iv_call_25d'] - 2 * features['atm_iv']
        )
    else:
        features['smile_curvature_25d'] = np.nan
    
    return features


def compute_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute surface features for each (date, expiry) group and merge back.
    
    Args:
        df: DataFrame with IV and delta computed
    
    Returns:
        DataFrame with added group-level feature columns
    """
    print("\nComputing surface features per (date, expiry) group...")
    
    # Group by date and expiry
    groups = df.groupby(['date', 'expiry'], group_keys=False)
    n_groups = len(groups)
    print(f"  Processing {n_groups:,} groups...")
    
    # Compute features for each group
    group_features_list = []
    group_keys_list = []
    
    for idx, (key, group) in enumerate(groups):
        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1:,} / {n_groups:,} groups ({(idx + 1)/n_groups*100:.1f}%)")
        
        features = compute_surface_features_group(group)
        group_features_list.append(features)
        group_keys_list.append(key)
    
    # Create features dataframe
    features_df = pd.DataFrame(group_features_list)
    features_df.index = pd.MultiIndex.from_tuples(group_keys_list, names=['date', 'expiry'])
    features_df = features_df.reset_index()
    
    print(f"  ✓ Completed surface features computation")
    
    # Merge back to original dataframe
    print(f"  Merging features back to dataframe...")
    df_merged = df.merge(
        features_df,
        on=['date', 'expiry'],
        how='left'
    )
    
    print(f"  ✓ Features merged")
    
    return df_merged


# ============================================================================
# LOAD AND ENFORCE TYPES
# ============================================================================

def load_dataset(file_path: Path) -> tuple[pd.DataFrame, str]:
    """
    Load dataset (prefer parquet, fallback to csv).
    
    Returns:
        (DataFrame, file_type)
    """
    parquet_path = file_path.with_suffix('.parquet')
    csv_path = file_path.with_suffix('.csv')
    
    if parquet_path.exists():
        print(f"Loading from Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        return df, 'parquet'
    elif csv_path.exists():
        print(f"Loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        return df, 'csv'
    else:
        raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} found")


def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce proper data types.
    """
    print("\nEnforcing data types...")
    
    df = df.copy()
    
    # Date columns
    for col in ['date', 'expiry']:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Numeric columns
    numeric_cols = ['strike', 'close', 'underlying_value', 'dte_days']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Option type
    if 'option_type' in df.columns:
        df['option_type'] = df['option_type'].astype(str).str.strip().str.upper()
    
    # Boolean columns
    for col in ['is_liquid']:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({
                    'True': True, 'true': True, 'TRUE': True, '1': True,
                    'False': False, 'false': False, 'FALSE': False, '0': False
                })
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('boolean')
    
    print(f"  ✓ Types enforced")
    
    return df


# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_dataset(df: pd.DataFrame, file_path: Path, file_type: str):
    """
    Save dataset (parquet and csv for consistency).
    """
    parquet_path = file_path.with_suffix('.parquet')
    csv_path = file_path.with_suffix('.csv')
    
    # Save parquet
    print(f"\nSaving to Parquet: {parquet_path}")
    df.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy')
    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")
    
    # Save csv
    print(f"\nSaving to CSV: {csv_path}")
    df.to_csv(csv_path, index=False)
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")


# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

def print_summary(df: pd.DataFrame, dataset_name: str):
    """
    Print comprehensive summary statistics.
    """
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS - {dataset_name}")
    print("="*80)
    
    # Row-level stats
    print(f"\nRow-level Statistics:")
    iv_valid = df['iv'].notna()
    iv_pct = iv_valid.sum() / len(df) * 100
    print(f"  IV: {iv_valid.sum():,} valid ({iv_pct:.2f}%)")
    
    if iv_valid.any():
        iv_values = df.loc[iv_valid, 'iv']
        print(f"    Min: {iv_values.min():.4f}")
        print(f"    Median: {iv_values.median():.4f}")
        print(f"    Max: {iv_values.max():.4f}")
    
    delta_valid = df['delta'].notna()
    if delta_valid.any():
        delta_values = df.loc[delta_valid, 'delta']
        print(f"\n  Delta: {delta_valid.sum():,} valid")
        print(f"    Min: {delta_values.min():.4f}")
        print(f"    Max: {delta_values.max():.4f}")
    
    # Group-level stats
    print(f"\nGroup-level Statistics:")
    groups = df.groupby(['date', 'expiry'])
    n_groups = len(groups)
    print(f"  Number of (date, expiry) groups: {n_groups:,}")
    
    # Get first row of each group for group features
    group_first = groups.first()
    
    if 'atm_straddle_price' in group_first.columns:
        atm_straddle_valid = group_first['atm_straddle_price'].notna()
        atm_straddle_pct = atm_straddle_valid.sum() / n_groups * 100
        print(f"  ATM straddle price: {atm_straddle_valid.sum():,} valid ({atm_straddle_pct:.2f}%)")
    
    if 'skew_25d' in group_first.columns:
        skew_valid = group_first['skew_25d'].notna()
        skew_pct = skew_valid.sum() / n_groups * 100
        print(f"  Skew 25Δ: {skew_valid.sum():,} valid ({skew_pct:.2f}%)")
        
        if skew_valid.any():
            skew_values = group_first.loc[skew_valid, 'skew_25d']
            print(f"    Min: {skew_values.min():.4f}")
            print(f"    Median: {skew_values.median():.4f}")
            print(f"    Max: {skew_values.max():.4f}")
    
    if 'smile_curvature_25d' in group_first.columns:
        curvature_valid = group_first['smile_curvature_25d'].notna()
        curvature_pct = curvature_valid.sum() / n_groups * 100
        print(f"  Smile curvature 25Δ: {curvature_valid.sum():,} valid ({curvature_pct:.2f}%)")
        
        if curvature_valid.any():
            curvature_values = group_first.loc[curvature_valid, 'smile_curvature_25d']
            print(f"    Min: {curvature_values.min():.4f}")
            print(f"    Median: {curvature_values.median():.4f}")
            print(f"    Max: {curvature_values.max():.4f}")
    
    # Top 5 groups by absolute skew
    if 'skew_25d' in group_first.columns:
        print(f"\n  Top 5 groups by absolute skew_25d:")
        top_skew = group_first.nlargest(5, 'skew_25d', keep='all')
        for idx, row in top_skew.iterrows():
            print(f"    {idx[0]} | {idx[1]}: skew = {row['skew_25d']:.4f}")
    
    # Top 5 groups by smile curvature
    if 'smile_curvature_25d' in group_first.columns:
        print(f"\n  Top 5 groups by smile_curvature_25d:")
        top_curvature = group_first.nlargest(5, 'smile_curvature_25d', keep='all')
        for idx, row in top_curvature.iterrows():
            print(f"    {idx[0]} | {idx[1]}: curvature = {row['smile_curvature_25d']:.4f}")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_dataset(file_path: Path):
    """
    Process a single tradable universe dataset.
    """
    dataset_name = file_path.stem
    
    print("\n\n" + "="*80)
    print(f"PROCESSING: {dataset_name}")
    print("="*80)
    
    # Load dataset
    df, file_type = load_dataset(file_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Enforce types
    df = enforce_types(df)
    
    # Sort
    print("\nSorting dataset...")
    df = df.sort_values(['date', 'expiry', 'strike', 'option_type']).reset_index(drop=True)
    print(f"  ✓ Sorted")
    
    # Compute IV and delta
    df = compute_iv_delta(df)
    
    # Compute surface features
    df = compute_surface_features(df)
    
    # Print summary
    print_summary(df, dataset_name)
    
    # Save output
    print(f"\nSaving updated dataset...")
    save_dataset(df, file_path, file_type)
    
    print(f"\n✓ Completed processing {dataset_name}")


def main():
    """
    Main execution pipeline.
    """
    print("\n" + "="*80)
    print("NIFTY OPTIONS DATA - IV, DELTA, AND SURFACE FEATURES CALCULATOR")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    print(f"  Risk-free rate (r): {R}")
    print(f"  Dividend yield (q): {Q}")
    print(f"  IV solver: bisection (sigma_low={SIGMA_LOW}, sigma_high={SIGMA_HIGH})")
    
    try:
        for file_path in INPUT_FILES:
            process_dataset(file_path)
        
        print("\n\n" + "="*80)
        print("✓ ALL PROCESSING COMPLETE")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

