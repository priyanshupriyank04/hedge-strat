"""
NIFTY Options Data - Tradable Universe Builder

This script filters the unified options master dataset to create tradable universes
with different DTE thresholds (30, 45, 60 days).

Usage:
    python build_tradable_universe.py

Input:
    - options_output/options_master_long.parquet (preferred)
    - options_output/options_master_long.csv (fallback)

Output:
    - options_output/options_tradable_universe_30d.csv
    - options_output/options_tradable_universe_45d.csv
    - options_output/options_tradable_universe_60d.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PARQUET = Path("options_output/options_master_long.parquet")
INPUT_CSV = Path("options_output/options_master_long.csv")

OUTPUT_DIR = Path("options_output")

# DTE thresholds to create separate tradable universes
DTE_THRESHOLDS = [30, 45, 60]


# ============================================================================
# LOAD MASTER DATASET
# ============================================================================

def load_master() -> pd.DataFrame:
    """
    Load the unified options master dataset.
    Tries parquet first, falls back to CSV if parquet is missing.
    
    Returns:
        DataFrame with master dataset
    """
    print("\n" + "="*80)
    print("LOADING MASTER DATASET")
    print("="*80)
    
    # Try parquet first
    if INPUT_PARQUET.exists():
        print(f"\nLoading from Parquet: {INPUT_PARQUET}")
        try:
            df = pd.read_parquet(INPUT_PARQUET)
            print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"  ✗ Error reading Parquet: {e}")
            print(f"  Falling back to CSV...")
    
    # Fallback to CSV
    if INPUT_CSV.exists():
        print(f"\nLoading from CSV: {INPUT_CSV}")
        try:
            df = pd.read_csv(INPUT_CSV, low_memory=False)
            print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"  ✗ Error reading CSV: {e}")
            raise
    
    # Neither file exists
    raise FileNotFoundError(
        f"Neither {INPUT_PARQUET} nor {INPUT_CSV} found. "
        "Please run build_options_master_long.py first."
    )


# ============================================================================
# ENFORCE DATA TYPES
# ============================================================================

def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure date/expiry are datetime and flags are boolean.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with proper types
    """
    print("\n" + "="*80)
    print("ENFORCING DATA TYPES")
    print("="*80)
    
    df = df.copy()
    
    # Parse date columns
    for date_col in ['date', 'expiry']:
        if date_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                print(f"\nConverting {date_col} to datetime...")
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
                null_count = df[date_col].isna().sum()
                if null_count > 0:
                    print(f"  ⚠ Warning: {null_count:,} rows with invalid {date_col}")
                else:
                    print(f"  ✓ {date_col} converted to datetime")
            else:
                print(f"  ✓ {date_col} is already datetime")
    
    # Convert boolean flags
    for bool_col in ['is_liquid', 'is_tradable_30d']:
        if bool_col in df.columns:
            print(f"\nConverting {bool_col} to boolean...")
            # Handle string booleans
            if df[bool_col].dtype == 'object':
                # Convert string "True"/"False" to boolean
                df[bool_col] = df[bool_col].astype(str).str.strip()
                df[bool_col] = df[bool_col].replace({
                    'True': True, 'true': True, 'TRUE': True, '1': True,
                    'False': False, 'false': False, 'FALSE': False, '0': False
                })
                # Convert remaining to boolean (NaN will stay NaN)
                df[bool_col] = pd.to_numeric(df[bool_col], errors='coerce').astype('boolean')
            elif df[bool_col].dtype == 'int64':
                # Convert 0/1 to boolean
                df[bool_col] = df[bool_col].astype(bool)
            elif not pd.api.types.is_bool_dtype(df[bool_col]):
                # Try to convert to boolean
                df[bool_col] = df[bool_col].astype('boolean')
            
            print(f"  ✓ {bool_col} converted to boolean")
            true_count = df[bool_col].sum()
            print(f"    True values: {true_count:,} ({true_count/len(df)*100:.2f}%)")
    
    # Ensure dte_days is numeric
    if 'dte_days' in df.columns:
        print(f"\nEnsuring dte_days is numeric...")
        if not pd.api.types.is_numeric_dtype(df['dte_days']):
            df['dte_days'] = pd.to_numeric(df['dte_days'], errors='coerce')
        print(f"  ✓ dte_days is numeric")
        non_null = df['dte_days'].notna().sum()
        print(f"    Non-null values: {non_null:,} ({non_null/len(df)*100:.2f}%)")
    
    return df


# ============================================================================
# FILTER TO TRADABLE UNIVERSE
# ============================================================================

def filter_tradable_universe(df: pd.DataFrame, dte_threshold: int) -> pd.DataFrame:
    """
    Filter to tradable universe:
    - is_liquid == True
    - dte_days is not NaN AND dte_days >= 0 AND dte_days <= dte_threshold
    
    Args:
        df: Master dataset DataFrame
        dte_threshold: Maximum DTE days (30, 45, or 60)
    
    Returns:
        Filtered DataFrame
    """
    print("\n" + "="*80)
    print(f"FILTERING TO TRADABLE UNIVERSE (DTE <= {dte_threshold} days)")
    print("="*80)
    
    master_rows = len(df)
    print(f"\nMaster dataset rows: {master_rows:,}")
    
    # Apply filters
    df_filtered = df.copy()
    
    # Filter 1: is_liquid == True
    if 'is_liquid' in df_filtered.columns:
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['is_liquid'] == True]
        after = len(df_filtered)
        print(f"\nFilter 1: is_liquid == True")
        print(f"  Before: {before:,} rows")
        print(f"  After: {after:,} rows")
        print(f"  Removed: {before - after:,} rows ({(before - after)/before*100:.2f}%)")
    else:
        print(f"\n  ⚠ WARNING: is_liquid column not found, skipping filter")
    
    # Filter 2: dte_days is not NaN AND dte_days >= 0 AND dte_days <= dte_threshold
    if 'dte_days' in df_filtered.columns:
        before = len(df_filtered)
        dte_valid = (
            df_filtered['dte_days'].notna() &
            (df_filtered['dte_days'] >= 0) &
            (df_filtered['dte_days'] <= dte_threshold)
        )
        df_filtered = df_filtered[dte_valid]
        after = len(df_filtered)
        print(f"\nFilter 2: dte_days valid (not NaN, >= 0, <= {dte_threshold})")
        print(f"  Before: {before:,} rows")
        print(f"  After: {after:,} rows")
        print(f"  Removed: {before - after:,} rows ({(before - after)/before*100:.2f}%)")
    else:
        print(f"\n  ⚠ WARNING: dte_days column not found, skipping filter")
    
    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)
    
    tradable_rows = len(df_filtered)
    pct_retained = (tradable_rows / master_rows) * 100 if master_rows > 0 else 0
    
    print(f"\n" + "-"*80)
    print(f"TRADABLE UNIVERSE SUMMARY (DTE <= {dte_threshold} days)")
    print(f"-"*80)
    print(f"Master rows: {master_rows:,}")
    print(f"Tradable universe rows: {tradable_rows:,}")
    print(f"% retained: {pct_retained:.2f}%")
    
    return df_filtered


# ============================================================================
# VALIDATE UNIQUENESS
# ============================================================================

def validate_uniqueness(df: pd.DataFrame) -> int:
    """
    Check for duplicate keys: (date, expiry, strike, option_type)
    
    Args:
        df: DataFrame to check
    
    Returns:
        Number of duplicate rows (should be 0)
    """
    print("\n" + "="*80)
    print("VALIDATING UNIQUENESS")
    print("="*80)
    
    key_cols = ['date', 'expiry', 'strike', 'option_type']
    missing_cols = [col for col in key_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n  ⚠ WARNING: Missing key columns: {missing_cols}")
        print(f"  Cannot validate uniqueness")
        return -1
    
    duplicates = df.duplicated(subset=key_cols, keep=False)
    dup_count = duplicates.sum()
    
    if dup_count > 0:
        print(f"\n  ⚠ WARNING: Found {dup_count:,} duplicate rows")
        print(f"  Unique key: (date, expiry, strike, option_type)")
        print(f"  This should be 0. Please investigate.")
    else:
        print(f"\n  ✓ No duplicate keys found")
        print(f"  All rows have unique (date, expiry, strike, option_type)")
    
    return dup_count


# ============================================================================
# CLEAN DATA FOR SAVING
# ============================================================================

def clean_for_saving(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace +/-inf with NaN before saving.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("CLEANING DATA FOR SAVING")
    print("="*80)
    
    df_clean = df.copy()
    
    # Replace inf with NaN
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    inf_count = 0
    
    for col in numeric_cols:
        inf_mask = np.isinf(df_clean[col])
        if inf_mask.any():
            inf_count += inf_mask.sum()
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    if inf_count > 0:
        print(f"  Replaced {inf_count:,} inf values with NaN")
    else:
        print(f"  ✓ No inf values found")
    
    return df_clean


# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

def print_summary(df_master: pd.DataFrame, df_tradable: pd.DataFrame, dte_threshold: int):
    """
    Print comprehensive summary statistics.
    
    Args:
        df_master: Master dataset
        df_tradable: Tradable universe dataset
        dte_threshold: DTE threshold used for filtering
    """
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS (DTE <= {dte_threshold} days)")
    print("="*80)
    
    master_rows = len(df_master)
    tradable_rows = len(df_tradable)
    pct_retained = (tradable_rows / master_rows) * 100 if master_rows > 0 else 0
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Master rows: {master_rows:,}")
    print(f"  Tradable universe rows: {tradable_rows:,}")
    print(f"  % retained: {pct_retained:.2f}%")
    
    # Date range
    if 'date' in df_tradable.columns:
        date_min = df_tradable['date'].min()
        date_max = df_tradable['date'].max()
        print(f"\n📅 Date Range (Tradable Universe):")
        print(f"  Min: {date_min}")
        print(f"  Max: {date_max}")
        if pd.notna(date_min) and pd.notna(date_max):
            span_days = (date_max - date_min).days
            print(f"  Span: {span_days} days")
    
    # Expiry range
    if 'expiry' in df_tradable.columns:
        expiry_min = df_tradable['expiry'].min()
        expiry_max = df_tradable['expiry'].max()
        print(f"\n📅 Expiry Range (Tradable Universe):")
        print(f"  Min: {expiry_min}")
        print(f"  Max: {expiry_max}")
    
    # DTE distribution
    if 'dte_days' in df_tradable.columns:
        print(f"\n⏰ Days to Expiry (dte_days) Distribution:")
        dte_valid = df_tradable['dte_days'].dropna()
        if len(dte_valid) > 0:
            print(f"  Min: {dte_valid.min():.1f}")
            print(f"  Max: {dte_valid.max():.1f}")
            print(f"  Mean: {dte_valid.mean():.2f}")
            print(f"  Median: {dte_valid.median():.1f}")
            
            # Buckets (adjust based on threshold)
            if dte_threshold == 30:
                bucket_0_7 = ((dte_valid >= 0) & (dte_valid <= 7)).sum()
                bucket_8_15 = ((dte_valid >= 8) & (dte_valid <= 15)).sum()
                bucket_16_30 = ((dte_valid >= 16) & (dte_valid <= 30)).sum()
                print(f"\n  DTE Buckets:")
                print(f"    0-7 days: {bucket_0_7:,} rows ({bucket_0_7/len(dte_valid)*100:.1f}%)")
                print(f"    8-15 days: {bucket_8_15:,} rows ({bucket_8_15/len(dte_valid)*100:.1f}%)")
                print(f"    16-30 days: {bucket_16_30:,} rows ({bucket_16_30/len(dte_valid)*100:.1f}%)")
            elif dte_threshold == 45:
                bucket_0_7 = ((dte_valid >= 0) & (dte_valid <= 7)).sum()
                bucket_8_15 = ((dte_valid >= 8) & (dte_valid <= 15)).sum()
                bucket_16_30 = ((dte_valid >= 16) & (dte_valid <= 30)).sum()
                bucket_31_45 = ((dte_valid >= 31) & (dte_valid <= 45)).sum()
                print(f"\n  DTE Buckets:")
                print(f"    0-7 days: {bucket_0_7:,} rows ({bucket_0_7/len(dte_valid)*100:.1f}%)")
                print(f"    8-15 days: {bucket_8_15:,} rows ({bucket_8_15/len(dte_valid)*100:.1f}%)")
                print(f"    16-30 days: {bucket_16_30:,} rows ({bucket_16_30/len(dte_valid)*100:.1f}%)")
                print(f"    31-45 days: {bucket_31_45:,} rows ({bucket_31_45/len(dte_valid)*100:.1f}%)")
            elif dte_threshold == 60:
                bucket_0_7 = ((dte_valid >= 0) & (dte_valid <= 7)).sum()
                bucket_8_15 = ((dte_valid >= 8) & (dte_valid <= 15)).sum()
                bucket_16_30 = ((dte_valid >= 16) & (dte_valid <= 30)).sum()
                bucket_31_45 = ((dte_valid >= 31) & (dte_valid <= 45)).sum()
                bucket_46_60 = ((dte_valid >= 46) & (dte_valid <= 60)).sum()
                print(f"\n  DTE Buckets:")
                print(f"    0-7 days: {bucket_0_7:,} rows ({bucket_0_7/len(dte_valid)*100:.1f}%)")
                print(f"    8-15 days: {bucket_8_15:,} rows ({bucket_8_15/len(dte_valid)*100:.1f}%)")
                print(f"    16-30 days: {bucket_16_30:,} rows ({bucket_16_30/len(dte_valid)*100:.1f}%)")
                print(f"    31-45 days: {bucket_31_45:,} rows ({bucket_31_45/len(dte_valid)*100:.1f}%)")
                print(f"    46-60 days: {bucket_46_60:,} rows ({bucket_46_60/len(dte_valid)*100:.1f}%)")
    
    # Liquidity stats
    print(f"\n💧 Liquidity Statistics:")
    liquidity_stats = {}
    
    if 'volume' in df_tradable.columns:
        volume_gt_0 = (df_tradable['volume'] > 0).sum()
        liquidity_stats['volume > 0'] = volume_gt_0
        print(f"  Rows with volume > 0: {volume_gt_0:,} ({volume_gt_0/tradable_rows*100:.2f}%)")
    
    if 'oi' in df_tradable.columns:
        oi_gt_0 = (df_tradable['oi'] > 0).sum()
        liquidity_stats['oi > 0'] = oi_gt_0
        print(f"  Rows with oi > 0: {oi_gt_0:,} ({oi_gt_0/tradable_rows*100:.2f}%)")
    
    if 'no_of_contracts' in df_tradable.columns:
        contracts_gt_0 = (df_tradable['no_of_contracts'] > 0).sum()
        liquidity_stats['no_of_contracts > 0'] = contracts_gt_0
        print(f"  Rows with no_of_contracts > 0: {contracts_gt_0:,} ({contracts_gt_0/tradable_rows*100:.2f}%)")
    
    if not liquidity_stats:
        print(f"  (No liquidity columns found)")
    
    # Duplicate check
    print(f"\n🔑 Uniqueness Check:")
    dup_count = validate_uniqueness(df_tradable)
    if dup_count == 0:
        print(f"  ✓ All rows have unique (date, expiry, strike, option_type)")
    elif dup_count > 0:
        print(f"  ⚠ WARNING: {dup_count:,} duplicate rows found")
    
    print("\n" + "="*80)


# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_output(df: pd.DataFrame, dte_threshold: int):
    """
    Save tradable universe to CSV.
    
    Args:
        df: Tradable universe DataFrame
        dte_threshold: DTE threshold used (for file naming)
    """
    print("\n" + "="*80)
    print(f"SAVING OUTPUT (DTE <= {dte_threshold} days)")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate output file name
    out_csv = OUTPUT_DIR / f"options_tradable_universe_{dte_threshold}d.csv"
    
    # Save CSV
    print(f"\nSaving to CSV: {out_csv}")
    print(f"  (This may take a while for large datasets...)")
    df.to_csv(out_csv, index=False)
    file_size_mb = out_csv.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")
    
    print(f"\n✓ Output file saved successfully")
    print(f"  - {out_csv}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline.
    Creates separate tradable universe files for each DTE threshold (30, 45, 60 days).
    """
    print("\n" + "="*80)
    print("NIFTY OPTIONS DATA - TRADABLE UNIVERSE BUILDER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nCreating tradable universes for DTE thresholds: {DTE_THRESHOLDS}")
    
    try:
        # Step 1: Load master dataset
        df_master = load_master()
        
        # Step 2: Enforce data types
        df_master = enforce_types(df_master)
        
        # Step 3: Process each DTE threshold
        results_summary = []
        
        for dte_threshold in DTE_THRESHOLDS:
            print("\n\n" + "="*80)
            print(f"PROCESSING DTE THRESHOLD: {dte_threshold} DAYS")
            print("="*80)
            
            # Filter to tradable universe
            df_tradable = filter_tradable_universe(df_master, dte_threshold)
            
            # Validate uniqueness
            dup_count = validate_uniqueness(df_tradable)
            
            # Clean data (replace inf with NaN)
            df_tradable = clean_for_saving(df_tradable)
            
            # Print summary
            print_summary(df_master, df_tradable, dte_threshold)
            
            # Save output
            save_output(df_tradable, dte_threshold)
            
            # Store results for final summary
            results_summary.append({
                'dte_threshold': dte_threshold,
                'rows': len(df_tradable),
                'pct_retained': (len(df_tradable) / len(df_master)) * 100,
                'duplicates': dup_count
            })
        
        # Print final summary
        print("\n\n" + "="*80)
        print("FINAL SUMMARY - ALL DTE THRESHOLDS")
        print("="*80)
        print(f"\nMaster dataset rows: {len(df_master):,}")
        print(f"\nTradable Universe Comparison:")
        print(f"{'DTE':<10} {'Rows':<15} {'% Retained':<15} {'Duplicates':<15}")
        print("-" * 60)
        for result in results_summary:
            print(f"{result['dte_threshold']:<10} {result['rows']:<15,} {result['pct_retained']:<15.2f} {result['duplicates']:<15,}")
        
        print("\n" + "="*80)
        print("✓ PROCESS COMPLETE")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutput files created:")
        for dte_threshold in DTE_THRESHOLDS:
            out_file = OUTPUT_DIR / f"options_tradable_universe_{dte_threshold}d.csv"
            print(f"  - {out_file}")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

