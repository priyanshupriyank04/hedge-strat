"""
NIFTY Options Data - Unified Master Long Dataset Builder

This script combines calls and puts master datasets into a single unified table
with canonical schema and liquidity/tradability flags.

Usage:
    python build_options_master_long.py

Input:
    - output/calls_master_dataset.csv
    - output/puts_master_dataset.csv

Output:
    - options_output/options_master_long.parquet
    - options_output/options_master_long.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import re

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CALLS_INPUT = "output/calls_master_dataset.csv"
PUTS_INPUT = "output/puts_master_dataset.csv"

OUTPUT_DIR = Path("options_output")
OUT_PARQUET = OUTPUT_DIR / "options_master_long.parquet"
OUT_CSV = OUTPUT_DIR / "options_master_long.csv"


# ============================================================================
# CANONICAL SCHEMA DEFINITION
# ============================================================================

CANONICAL_COLUMNS = {
    # Core keys
    'symbol': 'string',
    'date': 'datetime',
    'expiry': 'datetime',
    'option_type': 'string',
    'strike': 'float',
    
    # Prices
    'open': 'float',
    'high': 'float',
    'low': 'float',
    'close': 'float',
    'ltp': 'float',
    'settle_price': 'float',
    
    # Liquidity
    'no_of_contracts': 'float',
    'turnover': 'float',
    'volume': 'float',
    'oi': 'float',
    
    # Underlying
    'underlying_value': 'float',
    
    # Source tracking
    'source_file': 'string',
}


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================

def normalize_col_name(name):
    """Normalize column name for matching: lowercase, remove spaces/punctuation"""
    if pd.isna(name):
        return ''
    # Normalize: lowercase, strip, replace multiple spaces with single space
    normalized = str(name).lower().strip()
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove spaces, underscores, dots, special chars for exact matching
    normalized = normalized.replace(' ', '').replace('_', '').replace('.', '').replace('*', '').replace('₹', '')
    return normalized


def find_underlying_column(df: pd.DataFrame) -> str:
    """
    Find the underlying value column using robust matching.
    Looks for columns containing both "underlying" and ("value" or "spot").
    
    Returns:
        Column name if found, None otherwise
    """
    for col in df.columns:
        col_normalized = normalize_col_name(col)
        # Check if column contains "underlying" and ("value" or "spot")
        has_underlying = 'underlying' in col_normalized
        has_value_or_spot = 'value' in col_normalized or 'spot' in col_normalized
        
        if has_underlying and has_value_or_spot:
            return col
    
    return None


def map_to_canonical_columns(df: pd.DataFrame) -> dict:
    """
    Map existing columns to canonical schema.
    Returns a dictionary mapping original column names to canonical names.
    """
    column_mapping = {}
    mapped_targets = set()
    
    # Define patterns for each canonical column
    col_patterns = {
        'symbol': ['symbol'],
        'date': ['date'],
        'expiry': ['expiry', 'expirydate'],
        'option_type': ['optiontype', 'type', 'option_type'],
        'strike': ['strikeprice', 'strike', 'strike_price'],
        'open': ['open'],
        'high': ['high'],
        'low': ['low'],
        'close': ['close'],
        'ltp': ['ltp', 'lasttradedprice', 'last_traded_price'],
        'settle_price': ['settleprice', 'settle', 'settle_price'],
        'no_of_contracts': ['noofcontracts', 'no_of_contracts', 'contracts'],
        'turnover': ['turnover'],
        'volume': ['volume', 'noofcontracts', 'no_of_contracts'],
        'oi': ['openint', 'openinterest', 'oi', 'open_int'],
    }
    
    # Special handling for underlying_value: use robust matching
    underlying_col = find_underlying_column(df)
    if underlying_col:
        column_mapping[underlying_col] = 'underlying_value'
        mapped_targets.add('underlying_value')
    
    # Find matching columns
    for target_col, patterns in col_patterns.items():
        if target_col in mapped_targets:
            continue  # Already mapped
        
        for col in df.columns:
            if col in column_mapping:
                continue  # Already mapped
            
            col_normalized = normalize_col_name(col)
            for pattern in patterns:
                if pattern == col_normalized:
                    column_mapping[col] = target_col
                    mapped_targets.add(target_col)
                    break
            if col in column_mapping:
                break
    
    return column_mapping


# ============================================================================
# LOAD AND STANDARDIZE
# ============================================================================

def load_and_standardize(path: str, forced_option_type: str, source_file: str) -> pd.DataFrame:
    """
    Load a master dataset file and standardize it to canonical schema.
    
    Args:
        path: Path to the CSV file
        forced_option_type: "CE" for calls, "PE" for puts
        source_file: Source file identifier for tracking
    
    Returns:
        Standardized DataFrame with canonical schema
    """
    print(f"\nLoading: {path}")
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise ValueError(f"Could not read {path} with any encoding")
    
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Check for underlying column before mapping (to log it)
    underlying_col = find_underlying_column(df)
    if underlying_col:
        print(f"  Found underlying column: '{underlying_col}' -> will map to 'underlying_value'")
    else:
        print(f"  ⚠ WARNING: No underlying value column found in {path}")
    
    # Map columns to canonical schema
    column_mapping = map_to_canonical_columns(df)
    print(f"  Mapped {len(column_mapping)} columns to canonical schema")
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Force option_type
    df['option_type'] = forced_option_type
    
    # Add source_file
    df['source_file'] = source_file
    
    # Ensure all canonical columns exist (create as NaN if missing)
    for col in CANONICAL_COLUMNS.keys():
        if col not in df.columns:
            df[col] = np.nan
    
    # Parse dates
    for date_col in ['date', 'expiry']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    
    # Convert numeric columns
    numeric_cols = ['strike', 'open', 'high', 'low', 'close', 'ltp', 'settle_price',
                    'no_of_contracts', 'turnover', 'volume', 'oi', 'underlying_value']
    for col in numeric_cols:
        if col in df.columns:
            # Replace common placeholders
            df[col] = df[col].replace(['-', '', ' ', 'nan', 'NaN', 'None', 'null', 'NULL'], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Log underlying_value status
    if 'underlying_value' not in df.columns or df['underlying_value'].isna().all():
        print(f"  ⚠ WARNING: underlying_value column not found or all NaN in {path}")
    else:
        non_null_count = df['underlying_value'].notna().sum()
        print(f"  underlying_value: {non_null_count:,} non-null values ({non_null_count/len(df)*100:.1f}%)")
    
    # Normalize symbol
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
    
    # Select only canonical columns (in order)
    canonical_cols = [col for col in CANONICAL_COLUMNS.keys() if col in df.columns]
    df = df[canonical_cols]
    
    print(f"  Standardized to {len(df.columns)} canonical columns")
    
    return df


# ============================================================================
# DEDUPLICATION
# ============================================================================

def calculate_liquidity_score(row: pd.Series) -> float:
    """
    Calculate liquidity score for a row.
    Higher score = more liquid.
    """
    score = 0.0
    
    # Add liquidity indicators
    for col in ['no_of_contracts', 'volume', 'turnover', 'oi', 'close']:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and val > 0:
                score += float(val)
    
    return score


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates based on (date, expiry, strike, option_type).
    Keep the row with highest liquidity_score.
    """
    print(f"\nDeduplicating...")
    
    # Check for duplicates before
    key_cols = ['date', 'expiry', 'strike', 'option_type']
    duplicates_before = df.duplicated(subset=key_cols, keep=False).sum()
    print(f"  Duplicates before dedup: {duplicates_before:,}")
    
    if duplicates_before == 0:
        print(f"  No duplicates found, skipping deduplication")
        return df
    
    # Calculate liquidity score for each row
    print(f"  Calculating liquidity scores...")
    df['_liquidity_score'] = df.apply(calculate_liquidity_score, axis=1)
    
    # Sort by key columns and liquidity score (descending)
    # This ensures highest liquidity row comes first for each key group
    sort_cols = key_cols + ['_liquidity_score']
    df = df.sort_values(by=sort_cols, ascending=[True]*len(key_cols) + [False], na_position='last')
    
    # Keep first row for each key (highest liquidity)
    df = df.drop_duplicates(subset=key_cols, keep='first')
    
    # Remove temporary column
    df = df.drop(columns=['_liquidity_score'])
    
    # Check for duplicates after
    duplicates_after = df.duplicated(subset=key_cols, keep=False).sum()
    print(f"  Duplicates after dedup: {duplicates_after:,}")
    
    return df.reset_index(drop=True)


# ============================================================================
# ADD FLAGS
# ============================================================================

def add_is_liquid_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_liquid flag: True if ANY liquidity indicator is positive.
    """
    print(f"\nAdding is_liquid flag...")
    
    conditions = []
    
    # Check each liquidity indicator
    liquidity_cols = ['close', 'ltp', 'no_of_contracts', 'volume', 'turnover', 'oi']
    for col in liquidity_cols:
        if col in df.columns:
            conditions.append((df[col].notna()) & (df[col] > 0))
    
    if conditions:
        df['is_liquid'] = pd.concat(conditions, axis=1).any(axis=1)
    else:
        df['is_liquid'] = False
        print(f"  Warning: No liquidity columns found, all marked as not liquid")
    
    liquid_count = df['is_liquid'].sum()
    liquid_pct = (liquid_count / len(df)) * 100
    print(f"  is_liquid=True: {liquid_count:,} rows ({liquid_pct:.2f}%)")
    
    return df


def add_is_tradable_30d_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_tradable_30d flag and dte_days column.
    is_tradable_30d = True if 0 <= dte <= 30
    """
    print(f"\nAdding is_tradable_30d flag...")
    
    # Calculate days to expiry
    if 'date' in df.columns and 'expiry' in df.columns:
        df['dte_days'] = (df['expiry'] - df['date']).dt.days
        
        # is_tradable_30d: 0 <= dte <= 30
        df['is_tradable_30d'] = (df['dte_days'] >= 0) & (df['dte_days'] <= 30)
        
        tradable_count = df['is_tradable_30d'].sum()
        tradable_pct = (tradable_count / len(df)) * 100
        print(f"  is_tradable_30d=True: {tradable_count:,} rows ({tradable_pct:.2f}%)")
        
        # Check for negative DTE
        negative_dte = (df['dte_days'] < 0).sum()
        if negative_dte > 0:
            print(f"  Warning: {negative_dte:,} rows with negative dte_days")
    else:
        df['dte_days'] = np.nan
        df['is_tradable_30d'] = False
        print(f"  Warning: Missing date/expiry columns, cannot calculate dte")
    
    return df


# ============================================================================
# SORTING
# ============================================================================

def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by: date, expiry, strike, option_type
    """
    print(f"\nSorting dataset...")
    
    sort_cols = []
    for col in ['date', 'expiry', 'strike', 'option_type']:
        if col in df.columns:
            sort_cols.append(col)
    
    if sort_cols:
        df = df.sort_values(by=sort_cols, na_position='last').reset_index(drop=True)
        print(f"  Sorted by: {', '.join(sort_cols)}")
    else:
        print(f"  Warning: No sort columns found")
    
    return df


# ============================================================================
# VALIDATE UNDERLYING CONSISTENCY
# ============================================================================

def validate_underlying_consistency(df: pd.DataFrame, tolerance: float = 0.5) -> dict:
    """
    Validate that underlying_value is consistent across calls vs puts for the same date.
    
    Args:
        df: Combined dataframe with calls and puts
        tolerance: Maximum allowed difference between min and max underlying_value per date
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*80)
    print("VALIDATING UNDERLYING VALUE CONSISTENCY")
    print("="*80)
    
    if 'underlying_value' not in df.columns or 'date' not in df.columns:
        print("  ⚠ WARNING: Missing underlying_value or date column, skipping validation")
        return {
            'underlying_nan_pct': None,
            'inconsistent_dates_count': None,
            'problematic_dates': []
        }
    
    # Calculate % NaN
    underlying_nan_count = df['underlying_value'].isna().sum()
    underlying_nan_pct = (underlying_nan_count / len(df)) * 100
    print(f"\nUnderlying value statistics:")
    print(f"  Rows with underlying_value NaN: {underlying_nan_count:,} ({underlying_nan_pct:.2f}%)")
    
    # Group by date and compute statistics
    date_stats = df.groupby('date')['underlying_value'].agg([
        ('count_non_null', lambda x: x.notna().sum()),
        ('min_underlying', 'min'),
        ('max_underlying', 'max'),
    ]).reset_index()
    
    # Calculate max_minus_min
    date_stats['max_minus_min'] = date_stats['max_underlying'] - date_stats['min_underlying']
    
    # Find dates with inconsistency > tolerance
    inconsistent_dates = date_stats[
        (date_stats['count_non_null'] > 0) & 
        (date_stats['max_minus_min'] > tolerance)
    ].copy()
    
    inconsistent_dates_count = len(inconsistent_dates)
    print(f"\nUnderlying consistency check (tolerance: {tolerance}):")
    print(f"  Dates with inconsistency > tolerance: {inconsistent_dates_count:,}")
    
    if inconsistent_dates_count > 0:
        print(f"\n  ⚠ WARNING: Found {inconsistent_dates_count} dates with underlying inconsistency")
        print(f"  Top 10 problematic dates:")
        
        # Sort by max_minus_min descending and show top 10
        top_problematic = inconsistent_dates.nlargest(10, 'max_minus_min')
        for idx, row in top_problematic.iterrows():
            print(f"    Date: {row['date']}, min: {row['min_underlying']:.2f}, "
                  f"max: {row['max_underlying']:.2f}, diff: {row['max_minus_min']:.2f}, "
                  f"count: {row['count_non_null']:.0f}")
        
        problematic_dates_list = top_problematic[['date', 'min_underlying', 'max_underlying', 
                                                   'max_minus_min', 'count_non_null']].to_dict('records')
    else:
        print(f"  ✓ All dates have consistent underlying_value (within tolerance)")
        problematic_dates_list = []
    
    return {
        'underlying_nan_pct': underlying_nan_pct,
        'inconsistent_dates_count': inconsistent_dates_count,
        'problematic_dates': problematic_dates_list
    }


# ============================================================================
# VALIDATION AND LOGGING
# ============================================================================

def validate_and_log(df: pd.DataFrame, underlying_validation: dict = None):
    """
    Print validation statistics and summary.
    """
    print("\n" + "="*80)
    print("VALIDATION & SUMMARY")
    print("="*80)
    
    # Total rows
    print(f"\nTotal rows: {len(df):,}")
    
    # Duplicate check
    key_cols = ['date', 'expiry', 'strike', 'option_type']
    duplicates = df.duplicated(subset=key_cols, keep=False).sum()
    print(f"Unique key duplicates: {duplicates:,} (should be 0)")
    
    # Flag percentages
    if 'is_liquid' in df.columns:
        liquid_pct = (df['is_liquid'].sum() / len(df)) * 100
        print(f"is_liquid=True: {df['is_liquid'].sum():,} ({liquid_pct:.2f}%)")
    
    if 'is_tradable_30d' in df.columns:
        tradable_pct = (df['is_tradable_30d'].sum() / len(df)) * 100
        print(f"is_tradable_30d=True: {df['is_tradable_30d'].sum():,} ({tradable_pct:.2f}%)")
    
    # Underlying value statistics
    if underlying_validation:
        print(f"\nUnderlying value:")
        if underlying_validation['underlying_nan_pct'] is not None:
            print(f"  underlying_value NaN: {underlying_validation['underlying_nan_pct']:.2f}%")
        if underlying_validation['inconsistent_dates_count'] is not None:
            print(f"  Dates with inconsistency > tolerance: {underlying_validation['inconsistent_dates_count']:,}")
    
    # Negative DTE check
    if 'dte_days' in df.columns:
        negative_dte = (df['dte_days'] < 0).sum()
        print(f"Rows with dte_days < 0: {negative_dte:,}")
        if negative_dte > 0:
            print(f"  ⚠ Warning: Found {negative_dte:,} rows with negative dte_days")
    
    # Date ranges
    if 'date' in df.columns:
        date_min = df['date'].min()
        date_max = df['date'].max()
        print(f"\nDate range:")
        print(f"  Min: {date_min}")
        print(f"  Max: {date_max}")
    
    if 'expiry' in df.columns:
        expiry_min = df['expiry'].min()
        expiry_max = df['expiry'].max()
        print(f"\nExpiry range:")
        print(f"  Min: {expiry_min}")
        print(f"  Max: {expiry_max}")
    
    # Sample date analysis
    if 'date' in df.columns and 'is_tradable_30d' in df.columns:
        last_date = df['date'].max()
        if pd.notna(last_date):
            last_date_df = df[df['date'] == last_date].copy()
            
            # Find front expiry (minimum expiry for this date)
            if 'expiry' in last_date_df.columns:
                front_expiry = last_date_df['expiry'].min()
                if pd.notna(front_expiry):
                    front_expiry_df = last_date_df[last_date_df['expiry'] == front_expiry]
                    tradable_strikes = front_expiry_df[front_expiry_df['is_tradable_30d'] == True]
                    
                    print(f"\nSample date analysis (last date: {last_date}):")
                    print(f"  Front expiry: {front_expiry}")
                    print(f"  Strikes with is_tradable_30d=True: {len(tradable_strikes):,}")
    
    print("\n" + "="*80)


# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_output(df: pd.DataFrame):
    """
    Save dataset to parquet and CSV.
    """
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean mixed-type columns for Parquet compatibility
    print("\nCleaning mixed-type columns for Parquet...")
    df_clean = df.copy()
    
    # Convert object columns that should be numeric
    object_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    exclude_cols = ['symbol', 'option_type', 'source_file']
    object_cols = [col for col in object_cols if col not in exclude_cols]
    
    for col in object_cols:
        try:
            cleaned = df_clean[col].replace(['-', '', ' ', 'nan', 'NaN', 'None', 'null', 'NULL'], np.nan)
            numeric_series = pd.to_numeric(cleaned, errors='coerce')
            non_null_before = cleaned.notna().sum()
            non_null_after = numeric_series.notna().sum()
            
            if non_null_before > 0 and non_null_after / non_null_before > 0.5:
                df_clean[col] = numeric_series
        except Exception:
            pass
    
    # Save Parquet
    print(f"\nSaving to Parquet: {OUT_PARQUET}")
    df_clean.to_parquet(OUT_PARQUET, index=False, engine='pyarrow', compression='snappy')
    file_size_mb = OUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")
    
    # Save CSV
    print(f"\nSaving to CSV: {OUT_CSV}")
    print(f"  (This may take a while for large datasets...)")
    df_clean.to_csv(OUT_CSV, index=False)
    file_size_mb = OUT_CSV.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")
    
    print(f"\n✓ Output files saved successfully")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline.
    """
    print("\n" + "="*80)
    print("NIFTY OPTIONS DATA - UNIFIED MASTER LONG DATASET BUILDER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load and standardize calls
        print("\n" + "="*80)
        print("STEP 1: LOADING CALLS")
        print("="*80)
        df_calls = load_and_standardize(
            CALLS_INPUT,
            forced_option_type="CE",
            source_file="calls_master_dataset.csv"
        )
        
        # Step 2: Load and standardize puts
        print("\n" + "="*80)
        print("STEP 2: LOADING PUTS")
        print("="*80)
        df_puts = load_and_standardize(
            PUTS_INPUT,
            forced_option_type="PE",
            source_file="puts_master_dataset.csv"
        )
        
        # Step 3: Combine calls and puts
        print("\n" + "="*80)
        print("STEP 3: COMBINING CALLS AND PUTS")
        print("="*80)
        df_combined = pd.concat([df_calls, df_puts], ignore_index=True)
        print(f"Combined dataset: {len(df_combined):,} rows")
        
        # Step 3.5: Validate underlying consistency
        underlying_validation = validate_underlying_consistency(df_combined, tolerance=0.5)
        
        # Step 4: Deduplicate
        print("\n" + "="*80)
        print("STEP 4: DEDUPLICATION")
        print("="*80)
        df_combined = deduplicate(df_combined)
        
        # Step 5: Add flags
        print("\n" + "="*80)
        print("STEP 5: ADDING FLAGS")
        print("="*80)
        df_combined = add_is_liquid_flag(df_combined)
        df_combined = add_is_tradable_30d_flag(df_combined)
        
        # Step 6: Sort
        print("\n" + "="*80)
        print("STEP 6: SORTING")
        print("="*80)
        df_combined = sort_dataset(df_combined)
        
        # Step 7: Validate and log
        validate_and_log(df_combined, underlying_validation=underlying_validation)
        
        # Step 8: Save output
        save_output(df_combined)
        
        print("\n" + "="*80)
        print("✓ PROCESS COMPLETE")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutput files:")
        print(f"  - {OUT_PARQUET}")
        print(f"  - {OUT_CSV}")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

