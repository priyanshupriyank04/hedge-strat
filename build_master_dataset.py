"""
NIFTY Options Data Master Dataset Builder

This script merges all raw CSV files into a single canonical master dataset
with normalized schema, derived columns, flags, and validations.

Usage:
    python build_master_dataset.py

Input:
    Raw CSV files in data/Calls/ and data/Puts/

Output:
    - master_dataset.parquet (primary output)
    - master_dataset.csv (optional CSV export)
    - Summary statistics printed to console
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR = Path("data")
CALLS_DIR = INPUT_DIR / "Calls"
PUTS_DIR = INPUT_DIR / "Puts"
OUTPUT_DIR = Path("output")
OUTPUT_PARQUET = OUTPUT_DIR / "master_dataset.parquet"
OUTPUT_CSV = OUTPUT_DIR / "master_dataset.csv"

# Liquid candidate thresholds (configurable)
LIQUID_VOLUME_THRESHOLD = 1
LIQUID_OI_THRESHOLD = 1


# ============================================================================
# STEP 1: DATA INGESTION
# ============================================================================

def read_all_csvs(input_dir: Path, calls_dir: Path, puts_dir: Path) -> pd.DataFrame:
    """
    Read all CSV files from calls and puts directories and merge into one DataFrame.
    
    Returns:
        Raw DataFrame with all rows from all CSV files
    """
    print("\n" + "="*80)
    print("STEP 1: DATA INGESTION")
    print("="*80)
    
    all_dfs = []
    total_files = 0
    
    # Read calls directory
    if calls_dir.exists():
        csv_files = list(calls_dir.glob("*.csv"))
        print(f"\nFound {len(csv_files)} CSV files in Calls/ directory")
        for csv_file in csv_files:
            try:
                print(f"  Reading: {csv_file.name}...", end=" ")
                # Try multiple encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                for enc in encodings:
                    try:
                        df = pd.read_csv(csv_file, low_memory=False, encoding=enc)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                
                if df is None:
                    raise ValueError("Could not read file with any encoding")
                
                if len(df) == 0:
                    print(f"⚠ (empty file, skipping)")
                    continue
                df['_source_file'] = f"Calls/{csv_file.name}"
                all_dfs.append(df)
                print(f"✓ ({len(df)} rows)")
                total_files += 1
            except Exception as e:
                print(f"✗ ERROR reading {csv_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Read puts directory
    if puts_dir.exists():
        csv_files = list(puts_dir.glob("*.csv"))
        print(f"\nFound {len(csv_files)} CSV files in Puts/ directory")
        for csv_file in csv_files:
            try:
                print(f"  Reading: {csv_file.name}...", end=" ")
                # Try multiple encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                for enc in encodings:
                    try:
                        df = pd.read_csv(csv_file, low_memory=False, encoding=enc)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                
                if df is None:
                    raise ValueError("Could not read file with any encoding")
                
                if len(df) == 0:
                    print(f"⚠ (empty file, skipping)")
                    continue
                df['_source_file'] = f"Puts/{csv_file.name}"
                all_dfs.append(df)
                print(f"✓ ({len(df)} rows)")
                total_files += 1
            except Exception as e:
                print(f"✗ ERROR reading {csv_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    if not all_dfs:
        raise ValueError("No CSV files found in Calls/ or Puts/ directories!")
    
    print(f"\nMerging {len(all_dfs)} DataFrames...")
    try:
        df_raw = pd.concat(all_dfs, ignore_index=True, sort=False)
        print(f"✓ Total raw rows: {len(df_raw):,}")
        print(f"✓ Total columns: {len(df_raw.columns)}")
    except Exception as e:
        print(f"✗ ERROR merging DataFrames: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return df_raw


# ============================================================================
# STEP 2: COLUMN NORMALIZATION
# ============================================================================

def normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to consistent schema.
    Handles variations in column names, extra spaces, and optional columns.
    
    Returns:
        DataFrame with normalized column names
    """
    print("\n" + "="*80)
    print("STEP 2: COLUMN NORMALIZATION")
    print("="*80)
    
    df = df_raw.copy()
    
    # Strip whitespace from column names and handle duplicates
    new_columns = []
    seen = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in seen:
            seen[col_stripped] += 1
            new_columns.append(f"{col_stripped}_{seen[col_stripped]}")
        else:
            seen[col_stripped] = 0
            new_columns.append(col_stripped)
    df.columns = new_columns
    
    print(f"\nOriginal columns: {list(df.columns)}")
    
    # Column mapping: flexible matching for common variations
    column_mapping = {}
    mapped_columns = set()  # Track columns we've already mapped to avoid conflicts
    
    # Required columns with flexible matching (normalize by removing spaces, special chars)
    def normalize_col_name(name):
        """Normalize column name for matching: lowercase, remove spaces/punctuation"""
        return name.lower().strip().replace(' ', '').replace('_', '').replace('.', '').replace('*', '').replace('₹', '')
    
    col_patterns = {
        'symbol': ['symbol'],
        'date': ['date'],
        'expiry': ['expiry', 'expirydate'],
        'option_type': ['optiontype', 'type'],
        'strike': ['strikeprice', 'strike'],
        'open': ['open'],
        'high': ['high'],
        'low': ['low'],
        'close': ['close'],
        'ltp': ['ltp', 'lasttradedprice'],
        'settle_price': ['settleprice', 'settle'],
        'volume': ['noofcontracts', 'volume', 'contracts'],
        'oi': ['openint', 'openinterest', 'oi'],
    }
    
    # Find matching columns (case-insensitive, space/punctuation-tolerant)
    for target_col, patterns in col_patterns.items():
        for col in df.columns:
            if col in mapped_columns:  # Skip if already mapped
                continue
            col_normalized = normalize_col_name(col)
            for pattern in patterns:
                if pattern == col_normalized:
                    column_mapping[col] = target_col
                    mapped_columns.add(col)
                    break
            if col in column_mapping:
                break
    
    # Check for duplicate target names before renaming
    if len(column_mapping.values()) != len(set(column_mapping.values())):
        print(f"  Warning: Duplicate target column names detected in mapping")
        # Keep only the first occurrence of each target
        seen_targets = set()
        filtered_mapping = {}
        for col, target in column_mapping.items():
            if target not in seen_targets:
                filtered_mapping[col] = target
                seen_targets.add(target)
            else:
                print(f"    Skipping duplicate mapping: {col} -> {target}")
        column_mapping = filtered_mapping
    
    # Apply mapping (only normalize key columns, keep all others)
    df = df.rename(columns=column_mapping)
    
    # Keep ALL columns from original CSV files - don't filter any out
    # We only normalized the key column names for consistency
    print(f"\nNormalized key columns: {list(column_mapping.values())}")
    print(f"All columns preserved: {len(df.columns)} total columns")
    print(f"  {list(df.columns)}")
    print(f"✓ Column normalization complete (all original columns preserved)")
    
    return df


# ============================================================================
# STEP 3: DATA TYPE CONVERSION & CLEANING
# ============================================================================

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns to datetime and numeric columns to float.
    Handle '-' placeholders and missing values as NaN.
    
    Returns:
        DataFrame with proper data types
    """
    print("\n" + "="*80)
    print("STEP 3: DATA TYPE CONVERSION & CLEANING")
    print("="*80)
    
    df = df.copy()
    
    # Parse dates
    date_cols = ['date', 'expiry']
    for col in date_cols:
        if col in df.columns:
            try:
                print(f"\nParsing {col}...")
                # Try multiple date formats
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                null_count = df[col].isna().sum()
                if null_count > 0:
                    print(f"  Warning: {null_count:,} rows with invalid {col} format")
                else:
                    print(f"  ✓ All {col} values parsed successfully")
            except Exception as e:
                print(f"  ⚠ Warning: Could not parse {col} as datetime: {e}")
    
    # Convert numeric columns
    numeric_cols = ['strike', 'open', 'high', 'low', 'close', 'ltp', 'settle_price', 'volume', 'oi']
    for col in numeric_cols:
        if col in df.columns:
            try:
                print(f"\nConverting {col} to numeric...")
                # Replace '-' and empty strings with NaN
                df[col] = df[col].replace(['-', '', ' ', 'nan', 'NaN', 'None'], np.nan)
                # Convert to float
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Count non-null values
                non_null = df[col].notna().sum()
                print(f"  ✓ {non_null:,} non-null values ({non_null/len(df)*100:.1f}%)")
            except Exception as e:
                print(f"  ⚠ Warning: Could not convert {col} to numeric: {e}")
                # Keep as string/object type if conversion fails
    
    # Normalize option_type: CE/PE to uppercase, strip spaces
    if 'option_type' in df.columns:
        try:
            print(f"\nNormalizing option_type...")
            df['option_type'] = df['option_type'].astype(str).str.strip().str.upper()
            # Map common variations
            df['option_type'] = df['option_type'].replace({
                'CALL': 'CE', 'C': 'CE', 'CALLS': 'CE',
                'PUT': 'PE', 'P': 'PE', 'PUTS': 'PE'
            })
            print(f"  ✓ Option types: {df['option_type'].value_counts().to_dict()}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not normalize option_type: {e}")
    
    # Normalize symbol: strip and uppercase
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
    
    print(f"\n✓ Data type conversion complete")
    
    return df


# ============================================================================
# STEP 4: ADD DERIVED COLUMNS & FLAGS
# ============================================================================

def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns:
    - dte: days to expiry
    - has_valid_ohlc: all OHLC values present
    - is_traded_today: volume > 0 OR ltp > 0 OR close > 0
    - is_liquid_candidate: volume >= threshold AND oi >= threshold (if columns exist)
    
    Returns:
        DataFrame with added derived columns
    """
    print("\n" + "="*80)
    print("STEP 4: ADDING DERIVED COLUMNS & FLAGS")
    print("="*80)
    
    df = df.copy()
    
    # Calculate days to expiry (dte)
    if 'date' in df.columns and 'expiry' in df.columns:
        print("\nCalculating dte (days to expiry)...")
        df['dte'] = (df['expiry'] - df['date']).dt.days
        negative_dte = (df['dte'] < 0).sum()
        if negative_dte > 0:
            print(f"  Warning: {negative_dte:,} rows with negative dte (expiry < date)")
        print(f"  ✓ dte calculated: min={df['dte'].min():.0f}, max={df['dte'].max():.0f}, mean={df['dte'].mean():.1f}")
    
    # has_valid_ohlc: all OHLC present
    ohlc_cols = ['open', 'high', 'low', 'close']
    existing_ohlc = [col for col in ohlc_cols if col in df.columns]
    if existing_ohlc:
        print(f"\nCalculating has_valid_ohlc...")
        df['has_valid_ohlc'] = df[existing_ohlc].notna().all(axis=1)
        valid_count = df['has_valid_ohlc'].sum()
        print(f"  ✓ {valid_count:,} rows with valid OHLC ({valid_count/len(df)*100:.1f}%)")
    
    # is_traded_today: volume > 0 OR ltp > 0 OR close > 0
    print(f"\nCalculating is_traded_today...")
    trade_conditions = []
    
    if 'volume' in df.columns:
        trade_conditions.append(df['volume'] > 0)
    if 'ltp' in df.columns:
        trade_conditions.append((df['ltp'].notna()) & (df['ltp'] > 0))
    if 'close' in df.columns:
        trade_conditions.append((df['close'].notna()) & (df['close'] > 0))
    
    if trade_conditions:
        df['is_traded_today'] = pd.concat(trade_conditions, axis=1).any(axis=1)
        traded_count = df['is_traded_today'].sum()
        print(f"  ✓ {traded_count:,} rows marked as traded ({traded_count/len(df)*100:.1f}%)")
    else:
        df['is_traded_today'] = False
        print(f"  Warning: No volume/ltp/close columns found, all marked as not traded")
    
    # is_liquid_candidate: configurable thresholds
    print(f"\nCalculating is_liquid_candidate...")
    liquid_conditions = []
    
    if 'volume' in df.columns:
        liquid_conditions.append(df['volume'] >= LIQUID_VOLUME_THRESHOLD)
    if 'oi' in df.columns:
        liquid_conditions.append(df['oi'] >= LIQUID_OI_THRESHOLD)
    
    if len(liquid_conditions) == 2:
        # Both volume and OI must meet thresholds
        df['is_liquid_candidate'] = pd.concat(liquid_conditions, axis=1).all(axis=1)
    elif len(liquid_conditions) == 1:
        # Only one condition available
        df['is_liquid_candidate'] = liquid_conditions[0]
    else:
        # No conditions available
        df['is_liquid_candidate'] = False
        print(f"  Warning: No volume/OI columns found, all marked as not liquid")
    
    if 'is_liquid_candidate' in df.columns:
        liquid_count = df['is_liquid_candidate'].sum()
        print(f"  ✓ {liquid_count:,} rows marked as liquid candidates ({liquid_count/len(df)*100:.1f}%)")
    
    print(f"\n✓ Derived columns added")
    
    return df


# ============================================================================
# STEP 5: SORTING
# ============================================================================

def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataset deterministically:
    date asc → expiry asc → option_type (CE then PE) → strike asc
    
    Returns:
        Sorted DataFrame
    """
    print("\n" + "="*80)
    print("STEP 5: SORTING")
    print("="*80)
    
    df = df.copy()
    
    # Create sort key for option_type: CE=0, PE=1 (CE comes first)
    if 'option_type' in df.columns:
        df['_sort_option_type'] = df['option_type'].map({'CE': 0, 'PE': 1}).fillna(2)
    
    # Sort order: date → strike → option_type (CE then PE)
    sort_cols = []
    if 'date' in df.columns:
        sort_cols.append('date')
    if 'strike' in df.columns:
        sort_cols.append('strike')
    if '_sort_option_type' in df.columns:
        sort_cols.append('_sort_option_type')
    
    print(f"Sorting by: date → strike → option_type (CE then PE)")
    df = df.sort_values(by=sort_cols, na_position='last').reset_index(drop=True)
    
    # Drop temporary sort column
    if '_sort_option_type' in df.columns:
        df = df.drop(columns=['_sort_option_type'])
    
    print(f"✓ Dataset sorted: {len(df):,} rows")
    
    return df


# ============================================================================
# STEP 6: VALIDATION
# ============================================================================

def validate(df: pd.DataFrame) -> Dict:
    """
    Perform data quality checks and validations.
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*80)
    print("STEP 6: DATA VALIDATION")
    print("="*80)
    
    validation_results = {}
    
    # 1. Duplicate check
    print("\n1. Checking for duplicates...")
    key_cols = ['symbol', 'date', 'expiry', 'option_type', 'strike']
    existing_key_cols = [col for col in key_cols if col in df.columns]
    
    if len(existing_key_cols) == len(key_cols):
        duplicates = df.duplicated(subset=existing_key_cols, keep=False)
        dup_count = duplicates.sum()
        validation_results['duplicates'] = dup_count
        if dup_count > 0:
            print(f"  ⚠ Warning: {dup_count:,} duplicate rows found on key columns")
        else:
            print(f"  ✓ No duplicates found")
    else:
        print(f"  ⚠ Warning: Missing key columns for duplicate check")
        validation_results['duplicates'] = None
    
    # 2. Negative DTE check
    if 'dte' in df.columns:
        print("\n2. Checking for negative dte...")
        negative_dte = (df['dte'] < 0).sum()
        validation_results['negative_dte'] = negative_dte
        if negative_dte > 0:
            print(f"  ⚠ Warning: {negative_dte:,} rows with negative dte (expiry < date)")
        else:
            print(f"  ✓ No negative dte found")
    
    # 3. Negative price check
    print("\n3. Checking for negative prices...")
    price_cols = ['open', 'high', 'low', 'close', 'ltp', 'settle_price']
    existing_price_cols = [col for col in price_cols if col in df.columns]
    
    negative_prices = {}
    for col in existing_price_cols:
        neg_count = ((df[col].notna()) & (df[col] < 0)).sum()
        if neg_count > 0:
            negative_prices[col] = neg_count
            print(f"  ⚠ Warning: {neg_count:,} negative values in {col}")
    
    validation_results['negative_prices'] = negative_prices
    if not negative_prices:
        print(f"  ✓ No negative prices found")
    
    # 4. OHLC consistency check
    print("\n4. Checking OHLC consistency...")
    ohlc_cols = ['open', 'high', 'low', 'close']
    existing_ohlc = [col for col in ohlc_cols if col in df.columns]
    
    if len(existing_ohlc) == 4:
        # Check: high >= max(open, close, low) and low <= min(open, close, high)
        df_check = df[existing_ohlc].copy()
        df_check = df_check.dropna()
        
        if len(df_check) > 0:
            high_violations = (df_check['high'] < df_check[['open', 'close', 'low']].max(axis=1)).sum()
            low_violations = (df_check['low'] > df_check[['open', 'close', 'high']].min(axis=1)).sum()
            
            validation_results['ohlc_high_violations'] = high_violations
            validation_results['ohlc_low_violations'] = low_violations
            
            if high_violations > 0:
                print(f"  ⚠ Warning: {high_violations:,} rows where high < max(open,close,low)")
            if low_violations > 0:
                print(f"  ⚠ Warning: {low_violations:,} rows where low > min(open,close,high)")
            if high_violations == 0 and low_violations == 0:
                print(f"  ✓ OHLC consistency check passed")
        else:
            print(f"  ⚠ No rows with complete OHLC to check")
            validation_results['ohlc_high_violations'] = 0
            validation_results['ohlc_low_violations'] = 0
    else:
        print(f"  ⚠ Missing OHLC columns for consistency check")
        validation_results['ohlc_high_violations'] = None
        validation_results['ohlc_low_violations'] = None
    
    print(f"\n✓ Validation complete")
    
    return validation_results


# ============================================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================================

def print_summary(df: pd.DataFrame, validation_results: Dict):
    """
    Print comprehensive summary statistics.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    
    # Date range
    if 'date' in df.columns:
        date_min = df['date'].min()
        date_max = df['date'].max()
        print(f"\n📅 Date Range:")
        print(f"  From: {date_min}")
        print(f"  To: {date_max}")
        print(f"  Span: {(date_max - date_min).days} days")
    
    # Unique counts
    print(f"\n🔢 Unique Values:")
    if 'expiry' in df.columns:
        unique_expiries = df['expiry'].nunique()
        print(f"  Unique expiries: {unique_expiries:,}")
    if 'strike' in df.columns:
        unique_strikes = df['strike'].nunique()
        print(f"  Unique strikes: {unique_strikes:,}")
    if 'option_type' in df.columns:
        print(f"  Option types: {df['option_type'].value_counts().to_dict()}")
    
    # Missing data
    print(f"\n📉 Missing Data (%):")
    numeric_cols = ['open', 'high', 'low', 'close', 'ltp', 'settle_price', 'volume', 'oi']
    for col in numeric_cols:
        if col in df.columns:
            pct_missing = (df[col].isna().sum() / len(df)) * 100
            print(f"  {col}: {pct_missing:.1f}%")
    
    # Flags summary
    print(f"\n🏷️  Flags Summary:")
    if 'has_valid_ohlc' in df.columns:
        valid_ohlc_pct = (df['has_valid_ohlc'].sum() / len(df)) * 100
        print(f"  has_valid_ohlc: {df['has_valid_ohlc'].sum():,} ({valid_ohlc_pct:.1f}%)")
    if 'is_traded_today' in df.columns:
        traded_pct = (df['is_traded_today'].sum() / len(df)) * 100
        print(f"  is_traded_today: {df['is_traded_today'].sum():,} ({traded_pct:.1f}%)")
    if 'is_liquid_candidate' in df.columns:
        liquid_pct = (df['is_liquid_candidate'].sum() / len(df)) * 100
        print(f"  is_liquid_candidate: {df['is_liquid_candidate'].sum():,} ({liquid_pct:.1f}%)")
    
    # DTE stats
    if 'dte' in df.columns:
        print(f"\n⏰ Days to Expiry (dte):")
        print(f"  Min: {df['dte'].min():.0f}")
        print(f"  Max: {df['dte'].max():.0f}")
        print(f"  Mean: {df['dte'].mean():.1f}")
        print(f"  Median: {df['dte'].median():.1f}")
    
    # Validation summary
    print(f"\n✅ Validation Results:")
    if validation_results.get('duplicates') is not None:
        print(f"  Duplicates: {validation_results['duplicates']:,}")
    if validation_results.get('negative_dte') is not None:
        print(f"  Negative dte: {validation_results['negative_dte']:,}")
    if validation_results.get('negative_prices'):
        for col, count in validation_results['negative_prices'].items():
            print(f"  Negative {col}: {count:,}")
    if validation_results.get('ohlc_high_violations') is not None:
        print(f"  OHLC high violations: {validation_results['ohlc_high_violations']:,}")
    if validation_results.get('ohlc_low_violations') is not None:
        print(f"  OHLC low violations: {validation_results['ohlc_low_violations']:,}")
    
    print("\n" + "="*80)


# ============================================================================
# STEP 8: SAVE OUTPUT
# ============================================================================

def clean_mixed_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean columns with mixed types (object dtype) that should be numeric.
    Converts string placeholders to NaN and ensures proper data types for Parquet compatibility.
    """
    df = df.copy()
    
    print("\nCleaning mixed-type columns for Parquet compatibility...")
    
    # Identify object columns that might contain mixed types
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Exclude columns that should remain as strings (like _source_file, symbol)
    exclude_cols = ['_source_file', 'symbol', 'option_type']
    object_cols = [col for col in object_cols if col not in exclude_cols]
    
    for col in object_cols:
        try:
            # Check if column contains numeric values (even if mixed with strings)
            # Replace common placeholders with NaN
            cleaned = df[col].replace(['-', '', ' ', 'nan', 'NaN', 'None', 'null', 'NULL'], np.nan)
            
            # Try to convert to numeric
            numeric_series = pd.to_numeric(cleaned, errors='coerce')
            
            # If we successfully converted a reasonable portion, use numeric
            non_null_before = cleaned.notna().sum()
            non_null_after = numeric_series.notna().sum()
            
            if non_null_before > 0 and non_null_after / non_null_before > 0.5:
                # More than 50% converted successfully, use numeric
                df[col] = numeric_series
                print(f"  ✓ Converted {col} to numeric ({non_null_after:,} values)")
            else:
                # Keep as string but ensure it's proper string type
                df[col] = df[col].astype(str).replace('nan', np.nan)
                print(f"  ✓ Kept {col} as string type")
        except Exception as e:
            # If conversion fails, keep as string
            df[col] = df[col].astype(str)
            print(f"  ⚠ Kept {col} as string (conversion failed: {e})")
    
    print("✓ Mixed-type column cleaning complete")
    return df


def save_dataset(df: pd.DataFrame, parquet_path: Path, csv_path: Path):
    """
    Save dataset to Parquet (primary) and CSV (optional).
    """
    print("\n" + "="*80)
    print("STEP 7: SAVING OUTPUT")
    print("="*80)
    
    # Create output directory
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean mixed-type columns before saving to Parquet
    df_clean = clean_mixed_type_columns(df)
    
    # Save Parquet
    print(f"\nSaving to Parquet: {parquet_path}")
    df_clean.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy')
    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")
    
    # Save CSV (optional, may be large)
    print(f"\nSaving to CSV: {csv_path}")
    print("  (This may take a while for large datasets...)")
    df_clean.to_csv(csv_path, index=False)
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
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
    print("NIFTY OPTIONS DATA - MASTER DATASET BUILDER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Read all CSVs
        df_raw = read_all_csvs(INPUT_DIR, CALLS_DIR, PUTS_DIR)
        
        # Step 2: Normalize columns
        df = normalize_columns(df_raw)
        
        # Step 3: Convert data types
        df = convert_types(df)
        
        # Step 4: Add derived columns and flags
        df = add_flags(df)
        
        # Step 5: Sort dataset
        df = sort_dataset(df)
        
        # Step 6: Validate
        validation_results = validate(df)
        
        # Step 7: Print summary
        print_summary(df, validation_results)
        
        # Step 8: Save output
        save_dataset(df, OUTPUT_PARQUET, OUTPUT_CSV)
        
        print("\n" + "="*80)
        print("✓ PROCESS COMPLETE")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutput files:")
        print(f"  - {OUTPUT_PARQUET}")
        print(f"  - {OUTPUT_CSV}")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

