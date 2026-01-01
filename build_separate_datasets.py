"""
NIFTY Options Data - Separate Calls and Puts Dataset Builder

This script merges all raw CSV files separately for CALLS and PUTS,
with simple sorting by date → strike price.

Usage:
    python build_separate_datasets.py

Input:
    Raw CSV files in data/Calls/ and data/Puts/

Output:
    - calls_master_dataset.parquet
    - calls_master_dataset.csv
    - puts_master_dataset.parquet
    - puts_master_dataset.csv
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

INPUT_DIR = Path("data")
CALLS_DIR = INPUT_DIR / "Calls"
PUTS_DIR = INPUT_DIR / "Puts"
OUTPUT_DIR = Path("output")

CALLS_PARQUET = OUTPUT_DIR / "calls_master_dataset.parquet"
CALLS_CSV = OUTPUT_DIR / "calls_master_dataset.csv"
PUTS_PARQUET = OUTPUT_DIR / "puts_master_dataset.parquet"
PUTS_CSV = OUTPUT_DIR / "puts_master_dataset.csv"


# ============================================================================
# DATA INGESTION
# ============================================================================

def read_csvs_from_directory(directory: Path, option_type: str) -> pd.DataFrame:
    """
    Read all CSV files from a directory and merge into one DataFrame.
    
    Args:
        directory: Path to directory containing CSV files
        option_type: 'calls' or 'puts' for source tracking
    
    Returns:
        Raw DataFrame with all rows from all CSV files
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING {option_type.upper()} FILES")
    print(f"{'='*80}")
    
    if not directory.exists():
        print(f"  ⚠ Directory not found: {directory}")
        return pd.DataFrame()
    
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        print(f"  ⚠ No CSV files found in {directory}")
        return pd.DataFrame()
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    all_dfs = []
    total_files = 0
    
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
            
            df['_source_file'] = f"{option_type.capitalize()}/{csv_file.name}"
            all_dfs.append(df)
            print(f"✓ ({len(df):,} rows)")
            total_files += 1
        except Exception as e:
            print(f"✗ ERROR reading {csv_file.name}: {e}")
    
    if not all_dfs:
        print(f"  ⚠ No valid CSV files processed")
        return pd.DataFrame()
    
    print(f"\nMerging {len(all_dfs)} DataFrames...")
    df_raw = pd.concat(all_dfs, ignore_index=True, sort=False)
    print(f"  ✓ Total raw rows: {len(df_raw):,}")
    print(f"  ✓ Total columns: {len(df_raw.columns)}")
    
    return df_raw


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize key column names (date, strike) for sorting.
    Keeps all other columns as-is.
    """
    df = df.copy()
    
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
    
    # Normalize column names for matching
    def normalize_col_name(name):
        """Normalize column name for matching"""
        return name.lower().strip().replace(' ', '').replace('_', '').replace('.', '').replace('*', '').replace('₹', '')
    
    # Map only the columns we need for sorting
    column_mapping = {}
    
    col_patterns = {
        'date': ['date'],
        'strike': ['strikeprice', 'strike'],
    }
    
    # Find matching columns
    for target_col, patterns in col_patterns.items():
        for col in df.columns:
            # Skip if this column is already mapped
            if col in column_mapping:
                continue
            # Skip if target column already exists (already mapped from another column)
            if target_col in df.columns and df.columns.get_loc(target_col) != df.columns.get_loc(col):
                continue
            col_normalized = normalize_col_name(col)
            for pattern in patterns:
                if pattern == col_normalized:
                    column_mapping[col] = target_col
                    print(f"    Mapped '{col}' -> '{target_col}'")
                    break
            if col in column_mapping:
                break
    
    # Apply mapping
    df = df.rename(columns=column_mapping)
    
    print(f"\n  Normalized columns for sorting: {list(column_mapping.values())}")
    print(f"  Total columns preserved: {len(df.columns)}")
    
    # Verify required columns exist
    if 'date' not in df.columns:
        print(f"  ⚠ WARNING: 'date' column not found after normalization!")
        print(f"    Available columns: {list(df.columns)[:10]}...")
    if 'strike' not in df.columns:
        print(f"  ⚠ WARNING: 'strike' column not found after normalization!")
        print(f"    Available columns: {list(df.columns)[:10]}...")
    
    return df


# ============================================================================
# DATA TYPE CONVERSION
# ============================================================================

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date and strike columns to proper types for sorting.
    """
    df = df.copy()
    
    # Parse date
    if 'date' in df.columns:
        try:
            print(f"\n  Converting date column to datetime...")
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
            null_count = df['date'].isna().sum()
            if null_count > 0:
                print(f"    ⚠ Warning: {null_count:,} rows with invalid date format")
            else:
                print(f"    ✓ Date conversion successful")
        except Exception as e:
            print(f"    ⚠ Warning: Could not parse date: {e}")
    else:
        print(f"    ⚠ WARNING: 'date' column not found for conversion!")
    
    # Convert strike to numeric
    if 'strike' in df.columns:
        try:
            print(f"\n  Converting strike column to numeric...")
            # Check current type
            print(f"    Current strike dtype: {df['strike'].dtype}")
            
            # Replace placeholders with NaN
            df['strike'] = df['strike'].replace(['-', '', ' ', 'nan', 'NaN', 'None', 'null', 'NULL'], np.nan)
            
            # Convert to numeric
            df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
            
            # Verify conversion
            print(f"    After conversion dtype: {df['strike'].dtype}")
            non_null = df['strike'].notna().sum()
            print(f"    ✓ Strike conversion successful: {non_null:,} non-null values")
            
            # Show sample values to verify
            sample_strikes = df['strike'].dropna().head(5).tolist()
            print(f"    Sample strike values: {sample_strikes}")
        except Exception as e:
            print(f"    ⚠ Warning: Could not convert strike to numeric: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"    ⚠ WARNING: 'strike' column not found for conversion!")
        print(f"    Available columns: {[c for c in df.columns if 'strike' in c.lower() or 'price' in c.lower()]}")
    
    return df


# ============================================================================
# SORTING
# ============================================================================

def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataset by: date → strike price
    """
    df = df.copy()
    
    sort_cols = []
    if 'date' in df.columns:
        sort_cols.append('date')
        print(f"\n  Found 'date' column for sorting")
    else:
        print(f"\n  ⚠ WARNING: 'date' column not found!")
    
    # Check for strike column (try normalized name first, then original names)
    strike_col = None
    if 'strike' in df.columns:
        strike_col = 'strike'
    elif 'Strike Price' in df.columns:
        strike_col = 'Strike Price'
        print(f"  Found 'Strike Price' column, will use for sorting")
    elif 'Strike' in df.columns:
        strike_col = 'Strike'
        print(f"  Found 'Strike' column, will use for sorting")
    else:
        # Try to find any column with 'strike' in the name (case insensitive)
        for col in df.columns:
            if 'strike' in col.lower():
                strike_col = col
                print(f"  Found '{col}' column, will use for sorting")
                break
    
    if strike_col:
        sort_cols.append(strike_col)
        print(f"  Using '{strike_col}' column for sorting (dtype: {df[strike_col].dtype})")
        
        # Verify strike is numeric
        if not pd.api.types.is_numeric_dtype(df[strike_col]):
            print(f"    ⚠ WARNING: Strike column is not numeric! Converting...")
            df[strike_col] = df[strike_col].replace(['-', '', ' ', 'nan', 'NaN', 'None'], np.nan)
            df[strike_col] = pd.to_numeric(df[strike_col], errors='coerce')
            print(f"    ✓ Converted to numeric (dtype: {df[strike_col].dtype})")
    else:
        print(f"  ⚠ WARNING: 'strike' column not found!")
        print(f"    Available columns: {list(df.columns)[:15]}")
    
    if sort_cols:
        print(f"\n  Sorting by: {' → '.join(sort_cols)}")
        
        # Perform sort
        df = df.sort_values(by=sort_cols, na_position='last').reset_index(drop=True)
        
        # Validate sorting worked
        print(f"  ✓ Dataset sorted: {len(df):,} rows")
        
        # Sample check: verify first few rows are sorted correctly
        date_col = 'date' if 'date' in df.columns else None
        strike_col_check = strike_col if strike_col else ('strike' if 'strike' in df.columns else None)
        
        if date_col and strike_col_check:
            sample_cols = [date_col, strike_col_check]
            sample = df[sample_cols].head(10)
            print(f"\n  Sample sorted data (first 10 rows):")
            for idx, row in sample.iterrows():
                print(f"    Row {idx+1}: date={row[date_col]}, strike={row[strike_col_check]}")
            
            # Check if sorting is correct for same dates
            if len(df) > 1:
                def check_sorted(group):
                    values = group.dropna()
                    if len(values) <= 1:
                        return True
                    return values.is_monotonic_increasing
                
                same_date_groups = df.groupby(date_col)[strike_col_check].apply(check_sorted)
                unsorted_groups = same_date_groups[~same_date_groups]
                if len(unsorted_groups) > 0:
                    print(f"    ⚠ WARNING: Found {len(unsorted_groups)} date groups with unsorted strikes!")
                    print(f"    First few problematic dates: {unsorted_groups.head(3).index.tolist()}")
                else:
                    print(f"    ✓ Verified: All date groups have sorted strikes")
    else:
        print(f"  ⚠ ERROR: Could not find date or strike columns for sorting!")
        print(f"    Available columns: {list(df.columns)}")
    
    return df


# ============================================================================
# CLEAN MIXED TYPES FOR PARQUET
# ============================================================================

def clean_mixed_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean columns with mixed types for Parquet compatibility.
    """
    df = df.copy()
    
    print(f"\n  Cleaning mixed-type columns for Parquet...")
    
    # Identify object columns that might contain mixed types
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Exclude columns that should remain as strings
    exclude_cols = ['_source_file', 'symbol', 'option_type', 'Option type', 'Symbol']
    object_cols = [col for col in object_cols if col not in exclude_cols]
    
    for col in object_cols:
        try:
            # Replace common placeholders with NaN
            cleaned = df[col].replace(['-', '', ' ', 'nan', 'NaN', 'None', 'null', 'NULL'], np.nan)
            
            # Try to convert to numeric
            numeric_series = pd.to_numeric(cleaned, errors='coerce')
            
            # If we successfully converted a reasonable portion, use numeric
            non_null_before = cleaned.notna().sum()
            non_null_after = numeric_series.notna().sum()
            
            if non_null_before > 0 and non_null_after / non_null_before > 0.5:
                df[col] = numeric_series
        except Exception:
            # If conversion fails, keep as string
            df[col] = df[col].astype(str)
    
    print(f"  ✓ Column cleaning complete")
    return df


# ============================================================================
# SAVE DATASET
# ============================================================================

def save_dataset(df: pd.DataFrame, parquet_path: Path, csv_path: Path, dataset_name: str):
    """
    Save dataset to Parquet and CSV.
    """
    print(f"\n{'='*80}")
    print(f"SAVING {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    
    # Create output directory
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean mixed-type columns before saving to Parquet
    df_clean = clean_mixed_type_columns(df)
    
    # Save Parquet
    print(f"\n  Saving to Parquet: {parquet_path}")
    try:
        df_clean.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy')
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"    ✓ Saved: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"    ✗ ERROR saving Parquet: {e}")
        raise
    
    # Save CSV
    print(f"\n  Saving to CSV: {csv_path}")
    print(f"    (This may take a while for large datasets...)")
    df_clean.to_csv(csv_path, index=False)
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved: {file_size_mb:.2f} MB")
    
    print(f"\n  ✓ {dataset_name} dataset saved successfully")
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    Total rows: {len(df):,}")
    print(f"    Total columns: {len(df.columns)}")
    if 'date' in df.columns:
        date_min = df['date'].min()
        date_max = df['date'].max()
        print(f"    Date range: {date_min} to {date_max}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_dataset(directory: Path, option_type: str, parquet_path: Path, csv_path: Path):
    """
    Process a single dataset (calls or puts).
    """
    # Read all CSVs
    df_raw = read_csvs_from_directory(directory, option_type)
    
    if df_raw.empty:
        print(f"\n  ⚠ No data to process for {option_type}")
        return
    
    # Normalize columns
    df = normalize_columns(df_raw)
    
    # Convert types
    df = convert_types(df)
    
    # Sort
    df = sort_dataset(df)
    
    # Save
    save_dataset(df, parquet_path, csv_path, option_type)


def main():
    """
    Main execution pipeline.
    """
    print("\n" + "="*80)
    print("NIFTY OPTIONS DATA - SEPARATE CALLS & PUTS DATASET BUILDER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Process CALLS
        print("\n\n" + "="*80)
        print("PROCESSING CALLS")
        print("="*80)
        process_dataset(CALLS_DIR, "calls", CALLS_PARQUET, CALLS_CSV)
        
        # Process PUTS
        print("\n\n" + "="*80)
        print("PROCESSING PUTS")
        print("="*80)
        process_dataset(PUTS_DIR, "puts", PUTS_PARQUET, PUTS_CSV)
        
        print("\n\n" + "="*80)
        print("✓ PROCESS COMPLETE")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutput files:")
        print(f"  - {CALLS_PARQUET}")
        print(f"  - {CALLS_CSV}")
        print(f"  - {PUTS_PARQUET}")
        print(f"  - {PUTS_CSV}")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

