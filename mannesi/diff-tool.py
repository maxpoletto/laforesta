#!/usr/bin/env python3
# Compares two Excel files, using specified key columns to match rows
# Usage: diff-tool.py file1.xlsx file2.xlsx sheet_name key_col1 [key_col2 ...]
import sys
import pandas as pd

def format_row_values(values):
    return '\t'.join(f"{float(val):.1f}" if str(val).find('.') != -1 and str(val).replace('.','').isdigit() else str(val) for val in values)

def diff_sheets(sheet_name, file1, file2, key_cols):
    print(f"Diffing sheet '{sheet_name}' between {file1} and {file2}", file=sys.stderr)
    print(f"Using key columns: {', '.join(key_cols)}", file=sys.stderr)
    
    df1 = pd.read_excel(file1, sheet_name=sheet_name)
    print(f"Loaded {file1}", file=sys.stderr)
    
    df2 = pd.read_excel(file2, sheet_name=sheet_name)
    print(f"Loaded {file2}", file=sys.stderr)

    # Verify columns match
    if set(df1.columns) != set(df2.columns):
        only_in_1 = set(df1.columns) - set(df2.columns)
        only_in_2 = set(df2.columns) - set(df1.columns)
        error_msg = "Files have different columns:\n"
        if only_in_1:
            error_msg += f"Only in {file1}: {', '.join(only_in_1)}\n"
        if only_in_2:
            error_msg += f"Only in {file2}: {', '.join(only_in_2)}"
        raise ValueError(error_msg)

    # Verify key columns exist
    missing_keys = [col for col in key_cols if col not in df1.columns]
    if missing_keys:
        raise ValueError(f"Key columns not found: {', '.join(missing_keys)}")

    # Create composite key from specified columns
    df1['_key'] = df1[key_cols].astype(str).agg('|'.join, axis=1)
    df2['_key'] = df2[key_cols].astype(str).agg('|'.join, axis=1)

    # Find rows unique to each file
    keys1 = set(df1['_key'])
    keys2 = set(df2['_key'])
    
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common_keys = keys1 & keys2

    differences = []
    
    if only_in_1:
        differences.append(f"\nRows only in {file1}:")
        for key in sorted(only_in_1):
            rows = df1[df1['_key'] == key]
            for _, row in rows.drop('_key', axis=1).iterrows():
                differences.append(format_row_values(row.values))
            differences.append("")

    if only_in_2:
        differences.append(f"\nRows only in {file2}:")
        for key in sorted(only_in_2):
            rows = df2[df2['_key'] == key]
            for _, row in rows.drop('_key', axis=1).iterrows():
                differences.append(format_row_values(row.values))
            differences.append("")

    differences.append(f"\nDifferences in rows common to {file1} and {file2}:")
    print(f"Checking {len(common_keys)} common rows...", file=sys.stderr)
    for i, key in enumerate(sorted(common_keys), 1):
        if i % 1000 == 0:
            print(f"\rProcessed {i}/{len(common_keys)} rows...", file=sys.stderr, end='')

        rows1 = df1[df1['_key'] == key]
        rows2 = df2[df2['_key'] == key]
        
        # Convert rows to sets of tuples, replacing NaN with a consistent value
        rows1_set = set(
            tuple(str(val) if pd.notna(val) else 'NaN' 
                  for val in row) 
            for row in rows1.drop('_key', axis=1).values
        )
        rows2_set = set(
            tuple(str(val) if pd.notna(val) else 'NaN' 
                  for val in row) 
            for row in rows2.drop('_key', axis=1).values
        )
        
        # Find rows that don't match
        unmatched1 = rows1_set - rows2_set
        unmatched2 = rows2_set - rows1_set
        
        if unmatched1 or unmatched2:
            if unmatched1:
                for row_tuple in unmatched1:
                    differences.append(format_row_values(row_tuple))
            if unmatched2:
                for row_tuple in unmatched2:
                    differences.append(format_row_values(row_tuple))
            differences.append("")
    print("done.", file=sys.stderr)
    return differences

def main():
    if len(sys.argv) < 4:
        print("Usage: diff-tool <workbook1_filename> <workbook2_filename> <sheet_name> <key_col1> [key_col2 ...]", file=sys.stderr)
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    sheet_name = sys.argv[3]
    key_cols = sys.argv[4:]
    try:
        differences = diff_sheets(sheet_name, file1, file2, key_cols)
        for diff in differences:
            print(diff)
        
        if not differences:
            print("No differences found.")
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
