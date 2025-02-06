#!/usr/bin/env python
import sys
import pandas as pd

def diff_sheets(sheet_name, file1, file2):
    print(f"Diffing sheet '{sheet_name}' between {file1} and {file2}")
    
    print(f"Loading {file1}...")
    df1 = pd.read_excel(file1, sheet_name=sheet_name)
    print(f"Loaded {file1}")
    
    print(f"Loading {file2}...")
    df2 = pd.read_excel(file2, sheet_name=sheet_name)
    print(f"Loaded {file2}")
    
    # Print sheet statistics
    print(f"\nSheet Statistics:")
    print(f"Sheet '{sheet_name}':")
    print(f"Workbook 1 ({file1}):")
    print(f"  Rows: {len(df1)}")
    print(f"  Columns: {len(df1.columns)}")
    print(f"  Total cells: {len(df1) * len(df1.columns)}")
    print(f"\nWorkbook 2 ({file2}):")
    print(f"  Rows: {len(df2)}")
    print(f"  Columns: {len(df2.columns)}")
    print(f"  Total cells: {len(df2) * len(df2.columns)}")
    print()

    # Ensure both dataframes have the same columns
    all_columns = list(set(df1.columns) | set(df2.columns))
    df1 = df1.reindex(columns=all_columns)
    df2 = df2.reindex(columns=all_columns)

    differences = []
    total_cells = len(df1) * len(df1.columns)
    cells_checked = 0

    for col in df1.columns:
        for row in range(len(df1)):
            cells_checked += 1
            if cells_checked % 1000 == 0:
                print(f"\rProgress: {cells_checked}/{total_cells} cells checked ({(cells_checked/total_cells)*100:.1f}%)", end="", flush=True)
            
            val1 = df1.iloc[row][col]
            val2 = df2.iloc[row][col]
            
            # Handle NaN comparisons
            if pd.isna(val1) and pd.isna(val2):
                continue
            elif pd.isna(val1) or pd.isna(val2) or val1 != val2:
                col_letter = chr(65 + list(df1.columns).index(col))
                differences.append(
                    f"Cell {col_letter}{row + 1}: '{val1}' vs '{val2}'"
                )
    
    print("\n")  # New line after progress bar
    return differences

def main():
    if len(sys.argv) != 4:
        print("Usage: diff-tool <sheet_name> <workbook1_filename> <workbook2_filename>")
        sys.exit(1)
    
    sheet_name = sys.argv[1]
    file1 = sys.argv[2]
    file2 = sys.argv[3]
    
    try:
        differences = diff_sheets(sheet_name, file1, file2)
        for diff in differences:
            print(diff)
        
        if not differences:
            print("No differences found.")
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
