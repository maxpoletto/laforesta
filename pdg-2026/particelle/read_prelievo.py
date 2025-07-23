#!/usr/bin/env python3
"""
Read the "Prelievo per particella" sheet from foresta.xlsx and export selected columns as CSV

To run: source ~/venv_3.13/bin/activate && python3 read_prelievo.py
"""

import pandas as pd

def read_and_export_prelievo():
    """
    Read the 'Prelievo per particella' sheet and export required columns as CSV
    """
    try:
        # Read the specific sheet from the Excel file
        df = pd.read_excel('foresta.xlsx', sheet_name='Prelievo per particella')
        
        print("Contents of 'Prelievo per particella' sheet:")
        print("=" * 50)
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Define the columns we need for the visualization
        required_columns = [
            'Compresa',
            'Particella', 
            'Governo',
            'Area (ha)',
            'Età media',
            'No. fustaia',
            'No. ceduo',
            'm3/ha nuovo',
            'Incr/ha nuovo'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print("Available columns:")
            for i, col in enumerate(df.columns):
                print(f"  {i+1}. {col}")
            return None
        
        # Extract the required columns
        export_df = df[required_columns].copy()
        
        # Remove rows where both Compresa and Particella are NaN (empty rows)
        export_df = export_df.dropna(subset=['Compresa', 'Particella'], how='all')
        
        print(f"\nExporting {len(export_df)} rows with {len(required_columns)} columns")
        print("\nColumn summary:")
        for col in required_columns:
            non_null = export_df[col].notna().sum()
            print(f"  {col}: {non_null}/{len(export_df)} non-null values")
        
        # Export to CSV
        csv_filename = 'prelievo_parcels.csv'
        export_df.to_csv(csv_filename, index=False)
        print(f"\nExported data to '{csv_filename}'")
        
        # Show first few rows
        print("\nFirst 10 rows of exported data:")
        print(export_df.head(10))
        
        return export_df
        
    except FileNotFoundError:
        print("Error: Could not find 'foresta.xlsx'")
        return None
    except ValueError as e:
        if "Worksheet named" in str(e):
            print(f"Error: Sheet 'Prelievo per particella' not found in the Excel file")
            print("Available sheets:")
            xl_file = pd.ExcelFile('foresta.xlsx')
            for sheet in xl_file.sheet_names:
                print(f"  - {sheet}")
        else:
            print(f"Error reading Excel file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    df = read_and_export_prelievo() 