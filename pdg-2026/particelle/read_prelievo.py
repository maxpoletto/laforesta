#!/usr/bin/env python3
"""
Simple script to read and display the "Prelievo per particella" sheet from foresta.xlsx
"""

import pandas as pd

def read_prelievo_sheet():
    """
    Read and display the 'Prelievo per particella' sheet from the Excel file
    """
    try:
        # Read the specific sheet from the Excel file
        df = pd.read_excel('foresta.xlsx', sheet_name='Prelievo per particella')
        
        print("Contents of 'Prelievo per particella' sheet:")
        print("=" * 50)
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print("\nColumn names:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        
        print("\nDataFrame contents:")
        print(df)
        
        return df
        
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
    df = read_prelievo_sheet() 