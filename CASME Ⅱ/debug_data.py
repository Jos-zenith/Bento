import pandas as pd
import numpy as np

df = pd.read_excel('CASME2-coding-20140508.xlsx')

print("Data types:")
print(df.dtypes)

print("\n--- Checking frame columns for issues ---")
for col in ['OnsetFrame', 'ApexFrame', 'OffsetFrame']:
    print(f"\n{col}:")
    print(f"  Type: {df[col].dtype}")
    print(f"  Unique values sample: {df[col].unique()[:10]}")
    print(f"  Contains slash: {df[col].astype(str).apply(lambda x: '/' in x).any()}")
    
    # Find rows with problematic values
    problematic = df[df[col].astype(str).apply(lambda x: not x.replace('-', '').replace('.', '').isdigit())]
    if len(problematic) > 0:
        print(f"  Problematic rows: {len(problematic)}")
        print(problematic.head())

print("\n--- Full dataframe info ---")
print(df.info())
print("\n--- First 10 rows ---")
print(df.head(10))
