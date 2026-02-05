import pandas as pd

def read_excel(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(how="all")
    if df.shape[1] < 2:
        raise ValueError("Excel must have at least 2 columns (inputs + output).")
    if df.empty:
        raise ValueError("Excel file has no usable rows.")
    return df
