import pandas as pd
import numpy as np
import os

def load_nacc_data(file_path, data_dict=None):
    """
    Load NACC UDS data with basic preprocessing
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    data_dict : dict, optional
        Data dictionary information for validation
    
    Returns:
    --------
    pd.DataFrame
        Loaded and preprocessed data
    """
    
    # Determine file type and load accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, low_memory=False)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel files.")
    
    print(f"Loaded data with shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    return df

def basic_data_quality_check(df):
    """
    Perform basic data quality checks
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Data quality metrics
    """
    
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.value_counts().to_dict()
    }
    
    # Column-wise missing values
    missing_cols = df.isnull().sum()
    quality_report['columns_with_high_missing'] = missing_cols[missing_cols > 0.5 * len(df)].to_dict()
    
    return quality_report

def summarize_categorical_variables(df, max_categories=50):
    """
    Summarize categorical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    max_categories : int
        Maximum number of categories to display per variable
    
    Returns:
    --------
    dict
        Summary of categorical variables
    """
    
    categorical_summary = {}
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < max_categories:
            value_counts = df[col].value_counts()
            categorical_summary[col] = {
                'n_unique': df[col].nunique(),
                'top_categories': value_counts.head(10).to_dict(),
                'missing': df[col].isnull().sum()
            }
    
    return categorical_summary

if __name__ == "__main__":
    # Example usage
    print("NACC UDS Data Loader Module")