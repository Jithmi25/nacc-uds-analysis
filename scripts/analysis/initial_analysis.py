import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils.data_loader import load_nacc_data, basic_data_quality_check, summarize_categorical_variables
import os

def create_analysis_report(df, output_dir='reports'):
    """
    Create comprehensive analysis report
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_dir : str
        Output directory for reports and figures
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Data quality report
    quality_report = basic_data_quality_check(df)
    
    # Categorical variables summary
    cat_summary = summarize_categorical_variables(df)
    
    # Generate basic statistics
    numerical_stats = df.describe()
    
    # Create summary report
    with open(os.path.join(output_dir, 'initial_analysis_report.txt'), 'w') as f:
        f.write("NACC UDS DATA ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATA QUALITY METRICS:\n")
        f.write("-" * 30 + "\n")
        for key, value in quality_report.items():
            if key != 'columns_with_high_missing':
                f.write(f"{key}: {value}\n")
        
        f.write(f"\nColumns with >50% missing values: {len(quality_report['columns_with_high_missing'])}\n")
        
        f.write(f"\nDATA SHAPE: {df.shape}\n")
        f.write(f"COLUMNS: {len(df.columns)}\n\n")
        
        f.write("CATEGORICAL VARIABLES SUMMARY (Top 10):\n")
        f.write("-" * 50 + "\n")
        for i, (col, summary) in enumerate(list(cat_summary.items())[:10]):
            f.write(f"{i+1}. {col}: {summary['n_unique']} categories, {summary['missing']} missing\n")
    
    # Create basic visualizations
    create_basic_visualizations(df, output_dir)
    
    return {
        'quality_report': quality_report,
        'categorical_summary': cat_summary,
        'numerical_stats': numerical_stats
    }

def create_basic_visualizations(df, output_dir):
    """
    Create basic visualizations for dataset overview
    """
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Missing values heatmap
    plt.figure(figsize=(12, 8))
    missing_data = df.isnull()
    sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Pattern')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'missing_values_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Data types distribution
    plt.figure(figsize=(10, 6))
    dtype_counts = df.dtypes.value_counts()
    dtype_counts.plot(kind='bar')
    plt.title('Data Types Distribution')
    plt.xlabel('Data Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'data_types_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_demographics(df):
    """
    Analyze demographic variables
    """
    
    # Common demographic variables from data dictionary
    demo_vars = ['SEX', 'EDUC', 'NACCAGE', 'RACE', 'HISPANIC', 'MARISTAT']
    
    available_demo_vars = [var for var in demo_vars if var in df.columns]
    
    if available_demo_vars:
        print("Available demographic variables:", available_demo_vars)
        
        # Create demographic summary
        demo_summary = {}
        for var in available_demo_vars:
            if df[var].dtype in ['int64', 'float64']:
                demo_summary[var] = {
                    'mean': df[var].mean(),
                    'median': df[var].median(),
                    'std': df[var].std(),
                    'min': df[var].min(),
                    'max': df[var].max(),
                    'missing': df[var].isnull().sum()
                }
            else:
                demo_summary[var] = {
                    'value_counts': df[var].value_counts().to_dict(),
                    'missing': df[var].isnull().sum()
                }
        
        return demo_summary
    else:
        print("No standard demographic variables found in dataset")
        return {}

if __name__ == "__main__":
    # Example usage
    print("NACC UDS Initial Analysis Script")
    
    # This will be replaced with actual data loading when data is available
    # df = load_nacc_data('data/raw/nacc_data.csv')
    # report = create_analysis_report(df)
    # demographics = analyze_demographics(df)
    
    print("Script ready for use with actual data.")