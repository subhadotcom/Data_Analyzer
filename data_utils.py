"""
Data Utilities Module for Interactive EDA Application
Provides utility functions for data loading, analysis, and preprocessing.
Includes functions for sample dataset loading, dataset information extraction,
missing value analysis, outlier detection, and statistical calculations.
"""

# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_sample_data(dataset_name):
    """
    Load sample datasets from the data directory based on user selection.
    
    Args:
        dataset_name (str): Name of the dataset to load. Options include:
            - "Sales Data": Sample sales dataset
            - "Iris Dataset": Classic iris flower dataset
            - "Housing Data": Sample housing market dataset
            
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame
        
    Raises:
        ValueError: If an unknown dataset name is provided
        FileNotFoundError: If the dataset file doesn't exist in the data directory
    """
    data_dir = "data"
    
    # Map dataset names to file paths
    if dataset_name == "Sales Data":
        file_path = os.path.join(data_dir, "sample_sales.csv")
    elif dataset_name == "Iris Dataset":
        file_path = os.path.join(data_dir, "sample_iris.csv")
    elif dataset_name == "Housing Data":
        file_path = os.path.join(data_dir, "sample_housing.csv")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Verify file exists before attempting to load
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample dataset not found: {file_path}")
    
    return pd.read_csv(file_path)

# =============================================================================
# DATASET ANALYSIS FUNCTIONS
# =============================================================================

def get_dataset_info(df):
    """
    Generate comprehensive information about each column in the dataset.
    Provides data type, null counts, unique values, and type-specific statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe to analyze
        
    Returns:
        pd.DataFrame: DataFrame containing detailed information for each column
    """
    info_data = []
    
    for col in df.columns:
        # Basic column information
        dtype = str(df[col].dtype)
        non_null_count = df[col].count()
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        # Generate type-specific additional information
        if df[col].dtype in ['int64', 'float64']:
            # Numerical column statistics
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            additional_info = f"Mean: {mean_val:.2f}, Std: {std_val:.2f}, Range: [{min_val}, {max_val}]"
        else:
            # Categorical column statistics
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            additional_info = f"Most frequent: {mode_val}"
        
        # Compile column information
        info_data.append({
            'Column': col,
            'Data Type': dtype,
            'Non-Null Count': non_null_count,
            'Null Count': null_count,
            'Unique Values': unique_count,
            'Additional Info': additional_info
        })
    
    return pd.DataFrame(info_data)


def handle_missing_values(df):
    """
    Perform comprehensive analysis of missing values in the dataset.
    Calculates overall missing value statistics and identifies columns with missing data.
    
    Args:
        df (pd.DataFrame): Input dataframe to analyze
        
    Returns:
        dict: Dictionary containing missing value analysis results with keys:
            - total_missing: Total number of missing values
            - percentage_missing: Percentage of missing values in the dataset
            - missing_by_column: Series showing missing values per column
            - columns_with_missing: List of column names that have missing values
    """
    # Calculate overall missing value statistics
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    percentage_missing = (total_missing / total_cells) * 100
    
    # Analyze missing values by column
    missing_by_column = df.isnull().sum()
    
    return {
        'total_missing': total_missing,
        'percentage_missing': percentage_missing,
        'missing_by_column': missing_by_column,
        'columns_with_missing': missing_by_column[missing_by_column > 0].index.tolist()
    }

# =============================================================================
# OUTLIER DETECTION FUNCTIONS
# =============================================================================

def detect_outliers_iqr(series):
    """
    Detect outliers in a numerical series using the Interquartile Range (IQR) method.
    Uses the 1.5 * IQR rule to identify outliers beyond the lower and upper bounds.
    
    Args:
        series (pd.Series): Numerical series to analyze for outliers
        
    Returns:
        tuple: (outliers, lower_bound, upper_bound)
            - outliers: Series containing outlier values
            - lower_bound: Lower threshold for outlier detection
            - upper_bound: Upper threshold for outlier detection
    """
    # Calculate quartiles and IQR
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(series, threshold=3):
    """
    Detect outliers in a numerical series using the Z-score method.
    Identifies values that deviate more than the specified threshold from the mean.
    
    Args:
        series (pd.Series): Numerical series to analyze for outliers
        threshold (float): Z-score threshold for outlier detection (default: 3)
        
    Returns:
        tuple: (outliers, z_scores)
            - outliers: Series containing outlier values
            - z_scores: Series containing Z-scores for all values
    """
    # Calculate Z-scores for all values
    z_scores = np.abs((series - series.mean()) / series.std())
    
    # Identify outliers based on threshold
    outliers = series[z_scores > threshold]
    return outliers, z_scores

# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def get_column_statistics(df, column):
    """
    Generate comprehensive statistical summary for a specific column.
    Provides different statistics based on the column's data type (numerical vs categorical).
    
    Args:
        df (pd.DataFrame): Input dataframe containing the column
        column (str): Name of the column to analyze
        
    Returns:
        dict: Dictionary containing comprehensive statistics for the column
    """
    stats = {}
    
    # Generate type-specific statistics
    if df[column].dtype in ['int64', 'float64']:
        # Numerical column statistics
        stats = {
            'count': df[column].count(),
            'mean': df[column].mean(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'median': df[column].median(),
            'q1': df[column].quantile(0.25),
            'q3': df[column].quantile(0.75),
            'skewness': df[column].skew(),
            'kurtosis': df[column].kurtosis()
        }
    else:
        # Categorical column statistics
        value_counts = df[column].value_counts()
        stats = {
            'count': df[column].count(),
            'unique': df[column].nunique(),
            'top_value': value_counts.index[0] if not value_counts.empty else None,
            'top_frequency': value_counts.iloc[0] if not value_counts.empty else 0,
            'mode': df[column].mode().iloc[0] if not df[column].mode().empty else None
        }
    
    # Add common statistics for all data types
    stats['null_count'] = df[column].isnull().sum()
    stats['null_percentage'] = (df[column].isnull().sum() / len(df)) * 100
    
    return stats
