import pandas as pd
import numpy as np
import os

def load_sample_data(dataset_name):
    """Load sample datasets based on the selection."""
    data_dir = "data"
    
    if dataset_name == "Sales Data":
        file_path = os.path.join(data_dir, "sample_sales.csv")
    elif dataset_name == "Iris Dataset":
        file_path = os.path.join(data_dir, "sample_iris.csv")
    elif dataset_name == "Housing Data":
        file_path = os.path.join(data_dir, "sample_housing.csv")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample dataset not found: {file_path}")
    
    return pd.read_csv(file_path)

def get_dataset_info(df):
    """Get comprehensive information about the dataset."""
    info_data = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_count = df[col].count()
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        # Additional statistics based on data type
        if df[col].dtype in ['int64', 'float64']:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            additional_info = f"Mean: {mean_val:.2f}, Std: {std_val:.2f}, Range: [{min_val}, {max_val}]"
        else:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            additional_info = f"Most frequent: {mode_val}"
        
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
    """Analyze missing values in the dataset."""
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    percentage_missing = (total_missing / total_cells) * 100
    
    missing_by_column = df.isnull().sum()
    
    return {
        'total_missing': total_missing,
        'percentage_missing': percentage_missing,
        'missing_by_column': missing_by_column,
        'columns_with_missing': missing_by_column[missing_by_column > 0].index.tolist()
    }

def detect_outliers_iqr(series):
    """Detect outliers using the IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(series, threshold=3):
    """Detect outliers using the Z-score method."""
    z_scores = np.abs((series - series.mean()) / series.std())
    outliers = series[z_scores > threshold]
    return outliers, z_scores

def get_column_statistics(df, column):
    """Get detailed statistics for a specific column."""
    stats = {}
    
    if df[column].dtype in ['int64', 'float64']:
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
        value_counts = df[column].value_counts()
        stats = {
            'count': df[column].count(),
            'unique': df[column].nunique(),
            'top_value': value_counts.index[0] if not value_counts.empty else None,
            'top_frequency': value_counts.iloc[0] if not value_counts.empty else 0,
            'mode': df[column].mode().iloc[0] if not df[column].mode().empty else None
        }
    
    # Common statistics
    stats['null_count'] = df[column].isnull().sum()
    stats['null_percentage'] = (df[column].isnull().sum() / len(df)) * 100
    
    return stats
