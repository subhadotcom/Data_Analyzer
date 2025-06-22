import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import ceil

def create_histogram(df, columns, bins=30):
    """Create histograms for selected numerical columns."""
    if not columns:
        return None
    
    n_cols = len(columns)
    n_rows = ceil(n_cols / 2) if n_cols > 1 else 1
    n_subplot_cols = 2 if n_cols > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(15, 5 * n_rows))
    
    # Ensure axes is always a list for consistent handling
    if n_cols == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        # Create histogram
        df[col].hist(bins=bins, ax=ax, alpha=0.7, edgecolor='black')
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.legend()
    
    # Hide empty subplots
    if len(axes) > n_cols:
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_boxplot(df, columns):
    """Create box plots for selected numerical columns."""
    if not columns:
        return None
    
    n_cols = len(columns)
    n_rows = ceil(n_cols / 2) if n_cols > 1 else 1
    n_subplot_cols = 2 if n_cols > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(15, 5 * n_rows))
    
    # Ensure axes is always a list for consistent handling
    if n_cols == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        # Create box plot using seaborn for better handling
        sns.boxplot(data=df, y=col, ax=ax)
        ax.set_title(f'Box Plot of {col}')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    if len(axes) > n_cols:
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None):
    """Create scatter plot with optional color coding."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_col and color_col in df.columns:
        # Color by categorical variable
        if df[color_col].dtype == 'object' or df[color_col].dtype.name == 'category':
            unique_values = df[color_col].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_values)))
            
            for i, value in enumerate(unique_values):
                mask = df[color_col] == value
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], 
                          c=[colors[i]], label=str(value), alpha=0.7)
            ax.legend()
        else:
            # Color by numerical variable
            scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col], 
                               cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=color_col)
    else:
        # Simple scatter plot
        ax.scatter(df[x_col], df[y_col], alpha=0.7)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'Scatter Plot: {y_col} vs {x_col}')
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df[x_col].corr(df[y_col])
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df, columns, method='pearson'):
    """Create correlation heatmap for selected columns."""
    # Calculate correlation matrix
    corr_matrix = df[columns].corr(method=method)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title(f'Correlation Heatmap ({method.title()} method)')
    plt.tight_layout()
    return fig

def create_distribution_plot(df, column):
    """Create distribution plot for a single column."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    if df[column].dtype in ['int64', 'float64']:
        # Numerical column
        
        # Histogram
        axes[0, 0].hist(df[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title(f'Histogram of {column}')
        axes[0, 0].set_xlabel(column)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        df.boxplot(column=column, ax=axes[0, 1])
        axes[0, 1].set_title(f'Box Plot of {column}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Density plot
        df[column].dropna().plot(kind='density', ax=axes[1, 0])
        axes[1, 0].set_title(f'Density Plot of {column}')
        axes[1, 0].set_xlabel(column)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(df[column].dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q Plot of {column}')
        axes[1, 1].grid(True, alpha=0.3)
        
    else:
        # Categorical column
        value_counts = df[column].value_counts()
        
        # Bar plot
        value_counts.head(20).plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title(f'Value Counts of {column} (Top 20)')
        axes[0, 0].set_xlabel(column)
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pie chart (for top categories)
        top_categories = value_counts.head(10)
        if len(value_counts) > 10:
            others_count = value_counts.iloc[10:].sum()
            top_categories['Others'] = others_count
        
        axes[0, 1].pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%')
        axes[0, 1].set_title(f'Distribution of {column} (Top 10)')
        
        # Horizontal bar plot
        value_counts.head(15).plot(kind='barh', ax=axes[1, 0])
        axes[1, 0].set_title(f'Value Counts of {column} (Top 15)')
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics text
        axes[1, 1].axis('off')
        stats_text = f"""
        Summary Statistics for {column}:
        
        Total Count: {df[column].count()}
        Unique Values: {df[column].nunique()}
        Missing Values: {df[column].isnull().sum()}
        Most Frequent: {df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A'}
        Frequency of Most Common: {value_counts.iloc[0] if not value_counts.empty else 0}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    return fig

def create_pairplot(df, columns, hue=None):
    """Create pair plot for selected columns."""
    if len(columns) > 6:
        columns = columns[:6]  # Limit to 6 columns for readability
    
    try:
        if hue and hue in df.columns:
            g = sns.pairplot(df[columns + [hue]], hue=hue, diag_kind='hist')
        else:
            g = sns.pairplot(df[columns], diag_kind='hist')
        
        g.fig.suptitle('Pair Plot of Selected Variables', y=1.02)
        return g.fig
    except Exception as e:
        # Fallback to simple scatter plot matrix
        fig, axes = plt.subplots(len(columns), len(columns), figsize=(15, 15))
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    axes[i, j].hist(df[col1].dropna(), alpha=0.7)
                    axes[i, j].set_title(f'{col1}')
                else:
                    axes[i, j].scatter(df[col2], df[col1], alpha=0.5)
                    if i == len(columns) - 1:
                        axes[i, j].set_xlabel(col2)
                    if j == 0:
                        axes[i, j].set_ylabel(col1)
        plt.tight_layout()
        return fig

def create_categorical_analysis(df, column):
    """Create comprehensive analysis for categorical columns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    value_counts = df[column].value_counts()
    
    # Bar plot
    value_counts.head(20).plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title(f'Top 20 Values in {column}')
    axes[0, 0].set_xlabel(column)
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pie chart
    top_categories = value_counts.head(8)
    if len(value_counts) > 8:
        others_count = value_counts.iloc[8:].sum()
        top_categories['Others'] = others_count
    
    axes[0, 1].pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%')
    axes[0, 1].set_title(f'Distribution of {column} (Top 8)')
    
    # Frequency distribution
    axes[1, 0].bar(range(len(value_counts)), value_counts.values)
    axes[1, 0].set_title(f'Frequency Distribution of {column}')
    axes[1, 0].set_xlabel('Category Rank')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Summary for {column}:
    
    Total entries: {len(df)}
    Non-null entries: {df[column].count()}
    Unique values: {df[column].nunique()}
    Missing values: {df[column].isnull().sum()}
    
    Most frequent value: {value_counts.index[0]}
    Frequency: {value_counts.iloc[0]}
    Percentage: {(value_counts.iloc[0] / len(df)) * 100:.2f}%
    
    Least frequent value: {value_counts.index[-1]}
    Frequency: {value_counts.iloc[-1]}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center')
    
    plt.tight_layout()
    return fig
