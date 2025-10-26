"""
Interactive Exploratory Data Analysis (EDA) Application
A comprehensive Streamlit app for data analysis and visualization.
Provides tools for dataset overview, statistical analysis, missing value detection,
and various visualization techniques including histograms, box plots, scatter plots,
correlation heatmaps, and distribution analysis.
"""

# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Local imports
from data_utils import load_sample_data, get_dataset_info, handle_missing_values
from visualization_utils import (
    create_histogram, 
    create_boxplot, 
    create_scatter_plot, 
    create_correlation_heatmap, 
    create_distribution_plot
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Interactive EDA Application",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure visualization styles
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# MAIN APPLICATION FUNCTION
# =============================================================================

def main():
    """
    Main application function that orchestrates the entire EDA workflow.
    Handles data loading, UI setup, and navigation between analysis sections.
    """
    st.title("📊 Interactive Exploratory Data Analysis")
    st.markdown(
        "Upload your dataset or use sample data to perform comprehensive EDA. "
        "Explore data patterns, detect anomalies, and generate insightful visualizations."
    )
    
    # =============================================================================
    # SIDEBAR DATA SELECTION
    # =============================================================================
    
    st.sidebar.header("Data Selection")
    
    # Data source selection (upload vs sample)
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV", "Use Sample Dataset"]
    )
    
    df = None
    
    # Handle CSV file upload
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file to analyze. File should contain structured data with headers."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"File uploaded successfully!")
                st.sidebar.write(f"Shape: {df.shape}")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                st.error("Please upload a valid CSV file with proper formatting.")
                return
    
    # Handle sample dataset selection
    else:  # Use Sample Dataset
        sample_dataset = st.sidebar.selectbox(
            "Select sample dataset:",
            ["Sales Data", "Iris Dataset", "Housing Data"],
            help="Choose from pre-loaded sample datasets for demonstration"
        )
        
        try:
            df = load_sample_data(sample_dataset)
            st.sidebar.success(f"Sample dataset loaded!")
            st.sidebar.write(f"Shape: {df.shape}")
        except Exception as e:
            st.sidebar.error(f"Error loading sample dataset: {str(e)}")
            return
    
    # =============================================================================
    # MAIN ANALYSIS INTERFACE
    # =============================================================================
    
    if df is not None:
        # Create tabbed interface for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Dataset Overview", 
            "📊 Descriptive Statistics", 
            "🔍 Missing Values Analysis",
            "📈 Visualizations", 
            "🎯 Targeted Analysis"
        ])
        
        # Dataset overview tab
        with tab1:
            show_dataset_overview(df)
        
        # Descriptive statistics tab
        with tab2:
            show_descriptive_statistics(df)
        
        # Missing values analysis tab
        with tab3:
            show_missing_values_analysis(df)
        
        # Visualizations tab
        with tab4:
            show_visualizations(df)
        
        # Targeted analysis tab
        with tab5:
            show_targeted_analysis(df)
    
    else:
        st.info("👆 Please upload a CSV file or select a sample dataset from the sidebar to begin analysis.")

# =============================================================================
# DATASET OVERVIEW FUNCTIONS
# =============================================================================

def show_dataset_overview(df):
    """
    Display comprehensive overview of the dataset including basic metrics,
    column information, and sample data.
    
    Args:
        df (pd.DataFrame): Input dataframe to analyze
    """
    st.header("Dataset Overview")
    
    # Display key metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Rows", df.shape[0])
    
    with col2:
        st.metric("Number of Columns", df.shape[1])
    
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Display column information
    st.subheader("Column Information")
    col_info = get_dataset_info(df)
    st.dataframe(col_info, use_container_width=True)
    
    # Display sample data
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Last 10 Rows")
    st.dataframe(df.tail(10), use_container_width=True)

# =============================================================================
# DESCRIPTIVE STATISTICS FUNCTIONS
# =============================================================================

def show_descriptive_statistics(df):
    """
    Display comprehensive descriptive statistics for both numerical
    and categorical variables in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe to analyze
    """
    st.header("Descriptive Statistics")
    
    # Analyze numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        st.subheader("Numerical Variables")
        st.dataframe(df[numerical_cols].describe(), use_container_width=True)
    
    # Analyze categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.subheader("Categorical Variables")
        for col in categorical_cols:
            with st.expander(f"Statistics for {col}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Value Counts:**")
                    value_counts = df[col].value_counts()
                    st.dataframe(value_counts.head(10))
                
                with col2:
                    st.write("**Summary:**")
                    st.write(f"Unique values: {df[col].nunique()}")
                    st.write(f"Most frequent: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
                    st.write(f"Missing values: {df[col].isnull().sum()}")

# =============================================================================
# MISSING VALUES ANALYSIS FUNCTIONS
# =============================================================================

def show_missing_values_analysis(df):
    """
    Analyze and visualize missing values in the dataset.
    Provides comprehensive statistics and visual representation of missing data.
    
    Args:
        df (pd.DataFrame): Input dataframe to analyze
    """
    st.header("Missing Values Analysis")
    
    # Get missing values information
    missing_data = handle_missing_values(df)
    
    # Display results based on missing data presence
    if missing_data['total_missing'] == 0:
        st.success("🎉 No missing values found in the dataset!")
    else:
        # Display missing values metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Missing Values", missing_data['total_missing'])
            st.metric("Percentage Missing", f"{missing_data['percentage_missing']:.2f}%")
        
        with col2:
            st.subheader("Missing Values by Column")
            missing_by_col = missing_data['missing_by_column']
            missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
            
            if not missing_by_col.empty:
                # Create detailed missing values table
                st.dataframe(pd.DataFrame({
                    'Column': missing_by_col.index,
                    'Missing Count': missing_by_col.values,
                    'Percentage': (missing_by_col.values / len(df) * 100).round(2)
                }))
                
                # Visualize missing values distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_by_col.plot(kind='bar', ax=ax)
                ax.set_title('Missing Values by Column')
                ax.set_ylabel('Number of Missing Values')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def show_visualizations(df):
    """
    Provide interactive visualization options including histograms, box plots,
    scatter plots, correlation heatmaps, and distribution plots.
    
    Args:
        df (pd.DataFrame): Input dataframe to visualize
    """
    st.header("Data Visualizations")
    
    # Identify column types for visualization options
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numerical_cols) == 0 and len(categorical_cols) == 0:
        st.warning("No suitable columns found for visualization.")
        return
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Histograms", "Box Plots", "Scatter Plots", "Correlation Heatmap", "Distribution Plots"]
    )
    
    # Handle histogram visualization
    if viz_type == "Histograms":
        if numerical_cols:
            selected_cols = st.multiselect(
                "Select numerical columns for histograms:",
                numerical_cols,
                default=numerical_cols[:min(4, len(numerical_cols))]
            )
            
            if selected_cols:
                bins = st.slider("Number of bins:", 10, 100, 30)
                fig = create_histogram(df, selected_cols, bins)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.warning("No numerical columns available for histograms.")
    
    # Handle box plot visualization
    elif viz_type == "Box Plots":
        if numerical_cols:
            selected_cols = st.multiselect(
                "Select numerical columns for box plots:",
                numerical_cols,
                default=numerical_cols[:min(4, len(numerical_cols))]
            )
            
            if selected_cols:
                fig = create_boxplot(df, selected_cols)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.warning("No numerical columns available for box plots.")
    
    # Handle scatter plot visualization
    elif viz_type == "Scatter Plots":
        if len(numerical_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("Select X-axis:", numerical_cols)
            
            with col2:
                y_col = st.selectbox("Select Y-axis:", [col for col in numerical_cols if col != x_col])
            
            # Optional color coding by categorical variable
            color_col = None
            if categorical_cols:
                color_col = st.selectbox("Color by (optional):", [None] + categorical_cols)
            
            if x_col and y_col:
                fig = create_scatter_plot(df, x_col, y_col, color_col)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.warning("Need at least 2 numerical columns for scatter plots.")
    
    # Handle correlation heatmap visualization
    elif viz_type == "Correlation Heatmap":
        if len(numerical_cols) >= 2:
            selected_cols = st.multiselect(
                "Select numerical columns for correlation:",
                numerical_cols,
                default=numerical_cols
            )
            
            if len(selected_cols) >= 2:
                method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
                fig = create_correlation_heatmap(df, selected_cols, method)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("Select at least 2 columns for correlation analysis.")
        else:
            st.warning("Need at least 2 numerical columns for correlation analysis.")
    
    # Handle distribution plot visualization
    elif viz_type == "Distribution Plots":
        all_cols = numerical_cols + categorical_cols
        if all_cols:
            selected_col = st.selectbox("Select column for distribution analysis:", all_cols)
            
            if selected_col:
                fig = create_distribution_plot(df, selected_col)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

# =============================================================================
# TARGETED ANALYSIS FUNCTIONS
# =============================================================================

def show_targeted_analysis(df):
    """
    Provide targeted analysis options including column summaries, outlier detection,
    unique values analysis, and data type analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe to analyze
    """
    st.header("Targeted Analysis")
    
    # Identify column types for analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = numerical_cols + categorical_cols
    
    if not all_cols:
        st.warning("No columns available for analysis.")
        return
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select analysis type:",
        ["Column Summary", "Outlier Detection", "Unique Values Analysis", "Data Type Analysis"]
    )
    
    # Handle column summary analysis
    if analysis_type == "Column Summary":
        selected_col = st.selectbox("Select column:", all_cols)
        
        if selected_col:
            st.subheader(f"Analysis for '{selected_col}'")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"Data type: {df[selected_col].dtype}")
                st.write(f"Non-null count: {df[selected_col].count()}")
                st.write(f"Null count: {df[selected_col].isnull().sum()}")
                st.write(f"Unique values: {df[selected_col].nunique()}")
            
            with col2:
                if df[selected_col].dtype in ['int64', 'float64']:
                    st.write("**Statistical Summary:**")
                    st.write(f"Mean: {df[selected_col].mean():.2f}")
                    st.write(f"Median: {df[selected_col].median():.2f}")
                    st.write(f"Std: {df[selected_col].std():.2f}")
                    st.write(f"Min: {df[selected_col].min()}")
                    st.write(f"Max: {df[selected_col].max()}")
                else:
                    st.write("**Top Values:**")
                    top_values = df[selected_col].value_counts().head(5)
                    for value, count in top_values.items():
                        st.write(f"{value}: {count}")
    
    # Handle outlier detection analysis
    elif analysis_type == "Outlier Detection":
        if numerical_cols:
            selected_col = st.selectbox("Select numerical column:", numerical_cols)
            
            if selected_col:
                method = st.selectbox("Outlier detection method:", ["IQR", "Z-Score"])
                
                if method == "IQR":
                    # Calculate IQR-based outliers
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                    
                    st.write(f"**IQR Method Results for '{selected_col}':**")
                    st.write(f"Lower bound: {lower_bound:.2f}")
                    st.write(f"Upper bound: {upper_bound:.2f}")
                    st.write(f"Number of outliers: {len(outliers)}")
                    st.write(f"Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")
                
                else:  # Z-Score method
                    # Calculate Z-score based outliers
                    z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
                    threshold = st.slider("Z-Score threshold:", 2.0, 4.0, 3.0, 0.1)
                    outliers = df[z_scores > threshold]
                    
                    st.write(f"**Z-Score Method Results for '{selected_col}':**")
                    st.write(f"Threshold: {threshold}")
                    st.write(f"Number of outliers: {len(outliers)}")
                    st.write(f"Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")
                
                # Option to display outlier values
                if len(outliers) > 0:
                    if st.checkbox("Show outlier values"):
                        st.dataframe(outliers[[selected_col]], use_container_width=True)
        else:
            st.warning("No numerical columns available for outlier detection.")
    
    # Handle unique values analysis
    elif analysis_type == "Unique Values Analysis":
        selected_col = st.selectbox("Select column:", all_cols)
        
        if selected_col:
            unique_values = df[selected_col].nunique()
            total_values = len(df)
            
            st.write(f"**Unique Values Analysis for '{selected_col}':**")
            st.write(f"Total unique values: {unique_values}")
            st.write(f"Total values: {total_values}")
            st.write(f"Uniqueness ratio: {unique_values/total_values:.2f}")
            
            # Display appropriate level of detail based on unique values count
            if unique_values <= 20:
                st.write("**All unique values:**")
                value_counts = df[selected_col].value_counts()
                st.dataframe(pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / total_values * 100).round(2)
                }))
            else:
                st.write("**Top 20 most frequent values:**")
                value_counts = df[selected_col].value_counts().head(20)
                st.dataframe(pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / total_values * 100).round(2)
                }))
    
    # Handle data type analysis
    elif analysis_type == "Data Type Analysis":
        st.subheader("Data Type Distribution")
        
        type_counts = df.dtypes.value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Types:**")
            for dtype, count in type_counts.items():
                st.write(f"{dtype}: {count} columns")
        
        with col2:
            # Visualize data type distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            type_counts.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Data Types')
            ax.set_ylabel('Number of Columns')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Display detailed column information
        st.subheader("Detailed Column Information")
        detailed_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(detailed_info, use_container_width=True)

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
