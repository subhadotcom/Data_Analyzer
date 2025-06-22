# Interactive Streamlit EDA Application

## Overview

This is an Interactive Exploratory Data Analysis (EDA) application built with Streamlit that enables users to upload CSV datasets or utilize built-in sample datasets for comprehensive data analysis and visualization. The application provides a web-based interface for performing statistical analysis, data visualization, and exploratory data analysis without requiring users to write code.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Interface**: Single-page application with sidebar navigation
- **Layout**: Wide layout with expandable sidebar for data selection
- **Responsive Design**: Adapts to different screen sizes
- **Configuration**: Custom theme and server settings via `.streamlit/config.toml`

### Backend Architecture
- **Language**: Python 3.11
- **Core Libraries**: 
  - Pandas for data manipulation
  - NumPy for numerical operations
  - Matplotlib and Seaborn for visualization
  - SciPy for statistical functions
- **Modular Design**: Separated into utility modules for data handling and visualization
- **Error Handling**: Comprehensive error handling for file uploads and data processing

### Data Storage Solutions
- **File-based Storage**: CSV files stored in local `data/` directory
- **Sample Datasets**: Three built-in datasets (Sales, Iris, Housing)
- **User Uploads**: Temporary processing of uploaded CSV files
- **No Database**: Currently uses file-based approach, suitable for adding database layer later

## Key Components

### Core Application (`app.py`)
- Main Streamlit application entry point
- Data source selection interface (upload vs sample data)
- File upload handling with validation
- Integration with utility modules

### Data Utilities (`data_utils.py`)
- `load_sample_data()`: Loads predefined sample datasets
- `get_dataset_info()`: Comprehensive dataset information extraction
- `handle_missing_values()`: Missing data analysis and handling
- Data type detection and statistical summaries

### Visualization Utilities (`visualization_utils.py`)
- `create_histogram()`: Multi-column histogram generation
- `create_boxplot()`: Box plot creation for outlier detection
- `create_scatter_plot()`: Scatter plots with optional color coding
- `create_correlation_heatmap()`: Correlation matrix visualizations
- `create_distribution_plot()`: Detailed distribution analysis

### Sample Datasets
- **Sales Data**: Transaction data with categories, regions, and sales representatives
- **Iris Dataset**: Classic machine learning dataset with flower measurements
- **Housing Data**: Real estate data with property characteristics and prices

## Data Flow

1. **Data Input**: User selects between file upload or sample dataset
2. **Data Loading**: Application loads data using appropriate utility functions
3. **Data Validation**: Error handling ensures data integrity
4. **Data Analysis**: Automated generation of descriptive statistics and data info
5. **Visualization**: Interactive charts and plots based on data characteristics
6. **User Interaction**: Streamlit widgets allow dynamic exploration and filtering

## External Dependencies

### Python Packages
- **streamlit**: Web application framework (v1.46.0+)
- **pandas**: Data manipulation and analysis (v2.3.0+)
- **numpy**: Numerical computing (v2.3.1+)
- **matplotlib**: Basic plotting library (v3.10.3+)
- **seaborn**: Statistical data visualization (v0.13.2+)
- **scipy**: Scientific computing library (v1.15.3+)

### System Dependencies
- Python 3.11 runtime environment
- Cairo, FFmpeg, and other system libraries for visualization rendering
- GTK3 for GUI components

## Deployment Strategy

### Replit Configuration
- **Deployment Target**: Autoscale deployment for automatic scaling
- **Runtime**: Python 3.11 with Nix package management
- **Port Configuration**: Streamlit server on port 5000
- **Workflow**: Parallel execution with dedicated Streamlit server process

### Environment Setup
- Uses Nix stable channel (24_05) for reproducible environments
- Comprehensive system package installation for visualization dependencies
- UV lock file ensures consistent Python package versions

### Server Configuration
- Headless server mode for production deployment
- Configured for external access (0.0.0.0 binding)
- Light theme as default user interface

## Changelog

- June 22, 2025: Initial EDA application setup with Streamlit
- June 22, 2025: Added comprehensive localhost configuration and setup guides
- June 22, 2025: Removed complex setup script, created simple run_localhost.py
- June 22, 2025: Fixed file upload permissions and localhost configuration
- June 22, 2025: Streamlined localhost setup to 2 simple steps
- June 22, 2025: Enhanced run_localhost.py with automatic dependency checking and installation
- June 22, 2025: Created comprehensive LOCALHOST_SETUP_GUIDE.md with troubleshooting

## User Preferences

- Preferred communication style: Simple, everyday language
- Deployment preference: Local development with localhost configuration
- Setup preference: Automated setup with minimal manual steps
- Documentation preference: Clear step-by-step guides with troubleshooting sections