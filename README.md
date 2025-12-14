# Finance Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)

Advanced Financial Risk Prediction System - A professional solution that predicts credit risks using machine learning with a focus on data leakage prevention.

## Features

### Core Features
- **Data Leakage Prevention**: Avoid using future information in risk calculations
- **Historical Performance**: Risk assessment based on real payment history
- **AutoML Pipeline**: Automatic model selection and optimization
- **Web Interface**: User-friendly interface with Streamlit
- **Batch Prediction**: Bulk risk analysis for all customers
- **Detailed Reporting**: Model performance and risk analysis reports

### ML Features
- Multi-algorithm support (LightGBM, CatBoost, XGBoost, Random Forest, Linear Models)
- Hyperparameter optimization with Optuna
- Temporal cross-validation strategy
- Feature engineering and selection
- Model validation and leakage control

### Risk Analysis
- Risk scoring based on historical performance
- Risk categories: Very Low Risk → High Risk
- Detailed customer profile analysis
- Excel report output

## Installation

### Requirements
- Python 3.8+
- pip

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/bertansezgin/FinanceRiskPredictor.git
cd FinanceRiskPredictor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare data file:**
   - Place `data/birlesik_risk_verisi.csv` in the project root directory
   - Customer and credit information in CSV format

4. **Run the system:**
```bash
# Command line application
python main_advanced.py

# Web interface
streamlit run streamlit_app.py
```

## Usage

### Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

1. **Model Training**: System automatically trains the best model
2. **Risk Analysis**: Performs risk predictions for all customers
3. **Excel Download**: Download results in Excel format

### Command Line
```bash
# Quick training
python main_advanced.py

# Help
python main_advanced.py --help
```

## Project Structure

```
FinanceRiskPredictor/
├── main_advanced.py          # Main application
├── streamlit_app.py          # Web interface
├── requirements.txt          # Dependencies
├── src/                      # Source code
│   ├── config.py            # Configuration
│   ├── loader.py            # Data loading
│   ├── automl_system.py     # AutoML pipeline
│   ├── advanced_models.py   # ML models
│   ├── feature_engineering.py # Feature engineering
│   ├── model_evaluation.py  # Model evaluation
│   └── batch_predict.py     # Batch prediction
├── models/                  # Trained models
├── reports/                 # Reports and outputs
├── logs/                    # Log files
└── data/                    # Data files (not included in git)
```

## Configuration

### Risk Calculation Methods
- **Historical Performance**: Based on real payment history (recommended)
- Zero data leakage risk, uses only past performance

### Features
- Safe Features: Only information known at credit inception
- Polynomial Features: 2nd degree polynomial features
- Feature Selection: Top 50 features using Mutual Information

## Data Format

Required columns:
- `ProjectId`: Project ID
- `ProjectDate`: Credit date
- `PrincipalAmount`: Principal amount
- `InstallmentCount`: Number of installments
- `PaymentAmount`: Payment amount
- `AmountTL`: Collected amount in TL
- `MaturityDate`: Maturity date
- `PaymentDate`: Payment date

## Models

### Supported Algorithms
1. **LightGBM** - Gradient boosting (fastest and most successful)
2. **CatBoost** - Categorical variable support
3. **XGBoost** - Extreme gradient boosting
4. **Random Forest** - Ensemble learning
5. **Linear Regression** - Simple and interpretable

### Evaluation Metrics
- R² Score
- RMSE (Root Mean Square Error)
- Cross-validation results
- Feature importance analysis

## Data Leakage Prevention

### Basic Rules
- No input features used in target calculation
- Only past performance data is used
- Future information is excluded
- Train-test split based on ProjectId

### Safe Features
- Credit application date and amount
- Number of installments and period
- Customer categories
- Branch and product information

## Outputs

### Reports
- Model performance report
- Feature importance analysis
- Risk distribution analysis
- Cross-validation results

### Excel Output
- Risk scores for all customers
- Risk categories
- Summary statistics

## Troubleshooting

### Common Problems
1. **Data file not found**: Check `data/birlesik_risk_verisi.csv` file
2. **Memory error**: Use chunk processing for large datasets
3. **Model loading error**: Check files in `models/` directory

### Log Files
- `logs/finance_risk.log`: Detailed log records

## Updates

### v2.0.0 (Current)
- Historical performance-based risk calculation
- AutoML pipeline integration
- Streamlit web interface
- Data leakage prevention system
- Detailed reporting

### v1.0.0
- Basic risk prediction system
- Simple ML models
- Console-based interface

---

