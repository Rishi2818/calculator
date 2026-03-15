# House Price Prediction Model

A sophisticated machine learning project for predicting house prices using advanced ensemble methods and feature engineering techniques.

## Project Overview

This project implements multiple machine learning algorithms (XGBoost, LightGBM, RandomForest, GradientBoosting, Ridge, Lasso) with an ensemble voting mechanism to achieve high accuracy in house price predictions.

### Key Features

- **Multiple ML Algorithms**: XGBoost, LightGBM, RandomForest, GradientBoosting, Ridge, Lasso
- **Ensemble Voting**: Combines top 3 models for improved accuracy
- **Robust Data Preprocessing**: Handles missing values, categorical encoding, feature scaling
- **Feature Engineering**: Creates polynomial features and optimized representations
- **Comprehensive Evaluation**: RMSE, MAE, R² Score metrics
- **Model Persistence**: Save and load trained models with pickle

## Project Structure

```
House price prediction/
├── solution.csv                      # Dataset with house prices
├── requirements.txt                  # Python dependencies
├── train_model.py                    # Main training script
├── predict.py                        # Make predictions on new data
├── analyze_data.py                   # Data exploration and visualization
├── house_price_ensemble_model.pkl    # Trained model (generated after training)
└── README.md                         # This file
```

## Installation & Setup

### Step 1: Install Python Dependencies

```powershell
# Open PowerShell in the project directory
cd "c:\Users\mekal\Downloads\House price prediction"

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Installation

```powershell
# Verify packages are installed
pip list | findstr "pandas numpy scikit-learn xgboost lightgbm"
```

## Usage

### 1. Explore the Data

First, analyze the dataset to understand features and distributions:

```powershell
# In VS Code Terminal
python analyze_data.py
```

This will:
- Display dataset statistics
- Show correlation with target variable
- Generate visualization plots
- Display categorical feature analysis

### 2. Train the Model

Train the machine learning model with optimized hyperparameters:

```powershell
# Train all models and create ensemble
python train_model.py
```

This will:
- Preprocess the data
- Engineer new features
- Train 6 different algorithms
- Create an ensemble voting model
- Display performance metrics for each model
- Save the best ensemble model

**Output Metrics:**
- RMSE (Root Mean Square Error) - measures prediction error in dollars
- MAE (Mean Absolute Error) - average absolute prediction error
- R² Score - how well the model explains price variance (0-1 scale)

### 3. Make Predictions

Use the trained model to predict prices on new data:

```powershell
# Make predictions on the test data
python predict.py
```

This will:
- Load the trained ensemble model
- Process new house data
- Generate predictions
- Save results to `predictions.csv`
- Display summary statistics

## Performance Optimization for Better Accuracy

### What Makes This Model More Accurate:

1. **Ensemble Methods**: Combines multiple algorithms for more robust predictions
2. **Hyperparameter Tuning**: Each model is tuned for optimal performance
3. **Robust Scaling**: Uses RobustScaler to handle outliers better
4. **Feature Engineering**: Creates polynomial features for non-linear relationships
5. **Multiple Algorithms**: Different algorithms capture different patterns:
   - **XGBoost**: Fast, accurate gradient boosting
   - **LightGBM**: Faster with less memory usage
   - **RandomForest**: Captures feature interactions
   - **GradientBoosting**: Sequential error correction
   - **Ridge/Lasso**: Linear models for comparison

### Tips to Further Improve Accuracy:

1. **More Data**: Collect more house data if available
2. **Better Features**: Add domain-specific features (neighborhood quality, school ratings)
3. **Advanced Engineering**: Create interaction features between important variables
4. **Cross-Validation**: Use k-fold cross-validation for more robust estimates
5. **Hyperparameter Search**: GridSearchCV or RandomSearchCV for optimal parameters
6. **Remove Outliers**: Identify and handle extreme price outliers

## File Descriptions

### `train_model.py`
Main training script with the `HousePricePredictor` class that handles:
- Data loading and preprocessing
- Feature engineering
- Model training
- Model evaluation
- Ensemble creation
- Model persistence

**Key Methods:**
- `load_data()`: Load CSV file
- `preprocess_data()`: Handle missing values and encode categoricals
- `feature_engineering()`: Create additional features
- `train_models()`: Train all algorithms
- `create_ensemble()`: Create voting ensemble
- `save_model()`: Save trained model

### `predict.py`
Inference script for making predictions on new data:
- Loads the trained ensemble model
- Preprocesses new data
- Makes predictions
- Saves results to CSV

### `analyze_data.py`
Exploratory data analysis script that:
- Loads and displays dataset info
- Analyzes target variable distribution
- Calculates feature correlations
- Generates visualization plots
- Analyzes categorical features

### `requirements.txt`
Lists all Python package dependencies with versions for reproducible environments.

## Running in VS Code

### Step 1: Open the Project
1. Launch VS Code
2. File → Open Folder
3. Select: `c:\Users\mekal\Downloads\House price prediction`

### Step 2: Open Terminal
- Use `Ctrl + ~` or View → Terminal

### Step 3: Run Commands
```powershell
# Run from the terminal in VS Code
python analyze_data.py     # Explore data
python train_model.py      # Train model
python predict.py          # Make predictions
```

### Step 4: View Results
- `target_analysis.png` - Target variable distribution
- `feature_correlation.png` - Feature correlation chart
- `predictions.csv` - Predicted prices
- `house_price_ensemble_model.pkl` - Saved model

## Advanced Usage

### Using the Model in Your Own Code

```python
from train_model import HousePricePredictor
import pandas as pd

# Load trained model
predictor = HousePricePredictor()
predictor.load_model('house_price_ensemble_model.pkl')

# Load new data
new_data = pd.read_csv('new_houses.csv')
X, _ = predictor.preprocess_data(new_data, target_col=None)

# Make predictions
predictions = predictor.predict(X)
print(f"Average predicted price: ${predictions.mean():,.2f}")
```

### Retraining with New Data

```python
# Combine old and new data
df = pd.concat([old_data, new_data], ignore_index=True)

# Retrain the model
predictor = HousePricePredictor()
X, y = predictor.preprocess_data(df)
X = predictor.feature_engineering(X)
X_train, X_test, y_train, y_test = predictor.prepare_training_data(X, y)

results = predictor.train_models(X_train, X_test, y_train, y_test)
ensemble = predictor.create_ensemble(X_train, y_train, results)
predictor.save_model(ensemble)
```

## Model Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
- **MAE (Mean Absolute Error)**: Average absolute error in dollars
- **R² Score**: Proportion of variance explained by the model (closer to 1 is better)

### Expected Performance
- **R² Score**: 0.85+ (excellent)
- **RMSE**: <$20,000 (depending on price range)
- **MAE**: <$10,000 (depending on price range)

## Troubleshooting

### Issue: "Module not found" errors
**Solution**: Ensure all packages are installed
```powershell
pip install -r requirements.txt
```

### Issue: "solution.csv not found"
**Solution**: Ensure the CSV file is in the project directory
```powershell
dir    # List files in current directory
```

### Issue: "Model file not found"
**Solution**: Train the model first using `train_model.py`

### Issue: Inaccurate predictions
**Possible Solutions**:
1. Check data quality and missing values
2. Verify feature scaling is applied
3. Ensure model is properly trained on relevant data
4. Check for data distribution changes

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: ML algorithms (Random Forest, Ridge, Lasso)
- **xgboost**: Gradient boosting algorithm
- **lightgbm**: Light Gradient Boosting Machine
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scipy**: Scientific computing

## Performance Tips

1. **Use GPU** (if available): XGBoost and LightGBM support GPU acceleration
2. **Parallel Processing**: Models use `n_jobs=-1` for multi-core processing
3. **Feature Selection**: Remove low-importance features if needed
4. **Hyperparameter Tuning**: Adjust learning rates and tree depths

## Project Development

### Version: 1.0
- Ensemble voting regressor
- Six different algorithms
- Comprehensive preprocessing
- Feature engineering

### Future Enhancements
- Neural network models (TensorFlow/PyTorch)
- Stacking ensemble with meta-learner
- Automated feature selection
- Real-time prediction API
- Web interface for predictions

## GitHub Repository

This project can be version controlled with Git:

```powershell
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial house price prediction project"

# Add to GitHub
git remote add origin https://github.com/yourusername/house-price-prediction.git
git push -u origin main
```

## License

This project is provided for educational purposes.

## Author

Created as a house price prediction solution using advanced machine learning techniques.

## Contact & Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the error messages in the console
3. Verify all dependencies are installed
4. Ensure data files are in the correct location

---

**Last Updated**: March 2026
**Python Version**: 3.8+
**Status**: Production Ready
