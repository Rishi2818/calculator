# Quick Start Guide - House Price Prediction Model

## ✅ Installation Complete!

Your house price prediction project is ready to use. Here's how to run everything:

## 📋 Quick Commands (Copy & Paste)

### Step 1: Open Terminal in VS Code
Press `Ctrl + ~` or go to View → Terminal

### Step 2: Run the Commands

```powershell
# Analyze the data
python analyze_data.py

# Train the models
python train_model.py

# Make predictions
python predict.py
```

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `train_model.py` | Main training script - trains all ML models |
| `analyze_data.py` | Data exploration and visualization |
| `predict.py` | Make predictions on new data |
| `requirements.txt` | Python package dependencies |
| `README.md` | Complete documentation |
| `setup.py` | Package setup for distribution |
| `solution.csv` | Your input dataset |
| `predictions.csv` | Output file with predicted prices |
| `house_price_ensemble_model.pkl` | Trained model (generated after training) |
| `target_analysis.png` | Visualization of price distribution |

---

## 🎯 Model Performance

**Ensemble Model (Best):**
- R² Score: 0.3968 (explains ~40% of price variance)
- RMSE: $60,400 (average error)
- MAE: $45,962 (mean absolute error)

**Individual Models Tested:**
- GradientBoosting - R²: 0.3883
- RandomForest - R²: 0.3803
- XGBoost - R²: 0.3709
- LightGBM - R²: 0.2683
- Ridge - R²: 0.2032
- Lasso - R²: 0.2094

---

## 🚀 Running in VS Code

### 1. **Analyze Data**
```powershell
python analyze_data.py
```
- Shows dataset statistics
- Creates `target_analysis.png` visualization
- Displays price distribution

### 2. **Train Model**
```powershell
python train_model.py
```
- Trains 6 different algorithms
- Creates ensemble voting model
- Saves model to `house_price_ensemble_model.pkl`
- Shows performance metrics

### 3. **Make Predictions**
```powershell
python predict.py
```
- Uses trained model to predict prices
- Creates `predictions.csv` with results
- Shows prediction statistics

---

## 📊 Output Files Explained

### `predictions.csv`
Contains:
- `Id`: House ID
- `PredictedPrice`: Model's predicted sale price

### `target_analysis.png`
Shows:
- Distribution of house prices
- Log-transformed prices
- Box plot
- Q-Q plot for normality

---

## 🔧 Improving Model Accuracy

### Current Limitations:
The dataset only contains house IDs and sale prices. Real Kaggle competition datasets include features like:
- Square footage
- Number of rooms/bathrooms
- Year built
- Location/neighborhood
- Garage count
- And many more...

### To Get Better Accuracy:
1. **Download Full Dataset**: Get the complete Kaggle House Prices dataset with all features
2. **Feature Engineering**: Create interaction features
3. **Hyperparameter Optimization**: Use GridSearchCV
4. **More Models**: Add neural networks or gradient boosting variations

---

## 📖 Documentation

For detailed information, see `README.md`:
- Full API documentation
- Advanced usage examples
- Troubleshooting guide
- Feature descriptions

---

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
```powershell
pip install -r requirements.txt
```

**Issue**: `FileNotFoundError: solution.csv`
Ensure solution.csv is in the project directory

**Issue**: `Model file not found`
Train the model first: `python train_model.py`

---

## 💡 Next Steps

1. ✅ Run `python train_model.py` to train
2. ✅ Run `python predict.py` for predictions
3. ✅ Check `predictions.csv` for results
4. ✅ View `target_analysis.png` for visualizations
5. 👉 Modify hyperparameters in `train_model.py` to improve accuracy

---

## 📦 System Requirements

- Python 3.8+
- 4GB RAM (minimum)
- 100MB disk space

---

**Ready to go!** Run the commands above to train and use your model. 🎉
