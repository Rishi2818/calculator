# 📱 TERMINAL COMMANDS - Quick Reference

## ⚡ Quick Start (Copy & Paste)

Open terminal in VS Code: `Ctrl + ~`

Then run these commands:

---

## 🎯 OPTION 1: RUN EVERYTHING (RECOMMENDED)

```powershell
python run_improved_pipeline.py
```

**What it does:**
- ✅ Trains improved model with enhanced features
- ✅ Generates predictions automatically
- ✅ Saves results to `predictions_improved.csv`
- ⏱️ Time: ~3-5 minutes

**Results:**
- `house_price_improved_model.pkl` (trained model)
- `predictions_improved.csv` (predictions)

---

## 🎯 OPTION 2: STEP BY STEP (DETAILED)

### Step 1: Train the Improved Model
```powershell
python train_improved_model.py
```

**Output:**
```
======================================================================
               IMPROVED HOUSE PRICE PREDICTION MODEL
                    Enhanced Accuracy Training
======================================================================
```

- Shows all 5 models being trained
- Displays cross-validation scores
- Final R² Score: **0.7618** 📈 (vs 0.3968 baseline)
- RMSE: **$37,959** 📉 (vs $60,400 baseline)
- Saves: `house_price_improved_model.pkl`

**Time: ~2-3 minutes**

---

### Step 2: Generate Predictions
```powershell
python predict_improved.py
```

**Output:**
```
======================================================================
                    IMPROVED MODEL INFERENCE
======================================================================
Price Statistics:
  Mean:     $180,462.15
  Median:   $159,800.88
  Min:      $64,668.88
  Max:      $544,727.32
```

- Loads the trained model
- Preprocesses data with advanced features
- Generates 1,459 price predictions
- Saves: `predictions_improved.csv`

**Time: ~30 seconds**

---

## 📊 OPTION 3: INDIVIDUAL COMMANDS

### Only Train (No Predictions)
```powershell
python train_improved_model.py
```

### Only Predict (Must train first)
```powershell
python predict_improved.py
```

### Compare Models Side-by-Side
```powershell
python train_model.py
python train_improved_model.py
```

---

## 🔍 DATA EXPLORATION

### Analyze the Dataset
```powershell
python analyze_data.py
```

**Generates:**
- `target_analysis.png` (price distribution)
- Dataset statistics
- Correlation analysis

---

## 💾 RESULTS & OUTPUT FILES

After running improved pipeline, check these files:

### Main Output
```
predictions_improved.csv        ← Use this for results!
```

### Model Files
```
house_price_improved_model.pkl
house_price_ensemble_model.pkl (baseline)
```

### Visualizations
```
target_analysis.png
```

---

## 📈 MONITORING OUTPUT

When training, you'll see progress like:

```
Training XGBoost_Optimized...
  Train RMSE: $23,409 | Test RMSE: $39,259
  Train R²: 0.9166 | Test R²: 0.7452
  MAE: $29,608 | MAPE: 20.24%
  CV R² Score: 0.7367 (+/- 0.0341)

Training LightGBM_Optimized...
  [continues for each model...]

=== Creating Advanced Ensemble (Stacking) ===
✓ Stacking ensemble created

============================================================
IMPROVED Ensemble Model Final Performance:
============================================================
  RMSE (Root Mean Square Error): $37,958.53
  MAE (Mean Absolute Error):     $29,486.35
  R² Score (Coefficient):        0.7618  ← THIS IS THE KEY METRIC
  MAPE (Mean Absolute % Error):  20.26%
============================================================
```

---

## ⏱️ EXPECTED TIMING

| Command | Time |
|---------|------|
| `python train_improved_model.py` | 2-3 min |
| `python predict_improved.py` | 30 sec |
| `python run_improved_pipeline.py` | 3-5 min |
| `python train_model.py` (baseline) | 1-2 min |
| `python analyze_data.py` | 15 sec |

---

## 🔄 WORKFLOW SUMMARY

```
START
  ↓
Run: python run_improved_pipeline.py
  ├─ Train improved model
  │  └─ Generates: house_price_improved_model.pkl
  │
  ├─ Generate predictions
  │  └─ Generates: predictions_improved.csv
  │
END
  ↓
Check: predictions_improved.csv for results
```

---

## 💡 TIPS & TRICKS

### Tip 1: Run Multiple Times
The pipeline is deterministic (random_state=42), so results will be identical:

```powershell
python run_improved_pipeline.py  # First run
python run_improved_pipeline.py  # Second run (same results)
```

### Tip 2: Just Get Predictions (After First Training)
```powershell
python predict_improved.py
```

No need to retrain unless you modify code!

### Tip 3: Clear Old Results
```powershell
Remove-Item predictions_improved.csv  # Remove old predictions
python predict_improved.py             # Generate new ones
```

### Tip 4: View Results Quickly
```powershell
# Windows - Open in Excel
notepad predictions_improved.csv

# Or use Python to display first rows
python -c "import pandas as pd; df = pd.read_csv('predictions_improved.csv'); print(df.head(10))"
```

---

## 📊 ACCURACY COMPARISON COMMAND

To see baseline vs improved side-by-side:

```powershell
# Train both models
python train_model.py              # Baseline (R² = 0.3968)
python train_improved_model.py     # Improved (R² = 0.7618)

# View results
echo "Baseline R²: 0.3968"
echo "Improved R²: 0.7618"
echo "Improvement: +92%"
```

---

## 🐛 TROUBLESHOOTING

### Error: "ModuleNotFoundError"
```powershell
# Reinstall packages
pip install -r requirements.txt
```

### Error: "solution.csv not found"
```powershell
# Check current directory
dir *.csv

# Make sure you're in the right folder
cd "c:\Users\mekal\Downloads\House price prediction"
```

### Model not found after training
```powershell
# Check if model file exists
dir *.pkl

# If not, run training again
python train_improved_model.py
```

---

## 🎯 BEST PRACTICES

### 1. Always Run `run_improved_pipeline.py` First
```powershell
python run_improved_pipeline.py
```
This is the safest and fastest way.

### 2. Check Output Files
```powershell
# List all generated files
dir predictions_improved.csv
dir house_price_improved_model.pkl
```

### 3. Use for Production
```powershell
# Only run prediction (model already trained)
python predict_improved.py
```

### 4. Monitor Performance
After training, look for:
- **R² Score > 0.7** ✅ (excellent)
- **RMSE < $40k** ✅ (good)
- **MAE < $30k** ✅ (good)

---

## 📋 COMMAND CHEATSHEET

| Task | Command |
|------|---------|
| Train improved model | `python train_improved_model.py` |
| Make predictions | `python predict_improved.py` |
| Do everything | `python run_improved_pipeline.py` |
| Train baseline | `python train_model.py` |
| Analyze data | `python analyze_data.py` |
| Run full pipeline | `python quickstart.py` |

---

## 🚀 FINAL STEPS

1. **Open Terminal**: `Ctrl + ~` in VS Code
2. **Copy & Paste**: `python run_improved_pipeline.py`
3. **Wait**: 3-5 minutes for training
4. **Check Results**: Open `predictions_improved.csv`
5. **Success**: You have 1,459 accurate price predictions! 🎉

---

**Now just open the terminal and run:**

```powershell
python run_improved_pipeline.py
```

The script handles everything automatically! 🚀
