"""
Make predictions using the improved trained model
Run: python predict_improved.py
"""

import pandas as pd
import numpy as np
import os
import pickle
from train_improved_model import ImprovedHousePricePredictor


def predict_with_improved_model(data_path, model_path='house_price_improved_model.pkl'):
    """
    Make predictions using improved model
    """
    
    print("\n" + "="*70)
    print(" "*20 + "IMPROVED MODEL INFERENCE")
    print("="*70)
    
    # Load the improved model
    script_dir = os.path.dirname(__file__)
    possible_paths = [
        os.path.join(script_dir, model_path),
        model_path
    ]
    
    full_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            full_model_path = path
            break
    
    if not full_model_path:
        print(f"❌ Error: Model not found")
        print(f"   Run 'python train_improved_model.py' first")
        return None
    
    print(f"✓ Loading improved model from {model_path}...")
    with open(full_model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    print(f"✓ Model loaded successfully")
    print(f"  Features: {len(feature_names)} dimensions")
    
    # Load data
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {df.shape}")
    
    # Preprocess data
    print("\n✓ Preprocessing data...")
    predictor = ImprovedHousePricePredictor()
    
    has_target = 'SalePrice' in df.columns
    X, _ = predictor.preprocess_data(df, target_col='SalePrice' if has_target else None)
    
    # Apply advanced feature engineering
    X = predictor.advanced_feature_engineering(X)
    
    # Ensure we have the same features as training
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    
    X = X[feature_names]
    X = X.fillna(0)
    
    # Check feature consistency
    print(f"✓ Features prepared: {X.shape[1]} features")
    
    # Scale features
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Make predictions
    print("✓ Generating predictions...")
    predictions = model.predict(X_scaled_df)
    
    # Create results
    results_df = df[['Id']].copy() if 'Id' in df.columns else pd.DataFrame()
    results_df['PredictedPrice'] = predictions
    
    # Save results
    output_path = os.path.join(script_dir, 'predictions_improved.csv')
    results_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to predictions_improved.csv")
    
    # Display summary
    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total predictions: {len(predictions)}")
    print(f"\nPrice Statistics:")
    print(f"  Mean:     ${predictions.mean():>15,.2f}")
    print(f"  Median:   ${np.median(predictions):>15,.2f}")
    print(f"  Min:      ${predictions.min():>15,.2f}")
    print(f"  Max:      ${predictions.max():>15,.2f}")
    print(f"  Std Dev:  ${predictions.std():>15,.2f}")
    
    print(f"\nFirst 15 predictions:")
    print(f"{'='*70}")
    display_df = results_df.head(15).copy()
    display_df['PredictedPrice'] = display_df['PredictedPrice'].apply(lambda x: f"${x:,.2f}")
    print(display_df.to_string(index=False))
    print(f"{'='*70}\n")
    
    return results_df


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'solution.csv')
    
    if os.path.exists(data_path):
        predict_with_improved_model(data_path)
    else:
        print(f"❌ Error: solution.csv not found at {data_path}")
