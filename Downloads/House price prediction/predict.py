"""
Make predictions using the trained house price model
"""

import pandas as pd
import numpy as np
import os
import pickle
from train_model import HousePricePredictor


def predict_house_prices(data_path, model_path='house_price_ensemble_model.pkl'):
    """
    Make predictions on new house data
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file with house features
    model_path : str
        Path to the saved model file
    """
    
    print("="*60)
    print("House Price Prediction - Inference")
    print("="*60)
    
    # Load the model
    script_dir = os.path.dirname(__file__)
    possible_model_paths = [
        os.path.join(script_dir, model_path),
        model_path
    ]
    
    full_model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            full_model_path = path
            break
    
    if not full_model_path:
        print(f"Error: Model file not found")
        print(f"Searched for: {model_path}")
        print("Please train the model first using train_model.py")
        return None
    
    with open(full_model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    print(f"Model loaded successfully!")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    
    # Preprocess data
    predictor = HousePricePredictor()
    
    # Check if we have SalePrice for preprocessing
    has_target = 'SalePrice' in df.columns
    X, _ = predictor.preprocess_data(df, target_col='SalePrice' if has_target else None)
    
    # If no actual features (just synthetic), make sure to use them
    # Apply feature engineering (same as training)
    X = predictor.feature_engineering(X)
    
    # Ensure we have the same features as training
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    
    # Reorder to match training features
    X = X[feature_names]
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    print(f"\nDebug - Training features: {sorted(feature_names)}")
    print(f"Debug - Current features: {sorted(X.columns.tolist())}")
    print(f"Debug - Feature count - Training: {len(feature_names)}, Current: {len(X.columns)}")
    
    # Scale features
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Make predictions
    predictions = model.predict(X_scaled_df)
    
    # Create results dataframe
    results_df = df[['Id']].copy() if 'Id' in df.columns else pd.DataFrame()
    results_df['PredictedPrice'] = predictions
    
    # Save results
    output_path = os.path.join(script_dir, 'predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Display summary
    print(f"\nPrediction Summary:")
    print(f"  Mean Predicted Price: ${predictions.mean():,.2f}")
    print(f"  Min Predicted Price: ${predictions.min():,.2f}")
    print(f"  Max Predicted Price: ${predictions.max():,.2f}")
    print(f"  Std Dev: ${predictions.std():,.2f}")
    
    print(f"\nFirst 10 predictions:")
    print(results_df.head(10))
    
    return results_df


if __name__ == "__main__":
    # Example usage
    # Modify the path to your actual data file
    data_path = "solution.csv"  # or path to your test data
    
    if os.path.exists(data_path):
        predict_house_prices(data_path)
    else:
        print(f"Error: Data file not found at {data_path}")
        print("Please provide the correct path to your data file.")
