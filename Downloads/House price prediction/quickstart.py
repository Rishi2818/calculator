"""
Quick Start Script - Run this to set up and train the model
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and display feedback"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {command}\n")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False)
        if result.returncode == 0:
            print(f"✓ {description} - SUCCESS")
            return True
        else:
            print(f"✗ {description} - FAILED")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Main quick setup"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "House Price Prediction - Quick Start" + " "*11 + "║")
    print("╚" + "="*58 + "╝")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    print(f"\nWorking Directory: {current_dir}")
    print(f"\nThis script will:")
    print("  1. Install required Python packages")
    print("  2. Analyze your data")
    print("  3. Train the machine learning model")
    print("  4. Generate predictions")
    
    # Check if solution.csv exists
    if not os.path.exists('solution.csv'):
        print("\n⚠️  WARNING: solution.csv not found!")
        print("   Please ensure solution.csv is in the project directory.")
        input("\nPress Enter to continue anyway, or Ctrl+C to exit...")
    
    # Step 1: Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "STEP 1: Installing Python Packages"
    )
    
    if not success:
        print("\n⚠️  Package installation had issues. Continuing anyway...")
    
    # Step 2: Analyze data
    if os.path.exists('solution.csv'):
        run_command(
            f"{sys.executable} analyze_data.py",
            "STEP 2: Analyzing Data"
        )
    else:
        print(f"\n⚠️  Skipped data analysis (solution.csv not found)")
    
    # Step 3: Train model
    if os.path.exists('solution.csv'):
        run_command(
            f"{sys.executable} train_model.py",
            "STEP 3: Training Machine Learning Model"
        )
    else:
        print(f"\n⚠️  Skipped model training (solution.csv not found)")
    
    # Step 4: Make predictions
    if os.path.exists('solution.csv') and os.path.exists('house_price_ensemble_model.pkl'):
        run_command(
            f"{sys.executable} predict.py",
            "STEP 4: Making Price Predictions"
        )
    else:
        print(f"\n⚠️  Skipped predictions (model not available)")
    
    # Summary
    print(f"\n{'='*60}")
    print("╔" + "="*58 + "╗")
    print("║" + " "*18 + "Quick Start Complete!" + " "*18 + "║")
    print("╚" + "="*58 + "╝")
    
    print("\nGenerated Files:")
    files_to_check = [
        'target_analysis.png',
        'feature_correlation.png',
        'house_price_ensemble_model.pkl',
        'predictions.csv'
    ]
    
    for file in files_to_check:
        exists = "✓" if os.path.exists(file) else "✗"
        print(f"  {exists} {file}")
    
    print("\nNext Steps:")
    print("  1. View generated PNG files to analyze the data")
    print("  2. Check predictions.csv for predicted house prices")
    print("  3. Use predict.py to make predictions on new data")
    print("  4. Read README.md for detailed usage instructions")
    
    print(f"\n{'='*60}\n")
    input("Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.")
        sys.exit(0)
