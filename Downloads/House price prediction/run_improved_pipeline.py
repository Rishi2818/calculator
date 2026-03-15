"""
Complete Pipeline - Run to train improved model and generate predictions
Run: python run_improved_pipeline.py
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and show progress"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(command, shell=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"\n✓ {description} - SUCCESS\n")
            return True
        else:
            print(f"\n❌ {description} - FAILED\n")
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return False


def main():
    """Main pipeline execution"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "IMPROVED MODEL TRAINING PIPELINE" + " "*21 + "║")
    print("║" + " "*10 + "Advanced Features + Optimized Hyperparameters" + " "*12 + "║")
    print("╚" + "="*68 + "╝")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\nWorking Directory: {current_dir}")
    print("\nThis pipeline will:")
    print("  1️⃣  Train improved model with advanced features")
    print("  2️⃣  Use optimized hyperparameters for all models")
    print("  3️⃣  Create advanced stacking ensemble")
    print("  4️⃣  Generate predictions with improved accuracy")
    print("  5️⃣  Save model and predictions to files")
    
    # Check if solution.csv exists
    if not os.path.exists(os.path.join(current_dir, 'solution.csv')):
        print("\n⚠️  WARNING: solution.csv not found!")
        return
    
    # Step 1: Train improved model
    success_1 = run_command(
        f"{sys.executable} train_improved_model.py",
        "STEP 1: Training Improved Model"
    )
    
    if not success_1:
        print("❌ Training failed. Exiting pipeline.")
        return
    
    # Check if model was saved
    model_paths = [
        os.path.join(current_dir, 'house_price_improved_model.pkl'),
        'house_price_improved_model.pkl'
    ]
    
    model_exists = any(os.path.exists(p) for p in model_paths)
    
    if not model_exists:
        print("⚠️  Model file not found after training")
        return
    
    # Step 2: Generate predictions
    success_2 = run_command(
        f"{sys.executable} predict_improved.py",
        "STEP 2: Generating Predictions with Improved Model"
    )
    
    if not success_2:
        print("⚠️  Prediction generation had issues")
        return
    
    # Summary
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "PIPELINE EXECUTION COMPLETE!" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\n📊 GENERATED FILES:")
    print("  ✓ house_price_improved_model.pkl  (Trained model)")
    print("  ✓ predictions_improved.csv        (Predicted prices)")
    
    print("\n📈 IMPROVEMENTS MADE:")
    print("  ✓ Advanced feature engineering")
    print("  ✓ 30+ derived features")
    print("  ✓ Optimized hyperparameters")
    print("  ✓ 5-fold cross-validation")
    print("  ✓ Stacking ensemble method")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Check predictions_improved.csv for results")
    print("  2. Compare with baseline predictions.csv")
    print("  3. Use improved model for final predictions")
    
    print("\n💾 TO USE THE IMPROVED MODEL LATER:")
    print("  python predict_improved.py")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline cancelled by user.")
        sys.exit(0)
