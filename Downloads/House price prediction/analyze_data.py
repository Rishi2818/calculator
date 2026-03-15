"""
Data Analysis and Visualization for House Price Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_explore(data_path):
    """Load and explore the dataset"""
    print("="*60)
    print("Data Exploration & Analysis")
    print("="*60)
    
    df = pd.read_csv(data_path)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values!")
    
    return df


def analyze_target_variable(df, target_col='SalePrice'):
    """Analyze the target variable"""
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found")
        return
    
    print(f"\n{'='*60}")
    print(f"Target Variable Analysis: {target_col}")
    print(f"{'='*60}")
    
    target = df[target_col]
    
    print(f"\nStatistics for {target_col}:")
    print(f"  Mean: ${target.mean():,.2f}")
    print(f"  Median: ${target.median():,.2f}")
    print(f"  Std Dev: ${target.std():,.2f}")
    print(f"  Min: ${target.min():,.2f}")
    print(f"  Max: ${target.max():,.2f}")
    print(f"  Skewness: {skew(target):.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution
    axes[0, 0].hist(target, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'Distribution of {target_col}')
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Log distribution (for better visualization)
    axes[0, 1].hist(np.log1p(target), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title(f'Log Distribution of {target_col}')
    axes[0, 1].set_xlabel('Log(Price)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Box plot
    axes[1, 0].boxplot(target)
    axes[1, 0].set_title(f'Box Plot of {target_col}')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy.stats import probplot
    probplot(target, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'target_analysis.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


def analyze_numerical_features(df, target_col='SalePrice'):
    """Analyze correlation with numerical features"""
    print(f"\n{'='*60}")
    print("Numerical Features Correlation Analysis")
    print(f"{'='*60}")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if 'Id' in numerical_cols:
        numerical_cols.remove('Id')
    
    if len(numerical_cols) == 0:
        print(f"\nNo additional numerical features found besides Id and {target_col}")
        print("This dataset appears to contain only ID and target price.")
        print("For feature correlation analysis, full Kaggle dataset is needed.")
        return
    
    if target_col in df.columns:
        correlations = df[numerical_cols + [target_col]].corr()[target_col]
        correlations = correlations.drop(target_col).sort_values(ascending=False)
        
        print(f"\nTop 15 positively correlated features:")
        print(correlations.head(15))
        
        print(f"\nTop 15 negatively correlated features:")
        print(correlations.tail(15))
        
        # Visualize correlations
        if len(correlations) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_features = pd.concat([correlations.head(10), correlations.tail(10)])
            colors = ['green' if x > 0 else 'red' for x in top_features.values]
            top_features.plot(kind='barh', ax=ax, color=colors)
            ax.set_title(f'Top Features Correlated with {target_col}')
            ax.set_xlabel('Correlation Coefficient')
            plt.tight_layout()
            
            output_path = os.path.join(os.path.dirname(__file__), 'feature_correlation.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"\nCorrelation chart saved to {output_path}")
            plt.close()


def analyze_categorical_features(df):
    """Analyze categorical features"""
    print(f"\n{'='*60}")
    print("Categorical Features Analysis")
    print(f"{'='*60}")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nCategorical columns: {categorical_cols}")
    
    for col in categorical_cols[:5]:  # Limit to first 5
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most common values:")
        print(f"  {df[col].value_counts().head(5)}")


def main():
    """Main execution"""
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'solution.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Load and explore
    df = load_and_explore(data_path)
    
    # Analyze target variable
    analyze_target_variable(df, target_col='SalePrice')
    
    # Analyze numerical features
    analyze_numerical_features(df, target_col='SalePrice')
    
    # Analyze categorical features
    analyze_categorical_features(df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
