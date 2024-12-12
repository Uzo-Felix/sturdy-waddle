from data_loader import load_heart_data, get_train_test_split
from binarization import create_binary_features
from models import train_evaluate_models
from neural_fca_model import train_neural_fca
from visualization import plot_target_distribution, plot_model_comparison
from sklearn.metrics import accuracy_score, classification_report
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    # Create directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. Load data
    print("Loading data...")
    data = load_heart_data()
    
    # 2. Create two different data representations
    # 2.1 Standard scaled data for traditional models
    X = data.drop('target', axis=1)
    y = data['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 2.2 Binarized data for Neural FCA
    X_binary = create_binary_features(data)
    
    # 3. Create train/test splits (using same random state for fair comparison)
    X_train, X_test, y_train, y_test = get_train_test_split(X_scaled, y)
    X_binary_train, X_binary_test, y_binary_train, y_binary_test = get_train_test_split(X_binary, y)
    
    # 4. Train and evaluate traditional models
    print("\nTraining traditional models...")
    traditional_results = train_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 5. Train and evaluate Neural FCA
    print("\nTraining Neural FCA...")
    y_pred_nfca, network = train_neural_fca(X_binary_train, X_binary_test, y_binary_train, y_binary_test)
    
    # 6. Combine all results
    results = traditional_results
    results['Neural FCA'] = {
        'accuracy': accuracy_score(y_binary_test, y_pred_nfca),
        'report': classification_report(y_binary_test, y_pred_nfca)
    }
    
    # 7. Generate visualizations
    print("\nGenerating visualizations...")
    plot_target_distribution(data, 'figures/target_distribution.png')
    plot_model_comparison(results, 'figures/model_comparison.png')
    
    # 8. Save detailed results
    print("\nSaving results...")
    with open('results/model_performance.txt', 'w') as f:
        f.write("Heart Disease Classification Results\n")
        f.write("==================================\n\n")
        f.write("Data Information:\n")
        f.write(f"Total samples: {len(data)}\n")
        f.write(f"Binary features: {X_binary.columns.tolist()}\n\n")
        
        f.write("Model Performance:\n")
        for model, metrics in results.items():
            f.write(f"\n{model} Results:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(f"{metrics['report']}\n")
            
    print("\nAnalysis complete! Check 'results' and 'figures' directories for outputs.")
    
def visualize_binarization(data, binary_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original distributions
    sns.histplot(data=data, x='age', ax=axes[0,0])
    sns.histplot(data=data, x='trestbps', ax=axes[0,1])
    
    # Binarized features
    binary_age = binary_data[['age_30plus', 'age_45plus', 'age_60plus']].mean()
    binary_bp = binary_data[['bp_120plus', 'bp_140plus', 'bp_160plus']].mean()
    
    binary_age.plot(kind='bar', ax=axes[1,0])
    binary_bp.plot(kind='bar', ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('figures/binarization_process.png')
    plt.close()


if __name__ == "__main__":
    main()
