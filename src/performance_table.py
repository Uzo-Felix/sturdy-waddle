import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def create_performance_heatmap():
    # Create DataFrame with all metrics
    metrics_data = {
        'Model': ['Random Forest', 'XGBoost', 'Logistic Regression', 'KNN', 
                 'Decision Tree', 'Naive Bayes', 'CatBoost', 'Neural FCA'],
        'Class_0_Precision': [0.91, 0.91, 0.89, 0.84, 0.90, 0.90, 0.89, 0.60],
        'Class_0_Recall': [0.89, 0.81, 0.89, 0.89, 0.75, 0.97, 0.86, 1.00],
        'Class_0_F1': [0.90, 0.85, 0.89, 0.86, 0.82, 0.93, 0.87, 0.75],
        'Class_1_Precision': [0.84, 0.75, 0.83, 0.82, 0.70, 0.95, 0.80, 0.00],
        'Class_1_Recall': [0.88, 0.88, 0.83, 0.75, 0.88, 0.83, 0.83, 0.00],
        'Class_1_F1': [0.86, 0.81, 0.83, 0.78, 0.78, 0.89, 0.82, 0.00]
    }
    
    df = pd.DataFrame(metrics_data)
    df_melted = df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(15, 8))
    heatmap_data = df.set_index('Model').T
    
    # Create heatmap with custom colormap
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', 
                fmt='.2f', center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Score'})
    
    plt.title('Model Performance Metrics Comparison', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('figures/performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    create_performance_heatmap()