import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(data, save_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='target')
    plt.title('Distribution of Heart Disease Cases')
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(results, save_path):
    plt.figure(figsize=(12, 6))
    accuracies = [results[model]['accuracy'] for model in results]
    plt.bar(results.keys(), accuracies)
    plt.title('Model Comparison - Heart Disease Classification')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
