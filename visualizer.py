import os
import json
import matplotlib.pyplot as plt

def load_history(model_name, results_dir='results'):
    history_path = os.path.join(results_dir, model_name + '_history.json')
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history

def plot_comparison(histories, metric, title, ylabel, dataset, kind):
    for model_name, history in histories.items():
        plt.plot(history[kind + metric], label=f'{model_name} {kind.capitalize()} {metric.capitalize()}')
    
    plt.title(f'{title} for {dataset}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.ylim(0.55, 0.85)  # Set the y-axis limits
    plt.legend()
    plt.show()

def visualize_results_by_dataset(model_names, results_dir='results'):
    # Group model names by dataset
    datasets = {}
    for model_name in model_names:
        dataset = model_name.split('_')[0]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(model_name)
    
    # Visualize results for each dataset
    for dataset, models in datasets.items():
        print(f"Visualizing results for dataset: {dataset}")
        histories = {model: load_history(model, results_dir) for model in models}
        
        # plot_comparison(histories, 'accuracy', 'Model Accuracy Comparison (Training)', 'Accuracy', dataset, '')
        plot_comparison(histories, 'accuracy', 'Model Accuracy Comparison (Validation)', 'Accuracy', dataset, 'val_')
        # plot_comparison(histories, 'loss', 'Model Loss Comparison (Training)', 'Loss', dataset, '')
        # plot_comparison(histories, 'loss', 'Model Loss Comparison (Validation)', 'Loss', dataset, 'val_')

if __name__ == '__main__':
    # List the models you want to visualize
    model_names = [
        'cifar10_complex_cnn_dropout_epochs50',
    ]
    
    visualize_results_by_dataset(model_names)
