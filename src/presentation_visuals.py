import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from data_loader import load_heart_data
from binarization import create_binary_features
from neural_lib import ConceptNetwork
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

def visualize_neural_fca_architecture(network):
    plt.figure(figsize=(15, 10))
    colors = plt.cm.plasma(np.linspace(0, 1, network.poset[network.poset.bottoms[0]].level + 1))
    
    G = nx.DiGraph()
    pos = {}
    
    # Add nodes with different colors per level
    max_level = network.poset[network.poset.bottoms[0]].level
    nodes_per_levels = {lvl: [] for lvl in range(max_level + 1)}
    for node_i, node in enumerate(network.poset):
        nodes_per_levels[node.level].append(node_i)
    
    for level, nodes in nodes_per_levels.items():
        y = max_level - level
        for i, node in enumerate(nodes):
            x = i - len(nodes)/2
            G.add_node(node)
            pos[node] = (x, y)
            
        nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                             node_color=[colors[level]]*len(nodes),
                             node_size=1500)
    
    # Add edges with weights
    edge_weights = network.edge_weights_from_network()
    for (source, target), weight in edge_weights.items():
        G.add_edge(source, target, weight=abs(weight))
    
    # Draw edges with width based on weights
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', 
                          alpha=0.6, arrows=True, arrowsize=20)
    
    nx.draw_networkx_labels(G, pos)
    plt.title('Neural FCA Network Architecture', size=16, pad=20)
    plt.savefig('figures/neural_fca_architecture_colored.png')
    plt.close()
    
def visualize_traditional_architectures():
    # Random Forest
    plt.figure(figsize=(15, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    layers = {
        'Input': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs'],
        'Trees': [f'Tree {i}' for i in range(1, 6)],
        'Aggregation': ['Majority Vote'],
        'Output': ['Prediction']
    }
    create_architecture_plot(layers, colors, 'Random Forest Architecture', 'figures/rf_architecture.png')
    
    # XGBoost
    plt.figure(figsize=(15, 10))
    colors = plt.cm.magma(np.linspace(0, 1, 4))
    layers = {
        'Input': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs'],
        'Boosted Trees': [f'Boost {i}' for i in range(1, 6)],
        'Weighted Sum': ['Aggregation'],
        'Output': ['Prediction']
    }
    create_architecture_plot(layers, colors, 'XGBoost Architecture', 'figures/xgb_architecture.png')
    
    # Neural Network
    plt.figure(figsize=(15, 10))
    colors = plt.cm.cool(np.linspace(0, 1, 4))
    layers = {
        'Input': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs'],
        'Hidden1': [f'h1_{i}' for i in range(1, 5)],
        'Hidden2': [f'h2_{i}' for i in range(1, 4)],
        'Output': ['Prediction']
    }
    create_architecture_plot(layers, colors, 'Neural Network Architecture', 'figures/nn_architecture.png')

def create_architecture_plot(layers, colors, title, filename):
    pos = {}
    for i, (layer_name, nodes) in enumerate(layers.items()):
        y = -i
        for j, node in enumerate(nodes):
            x = j - len(nodes)/2
            pos[node] = (x, y)
    
    G = nx.DiGraph()
    for layer_nodes in layers.values():
        for node in layer_nodes:
            G.add_node(node)
    
    for i in range(len(list(layers.values()))-1):
        current_layer = list(layers.values())[i]
        next_layer = list(layers.values())[i+1]
        for node1 in current_layer:
            for node2 in next_layer:
                G.add_edge(node1, node2)
    
    for i, (layer_name, nodes) in enumerate(layers.items()):
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                             node_color=[colors[i]]*len(nodes),
                             node_size=2000)
    
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title, size=16, pad=20)
    plt.savefig(filename)
    plt.close()

def create_all_visuals():
    print("Loading data...")
    data = load_heart_data()
    binary_data = create_binary_features(data)
    binary_data = binary_data.astype(bool)
    
    print("\nCreating visualizations...")
    
    # Dataset visualizations
    create_dataset_visualizations(data)
    
    # Feature distributions
    create_feature_distributions(data)
    
    # Binarization process
    create_binarization_visualizations(data, binary_data)
    
    # Model architectures
    print("Generating model architectures...")
    visualize_traditional_architectures()
    
    # Neural FCA
    print("Creating Neural FCA network...")
    object_names = [f'obj_{i}' for i in range(len(binary_data))]
    context = FormalContext(binary_data.values, 
                          attribute_names=binary_data.columns,
                          object_names=object_names)
    lattice = ConceptLattice.from_context(context)
    network = ConceptNetwork.from_lattice(lattice, list(range(len(lattice))), (0, 1))
    network.fit(binary_data, data['target'], n_epochs=100)
    visualize_neural_fca_architecture(network)

def create_dataset_visualizations(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x='target', palette='Set2')
    plt.title('Heart Disease Distribution')
    plt.xlabel('Disease Present (1) vs Absent (0)')
    plt.ylabel('Number of Patients')
    plt.savefig('figures/target_distribution.png')
    plt.close()

def create_feature_distributions(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(data=data, x='age', bins=20, ax=axes[0,0], color='skyblue')
    sns.histplot(data=data, x='trestbps', bins=20, ax=axes[0,1], color='lightgreen')
    sns.countplot(data=data, x='cp', ax=axes[1,0], palette='Set3')
    sns.histplot(data=data, x='chol', bins=20, ax=axes[1,1], color='salmon')
    
    axes[0,0].set_title('Age Distribution')
    axes[0,1].set_title('Blood Pressure Distribution')
    axes[1,0].set_title('Chest Pain Types')
    axes[1,1].set_title('Cholesterol Distribution')
    
    plt.tight_layout()
    plt.savefig('figures/feature_distributions.png')
    plt.close()

def create_binarization_visualizations(data, binary_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.histplot(data=data, x='age', ax=axes[0,0], color='lightblue')
    axes[0,0].set_title('Original Age Distribution')
    
    binary_data[['age_30plus', 'age_45plus', 'age_60plus']].mean().plot(
        kind='bar', ax=axes[0,1], color=['skyblue', 'lightgreen', 'salmon'])
    axes[0,1].set_title('Binarized Age Features')
    
    sns.histplot(data=data, x='trestbps', ax=axes[1,0], color='lightgreen')
    axes[1,0].set_title('Original Blood Pressure Distribution')
    
    binary_data[['bp_120plus', 'bp_140plus', 'bp_160plus']].mean().plot(
        kind='bar', ax=axes[1,1], color=['purple', 'orange', 'pink'])
    axes[1,1].set_title('Binarized Blood Pressure Features')
    
    plt.tight_layout()
    plt.savefig('figures/binarization_process.png')
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    create_all_visuals()
    print("\nAll presentation visuals have been generated in the 'figures' directory.")
