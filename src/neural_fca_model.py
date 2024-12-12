from neural_lib import ConceptNetwork
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
import numpy as np
# torch
import torch

def train_neural_fca(X_binary_train, X_binary_test, y_binary_train, y_binary_test):
    # Convert integer values to boolean
    X_binary_train_bool = X_binary_train.astype(bool)
    X_binary_test_bool = X_binary_test.astype(bool)
    
    # Create object names for each row
    object_names = [f"obj_{i}" for i in range(len(X_binary_train_bool))]
    
    print("Creating formal context...")
    context = FormalContext(
        data=X_binary_train_bool.values,
        attribute_names=X_binary_train_bool.columns,
        object_names=object_names
    )
    
    print("Building concept lattice...")
    lattice = ConceptLattice.from_context(context)
    
    # Use list() to get all concepts
    best_concepts_indices = list(range(len(lattice)))
    
    print(f"Number of concepts: {len(best_concepts_indices)}")
    
    targets = tuple(sorted(set(y_binary_train)))
    
    print("Creating and training network...")
    network = ConceptNetwork.from_lattice(lattice, best_concepts_indices, targets)
    network.fit(X_binary_train_bool, y_binary_train, n_epochs=2000)
    
    y_pred = network.predict(X_binary_test_bool)
    # Convert prediction to numpy array
    if isinstance(y_pred, tuple):
        y_pred = y_pred[1]  # Get class predictions if tuple returned
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return y_pred, network
