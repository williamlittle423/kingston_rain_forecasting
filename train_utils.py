# train_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20, patience=5):
    """
    Train the neural network model with early stopping.

    Parameters:
        model (nn.Module): The neural network model.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Maximum number of epochs.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        model (nn.Module): The trained model with best weights.
        history (dict): Training and validation loss and accuracy history.
    """
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            batch_acc = (preds == targets).float().mean()
            total_acc += batch_acc.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_acc += (preds == targets).float().mean().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Check for improvement
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print("Early stopping triggered.")
            break
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set and print metrics.

    Parameters:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        f1 (float): F1-Score for the "Rain" class.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            logits = model(inputs)
            preds = (torch.sigmoid(logits) > 0.5).int()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.int().cpu().numpy().flatten())
    
    # Compute confusion matrix and classification report
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=['No Rain', 'Rain'])
    
    # Plot confusion matrix
    labels = ['No Rain', 'Rain']
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Return F1-Score for "Rain" class
    f1 = f1_score(all_targets, all_preds, pos_label=1)
    return f1

def evaluate_f1(model, X_val, y_val):
    """
    Evaluate the F1-Score for the "Rain" class on a validation set.

    Parameters:
        model (nn.Module): The trained model.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.

    Returns:
        f1 (float): F1-Score for the "Rain" class.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32)
        logits = model(inputs)
        preds = (torch.sigmoid(logits) > 0.5).int().numpy().flatten()
        y_val_np = y_val.flatten()
        f1 = f1_score(y_val_np, preds, pos_label=1)
    return f1
