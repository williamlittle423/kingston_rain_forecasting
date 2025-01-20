# optimize.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preparation import load_and_preprocess_data, split_and_scale, handle_class_imbalance
from model import PrecipitationClassifier
from train_utils import train_model, evaluate_f1
import copy
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import joblib
from sklearn.preprocessing import StandardScaler

def objective(trial: Trial, X_train, X_val, y_train, y_val):
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
        trial (Trial): Optuna trial object.
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        y_train (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.

    Returns:
        f1 (float): F1-Score on the validation set.
    """
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 64, 1024, step=64)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.7)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Handle class imbalance with SMOTE
    X_train_res, y_train_res = handle_class_imbalance(X_train, y_train, method='smote')
    
    # Scale the resampled data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_res, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model
    input_size = X_train.shape[1]
    model = PrecipitationClassifier(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate)
    
    # Define loss with pos_weight
    no_rain = (y_train_res == 0).sum()
    rain = (y_train_res == 1).sum()
    pos_weight = torch.tensor([no_rain / rain])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    num_epochs = 50
    patience = 10
    model, history = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs, patience=patience)
    
    # Evaluate on validation set
    val_f1 = evaluate_f1(model, X_val_scaled, y_val)
    
    return val_f1

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split into train and test
    X_train_full, X_test, y_train_full, y_test, scaler = split_and_scale(X, y, test_size=0.2)

    # Further split train into train and validation for Optuna
    split_index = int(len(X_train_full) * 0.8)
    X_train, X_val = X_train_full[:split_index], X_train_full[split_index:]
    y_train, y_val = y_train_full[:split_index], y_train_full[split_index:]
    
    # Create an Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Optimize
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=50)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Retrain the model with the best hyperparameters on the full training set
    best_params = trial.params
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    
    # Handle class imbalance with SMOTE on full training data
    X_train_res, y_train_res = handle_class_imbalance(X_train_full, y_train_full, method='smote')
    
    # Scale the resampled data
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_res)
    X_test_scaled = scaler_final.transform(X_test)
    
    # Save the scaler for future use
    scaler_path = 'scaler_optimize.pth'
    joblib.dump(scaler_final, scaler_path)
    print(f"Scaler saved at '{scaler_path}'")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_res, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define the best model
    model = PrecipitationClassifier(input_size=X_train_full.shape[1],
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate)
    
    # Define loss with pos_weight
    no_rain = (y_train_res == 0).sum()
    rain = (y_train_res == 1).sum()
    pos_weight = torch.tensor([no_rain / rain])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the best model
    model, history = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=50, patience=patience)
    
    # Save the trained model
    model_path = 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved at '{model_path}'")
    
    # Evaluate on test set
    f1 = evaluate_f1(model, X_test_scaled, y_test.numpy())
    print(f"Test F1-Score: {f1:.4f}")
    
    # Optionally, plot the final confusion matrix
    evaluate_model(model, test_loader)
    
if __name__ == "__main__":
    main()
