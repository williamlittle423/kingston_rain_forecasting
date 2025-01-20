# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preparation import load_and_preprocess_data, split_and_scale, handle_class_imbalance
from model import PrecipitationClassifier
from train_utils import train_model, evaluate_model
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split into train and test
    X_train_full, X_test, y_train_full, y_test, scaler = split_and_scale(X, y, test_size=0.2)

    # Handle class imbalance with SMOTE on full training data
    X_train_res, y_train_res = handle_class_imbalance(X_train_full, y_train_full, method='smote')
    
    # Scale the resampled data
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_res)
    X_test_scaled = scaler_final.transform(X_test)
    
    # Save the scaler for future use
    scaler_path = 'scaler_hardcoded.pth'
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
    batch_size = 128  # Hardcoded hyperparameters
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model with hardcoded hyperparameters
    input_size = X_train_full.shape[1]
    hidden_size = 512
    num_layers = 4
    dropout_rate = 0.5858298634189862
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
    learning_rate = 0.003995856030068804
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    num_epochs = 50
    patience = 10
    model, history = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=num_epochs, patience=patience)
    
    # Save the trained model
    model_path = 'model_hardcoded.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at '{model_path}'")
    
    # Evaluate on test set
    f1 = evaluate_model(model, test_loader)
    print(f"Test F1-Score: {f1:.4f}")
    
if __name__ == "__main__":
    main()
