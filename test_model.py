# test_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_preparation import load_and_preprocess_data, split_and_scale
from model import PrecipitationClassifier
from train_utils import evaluate_model
import joblib
from sklearn.preprocessing import StandardScaler

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split into train and test
    X_train_full, X_test, y_train_full, y_test, scaler = split_and_scale(X, y, test_size=0.2)
    
    # Load the scaler used during training
    scaler_path = 'scaler_hardcoded.pth'  # Ensure this path matches where the scaler was saved
    scaler_final = joblib.load(scaler_path)
    
    # Scale the test data
    X_test_scaled = scaler_final.transform(X_test)
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = 128  # Must match the batch_size used during training
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define hardcoded hyperparameters (must match the ones used during training)
    input_size = X_train_full.shape[1]
    hidden_size = 512
    num_layers = 4
    dropout_rate = 0.5858298634189862
    model = PrecipitationClassifier(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate)
    
    # Load the trained model
    model_path = 'model_hardcoded.pth'  # Ensure this path matches where the model was saved
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Define loss with pos_weight (must match the one used during training)
    no_rain = (y_train_full == 0).sum()
    rain = (y_train_full == 1).sum()
    pos_weight = torch.tensor([no_rain / rain])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Evaluate the model
    f1 = evaluate_model(model, test_loader)
    print(f"Test F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
