import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# ------------------------------
# 1. Data Preparation Functions
# ------------------------------

def load_and_preprocess_data(file_pattern='climate-hourly_{}.csv', num_files=7):
    # 1. Read multiple CSVs and concatenate
    df_list = []
    for i in range(num_files):
        file_name = file_pattern.format(i)
        df_list.append(pd.read_csv(file_name))
    climate_data = pd.concat(df_list, ignore_index=True)
    
    # 2. Create a proper datetime index
    climate_data['LOCAL_DATETIME'] = pd.to_datetime(
        climate_data[['LOCAL_YEAR','LOCAL_MONTH','LOCAL_DAY','LOCAL_HOUR']]  
        .rename(columns={
            'LOCAL_YEAR': 'year',
            'LOCAL_MONTH': 'month',
            'LOCAL_DAY': 'day',
            'LOCAL_HOUR': 'hour'
        })
    )
    climate_data.set_index('LOCAL_DATETIME', inplace=True)
    climate_data.sort_index(inplace=True)
    
    # 3. Resample hourly (so missing hours appear as NaN rows)
    climate_data = climate_data.resample('h').asfreq()
    
    # 4. Filter the columns you need
    filtered_climate_data = climate_data[[
        'TEMP', 'DEW_POINT_TEMP', 'RELATIVE_HUMIDITY',
        'STATION_PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION',
        'PRECIP_AMOUNT'
    ]].copy()
    
    # 5. Drop rows with missing values
    filtered_climate_data.dropna(subset=[
        'TEMP','DEW_POINT_TEMP','RELATIVE_HUMIDITY',
        'STATION_PRESSURE','WIND_SPEED','WIND_DIRECTION',
        'PRECIP_AMOUNT'
    ], inplace=True)
    
    # 6. Shift precip to the next hour
    filtered_climate_data['PRECIP_AMOUNT_NEXT_HOUR'] = filtered_climate_data['PRECIP_AMOUNT'].shift(-1)
    
    # 7. Drop any rows where next-hour precip is NaN (last row(s))
    filtered_climate_data.dropna(subset=['PRECIP_AMOUNT_NEXT_HOUR'], inplace=True)
    
    # 8. Convert next-hour precipitation to a binary label
    filtered_climate_data['RAIN_NEXT_HOUR'] = (
        filtered_climate_data['PRECIP_AMOUNT_NEXT_HOUR'] > 0
    ).astype(int)
    
    # 9. Prepare features (X) and target (y)
    X = filtered_climate_data[[
        'TEMP','DEW_POINT_TEMP','RELATIVE_HUMIDITY',
        'STATION_PRESSURE','WIND_SPEED','WIND_DIRECTION', 'PRECIP_AMOUNT'
    ]].to_numpy()
    
    y = filtered_climate_data['RAIN_NEXT_HOUR'].to_numpy()
    
    return X, y

def split_and_scale(X, y, test_size=0.2):
    # 10. Time-based split (80% train, 20% test) — no shuffle
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # 11. Scale the training data, then transform test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def handle_class_imbalance(X, y, method='smote'):
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    elif method == 'none':
        return X, y
    else:
        raise ValueError("Unsupported imbalance handling method.")

# ------------------------------
# 2. Model Definition
# ------------------------------

class PrecipitationClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=3, dropout_rate=0.5):
        super(PrecipitationClassifier, self).__init__()
        layers = []
        current_input = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(current_input, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_input = hidden_size
        layers.append(nn.Linear(hidden_size, 1))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ------------------------------
# 3. Training and Evaluation Functions
# ------------------------------

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20, patience=5):
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

# ------------------------------
# 4. Hyperparameter Optimization with Optuna
# ------------------------------

def objective(trial: Trial, X_train, X_val, y_train, y_val):
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
    model, history = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50, patience=10)
    
    # Evaluate on validation set
    val_f1 = evaluate_f1(model, X_val_scaled, y_val)
    
    return val_f1

def evaluate_f1(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32)
        logits = model(inputs)
        preds = (torch.sigmoid(logits) > 0.5).int().numpy().flatten()
        y_val_np = y_val.flatten()
        f1 = f1_score(y_val_np, preds, pos_label=1)
    return f1

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split into train and test
    X_train_full, X_test, y_train_full, y_test, scaler = split_and_scale(X, y, test_size=0.2)

    split_index = int(len(X_train_full) * 0.8)
    X_train, X_val = X_train_full[:split_index], X_train_full[split_index:]
    y_train, y_val = y_train_full[:split_index], y_train_full[split_index:]
    
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
        print("    {}: {}".format(key, value))
    
    # Retrain the model with the best hyperparameters on the full training set
    best_params = trial.params
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']

    hidden_size = 512
    num_layers = 4
    dropout_rate = 0.5858
    learning_rate = 0.0039958
    batch_size = 128
    
    # Handle class imbalance with SMOTE on full training data
    X_train_res, y_train_res = handle_class_imbalance(X_train_full, y_train_full, method='smote')
    
    # Scale the resampled data
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_res)
    X_test_scaled = scaler_final.transform(X_test)
    
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
    model, history = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=50, patience=10)
    
    # Evaluate on test set
    test_f1 = evaluate_f1(model, X_test_scaled, y_test)
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Optionally, plot the final confusion matrix
    evaluate_model(model, test_loader)
    
if __name__ == "__main__":
    main()
