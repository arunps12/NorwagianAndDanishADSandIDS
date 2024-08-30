import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import make_scorer
from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# xgboost model trainig
def train_test_xgboost(X_train, y_train, X_test, y_test, num_class):
    # Define the model
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_class)
    
    # Hyperparameters to tune
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # F1 score as the metric
    f1 = make_scorer(f1_score, average='macro')

    # Randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, scoring=f1, cv=3, random_state=42, error_score='raise')
    
    try:
        random_search.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during RandomizedSearchCV: {e}")
        raise

    # Best model
    best_model = random_search.best_estimator_

    # Evaluate on test data
    try:
        y_pred = best_model.predict(X_test)
        f1_test = f1_score(y_test, y_pred, average='macro')
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise

    return f1_test

# function for loading speaker-specific data by age for xgboost
def xgboost_load_data_compute_f1score_for_speaker_age(train_csv_file_path, test_csv_file_path, feature_column_names, label_column_name):
    df_train = pd.read_csv(train_csv_file_path)
    df_test = pd.read_csv(test_csv_file_path)
    X_test = df_test[feature_column_names].values
    y_test = df_test[label_column_name].values

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Combine y_train and y_test to fit the encoder
    combined_labels = pd.concat([pd.Series(df_train[label_column_name]), pd.Series(df_test[label_column_name])], ignore_index=True)
    le.fit(combined_labels)

    # Encode the labels
    y_train_encoded = le.transform(df_train[label_column_name])
    y_test_encoded = le.transform(y_test)

    # Number of classes in the training set
    train_classes = set(y_train_encoded)
    num_class = len(train_classes)

    # Filter out unseen classes in y_test
    mask = np.isin(y_test_encoded, list(train_classes))
    y_test_encoded = y_test_encoded[mask]
    X_test = X_test[mask]

    results = []
    for (spkid, age), group in df_train.groupby(['spkid', 'AgeMonth']):
        X_train = group[feature_column_names].values
        y_train = group[label_column_name].values
        y_train_encoded = le.transform(y_train)
        
        # Ensure y_train_encoded has the same number of classes as num_class
        train_classes_in_group = set(y_train_encoded)
        if len(train_classes_in_group) != num_class:
            print(f"Skipping speaker {spkid}, age {age} due to missing classes.")
            continue

        # Train and test the model
        f1_score_value = train_test_xgboost(X_train, y_train_encoded, X_test, y_test_encoded, num_class)
        results.append({
            'spkid': spkid,
            'AgeMonth': age,
            '#train_samples': X_train.shape[0],
            'F1_Score': "{:.4f}".format(f1_score_value)
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.dropna(inplace=True)
    return results_df
# Dataset class for CNN model
class SpeechDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# CNN model definition
class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * ((input_dim - 2) // 2 - 2) // 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# Training and Evaluation Functions for CNN:
def train_cnn(X_train, y_train, X_test, y_test, input_dim):
    train_dataset = SpeechDataset(X_train, y_train)
    test_dataset = SpeechDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNNModel(input_dim=input_dim, num_classes=6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            features, labels = batch
            features = features.unsqueeze(1).to(device)  # Adding channel dimension
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch
            features = features.unsqueeze(1).to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    f1_test = f1_score(y_true, y_pred, average='macro')
    return f1_test, model

