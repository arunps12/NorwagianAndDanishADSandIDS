import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import make_scorer

from tqdm import tqdm 
import torch.nn.init as init

import optuna
from optuna.samplers import TPESampler

import logging

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Device configuration
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
    def __init__(self, input_dim, num_classes, conv_out_channels, fc_hidden_units, kernel_sizes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_out_channels[0], kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=conv_out_channels[0], out_channels=conv_out_channels[1], kernel_size=kernel_sizes[1])
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size of the input to the first fully connected layer
        conv_out_dim = input_dim
        
        # Apply conv1
        conv_out_dim = (conv_out_dim - kernel_sizes[0]) + 1  # Conv1
        
        # Apply conv2 followed by pooling
        conv_out_dim = (conv_out_dim - kernel_sizes[1]) + 1  # Conv2
        conv_out_dim = (conv_out_dim - 2) // 2 + 1  # Pool2

        # Ensure conv_out_dim is positive before defining the fully connected layer
        assert conv_out_dim > 0, f'Calculated dimension {conv_out_dim} is too small for the FC layer.'
        
        self.fc1 = nn.Linear(conv_out_channels[1] * conv_out_dim, fc_hidden_units)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc_hidden_units, num_classes)
        
        # Initialize layers
        self._initialize_weights()

    def forward(self, x):
        #print(f'Input size: {x.size()}')  # Log the input size
        x = F.relu(self.conv1(x))
        #print(f'After conv1: {x.size()}')  # Log size after conv1
        x = self.pool(F.relu(self.conv2(x)))
        #print(f'After conv2: {x.size()}')  # Log size after conv2
        x = x.view(x.size(0), -1)
        #print(f'After flatten: {x.size()}')  # Log size after flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        # Initialize conv1 and conv2 with Kaiming initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)

        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)

        # Initialize fc1 and fc2 with Xavier initialization
        nn.init.xavier_normal_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
            
        nn.init.xavier_normal_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

# Custom Estimator Wrapper
class CNNWrapper:
    def __init__(self, input_dim, num_class, batch_size=32, learning_rate=0.001, num_epochs=20, conv_out_channels=[32, 64], fc_hidden_units=128, kernel_sizes = [2, 2]):
        self.input_dim = input_dim
        self.num_class = num_class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.conv_out_channels = conv_out_channels
        self.fc_hidden_units = fc_hidden_units
        self.kernel_sizes = kernel_sizes
        self.model = CNNModel(
            input_dim=input_dim,
            num_classes=num_class,
            conv_out_channels=conv_out_channels,
            fc_hidden_units=fc_hidden_units,
            kernel_sizes = kernel_sizes
        ).to(device)

    def fit(self, X, y):
        train_dataset = SpeechDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}', leave=False)
            for features, labels in progress_bar:
                features = features.unsqueeze(1).to(device)  # Adding channel dimension
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def score(self, X, y):
        test_dataset = SpeechDataset(X, y)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.unsqueeze(1).to(device)
                labels = labels.to(device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        return f1_score(y_true, y_pred, average='macro')

# Optuna objective function
def create_objective_function(input_dim, num_class, X_train, y_train, X_test, y_test, kernel_sizes):
    def objective(trial):
        # Hyperparameters to optimize
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        num_epochs = trial.suggest_int('num_epochs', 10, 50)
        conv_out_channels = [trial.suggest_categorical('conv1_out_channels', [16, 32, 64]), trial.suggest_categorical('conv2_out_channels', [32, 64, 128])]
        fc_hidden_units = trial.suggest_categorical('fc_hidden_units', [64, 128, 256])

        # Create CNNWrapper instance with the suggested hyperparameters
        model_wrapper = CNNWrapper(
            input_dim=input_dim,
            num_class=num_class,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            conv_out_channels=conv_out_channels,
            fc_hidden_units=fc_hidden_units,
            kernel_sizes = kernel_sizes
        )

        # Train the model
        model_wrapper.fit(X_train, y_train)

        # Evaluate the model
        f1 = model_wrapper.score(X_test, y_test)
        return f1
    return objective

# Function to load speaker-specific data and compute F1 score with CNN
def cnn_load_data_compute_f1score_for_speaker_age(train_csv_file_path, test_csv_file_path, feature_column_names, label_column_name, kernel_sizes):
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

    # Iterate over each speaker and age group
    for (spkid, age), group in df_train.groupby(['spkid', 'AgeMonth']):
        X_train = group[feature_column_names].values
        y_train = group[label_column_name].values
        y_train_encoded = le.transform(y_train)

        # Ensure y_train_encoded has the same number of classes as num_class
        train_classes_in_group = set(y_train_encoded)
        if len(train_classes_in_group) != num_class:
            print(f"Skipping speaker {spkid}, age {age} due to missing classes.")
            continue

        input_dim = len(feature_column_names)
        #print(input_dim, num_class)
        X_train = X_train
        y_train = y_train_encoded
        X_test = X_test
        y_test = y_test_encoded
        kernel_sizes = kernel_sizes
        
        # Optimize hyperparameters using Optuna
        objective = create_objective_function(input_dim, num_class, X_train, y_train, X_test, y_test, kernel_sizes)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=20)

        print(f'Best trial for speaker {spkid}, age {age}: {study.best_trial.params}')
        print(f'Best F1 score for speaker {spkid}, age {age}: {study.best_value:.4f}')

        results.append({
            'spkid': spkid,
            'AgeMonth': age,
            '#train_samples': X_train.shape[0],
            'F1_Score': "{:.4f}".format(study.best_value)
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.dropna(inplace=True)
    return results_df

