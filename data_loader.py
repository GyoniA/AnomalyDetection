from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from imblearn.combine import SMOTEENN


def process_dates(df):
    """
    Process the 'trans_date_trans_time' column (convert date to useful numerical features)
    """
    # We'll extract month, day, hour minute second from the date to get more usable information
    # Format: YYYY-MM-DD HH:MM:SS
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
    df['month'] = df['trans_date_trans_time'].dt.month
    df['day'] = df['trans_date_trans_time'].dt.day
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['minute'] = df['trans_date_trans_time'].dt.minute
    df['second'] = df['trans_date_trans_time'].dt.second
    df = df.drop(columns=['trans_date_trans_time'])  # Drop the original date column after processing
    return df


def process_dob(df):
    """
    Process the 'dob' column (convert date of birth to useful numerical features)
    """
    # We'll extract year, month and day from the date to get more usable information
    # Format: YYYY-MM-DD
    df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d')
    df['dob_year'] = df['dob'].dt.year
    df['dob_month'] = df['dob'].dt.month
    df = df.drop(columns=['dob'])  # Drop the original date of birth column after processing
    return df


def load_pointe77_data(drop_string_columns=True, limit=None, path='data/pointe77/credit-card-transaction/'):
    """
    Load the 'credit-card-transaction' dataset from 'pointe77'

    :param drop_string_columns: Drop columns that have string values
    :param limit: Limit the number of rows to speed up training during development
    :param path: Path to the dataset files
    """
    # Load the train and test datasets
    start_time = datetime.now()
    print("Loading datasets...")
    train_path = path + 'credit_card_transaction_train.csv'
    test_path = path + 'credit_card_transaction_test.csv'

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if limit:
        # Limit training/testing data, to speed up training during development
        train_data = train_data.head(limit)
        test_data = test_data.head(limit)

    # Drop unnecessary columns
    if drop_string_columns:
        drop_columns = ['Unnamed: 0', 'cc_num', 'trans_num', 'first', 'last', 'street']
    else:
        drop_columns = ['Unnamed: 0', 'cc_num', 'trans_num']
    train_data = train_data.drop(columns=drop_columns)
    test_data = test_data.drop(columns=drop_columns)

    train_data = process_dates(train_data)
    train_data = process_dob(train_data)

    test_data = process_dates(test_data)
    test_data = process_dob(test_data)

    # Encode categorical columns
    categorical_columns = ['merchant', 'category', 'gender', 'city', 'state', 'job']

    for column in categorical_columns:
        # Combine train and test columns to ensure consistency of encoding
        combined_data = pd.concat([train_data[column], test_data[column]], axis=0)
        # Convert to categorical type and encode as numerical
        train_data[column] = pd.Categorical(train_data[column], categories=combined_data.unique()).codes
        test_data[column] = pd.Categorical(test_data[column], categories=combined_data.unique()).codes

    # Handle missing values (NaNs)
    # Impute numerical features with the mean and categorical features with the most frequent value
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Separate numeric and categorical columns
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    available_categorical_columns = train_data.select_dtypes(
        include=['category', 'object']).columns.tolist()  # Refresh the list

    # Apply imputers to both train and test data if the columns exist
    if numeric_columns:
        train_data[numeric_columns] = num_imputer.fit_transform(train_data[numeric_columns])
        test_data[numeric_columns] = num_imputer.transform(test_data[numeric_columns])

    if available_categorical_columns:  # Only apply imputation if there are categorical columns
        train_data[available_categorical_columns] = cat_imputer.fit_transform(train_data[available_categorical_columns])
        test_data[available_categorical_columns] = cat_imputer.transform(test_data[available_categorical_columns])

    # Separate features and target ('is_fraud' is the target)
    x_train = train_data.drop(columns=['is_fraud'])
    y_train = train_data['is_fraud']

    x_test = test_data.drop(columns=['is_fraud'])
    y_test = test_data['is_fraud']

    # Normalize the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print(f"Data loading and preprocessing completed in {(datetime.now() - start_time).seconds} seconds")
    return x_train_scaled, x_test_scaled, y_train, y_test


def load_mlg_ulb_data(csv_path='data/mlg-ulb/credit-card-fraud/creditcard.csv', test_size=0.25, apply_smote_enn=True):
    """
    Load and preprocess the "Credit Card Fraud Detection" dataset from mlg-ulb

    :param csv_path: Path to the credit card dataset CSV file
    :param test_size: Proportion of the dataset to reserve for testing (default is 0.25 for a 25% test set)
    :param apply_smote_enn: Whether to apply SMOTE-ENN to the train dataset or not
    :return: Preprocessed training and test sets (scaled), along with labels
    """
    # Load the dataset
    start_time = datetime.now()
    print("Loading the credit card dataset...")

    data = pd.read_csv(csv_path)

    # Perform time-based split based on the 'Time' feature, to mirror real-world data availability
    data = data.sort_values(by='Time').reset_index(drop=True)
    split_index = int(len(data) * (1 - test_size))

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    # Separate features and target ('Class' is the target for fraud detection)
    x_train = train_data.drop(columns=['Class'])
    y_train = train_data['Class']

    x_test = test_data.drop(columns=['Class'])
    y_test = test_data['Class']

    if apply_smote_enn:
        print("Applying SMOTE-ENN resampling to the training set...")
        smote_enn = SMOTEENN(random_state=0, sampling_strategy=0.015)
        x_train, y_train = smote_enn.fit_resample(x_train, y_train)
        print(f"SMOTE-ENN applied: Resampled training set size: {len(x_train)} samples, original training set size: {len(train_data)}")
        print(f"Resampling completed in {(datetime.now() - start_time).seconds} seconds.")

    # Normalize the features (standardize to zero mean and unit variance)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Report distribution of anomalies in train and test sets
    train_fraud_count = y_train.sum()
    test_fraud_count = y_test.sum()

    print(f"Data loaded and processed in {(datetime.now() - start_time).seconds} seconds.")
    print(
        f"Training set: {len(x_train)} samples, with {train_fraud_count} fraud cases ({100 * train_fraud_count / len(y_train):.4f}%)."
    )
    print(
        f"Test set: {len(x_test)} samples, with {test_fraud_count} fraud cases ({100 * test_fraud_count / len(y_test):.4f}%)."
    )

    return x_train_scaled, x_test_scaled, y_train, y_test


def create_dataloader(x_train, x_test, batch_size=64):
    """
    Convert NumPy arrays to PyTorch tensors and then to DataLoaders
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor)
    test_dataset = TensorDataset(x_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
