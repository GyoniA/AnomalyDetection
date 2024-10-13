import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from imblearn.combine import SMOTEENN
from sdv.single_table import CTGANSynthesizer


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


def ctgan_resample(train_data, y_train, target_column, random_state, desired_fraud_ratio=0.01, epochs=100):
    start_time = datetime.now()
    current_fraud_ratio = y_train.sum() / len(train_data)

    numerator = desired_fraud_ratio * len(train_data) - y_train.sum()
    denominator = 1 - desired_fraud_ratio
    num_samples = int(np.ceil(numerator / denominator)) if denominator != 0 else 0

    if num_samples <= 0:
        print(
            f"Desired fraud ratio of {desired_fraud_ratio * 100}% is already achieved or cannot be achieved with the current data."
        )
        augmented_train = train_data
    else:
        ctgan_save_path = f'data/mlg-ulb/credit-card-fraud/ctgan_{epochs}_epochs.pkl'
        print(f"Number of synthetic fraud samples to generate: {num_samples}")
        ctgan_start_time = datetime.now()
        if os.path.exists(ctgan_save_path):
            print(f"Loading CTGAN from {ctgan_save_path}...")
            ctgan = joblib.load(ctgan_save_path)
            print(f"CTGAN loaded in {(datetime.now() - ctgan_start_time).seconds} seconds.")
        else:
            print(f"Training CTGAN...")
            metadata = Metadata.detect_from_dataframe(train_data)
            ctgan = CTGANSynthesizer(metadata=metadata, epochs=epochs, verbose=True)
            ctgan.fit(train_data)
            joblib.dump(ctgan, ctgan_save_path)
            print(f"CTGAN training completed in {(datetime.now() - ctgan_start_time).seconds} seconds.")
        ctgan_start_time = datetime.now()
        # Generate synthetic data
        synthetic_data = ctgan.sample(int(num_samples / current_fraud_ratio / 150))  # TODO: Check if this is too much
        synthetic_frauds = synthetic_data.query(f"{target_column} == 1")

        # Ensure that all synthetic samples are frauds
        num_frauds = synthetic_frauds[target_column].sum()
        print(f"Number of synthetic frauds generated: {num_frauds} in {(datetime.now() - ctgan_start_time).seconds} seconds.")
        if num_frauds > num_samples:
            print(f"Dropping synthetic frauds.")
            fraction = num_samples / num_frauds
            synthetic_frauds = synthetic_frauds.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
            num_frauds = synthetic_frauds[target_column].sum()
            print(f"Number of synthetic frauds after dropping: {num_frauds}")

        # Combine synthetic frauds with the original training data
        augmented_train = pd.concat([train_data, synthetic_frauds], ignore_index=True)

        # Shuffle the augmented training data
        augmented_train = augmented_train.sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(
            f"Augmented training set size: {len(augmented_train)} samples, with {augmented_train[target_column].sum()} " +
            f"fraud cases ({100 * augmented_train[target_column].sum() / len(augmented_train):.2f}%)."
        )
        print(f"Data generation completed in {(datetime.now() - start_time).seconds} seconds.")

    # Separate features and target after augmentation
    x_train = augmented_train.drop(columns=[target_column])
    y_train = augmented_train[target_column]

    return x_train, y_train


def load_mlg_ulb_data(csv_path='data/mlg-ulb/credit-card-fraud/creditcard.csv', test_size=0.25, resampling=None, random_state=0):
    """
    Load and preprocess the "Credit Card Fraud Detection" dataset from mlg-ulb

    :param csv_path: Path to the credit card dataset CSV file
    :param test_size: Proportion of the dataset to reserve for testing (default is 0.25 for a 25% test set)
    :param resampling: Resampling method to use (default is None, which means no resampling), options are 'smote' and 'ctgan'
    :return: Preprocessed training and test sets (scaled), along with labels
    :param random_state: The random state to use for resampling
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

    target_column = 'Class'
    # Separate features and target ('Class' is the target for fraud detection)
    x_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    x_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    if resampling == 'smote':
        print("Applying SMOTE-ENN resampling to the training set...")
        smote_enn = SMOTEENN(random_state=random_state, sampling_strategy=0.015)
        x_train, y_train = smote_enn.fit_resample(x_train, y_train)
        print(f"SMOTE-ENN applied: Resampled training set size: {len(x_train)} samples, original training set size: {len(train_data)}")
        print(f"Resampling completed in {(datetime.now() - start_time).seconds} seconds.")

    if resampling == 'ctgan':
        print("Applying CTGAN-based data generation to the training set...")
        x_train, y_train = ctgan_resample(train_data, y_train, target_column, random_state)

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


def create_dataloader(x_train, x_test, batch_size=64, use_gpu=False):
    """
    Convert NumPy arrays to PyTorch tensors and then to DataLoaders
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor)
    test_dataset = TensorDataset(x_test_tensor)

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu, pin_memory_device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=use_gpu, pin_memory_device=device)

    return train_loader, test_loader


def create_classification_dataloader(x_train, x_test, y_train, y_test, batch_size=64):
    """
    Convert NumPy arrays to PyTorch tensors and then to DataLoaders for classification.

    :param x_train: Training features
    :param x_test: Test features
    :param y_train: Training labels
    :param y_test: Test labels
    :param batch_size: Batch size for DataLoaders
    :return: Training and test DataLoaders
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # Assuming y_train is a Pandas Series
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

