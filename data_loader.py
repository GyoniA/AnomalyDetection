from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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


def load_pointe77_data(drop_string_columns=True, limit=None):
    """
    Load the 'credit-card-transaction' dataset from 'pointe77'
    @param drop_string_columns: Drop columns that have string values
    @param limit: Limit the number of rows to speed up training during development
    """
    # Load the train and test datasets
    start_time = datetime.now()
    print("Loading datasets...")
    train_path = 'data/pointe77/credit-card-transaction/credit_card_transaction_train.csv'
    test_path = 'data/pointe77/credit-card-transaction/credit_card_transaction_test.csv'

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
