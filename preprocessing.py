from typing import final
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# NSL-KDD dataset files
CSV_FILES: final = [
    'KDDTrain+.txt',
    'KDDTest+.txt'
]

# NSL-KDD features
FEATURES: final = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# Features to standardize
FEATURES_TO_STANDARDIZE: final = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Mapping attack categories to numerical labels
MAPPING: final = {
    "normal": 0,
    "neptune": 1,
    "smurf": 2,
    "guess_passwd": 3,
    "buffer_overflow": 4,
    "back": 5,
    "satan": 6,
    "portsweep": 7,
    "warezclient": 8,
    "teardrop": 9,
    # Add other attack categories as needed
}

def pre_processing(dataset_path: str):
    # Init global dataframe
    df = pd.DataFrame(columns=FEATURES)

    # Create global df from each local df
    for csv_file in CSV_FILES:
        df_local = pd.read_csv(os.path.join(dataset_path, csv_file), header=None, low_memory=False)
        df_local.columns = FEATURES
        df = pd.concat([df, df_local])

    # Apply one-hot encoding on categorical features
    # Protocol type
    df['protocol_type'] = df['protocol_type'].apply(lambda x: x.lower())
    for value in ['tcp', 'udp', 'icmp']:
        df[f'protocol_{value}'] = df.apply(lambda row: 1 if row['protocol_type'] == value else 0, axis=1)
    df = df.drop('protocol_type', axis='columns')

    # Service
    df['service'] = df['service'].apply(lambda x: x.lower())
    top_services = ['http', 'smtp', 'ftp', 'domain_u', 'eco_i']  # Add more as needed
    for value in top_services:
        df[f'service_{value}'] = df.apply(lambda row: 1 if row['service'] == value else 0, axis=1)
    df['service_other'] = df.apply(lambda row: 1 if row['service'] not in top_services else 0, axis=1)
    df = df.drop('service', axis='columns')

    # Flag
    df['flag'] = df['flag'].apply(lambda x: x.lower())
    for value in ['sf', 'rej', 'rstr']:
        df[f'flag_{value}'] = df.apply(lambda row: 1 if row['flag'] == value else 0, axis=1)
    df['flag_other'] = df.apply(lambda row: 1 if row['flag'] not in ['sf', 'rej', 'rstr'] else 0, axis=1)
    df = df.drop('flag', axis='columns')

    # Map attack categories to numerical labels
    df['label'] = df['label'].str.strip()
    df['label'] = df['label'].replace(np.nan, 'normal')
    df['label'] = df['label'].map(MAPPING)

    # Use Min Max Scaler
    scaler = MinMaxScaler()
    df[FEATURES_TO_STANDARDIZE] = scaler.fit_transform(df[FEATURES_TO_STANDARDIZE])

    # Create train, validation, and test sets
    train, val = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    train, test = train_test_split(train, test_size=0.25, shuffle=True, random_state=42)
    train.to_csv(os.path.join(dataset_path, 'NSL-KDD-train.csv'), index=False)
    val.to_csv(os.path.join(dataset_path, 'NSL-KDD-val.csv'), index=False)
    test.to_csv(os.path.join(dataset_path, 'NSL-KDD-test.csv'), index=False)

    # Create train, validation, and test sets for binary classification
    df['label'] = df['label'].apply(lambda x: 1 if x != 0 else 0)
    train, val = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    train, test = train_test_split(train, test_size=0.25, shuffle=True, random_state=42)
    train.to_csv(os.path.join(dataset_path, 'NSL-KDD-train-binary.csv'), index=False)
    val.to_csv(os.path.join(dataset_path, 'NSL-KDD-val-binary.csv'), index=False)
    test.to_csv(os.path.join(dataset_path, 'NSL-KDD-test-binary.csv'), index=False)


if __name__ == '__main__':
    pre_processing(os.path.join('dataset', 'raw'))
