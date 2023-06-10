import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import os

# Define custom dataset class
class CICIDSDataset(Dataset):
    def __init__(self, data):
        self.features = data[:, :-2]  # All columns except the last one
        self.labels = data[:, -2:]  # Last column (labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = self.labels[idx]

        return feature, label

def load_data():
    if os.path.exists("data/preprocessed/data.csv.gz"):
        print("Loading Preprocessed Data")
        data = pd.read_csv('data/preprocessed/data.csv.gz', compression='gzip')
    
    else:
        # Path to the directory containing CICIDS 2017 dataset CSV files
        directory_path = 'data/'
        # Get a list of all CSV files in the directory
        csv_files = glob.glob(directory_path + '*.csv')

        # Initialize an empty list to store individual DataFrames
        dataframes = []
        # Load each CSV file and append its DataFrame to the list
        for file in csv_files:
            dataframe = pd.read_csv(file)
            dataframes.append(dataframe)
        # Concatenate all DataFrames into a single dataset
        data = pd.concat(dataframes, ignore_index=True)
        data.columns = data.columns.str.strip()

        data = data.dropna()
        # Set values larger than float64 maximum to maximum float64 value
        max_float64 = np.finfo(np.float64).max


        features = data.drop('Label', axis=1)
        print("Removing large Values")
        features = features.where(features <= max_float64, max_float64)
        labels = data['Label']
        data = pd.concat([features, labels], axis=1)
        # print("Beginning SMOTE")
        # # Split the dataset by label
        # benign_data = data[data['Label'] == 'Benign']
        # attack_data = data[data['Label'] != 'Benign']

        # # Create a nearest neighbor estimator with n_jobs
        # nn_estimator = NearestNeighbors(n_neighbors=5, n_jobs=-1)

        # # Apply SMOTE to the subset containing the minority classes
        # smote = SMOTE(random_state=42, k_neighbors=nn_estimator)
        # features_minority, labels_minority = smote.fit_resample(attack_data.drop('Label', axis=1), attack_data['Label'])

        # # Combine the majority class (benign_data) with the oversampled minority class
        # balanced_data = pd.concat([benign_data, pd.DataFrame(features_minority, columns=attack_data.drop('Label', axis=1).columns)], axis=0)
        # balanced_data['Label'] = pd.concat([benign_data['Label'], pd.Series(labels_minority)])

        # print("SMOTE Done")

        # Encode Labels
        print("Encoding Labels")
        data['Label'] = data['Label'].apply(lambda x: 'Attack' if x != 'BENIGN' else x)
        # Perform one-hot encoding and drop the original "Labels" column
        encoded_labels = pd.get_dummies(data['Label'], prefix='', prefix_sep='')
        data = pd.concat([data.drop('Label', axis=1), encoded_labels], axis=1)

        scaler = MinMaxScaler()
        print("Min Max Scaling")
        scaled_data = scaler.fit_transform(data)
        data = pd.DataFrame(data=scaled_data, columns=data.columns)
        print("Saving Data to File")
        data.to_csv('data/preprocessed/data.csv.gz', index=False, compression='gzip')

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_data.values, val_data.values

    return train_data, val_data

def get_data_loader(data, batch_size):

    # Create an instance of the custom dataset
    cicids_dataset = CICIDSDataset(data)
    
    return DataLoader(cicids_dataset, batch_size=batch_size, shuffle=True)
