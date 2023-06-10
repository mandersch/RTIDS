import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
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

        # Encode Labels
        print("Encoding Labels")
        data['Label'] = data['Label'].apply(lambda x: 'Attack' if x != 'BENIGN' else x)

        print("Beginning SMOTE")


        # Separate features and labels
        features = data.drop('Label', axis=1)
        labels = data['Label']

        features, labels = shuffle(features, labels, random_state=42)

        # Define the incremental SMOTE parameters
        nn_estimator = NearestNeighbors(n_neighbors=5, n_jobs=-1)
        smote = SMOTE(k_neighbors=nn_estimator, random_state=42)
        under_sampler = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
        batch_size = 10000

        # Create empty arrays to store balanced data
        balanced_features = np.empty((0, features.shape[1]))
        balanced_labels = np.empty((0,))

        # Perform incremental SMOTE
        for i in range(0, features.shape[0], batch_size):
            # Get a batch of samples
            batch_features = features[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Perform random undersampling on the combined samples
            sampled_features, sampled_labels = under_sampler.fit_resample(batch_features, batch_labels)
            
            # Perform SMOTE on the batch
            synthetic_features, synthetic_labels = smote.fit_resample(sampled_features, sampled_labels)

            # Combine the sampled data with previously balanced data
            balanced_features = np.vstack((balanced_features, synthetic_features))
            balanced_labels = np.concatenate((balanced_labels, synthetic_labels))

        # Shuffle the balanced data
        balanced_features, balanced_labels = shuffle(balanced_features, balanced_labels, random_state=42)

        # Convert balanced data back to pandas DataFrame
        balanced_data = pd.DataFrame(data=balanced_features, columns=features.columns)
        balanced_data['Label'] = balanced_labels
        data = balanced_data
        print("SMOTE Done")

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
