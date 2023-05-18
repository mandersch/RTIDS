import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, KBinsDiscretizer

# Define custom dataset class
class CICIDSDataset(Dataset):
    def __init__(self, data):
        self.features = data[:, :-1]  # All columns except the last one
        self.labels = data[:, -1]  # Last column (labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = self.labels[idx]

        return feature, label

def load_data(vocab_size=10000):

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

    # Encode labels
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])

    # Convert string values to float64
    for column in data.columns:
        if data[column].dtype == object:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna()
    data = data.clip(lower=-1e6, upper=1e6)


    # Select the columns to bin (exclude the last column)
    columns_to_bin = data.columns[:-1]
    # Define the number of desired bins for the embedding
    bucket_size = vocab_size  
    # Initialize the KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=bucket_size, encode='ordinal', strategy='uniform')
    # Perform binning on each column
    binned_data = data.copy()

    for column in columns_to_bin:
        column_array = np.array(data[column]).reshape(-1, 1)
        binned_values = discretizer.fit_transform(column_array).flatten()
        binned_data[column] = binned_values #/ bucket_size

    # Convert the dataset to a NumPy array
    dataset = binned_data.values

    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    return train_data, val_data

def get_data_loader(data, batch_size):

    # Create an instance of the custom dataset
    cicids_dataset = CICIDSDataset(data)
    
    return DataLoader(cicids_dataset, batch_size=batch_size, shuffle=True)
