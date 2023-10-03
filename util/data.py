import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os

label_mapping = {}

class CICIDSDataset(Dataset):
    def __init__(self, data):
        self.features = data[:, :-3]  # Feature Columns
        self.at_type = data[:, -3] # Attack Type Column
        self.labels = data[:, -2:]  # 1 Hot Encoded Label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = self.labels[idx]
        at_type = self.at_type[idx]

        return feature, label, at_type

def load_data():
    if os.path.exists("data/preprocessed/data.csv.gz"):
        print("Loading Preprocessed Data")
        data = pd.read_csv('data/preprocessed/data.csv.gz', compression='gzip')
    
    else:
        directory_path = 'data/' # Path to the directory containing CICIDS 2017 dataset CSV files
        csv_files = glob.glob(directory_path + '*.csv')

        dataframes = []
        for file in csv_files:
            dataframe = pd.read_csv(file)
            dataframes.append(dataframe)
        data = pd.concat(dataframes, ignore_index=True)
        data.columns = data.columns.str.strip()

        data = data.dropna()
        max_float64 = np.finfo(np.float64).max

        features = data.drop('Label', axis=1)
        features = features.where(features <= max_float64, max_float64)
        labels = data['Label']
        data = pd.concat([features, labels], axis=1)

        # Undersampling & SMOTE

        max_class_size = 100000 # Size of all Classes for Undersampling
        class_counts = data['Label'].value_counts()
        classes_to_undersample = class_counts[class_counts > max_class_size].index

        under_sampler = RandomUnderSampler(sampling_strategy={
                label: 7*max_class_size if label =="BENIGN" else max_class_size if label in classes_to_undersample else class_counts[label] for label in np.unique(data['Label'])
            }, random_state=42)
        nn_estimator = NearestNeighbors(n_neighbors=5, n_jobs=-1)
        smote = SMOTE(sampling_strategy={
                label: 7*max_class_size if label =="BENIGN" else max_class_size for label in np.unique(data['Label'])
            }, k_neighbors=nn_estimator, random_state=42)
        scaler = MinMaxScaler()

        features = data.drop('Label', axis=1)
        labels = data['Label']
        sampled_features, sampled_labels = under_sampler.fit_resample(features, labels)
        balanced_features, balanced_labels = smote.fit_resample(sampled_features, sampled_labels)
        scaled_data = scaler.fit_transform(balanced_features)

        data = pd.DataFrame(data=scaled_data, columns=features.columns)

        # Encode the actual Attack type for later analysis
        label_encoder = LabelEncoder()
        data['Label'] = label_encoder.fit_transform(balanced_labels)
        label_mapping = dict(enumerate(label_encoder.classes_))

        # 1 Hot Encoding
        data['Attack_Label'] = data['Label'].apply(lambda x: "Attack" if label_mapping[x] != "BENIGN" else "BENIGN")
        encoded_labels = pd.get_dummies(data['Attack_Label'], prefix='', prefix_sep='')
        data = pd.concat([data, encoded_labels], axis=1)
        data.drop("Attack_Label", axis=1, inplace=True)
        
        label_counts = data['Label'].value_counts()
        print(label_counts)
        data.to_csv('data/preprocessed/data.csv.gz', index=False, compression='gzip')
        print("saved")

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_data.values, val_data.values

    return train_data, val_data

def get_data_loader(data, batch_size):
    cicids_dataset = CICIDSDataset(data)
    
    return DataLoader(cicids_dataset, batch_size=batch_size, shuffle=True)
