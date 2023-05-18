# Robust Trasformer Based Intrusion Detection

This is an Implementation of a Transformer Based Network Intrusion Detection AI based on the Paper by Wu et Al.

## Training / Test Data

It is recommended to download the [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) into a folder named ```data``` in the root directory of the Project. This Dataset is the main source of training and validation data for this project.

## Requirements

You will need to install PyTorch to run this Project. Currently Cuda is not yet supported, however this will likely be updated later on.

## Usage

Running ```python main.py``` will automatically preprocess the Dataset and start the Training. This will be removed in the future in favor of seperate methods to Preprocess and store the data, Train the model and Evaluate the model. 