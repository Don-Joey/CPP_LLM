import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import random

def read_model_features(csv_path, mask=None):
    df = pd.read_csv(csv_path)
    df = df.iloc[:, 1:]
    encoder = OrdinalEncoder()
    if mask !=None:
        mask_id = None
        numerical_list = ['Pretraining Dataset Size(B)','Parameter Size(M)','GPUh','FLOPs','Layers','Number Heads','bottleneck activation size','Carbon Emission (tCO2Eq)']
        catogrical_list = ['family', 'Finetuning', 'Context window','batch size(M)']
        if mask in ['family', 'Finetuning', 'Context window','batch size(M)']:
            mask_id = catogrical_list.index(mask)+8
        elif mask in ['Pretraining Dataset Size(B)','Parameter Size(M)','GPUh','FLOPs','Layers','Number Heads','bottleneck activation size','Carbon Emission (tCO2Eq)']:
            mask_id = numerical_list.index(mask)
        encoded_data = encoder.fit_transform(df[catogrical_list])
        # Get feature names from encoder
        feature_names = encoder.get_feature_names_out(catogrical_list)
        # Create DataFrame with correct feature names
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
        
        # Check the DataFrame
        for column in numerical_list:
            df[column] = np.log1p(df[column])
        final_df = pd.concat([df[numerical_list], encoded_df], axis=1)
        final_matrix = final_df.values
        array_without_nans = np.nan_to_num(final_matrix, nan=0.0)
        array_without_nans[:, mask_id].fill(0.)
    else:
        encoded_data = encoder.fit_transform(df[['family', 'Finetuning', 'Context window','batch size(M)']])
        # Get feature names from encoder
        feature_names = encoder.get_feature_names_out(['family', 'Finetuning', 'Context window','batch size(M)'])
        # Create DataFrame with correct feature names
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

        # Check the DataFrame
        for column in ['Pretraining Dataset Size(B)','Parameter Size(M)','GPUh','FLOPs','Layers','Number Heads','bottleneck activation size','Carbon Emission (tCO2Eq)']:
            df[column] = np.log1p(df[column])
        final_df = pd.concat([df[['Pretraining Dataset Size(B)','Parameter Size(M)','GPUh','FLOPs','Layers','Number Heads','bottleneck activation size','Carbon Emission (tCO2Eq)']], encoded_df], axis=1)
        final_matrix = final_df.values
        array_without_nans = np.nan_to_num(final_matrix, nan=0.0)
    return array_without_nans

def read_benchmark_features(csv_path, mask=None):
    df = pd.read_csv(csv_path)
    df = df.iloc[:, 1:]
    df = df.T
    encoder = OrdinalEncoder()
    if mask!=None:
        encoded_data = encoder.fit_transform(df[[0, 1, 2, 3]])
        # Get feature names from encoder
        feature_names = encoder.get_feature_names_out([0, 1, 2,3])
        # Create DataFrame with correct feature names
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
        buffer =encoded_df.values
        buffer[:, mask].fill(0)
        return buffer
    else:
        encoded_data = encoder.fit_transform(df[[0, 1, 2,3]])
        # Get feature names from encoder
        feature_names = encoder.get_feature_names_out([0, 1, 2,3])
        # Create DataFrame with correct feature names
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
    return encoded_df.values



def train_test_shapley_each_model_split(non_nan_indices, random_state, test_size, mask_model_indice, valid_model_indice):
    validate_positions = []
    train_positions = []
    model_indices = {}
    for indice in non_nan_indices:
        if indice[0] not in model_indices.keys():
            model_indices[indice[0]] = [indice[1]]
        else:
            model_indices[indice[0]].append(indice[1])
    random.seed(random_state)
    random.shuffle(model_indices[valid_model_indice])
    sample_model_indice =  model_indices[valid_model_indice][int(test_size):]
    for indice in non_nan_indices:
        if indice[1] in sample_model_indice and indice[0] == valid_model_indice:
            validate_positions.append(indice)
        elif indice[0] != mask_model_indice:
            train_positions.append(indice)
    return train_positions, validate_positions

def train_test_shapley_each_benchmark_split(non_nan_indices, random_state, test_size, mask_benchmark_indice, valid_benchmark_indice):
    validate_positions = []
    train_positions = []
    benchmark_indices = {}
    for indice in non_nan_indices:
        if indice[1] not in benchmark_indices.keys():
            benchmark_indices[indice[1]] = [indice[0]]
        else:
            benchmark_indices[indice[1]].append(indice[0])
    random.seed(random_state)
    random.shuffle(benchmark_indices[valid_benchmark_indice])
    sample_model_indice =  benchmark_indices[valid_benchmark_indice][int(test_size):]
    for indice in non_nan_indices:
        if indice[0] in sample_model_indice and indice[1] == valid_benchmark_indice:
            validate_positions.append(indice)
        elif indice[1] != mask_benchmark_indice:
            train_positions.append(indice)
    return train_positions, validate_positions


def train_test_large_model_split(non_nan_indices, random_state, have_size, valid_model_indice):
    validate_positions = []
    train_positions = []
    model_indices = {}
    for indice in non_nan_indices:
        if indice[0] not in model_indices.keys():
            model_indices[indice[0]] = [indice[1]]
        else:
            model_indices[indice[0]].append(indice[1])
    random.seed(random_state)
    random.shuffle(model_indices[valid_model_indice])
    sample_model_indice =  model_indices[valid_model_indice][int(have_size):]
    for indice in non_nan_indices:
        if indice[1] in sample_model_indice and indice[0] == valid_model_indice:
            validate_positions.append(indice)
        else:
            train_positions.append(indice)
    return train_positions, validate_positions
