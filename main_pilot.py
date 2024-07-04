import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import fire

from src.mf import MatrixFactorization
from src.ncf import NeuralCollaborativeFiltering

class SparseMatrixDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

def train(model, optimizer, criterion, data_loader, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for user, item, rating in data_loader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        if epoch % 1000 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')
    
#normalization
def normalize_scores(scores):
    """
    Normalize the scores in each row of the scores matrix to the range [0, 1].
    
    Parameters:
    - scores: A 2D numpy array where rows represent benchmarks and columns represent models.
    
    Returns:
    - A 2D numpy array with the scores normalized within each row.
    """
    # Find the min and max values in each row
    min_values = scores.min(axis=1, keepdims=True)
    max_values = scores.max(axis=1, keepdims=True)
    
    # Normalize the scores within each row
    normalized_scores = (scores - min_values) / (max_values - min_values)
    
    return normalized_scores, max_values - min_values, min_values

# Example sparse data (user_ids, item_ids, ratings for observed entries)
def read_data(csv_path, random_state, mask_size):

    score_data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)[:, 1:]
    non_nan_indices = np.argwhere(~np.isnan(score_data))
    np.random.shuffle(non_nan_indices)
    sample_size = int(mask_size * len(non_nan_indices))
    # Select the first 'sample_size' elements after shuffling
    train_positions, validate_positions = train_test_split(non_nan_indices[:sample_size], random_state=random_state, test_size=0.1)
    validate_positions = np.concatenate((validate_positions, non_nan_indices[sample_size:]))
    original_nan_positions = np.isnan(score_data)
    array_temp = np.where(original_nan_positions, 0, score_data)
    normalized_array, scale, min_value = normalize_scores(array_temp)
    normalized_array[original_nan_positions] = np.nan
    score_data = normalized_array
    train_array_positions = [pos.tolist() for pos in train_positions]
    validate_array_positions = [pos.tolist() for pos in validate_positions]
    return score_data, train_array_positions, validate_array_positions, scale, min_value

def load_data(R, subset, scale, min_value):
    user_ids = []
    item_ids = []
    ratings = []
    scales_list = []
    min_values_list = []
    for i in range(len(R)):
        for j in range(len(R[i])):
            if not np.isnan(R[i][j]) and np.any(np.all(np.array([i,j]) == subset, axis=1)): 
                user_ids.append(i)
                item_ids.append(j)
                ratings.append(R[i][j])
                scales_list.append(scale[i])
                min_values_list.append(min_value[i])
    print("Load Finished")
    return torch.LongTensor(user_ids), torch.LongTensor(item_ids), torch.FloatTensor(ratings), scales_list, min_values_list



def create_score_table(data):
    """
    Creates a table where rows are item_ids, columns are user_ids, and
    cells contain 'label_score/predicted_score'.

    :param data: List of tuples (user_id, item_id, label_score, predicted_score)
    :return: Pandas DataFrame
    """
    # Create a dictionary where keys are item_ids and values are dictionaries,
    # where the keys of these sub-dictionaries are user_ids and values are the score strings
    table_dict = {}
    
    for user_id, item_id, label_score, predicted_score, scale, min_value in data:
        if user_id not in table_dict:
            table_dict[user_id] = {}
        # Format the cell value as 'label_score/predicted_score'
        table_dict[user_id][item_id] = f"{label_score*scale+min_value}/{predicted_score*scale+min_value}"
        
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(table_dict, orient='index').sort_index()
    # Optionally, fill NaN with some placeholder if any cells are empty
    df.fillna('-', inplace=True)
    df_transposed = df.T
    df_transposed = df_transposed.sort_index()
    df = df_transposed.T
    
    return df


def evaluate(model, valid_data_loader, criterion, valid_user_ids, valid_item_ids, valid_ratings, device, valid_scales, valid_min_values, model_name, random_state, mask_size):
    #MSE Loss
    total_loss = 0
    for user, item, rating in valid_data_loader:
        user = user.to(device)
        item = item.to(device)
        rating = rating.to(device)
        prediction = model(user, item)
        loss = criterion(prediction, rating)
        total_loss += loss.item()
    print("validate_loss", total_loss/len(valid_data_loader))
    #Visualization
    extracted_data = []
    for user_id, item_id, rating, scale, min_value in zip(valid_user_ids, valid_item_ids, valid_ratings, valid_scales, valid_min_values):
        predicted_scores = model(user_id.unsqueeze(0).to(device), item_id.unsqueeze(0).to(device))
        extracted_data.append((user_id.item(), item_id.item(), predicted_scores.item(), rating.item(), scale, min_value))
    df = create_score_table(extracted_data)
    df.to_csv('pilot_'+model_name+"-"+str(random_state)+"-"+str(int(mask_size*100))+"-factor-10"+'.csv')
    

def main(data_name, mask_size, model_name, random_state):
    data_save_root = "./data/"
    model_save_root = "./model/"
    if data_name == "helm_core":
        csv_path = data_save_root+"/helm_core.csv"

    random_states = [random_state]
    mask_size = mask_size
    for random_state in random_states:
        score_data, train_array_positions, validate_array_positions, scale, min_value = read_data(csv_path=csv_path, random_state=random_state, mask_size=mask_size)
        user_ids, item_ids, ratings, _, _ = load_data(score_data, train_array_positions, scale, min_value)
        valid_user_ids, valid_item_ids, valid_ratings, valid_scales, valid_min_values = load_data(score_data, validate_array_positions, scale, min_value)

        # Hyperparameters
        learning_rate = 0.01
        num_epochs = 200000
        num_models = len(score_data)   # example number of users
        num_tasks = len(score_data[0])   # example number of items
        num_factors = 10   # number of latent factors


        # Create a DataLoader for sparse matrix
        dataset = SparseMatrixDataset(user_ids, item_ids, ratings)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        valid_dataset = SparseMatrixDataset(valid_user_ids, valid_item_ids, valid_ratings)
        valid_data_loader = DataLoader(valid_dataset,  batch_size=16, shuffle=False)

        # Initialize model, optimizer, and loss function
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name == "matrix_factorization":
            model = MatrixFactorization(num_models, num_tasks, num_factors).to(device)
        elif model_name == "neural_collaborative_filtering":
            model = NeuralCollaborativeFiltering(num_models, num_tasks, num_factors).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate) #AdamW
        criterion = nn.MSELoss()

        # Train the model
        train(model, optimizer, criterion, data_loader, num_epochs, device)
        if model_name == "matrix_factorization":
            torch.save(model.state_dict(), model_save_root+"mf/"+data_name+"-"+model_name+"-"+str(num_epochs)+"-"+str(random_state)+"-"+str(learning_rate)[2:]+"-"+str(num_factors)+"-SGD-mask"+str(int(mask_size*100))+'.pth')
        elif model_name == "neural_collaborative_filtering":
            torch.save(model.state_dict(), model_save_root+"ncf/"+data_name+"-"+model_name+"-"+str(num_epochs)+"-"+str(random_state)+"-"+str(learning_rate)[2:]+"-"+str(num_factors)+"-SGD-mask"+str(int(mask_size*100))+'.pth')

        # Example prediction
        evaluate(model, valid_data_loader, criterion, valid_user_ids, valid_item_ids, valid_ratings, device, valid_scales, valid_min_values, model_name, random_state)
    
if __name__ == '__main__':
    fire.Fire(main)