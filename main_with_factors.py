import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import fire

from src.ncf_f import EnhancedNeuralCollaborativeFiltering

class SparseMatrixDataset(Dataset):
    def __init__(self, model_ids, task_ids, ratings, bench_features, model_features):
        self.model_ids = model_ids
        self.task_ids = task_ids
        self.ratings = ratings
        self.bench_features = bench_features
        self.model_features = model_features
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.model_ids[idx], self.task_ids[idx], self.ratings[idx], self.bench_features[idx], self.model_features[idx]

def train(model, optimizer, criterion, data_loader, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for model, task, rating, model_feature, bench_feature in data_loader:
            model = model.to(device)
            task = task.to(device)
            rating = rating.to(device)
            bench_feature = bench_feature.to(device)
            model_feature = model_feature.to(device)
            
            optimizer.zero_grad()
            prediction = model(model, task, model_feature, bench_feature)
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

# Example sparse data (model_ids, task_ids, ratings for observed entries)
def read_data(csv_path, random_state, mask_size):

    score_data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)[:, 1:]
    non_nan_indices = np.argwhere(~np.isnan(score_data))
    train_positions, validate_positions = train_test_split(non_nan_indices, random_state=random_state, test_size=mask_size)
    original_nan_positions = np.isnan(score_data)
    array_temp = np.where(original_nan_positions, 0, score_data)
    normalized_array, scale, min_value = normalize_scores(array_temp)
    normalized_array[original_nan_positions] = np.nan
    score_data = normalized_array
    train_array_positions = [pos.tolist() for pos in train_positions]
    validate_array_positions = [pos.tolist() for pos in validate_positions]
    from src.preprocess_feature import read_benchmark_features, read_model_features
    benchmark_features = read_benchmark_features("./data/benchmark_feature.csv")
    bench_feature_dim = []
    for feature_i in range(len(benchmark_features[0])):
        bench_feature_dim.append(int(np.amax(benchmark_features[:, feature_i]))+1)
    model_features = read_model_features("./data/model_feature.csv")
    model_cate_feature_dim = []
    for feature_i in range(len(model_features[0])-4,len(model_features[0])):
        model_cate_feature_dim.append(int(np.amax(model_features[:, feature_i]))+1)
    model_num_feature_dim = len(model_features[0])-4
    return score_data, train_array_positions, validate_array_positions, scale, min_value, bench_feature_dim, model_cate_feature_dim, model_num_feature_dim

def load_data(R, subset, scale, min_value):
    model_ids = []
    task_ids = []
    ratings = []
    scales_list = []
    min_values_list = []
    benchmark_feature_list = []
    model_feature_list = []
    from src.preprocess_feature import read_benchmark_features, read_model_features
    benchmark_features = read_benchmark_features("./data/benchmark_feature.csv")
    model_features = read_model_features("./data/model_feature.csv")
    for i in range(len(R)):
        for j in range(len(R[i])):
            if not np.isnan(R[i][j]) and np.any(np.all(np.array([i,j]) == subset, axis=1)): 
                model_ids.append(i)
                task_ids.append(j)
                ratings.append(R[i][j])
                scales_list.append(scale[i])
                min_values_list.append(min_value[i])
                model_feature_list.append(model_features[i])
                benchmark_feature_list.append(benchmark_features[j])
    benchmark_feature_list = np.vstack(benchmark_feature_list)
    model_feature_list = np.vstack(model_feature_list)
    print("Load Finished")
    return torch.LongTensor(model_ids), torch.LongTensor(task_ids), torch.FloatTensor(ratings), scales_list, min_values_list, torch.from_numpy(benchmark_feature_list).int(), torch.from_numpy(model_feature_list).float()



def create_score_table(data):
    """
    Creates a table where rows are task_ids, columns are model_ids, and
    cells contain 'label_score/predicted_score'.

    :param data: List of tuples (model_id, task_id, label_score, predicted_score)
    :return: Pandas DataFrame
    """
    # Create a dictionary where keys are task_ids and values are dictionaries,
    # where the keys of these sub-dictionaries are model_ids and values are the score strings
    table_dict = {}
    
    for model_id, task_id, label_score, predicted_score, scale, min_value in data:
        if model_id not in table_dict:
            table_dict[model_id] = {}
        # Format the cell value as 'label_score/predicted_score'
        table_dict[model_id][task_id] = f"{label_score*scale+min_value}/{predicted_score*scale+min_value}"
        
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(table_dict, orient='index').sort_index()
    # Optionally, fill NaN with some placeholder if any cells are empty
    df.fillna('-', inplace=True)
    df_transposed = df.T
    df_transposed = df_transposed.sort_index()
    df = df_transposed.T
    return df


def evaluate(model, valid_data_loader, criterion, device):
    #MSE Loss
    total_loss = 0
    for model, task, rating, model_feature, bench_feature in valid_data_loader:
        model = model.to(device)
        task = task.to(device)
        rating = rating.to(device)
        bench_feature = bench_feature.to(device)
        model_feature = model_feature.to(device)
        prediction = model(model, task, model_feature, bench_feature)
        loss = criterion(prediction, rating)
        total_loss += loss.item()
    print("validate_loss", total_loss/len(valid_data_loader))
    

def main(data_name, mask_size, model_name, random_state):
    data_save_root = "./data/"
    model_save_root = "./model/"
    
    csv_path = data_save_root+"/crowdsource_performance.csv"
    random_states = [random_state]
    mask_size = mask_size
    for random_state in random_states:
        score_data, train_array_positions, validate_array_positions, scale, min_value, bench_feature_dim, model_cate_feature_dim, model_num_feature_dim = read_data(csv_path=csv_path, random_state=random_state, mask_size=mask_size)
        model_ids, task_ids, ratings, _, _, train_bench_features, train_model_features = load_data(score_data, train_array_positions, scale, min_value)
        valid_model_ids, valid_task_ids, valid_ratings, _, _, valid_bench_features, valid_model_features = load_data(score_data, validate_array_positions, scale, min_value)
        # Hyperparameters
        learning_rate = 0.01
        num_epochs = 250000
        num_models = len(score_data)   # example number of models
        num_tasks = len(score_data[0])   # example number of tasks
        num_factors = 10   # number of latent factors #8
        # Create a DataLoader for sparse matrix
        dataset = SparseMatrixDataset(model_ids, task_ids, ratings, train_bench_features, train_model_features)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        valid_dataset = SparseMatrixDataset(valid_model_ids, valid_task_ids, valid_ratings, valid_bench_features, valid_model_features)
        valid_data_loader = DataLoader(valid_dataset,  batch_size=16, shuffle=False)

        # Initialize model, optimizer, and loss function
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnhancedNeuralCollaborativeFiltering(num_models, num_tasks, bench_feature_dim, model_cate_feature_dim, model_num_feature_dim, num_factors)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate) #AdamW
        criterion = nn.MSELoss()

        # Train the model
        train(model, optimizer, criterion, data_loader, num_epochs, device)
        torch.save(model.state_dict(), model_save_root+"ncf_f/"+data_name+"-"+model_name+"-"+str(num_epochs)+"-"+str(random_state)+"-"+str(learning_rate)[2:]+"-"+str(num_factors)+"-SGD-mask"+str(int(mask_size*100))+'.pth')

        # Example prediction
        evaluate(model, valid_data_loader, criterion, device)
    
if __name__ == '__main__':
    fire.Fire(main)