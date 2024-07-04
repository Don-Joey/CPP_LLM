import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_models, num_tasks, num_factors):
        super(MatrixFactorization, self).__init__()
        self.model_factors = nn.Embedding(num_models, num_factors)
        self.task_factors = nn.Embedding(num_tasks, num_factors)
        
        nn.init.normal_(self.model_factors.weight, std=0.01)
        nn.init.normal_(self.task_factors.weight, std=0.01)
    
    def forward(self, model, task):
        model_embedding = self.model_factors(model)
        task_embedding = self.task_factors(task)
        return (model_embedding * task_embedding).sum(1)
