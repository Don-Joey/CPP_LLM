import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_models, num_tasks, embedding_size=8):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.model_embedding_gmf = nn.Embedding(num_models, embedding_size)
        self.task_embedding_gmf = nn.Embedding(num_tasks, embedding_size)
        self.model_embedding_mlp = nn.Embedding(num_models, embedding_size * 2)
        self.task_embedding_mlp = nn.Embedding(num_tasks, embedding_size * 2)

        self.mlp_layers = nn.Sequential(
            nn.Linear(4 * embedding_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(embedding_size + 16, 1)

    def forward(self, model_ids, task_ids):
        model_gmf = self.model_embedding_gmf(model_ids)
        task_gmf = self.task_embedding_gmf(task_ids)
        gmf = model_gmf * task_gmf

        model_mlp = self.model_embedding_mlp(model_ids)
        task_mlp = self.task_embedding_mlp(task_ids)
        mlp = torch.cat((model_mlp, task_mlp), -1)
        mlp = self.mlp_layers(mlp)

        concat = torch.cat((gmf, mlp), -1)
        output = self.output_layer(concat)
        return output.squeeze()