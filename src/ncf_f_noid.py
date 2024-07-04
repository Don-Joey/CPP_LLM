import torch
import torch.nn as nn


class EnhancedNeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_models, num_tasks, bench_feature_dim, model_cate_feature_dim, model_num_feature_dim, embedding_size=8):
        super(EnhancedNeuralCollaborativeFiltering, self).__init__()
        self.model_embedding = nn.Embedding(num_models, embedding_size) 
        self.task_embedding = nn.Embedding(num_tasks, embedding_size)

        self.model_embedding_mlp = nn.Embedding(num_models, embedding_size * 2)
        self.task_embedding_mlp = nn.Embedding(num_tasks, embedding_size * 2)

        #bench_dim
        
        self.task_feat1_embedding = nn.Embedding(bench_feature_dim[0], embedding_size * 2) #'ability'
        self.task_feat2_embedding = nn.Embedding(bench_feature_dim[1], embedding_size * 2) #'family'
        self.task_feat3_embedding = nn.Embedding(bench_feature_dim[2], embedding_size * 2) #'output'
        self.task_feat4_embedding = nn.Embedding(bench_feature_dim[3], embedding_size * 2) #'few-shot'

        #model_numerical_dim
        self.model_numerical_feat_mlp = nn.Sequential(
            nn.Linear(model_num_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embedding_size*2),
            nn.ReLU()
        )
        #model_categorical_dim
        #in our data, there are 4 types of catogrical factors
        self.model_feat1_embedding = nn.Embedding(model_cate_feature_dim[0], embedding_size * 2) #'family'
        self.model_feat2_embedding = nn.Embedding(model_cate_feature_dim[1], embedding_size * 2) #'Finetuning'
        self.model_feat3_embedding = nn.Embedding(model_cate_feature_dim[2], embedding_size * 2) #'Context window'
        self.model_feat4_embedding = nn.Embedding(model_cate_feature_dim[3], embedding_size * 2) #'batch size(M)'

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_size * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embedding_size * 2),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(embedding_size* 2, 1)

    def forward(self, task_features, model_features):
        
        model_numerical_feat = self.model_numerical_feat_mlp(model_features[:, :-4])
        model_cate_feat1 = self.model_feat1_embedding(model_features[:,-4:-3].int().squeeze(1))
        model_cate_feat2 = self.model_feat2_embedding(model_features[:,-3:-2].int().squeeze(1))
        model_cate_feat3 = self.model_feat3_embedding(model_features[:,-2:-1].int().squeeze(1))
        model_cate_feat4 = self.model_feat4_embedding(model_features[:,-1:].int().squeeze(1))
        model_features = model_numerical_feat+model_cate_feat1+model_cate_feat2+model_cate_feat3+model_cate_feat4

        task_feat1 = self.task_feat1_embedding(task_features[:, :1].squeeze(1))
        task_feat2 = self.task_feat2_embedding(task_features[:, 1:2].squeeze(1))
        task_feat3 = self.task_feat3_embedding(task_features[:, 2:3].squeeze(1))
        task_feat4 = self.task_feat4_embedding(task_features[:, 3:].squeeze(1))
        task_features = task_feat1+task_feat2+task_feat3+task_feat4
        mlp = torch.cat((model_features, task_features), -1)
        mlp = self.fc_layers(mlp)

        output = self.output_layer(mlp)
        return output.squeeze()