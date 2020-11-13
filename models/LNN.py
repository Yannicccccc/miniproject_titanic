import torch.nn as nn


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'SimpleNN'
        self.num_hidden = 100
        self.num_features = 9
        self.num_classes = 2
        self.dropout = 0.5
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.batch_size = 32
        self.num_epochs = 20
        self.require_improvement = 20
        self.save_path = 'data/saved_dict/' + dataset + '/' + self.model_name + '.ckpt'
        self.log_path = 'data/log/' + dataset + '/' + self.model_name


'''Simple 2-Layer Neural Networks'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(config.num_features, config.num_hidden)
        self.fc2 = nn.Linear(config.num_hidden, config.num_classes)


    def forward(self, x):
        out = self.relu(x)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.sigmoid(out)
        out = self.fc2(out)
        # out = self.dropout(out)
        # out = self.sigmoid(out)
        # out = self.fc3(out)
        return out



