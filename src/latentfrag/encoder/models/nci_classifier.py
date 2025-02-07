import torch
import torch.nn as nn

class NCIClassifier(nn.Module):
    '''
    Multi-label classifier for NCI types.
    '''
    def __init__(self, embedding_dim, num_classes):
        super(NCIClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
