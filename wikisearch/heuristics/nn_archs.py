import torch.nn as nn
import torch.nn.functional as F


class EmbeddingsDistance(nn.Module):
    def __init__(self, embed_dim):
        super(EmbeddingsDistance, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 128)
        self.conv1 = nn.Conv1d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3)
        self.conv3 = nn.Conv1d(16, 1, kernel_size=1)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.max_pool1d(self.conv1(x), 2)
        x = F.max_pool1d(self.conv2(x), 2)
        x = F.max_pool1d(self.conv3(x), 2)
        x = self.fc3(x).squeeze(2)

        return x