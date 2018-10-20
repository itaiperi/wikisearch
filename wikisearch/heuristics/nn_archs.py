import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingsDistance(nn.Module):
    def __init__(self, embed_dim):
        super(EmbeddingsDistance, self).__init__()
        self.siamese_fc1 = nn.Linear(embed_dim, 128)
        self.siamese_conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.siamese_conv2 = nn.Conv1d(32, 16, kernel_size=3)
        self.conv1 = nn.Conv1d(32, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=1)
        self.fc1 = nn.Linear(14, 1)

    def forward(self, x1, x2):
        x1, x2 = self.siamese_fc1(x1), self.siamese_fc1(x2)
        x1, x2 = F.relu(F.max_pool1d(self.siamese_conv1(x1), 2)), F.relu(F.max_pool1d(self.siamese_conv1(x2), 2))
        x1, x2 = F.relu(F.max_pool1d(self.siamese_conv2(x1), 2)), F.relu(F.max_pool1d(self.siamese_conv2(x2), 2))
        x = torch.stack([x1, x2], dim=1)
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = self.fc1(x).squeeze(2)

        return x