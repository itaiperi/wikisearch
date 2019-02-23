from abc import ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F

NN_ARCHS = [
    "EmbeddingsDistance1",
    "EmbeddingsDistance2",
]


class EmbeddingsDistance(nn.Module, metaclass=ABCMeta):
    def __init__(self, embed_dim):
        super(EmbeddingsDistance, self).__init__()
        self._embed_dim = embed_dim

    def get_metadata(self):
        """
        Returns metadata relevant to the model
        :return: dictionary with key-metadata pairs.
            Compulsory keys:
                * architecture - description of architecture of the model's network
        """
        return {
            'arch_type': self.__class__.__name__,
            'architecture': [k + ": " + repr(v) for k, v in self._modules.items()],
            'embed_dim': self._embed_dim,
        }


# TODO: Add documentation
class EmbeddingsDistance1(EmbeddingsDistance):
    def __init__(self, embed_dim):
        super(EmbeddingsDistance1, self).__init__(embed_dim)
        # Architecture of Siamese Network fed into a Sequential one
        siamese_fc1_size = 128
        self.siamese_fc1 = nn.Linear(self._embed_dim, siamese_fc1_size)
        self.siamese_conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.siamese_conv2 = nn.Conv1d(32, 16, kernel_size=3)
        self.conv1 = nn.Conv1d(32, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=1)
        # conv5 -> pool2 -> conv3 -> pool2 -> conv3 -> pool2 -> concatenate -> conv1 -> pool2
        linear_size_float = (((((siamese_fc1_size - 4) / 2) - 2) / 2) - 2) / 2 / 2
        linear_size = int(linear_size_float)
        assert linear_size == linear_size_float
        self.fc1 = nn.Linear(linear_size, 1)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x1, x2 = self.siamese_fc1(x1), self.siamese_fc1(x2)
        x1, x2 = F.relu(F.max_pool1d(self.siamese_conv1(x1), 2)), F.relu(F.max_pool1d(self.siamese_conv1(x2), 2))
        x1, x2 = F.relu(F.max_pool1d(self.siamese_conv2(x1), 2)), F.relu(F.max_pool1d(self.siamese_conv2(x2), 2))
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = self.fc1(x).squeeze(2)

        return x


# TODO add documentation
class EmbeddingsDistance2(EmbeddingsDistance):
    """
    Like EmbeddingsDistance1, but with batch normalization after each convolution, for regularization
    """
    def __init__(self, embed_dim):
        super(EmbeddingsDistance2, self).__init__(embed_dim)
        # Architecture of Siamese Network fed into a Sequential one
        siamese_fc1_size = 128
        self.siamese_fc1 = nn.Linear(embed_dim, siamese_fc1_size)
        self.siamese_conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.siamese_batchnorm1 = nn.BatchNorm1d(32)
        self.siamese_conv2 = nn.Conv1d(32, 16, kernel_size=3)
        self.siamese_batchnorm2 = nn.BatchNorm1d(16)
        self.conv1 = nn.Conv1d(32, 16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm1d(1)
        # conv5 -> pool2 -> conv3 -> pool2 -> conv3 -> pool2 -> concatenate -> conv1 -> pool2
        linear_size_float = (((((siamese_fc1_size - 4) / 2) - 2) / 2) - 2) / 2 / 2
        linear_size = int(linear_size_float)
        assert linear_size == linear_size_float
        self.fc1 = nn.Linear(linear_size, 1)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x1, x2 = self.siamese_fc1(x1), self.siamese_fc1(x2)
        x1, x2 = F.relu(F.max_pool1d(self.siamese_batchnorm1(self.siamese_conv1(x1)), 2)),\
                 F.relu(F.max_pool1d(self.siamese_batchnorm1(self.siamese_conv1(x2)), 2))
        x1, x2 = F.relu(F.max_pool1d(self.siamese_batchnorm2(self.siamese_conv2(x1)), 2)),\
                 F.relu(F.max_pool1d(self.siamese_batchnorm2(self.siamese_conv2(x2)), 2))
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(F.max_pool1d(self.batchnorm1(self.conv1(x)), 2))
        x = F.relu(F.max_pool1d(self.batchnorm2(self.conv2(x)), 2))
        x = self.fc1(x).squeeze(2)

        return x


class EmbeddingsDistanceMultiHotCategories(EmbeddingsDistance):
    """
    Like EmbeddingsDistance2, but with categories as inputs as well.
    """
    def __init__(self, embed_dim, categories_dim):
        super(EmbeddingsDistanceMultiHotCategories, self).__init__(embed_dim + categories_dim)
        self._categories_dim = categories_dim
        # Architecture of Siamese Network fed into a Sequential one
        siamese_fc1_size = 128
        siamese_categories_fc1_size = 512
        siamese_categories_fc2_size = 128
        self.siamese_fc1 = nn.Linear(embed_dim, siamese_fc1_size)
        self.siamese_categories_fc1 = nn.Linear(categories_dim, siamese_categories_fc1_size)
        self.siamese_categories_fc2 = nn.Linear(siamese_categories_fc1_size, siamese_categories_fc2_size)
        self.siamese_conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.siamese_batchnorm1 = nn.BatchNorm1d(32)
        self.siamese_conv2 = nn.Conv1d(32, 16, kernel_size=3)
        self.siamese_batchnorm2 = nn.BatchNorm1d(16)
        self.conv1 = nn.Conv1d(32, 16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm1d(1)
        # conv5 -> pool2 -> conv3 -> pool2 -> conv3 -> pool2 -> concatenate -> conv1 -> pool2
        linear_size_float = ((((((siamese_fc1_size + siamese_categories_fc2_size) - 4) / 2) - 2) / 2) - 2) / 2 / 2
        linear_size = int(linear_size_float)
        assert linear_size == linear_size_float
        self.fc1 = nn.Linear(linear_size, 1)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x1_embed, x1_categories = x1.split([self._embed_dim, self._categories_dim])
        x2_embed, x2_categories = x2.split([self._embed_dim, self._categories_dim])
        x1_embed, x2_embed = self.siamese_fc1(x1_embed), self.siamese_fc1(x2_embed)
        x1_categories = self.siamese_categories_fc2(F.relu(self.siamese_categories_fc1(x1_categories)))
        x2_categories = self.siamese_categories_fc2(F.relu(self.siamese_categories_fc1(x2_categories)))
        x1 = torch.cat((x1_embed, x1_categories), dim=2)
        x2 = torch.cat((x2_embed, x2_categories), dim=2)
        x1, x2 = F.relu(F.max_pool1d(self.siamese_batchnorm1(self.siamese_conv1(x1)), 2)),\
                 F.relu(F.max_pool1d(self.siamese_batchnorm1(self.siamese_conv1(x2)), 2))
        x1, x2 = F.relu(F.max_pool1d(self.siamese_batchnorm2(self.siamese_conv2(x1)), 2)),\
                 F.relu(F.max_pool1d(self.siamese_batchnorm2(self.siamese_conv2(x2)), 2))
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(F.max_pool1d(self.batchnorm1(self.conv1(x)), 2))
        x = F.relu(F.max_pool1d(self.batchnorm2(self.conv2(x)), 2))
        x = self.fc1(x).squeeze(2)

        return x
