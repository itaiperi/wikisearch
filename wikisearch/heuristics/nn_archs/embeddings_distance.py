from abc import ABCMeta

from torch import nn


class EmbeddingsDistance(nn.Module, metaclass=ABCMeta):
    def __init__(self, dims):
        super(EmbeddingsDistance, self).__init__()
        self._dims = dims
        self._embed_dim = dims['embed_dim']

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
            'dims': self._dims,
        }
