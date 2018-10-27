import argparse
import time
from importlib import import_module

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import CSV_SEPARATOR, WIKI_LANG, PAGES
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS, EMBEDDINGS_MODULES
from wikisearch.heuristics.nn_archs import EmbeddingsDistance


class DistanceDataset(torch.utils.data.Dataset):
    """
    Dataset class.
    Supposed to give the whole size of the dataset, and return elements when accessed through [i]
    """
    def __init__(self, path, embedder):
        super(DistanceDataset, self).__init__()
        self._path = path
        # Read Dataset file into dataframe
        self._df = pd.read_csv(self._path, sep=CSV_SEPARATOR)
        # Embedder to be used to embed wikipedia pages
        self._embedder = embedder

    def __len__(self):
        return len(self._df)

    def __getitem__(self, item):
        # Get source, destination and min_distance at index item
        item_row = self._df.iloc[item]
        # Embed source and destination, and cast distance from string to integer.
        return self._embedder.embed(item_row['source']), self._embedder.embed(item_row['destination']), int(item_row[
            'min_distance'])


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Training function for pytorch models
    :param args: arguments to be used (for example, from __main__)
    :param model: model to be trained
    :param device: device to train on (cuda(gpu) or cpu)
    :param train_loader: loader of data
    :param optimizer: optimizer to use
    :param epoch: current epoch
    :return: None
    """
    model.train()
    start = time.time()
    for batch_idx, (source, destination, min_distance) in enumerate(train_loader, 1):
        # Move tensors to relevant devices, and handle distances tensor.
        source, destination, min_distance = source.to(device), destination.to(device), min_distance.float().to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(source, destination)
        loss = F.mse_loss(output, min_distance)
        loss.backward()
        optimizer.step()
        print_progress_bar(min(batch_idx * train_loader.batch_size, len(train_loader.dataset)),
                           len(train_loader.dataset), time.time() - start, prefix=f'Epoch {epoch + 1},',
                           suffix=f'Loss: {loss.item():.1f}', length=50)
    print()


def test(args, model, device, test_loader):
    """
    Training function for pytorch models
    :param args: arguments to be used (for example, from __main__)
    :param model: model to be trained
    :param device: device to train on (cuda(gpu) or cpu)
    :param test_loader: loader of data
    :return: None
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for source, destination, min_distance in test_loader:
            # Move tensors to relevant devices, and handle distances tensor.
            source, destination, min_distance = source.to(device), destination.to(device), min_distance.float().to(device).unsqueeze(1)
            output = model(source, destination)
            test_loss += F.mse_loss(output, min_distance, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', help='Path to training file')
    parser.add_argument('-te', '--test', help='Path to testing file')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('-o', '--out', required=True)
    parser.add_argument('--embedding', required=True, choices=AVAILABLE_EMBEDDINGS)

    args = parser.parse_args()
    # Dynamically load the relevant embedding class. This is used for embedding entries later on!
    embedding_module = import_module('.'.join(['wikisearch', 'embeddings', EMBEDDINGS_MODULES[args.embedding]]),
                                     package='wikisearch')
    embedding_class = getattr(embedding_module, args.embedding)

    embedder = embedding_class(WIKI_LANG, PAGES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(300).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = torch.utils.data.DataLoader(DistanceDataset(args.train, embedder), batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(DistanceDataset(args.test, embedder), batch_size=args.batch_size)
    # Do train-test iterations, to train and check efficiency of model
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        # TODO save the best model here! should use return value from test function to see which model is best.

    torch.save(model.state_dict(), args.out)