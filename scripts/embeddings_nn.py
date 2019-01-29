import argparse
import json
import os
import time
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import CSV_SEPARATOR, WIKI_LANG, PAGES
from wikisearch.consts.nn import EMBEDDING_VECTOR_SIZE
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS, EMBEDDINGS_MODULES
from wikisearch.heuristics.nn_archs import EmbeddingsDistance


class DistanceDataset(torch.utils.data.Dataset):
    """
    Dataset class.
    Supposed to give the whole size of the dataset, and returns the vectored elements when accessed
    through [i]
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
        return \
            self._embedder.embed(item_row['source']), \
            self._embedder.embed(item_row['destination']), \
            int(item_row['min_distance'])


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
    batch_losses = []
    for batch_idx, (source, destination, min_distance) in enumerate(train_loader, 1):
        # Move tensors to relevant devices, and handle distances tensor.
        source, destination, min_distance = \
            source, destination, min_distance.float().to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(source, destination)
        loss = F.mse_loss(output, min_distance)
        batch_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print_progress_bar(min(batch_idx * train_loader.batch_size, len(train_loader.dataset)),
                           len(train_loader.dataset), time.time() - start, prefix=f'Epoch {epoch + 1},',
                           suffix=f'Average Loss: {np.mean(batch_losses):.1f}', length=50)

    train_loss = np.mean(batch_losses)
    return train_loss


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
    test_start_time = time.time()
    with torch.no_grad():
        for source, destination, min_distance in test_loader:
            # Move tensors to relevant devices, and handle distances tensor.
            source, destination, min_distance = \
                source, destination, min_distance.float().to(device).unsqueeze(1)
            output = model(source, destination)
            test_loss += F.mse_loss(output, min_distance, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('-STAT- Test set: Average loss: {:.4f}, Time elapsed: {:.1f}s'.format(test_loss, time.time() - test_start_time))
    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', help='Path to training file')
    parser.add_argument('-te', '--test', help='Path to testing file')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--opt', choices=['SGD', 'Adam'], default='SGD')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--sgd-momentum', type=float, default=0.9)
    parser.add_argument('--adam-betas', nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument('--adam-amsgrad', action='store_true')
    parser.add_argument('-o', '--out', required=True)
    parser.add_argument('--embedding', required=True, choices=AVAILABLE_EMBEDDINGS)

    args = parser.parse_args()
    # Loads dynamically the relevant embedding class. This is used for embedding entries later on!
    embedding_module = import_module('.'.join(['wikisearch', 'embeddings', EMBEDDINGS_MODULES[args.embedding]]),
                                     package='wikisearch')
    embedding_class = getattr(embedding_module, args.embedding)

    embedder = embedding_class(WIKI_LANG, PAGES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(EMBEDDING_VECTOR_SIZE).to(device)
    train_loader = torch.utils.data.DataLoader(DistanceDataset(args.train, embedder), batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(DistanceDataset(args.test, embedder), batch_size=args.batch_size)
    optimizer = None
    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum)
    elif args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.adam_betas, amsgrad=args.adam_amsgrad)
    optimizer_meta = {"type": optimizer.__class__.__name__}
    optimizer_meta.update(optimizer.defaults)
    metadata = {
        "training_set": args.train,
        "validation_set": args.test,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": optimizer_meta,
    }
    metadata['model'] = model.get_metadata()
    metadata['embedder'] = embedder.get_metadata()
    model_name = os.path.splitext(args.out)[0]
    with open(model_name + ".meta", 'w') as meta_file:
        json.dump(metadata, meta_file, indent=2)

    # Do train-test iterations, to train and check efficiency of model
    train_losses = []
    test_losses = []

    start_of_all = time.time()
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        # Test the model on train and test sets, for progress tracking
        train_losses.append(test(args, model, device, train_loader))
        test_losses.append(test(args, model, device, test_loader))
        print()
        # TODO save the best model here! should use return value from test function to see which model is best
        plt.clf()
        plt.plot(range(1, epoch + 2), train_losses, range(1, epoch + 2), test_losses)
        plt.legend(['Average train loss', 'Average test loss'])
        plt.savefig(model_name + '_losses.jpg')
        torch.save(model.state_dict(), args.out)

    total_time = time.time() - start_of_all
    print(f"-TIME- Total time took to train the model: {total_time - start_of_all:.1f}s -> "
          f"{total_time / 60}m -> {total_time / 3600}h")
