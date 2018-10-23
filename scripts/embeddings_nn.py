import argparse
from importlib import import_module

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from wikisearch.consts.mongo import CSV_SEPARATOR, WIKI_LANG, PAGES
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS, EMBEDDINGS_MODULES
from wikisearch.heuristics.nn_archs import EmbeddingsDistance


class DistanceDataset(torch.utils.data.Dataset):
    def __init__(self, path, embedder):
        super(DistanceDataset, self).__init__()
        self._path = path
        self._df = pd.read_csv(self._path, sep=CSV_SEPARATOR)
        self._embedder = embedder

    def __len__(self):
        return len(self._df)

    def __getitem__(self, item):
        item_row = self._df.iloc[item]
        if not self._embedder.embed(item_row['source']).size() or not self._embedder.embed(item_row['destination']).size():
            print(item_row['source'], '->', item_row['destination'])
        return self._embedder.embed(item_row['source']), self._embedder.embed(item_row['destination']), int(item_row[
            'min_distance'])


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (source, destination, min_distance) in enumerate(train_loader, 1):
        source, destination, min_distance = source.to(device), destination.to(device), min_distance.to(device)
        optimizer.zero_grad()
        output = model(source, destination)
        loss = F.mse_loss(output, min_distance)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(source), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train')
    parser.add_argument('-te', '--test')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--log-interval', type=int, default=10)
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
    for epoch in range(args.epochs):
        print(epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    torch.save(model.state_dict(), args.out)
