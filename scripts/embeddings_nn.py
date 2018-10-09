import argparse

# import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from wikisearch.heuristics.nn_archs import EmbeddingsDistance
# from wikisearch.utils.consts import CSV_SEPARATOR


class DistanceDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super(DistanceDataset, self).__init__()
        self._path = path
        # self._df = pd.read_csv(self._path, sep=CSV_SEPARATOR)

        self._test_tensor = torch.randn(32, 2, 300)
        self._test_result = torch.Tensor([[2] * 32]).transpose(1, 0)

    def __len__(self):
        # return len(self._df)
        return self._test_tensor.size(0)

    def __getitem__(self, item):
        # return self._df.iloc[item]
        return self._test_tensor[item], self._test_result[item]


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
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

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(300).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = torch.utils.data.DataLoader(DistanceDataset(args.train), batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(DistanceDataset(args.test), batch_size=args.batch_size)
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    torch.save(model.state_dict(), args.out)
