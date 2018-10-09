import argparse

# from wikisearch.utils.consts import CSV_SEPARATOR
# import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class DistanceDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super(DistanceDataset, self).__init__()
        self._path = path
        # self._df = pd.read_csv(self._path, sep=CSV_SEPARATOR)

        self._test_tensor = torch.randn(2, 300)
        self._test_result = torch.Tensor([[2]])

    def __len__(self):
        # return len(self._df)
        return 1

    def __getitem__(self, item):
        # return self._df.iloc[item]
        return self._test_tensor, self._test_result


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
        x = self.fc3(x)

        return x


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
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--log-interval', type=int, default=10)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(300).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = torch.utils.data.DataLoader(DistanceDataset(args.train), batch_size=1)
    test_loader = torch.utils.data.DataLoader(DistanceDataset(args.test), batch_size=1)
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
