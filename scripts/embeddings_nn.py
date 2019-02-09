import argparse
import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn import MSELoss

from scripts.loaders import load_embedder_by_name
from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import CSV_SEPARATOR
from wikisearch.consts.nn import EMBEDDING_VECTOR_SIZE
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS
from wikisearch.heuristics.nn_archs import EmbeddingsDistance

CRITERION_OPTIONS = ["MSELoss", "AsymmetricMSELoss"]
OPTIMIZER_OPTIONS = ["SGD", "Adam"]


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
            self._embedder.embed(item_row["source"]), \
            self._embedder.embed(item_row["destination"]), \
            int(item_row["min_distance"])


def early_stop(val_losses, best_val_loss, consequent_deteriorations):
    if consequent_deteriorations == 0 or len(val_losses) <= consequent_deteriorations:
        return False
    loss_differences = np.array(val_losses[-consequent_deteriorations:]) - best_val_loss
    # If all differences are > 0, then there was no improvement in the last <consequent_deteriorations> epochs compared
    # to best validation loss
    return np.all(loss_differences > 0)


def AsymmetricMSELoss(alphas):
    """
    Creates an asymmetric MSE loss function, where alphas are the weights by which the loss gets multiplied for
    loss < 0, loss > 0 respectively
    :param alphas: weights. alphas[0] for loss < 0, alphas[1] for loss > 0
    :return: function which accepts (input, target) and returns asymmetric loss
    """
    def asymmetric_mse_loss(input, target, reduction="mean"):
        loss = (input - target) ** 2
        sign = (input - target).sign()
        # Construct a vector of alpha coefficients depending on result of sign
        alpha_coefficients = (sign < 0).type(input.dtype) * alphas[0] + (sign > 0).type(input.dtype) * alphas[1]
        # Element-wise multiplication with coefficients
        return (loss * alpha_coefficients).mean() if reduction == "mean" else (loss * alpha_coefficients).sum()
    return asymmetric_mse_loss


def train(args, model, device, train_loader, criterion, optimizer, epoch):
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
        source, destination, min_distance = \
            source, destination, min_distance.float().to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(source, destination)
        loss = criterion(output, min_distance)
        loss.backward()
        optimizer.step()
        print_progress_bar(min(batch_idx * train_loader.batch_size, len(train_loader.dataset)),
                           len(train_loader.dataset), time.time() - start, prefix=f"Epoch {epoch},",
                           suffix=f"({batch_idx}/{math.ceil(len(train_loader.dataset) / train_loader.batch_size)}"
                                    f" batches) Loss (prev. batch): {loss.item():.4f}", length=50)


def test(args, model, criterion, test_loader, device):
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
            # Take with reduction="sum" to sum instead of take mean, because we later divide by num of samples
            test_loss += F.mse_loss(output, min_distance, reduction="sum").item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print("-STAT- Test set: MSE loss: {:.4f}, Time elapsed: {:.1f}s".format(test_loss, time.time() - test_start_time))
    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help="Path to training file")
    parser.add_argument("-te", "--test", help="Path to testing file")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("--crit", choices=CRITERION_OPTIONS, default="MSELoss", help="Criterion to calculate loss by")
    parser.add_argument("--opt", choices=OPTIMIZER_OPTIONS, default="SGD", help="Optimizer to use")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--early-stop-epochs", type=int, default=0,
                        help="Number of consecutive epochs with no improvemnet to stop training")
    parser.add_argument("--alphas", type=float, nargs=2, default=(1, 1),
                        help="Weights for loss < 0 and loss > 0 for Asymmetric MSE Loss")
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--adam-betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--adam-amsgrad", action="store_true")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--embedding", required=True, choices=AVAILABLE_EMBEDDINGS)

    args = parser.parse_args()

    embedder = load_embedder_by_name(args.embedding)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(EMBEDDING_VECTOR_SIZE).to(device)
    train_loader = torch.utils.data.DataLoader(DistanceDataset(args.train, embedder), batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(DistanceDataset(args.test, embedder), batch_size=args.batch_size)

    criterion = None
    criterion_meta = {}
    if args.crit == "MSELoss":
        criterion = MSELoss()
    elif args.crit == "AsymmetricMSELoss":
        criterion = AsymmetricMSELoss(args.alphas)
        criterion_meta["alphas"] = args.alphas
    criterion_meta["type"] = criterion.__class__.__name__

    optimizer = None
    if args.opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum)
    elif args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.adam_betas, amsgrad=args.adam_amsgrad)
    optimizer_meta = {"type": optimizer.__class__.__name__}
    optimizer_meta.update(optimizer.defaults)

    metadata = {
        "training_set": args.train,
        "validation_set": args.test,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "last_trained_epoch": 0,
        "criterion": criterion_meta,
        "optimizer": optimizer_meta,
    }
    metadata["model"] = model.get_metadata()
    metadata["embedder"] = embedder.get_metadata()
    model_name = os.path.splitext(args.out)[0]
    with open(model_name + ".meta", "w") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    # Do train-test iterations, to train and check efficiency of model
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    start_of_all = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        # Test the model on train and test sets, for progress tracking
        train_losses.append(round(test(args, model, criterion, train_loader, device), 4))
        val_losses.append(round(test(args, model, criterion, test_loader, device), 4))
        print()

        # Save best model
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            metadata["best_model"] = {
                "epoch": epoch,
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1],
            }
            # Save the new model if it"s better
            torch.save(model.state_dict(), args.out)

        # Plot train and val losses
        plt.clf()
        plt.plot(range(1, epoch + 1), train_losses, range(1, epoch + 1), val_losses)
        plt.legend(["Average train loss", "Average test loss"])
        plt.savefig(model_name + "_losses.jpg")

        # Save metadata again, with updated train and val loss history during training
        metadata["loss_history"] = {
            "train": train_losses,
            "val": val_losses,
        }
        metadata["last_trained_epoch"] = epoch
        with open(model_name + ".meta", "w") as meta_file:
            json.dump(metadata, meta_file, indent=2)

        # Early stop if there's no improvement in validation loss
        if early_stop(val_losses, best_val_loss, args.early_stop_epochs):
            print(f"-INFO- Early stop. last {args.early_stop_epochs} validation losses: {val_losses[-args.early_stop_epochs:]}")
            break

    total_time = time.time() - start_of_all
    print(f"-TIME- Total time took to train the model: {total_time:.1f}s -> "
          f"{total_time / 60:.2f}m -> {total_time / 3600:.3f}h")
