import argparse
import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import MSELoss

from scripts.loaders import load_embedder_by_name
from scripts.loaders.load_model import load_model_type
from scripts.utils import print_progress_bar
from wikisearch.consts.embeddings import EMBEDDING_VECTOR_SIZE
from wikisearch.consts.mongo import CSV_SEPARATOR
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS
from wikisearch.heuristics.nn_archs import NN_ARCHS

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


def AsymmetricMSELoss(alphas, reduction="mean"):
    """
    Creates an asymmetric MSE loss function, where alphas are the weights by which the loss gets multiplied for
    loss < 0, loss > 0 respectively
    :param alphas: weights. alphas[0] for loss < 0, alphas[1] for loss > 0
    :return: function which accepts (input, target) and returns asymmetric loss
    """
    def asymmetric_mse_loss(input, target):
        loss = (input - target) ** 2
        sign = (input - target).sign()
        # Construct a vector of alpha coefficients depending on result of sign
        alpha_coefficients = (sign < 0).type(input.dtype) * alphas[0] + (sign > 0).type(input.dtype) * alphas[1]
        # Element-wise multiplication with coefficients
        return (loss * alpha_coefficients).mean() if reduction == "mean" else (loss * alpha_coefficients).sum()
    return asymmetric_mse_loss


def train_epoch(args, model, device, train_loader, criterion, optimizer, epoch):
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
            # because reduction is mean, we want to "convert" it to sum by multiplying by batch size to be able to
            # handle uneven batch sizes (at end of data), and so we divide later by number of samples
            test_loss += criterion(output, min_distance).item() * source.size(0)  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print("-STAT- Test set: MSE loss: {:.4f}, Time elapsed: {:.1f}s".format(test_loss, time.time() - test_start_time))
    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help="Path to training file")
    parser.add_argument("-te", "--test", help="Path to testing file")
    parser.add_argument("-a", "--arch", choices=NN_ARCHS, help="NN Architecture to use")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("--crit", choices=CRITERION_OPTIONS, default="MSELoss", help="Criterion to calculate loss by")
    parser.add_argument("--opt", choices=OPTIMIZER_OPTIONS, default="SGD", help="Optimizer to use")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--early-stop-epochs", type=int, default=0,
                        help="Number of consecutive epochs with no improvement to stop training")
    parser.add_argument("--alphas", type=float, nargs=2, default=(1, 1),
                        help="Weights for loss < 0 and loss > 0 for Asymmetric MSE Loss")
    parser.add_argument("--sgd-momentum", type=float, default=0)
    parser.add_argument("--adam-betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--adam-amsgrad", action="store_true")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--embedding", required=True, choices=AVAILABLE_EMBEDDINGS)

    args = parser.parse_args()

    embedder = load_embedder_by_name(args.embedding)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_type(args.arch, EMBEDDING_VECTOR_SIZE[embedder.type])
    train_loader = torch.utils.data.DataLoader(DistanceDataset(args.train, embedder), batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(DistanceDataset(args.test, embedder), batch_size=args.batch_size)

    criterion = None
    reduction = "mean"
    criterion_meta = {}
    if args.crit == "MSELoss":
        criterion = MSELoss(reduction=reduction)
    elif args.crit == "AsymmetricMSELoss":
        criterion = AsymmetricMSELoss(args.alphas, reduction=reduction)
        criterion_meta["alphas"] = args.alphas
    criterion_meta["type"] = criterion.__class__.__name__

    optimizer = None
    if args.opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum)
    elif args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.adam_betas, amsgrad=args.adam_amsgrad)
    optimizer_meta = {"type": optimizer.__class__.__name__}
    optimizer_meta.update(optimizer.defaults)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
    lr_scheduler_meta = lr_scheduler.state_dict()

    metadata = {
        "training_set": args.train,
        "validation_set": args.test,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "last_trained_epoch": 0,
        "criterion": criterion_meta,
        "optimizer": optimizer_meta,
        "lr_scheduler": lr_scheduler_meta,
    }
    metadata["model"] = model.get_metadata()
    metadata["embedder"] = embedder.get_metadata()
    model_name = os.path.splitext(args.out)[0]

    model_dir = os.path.dirname(args.out)
    os.makedirs(model_dir, exist_ok=True)

    with open(model_name + ".meta", "w") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    # Do train-test iterations, to train and check efficiency of model
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    start_of_all = time.time()
    for epoch in range(1, args.epochs + 1):
        train_epoch(args, model, device, train_loader, criterion, optimizer, epoch)
        # Test the model on train and test sets, for progress tracking
        train_losses.append(round(test(args, model, criterion, train_loader, device), 4))
        val_losses.append(round(test(args, model, criterion, test_loader, device), 4))
        # Update learning rate according to scheduler
        lr_scheduler.step(val_losses[-1])
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
        plt.axvline(metadata["best_model"]["epoch"], linestyle="--")
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
