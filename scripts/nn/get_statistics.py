import argparse
import random
from importlib import import_module

import pandas as pd
import tabulate
import torch.nn.functional as F
import torch.utils.data
from torch import Tensor

from scripts.embeddings_nn import DistanceDataset
from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS, EMBEDDINGS_MODULES
from wikisearch.heuristics.nn_archs import EmbeddingsDistance


def get_model_accuracy_statistics(model_pr, test_loader_pr, statistics_path_pr):
    """
    Calculates the accuracy statistics of the model
    :param model_pr: the model to create the statistics file for
    :param test_loader_pr: the expected results to compare to
    :param statistics_path_pr: the path where the statistics file will be saved to
    """

    losses = {
        'L1': F.l1_loss,
        'L2': F.mse_loss
    }
    model_distances = Tensor([]).to(device)
    actual_distances = Tensor([]).to(device)

    with torch.no_grad():
        count = 1
        for source, destination, actual_distance in test_loader_pr:
            count += 1
            # Move tensors to relevant devices, and handle distances tensor.
            source, destination, actual_distance = \
                source.to(device), destination.to(device), actual_distance.float().to(device).unsqueeze(1)
            model_distance = model_pr(source, destination)
            model_distances = torch.cat((model_distances, model_distance))
            actual_distances = torch.cat((actual_distances, actual_distance))

    dataset_size = len(test_loader_pr.dataset)
    dataset_idxs = range(dataset_size)
    idxs_per_50_percentage = random.choices(dataset_idxs, k=round(0.5 * dataset_size))
    dataset_idxs = list(set(dataset_idxs) - set(idxs_per_50_percentage))
    idxs_per_75_percentage = random.choices(dataset_idxs, k=round(0.25 * dataset_size)) + idxs_per_50_percentage
    dataset_idxs = list(set(dataset_idxs) - set(idxs_per_75_percentage))
    idxs_per_90_percentage = random.choices(dataset_idxs, k=round(0.15 * dataset_size)) + idxs_per_75_percentage

    idxs_per_50_percentage = Tensor(idxs_per_50_percentage).to(device).long()
    idxs_per_75_percentage = Tensor(idxs_per_75_percentage).to(device).long()
    idxs_per_90_percentage = Tensor(idxs_per_90_percentage).to(device).long()

    # Options used for printing dataset summaries and statistics
    pd.set_option('display.max_columns', 10)
    pd.set_option('precision', 2)
    statistics_df = pd.DataFrame(columns=['Loss Method', 'Average'])
    for loss_type, loss_method in losses.items():
        statistics_df = statistics_df.append(
            {
                'Loss Method': loss_type,
                'Average': loss_method(model_distances, actual_distances),
                '90% Avg': loss_method(
                    model_distances.take(idxs_per_90_percentage),
                    actual_distances.take(idxs_per_90_percentage)),
                '75% Avg': loss_method(
                    model_distances.take(idxs_per_75_percentage),
                    actual_distances.take(idxs_per_75_percentage)),
                '50% Avg': loss_method(
                    model_distances.take(idxs_per_50_percentage),
                    actual_distances.take(idxs_per_50_percentage))
            }, ignore_index=True)

    # Print out statistics to file
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')
    print(tabulate.tabulate(statistics_df, headers='keys', tablefmt='grid', floatfmt='.2f', showindex=False))
    with open(statistics_path_pr, 'w', encoding='utf8') as f:
        f.write(tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='grid', floatfmt='.2f'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to the model file')
    parser.add_argument('-t', '--test', help='Path to testing file')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--embedding', required=True, choices=AVAILABLE_EMBEDDINGS)
    parser.add_argument('-s', '--statistics', help='Path to the output statistics file')

    args = parser.parse_args()

    # Loads dynamically the relevant embedding class. This is used for embedding entries later on!
    embedding_module = import_module(
        '.'.join(['wikisearch', 'embeddings', EMBEDDINGS_MODULES[args.embedding]]),
        package='wikisearch')
    embedding_class = getattr(embedding_module, args.embedding)
    embedder = embedding_class(WIKI_LANG, PAGES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(300).to(device)
    model.load_state_dict(torch.load(args.model, map_location=None if torch.cuda.is_available() else 'cpu'))
    model.eval()
    test_dd = DistanceDataset(args.test, embedder)
    test_loader = torch.utils.data.DataLoader(test_dd, batch_size=args.batch_size)
    get_model_accuracy_statistics(model, test_loader, args.statistics)
