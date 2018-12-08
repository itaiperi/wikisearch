import argparse
from enum import Enum
from importlib import import_module

import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import tabulate

from scripts.embeddings_nn import DistanceDataset
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS, EMBEDDINGS_MODULES

from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.heuristics.nn_archs import EmbeddingsDistance


class LossFunction(Enum):
    VALUE = 1
    METHOD = 2


def get_model_accuracy_statistics(model_pr, test_loader_pr, statistics_path_pr):
    """
    Calculates the accuracy statistics of the model
    :param model_pr: the model which will crate the statistics file for
    :param test_loader_pr: the expected results to compare to
    :param statistics_path_pr: the path where the statistics file will be saved to
    """

    losses = {'L1_AVG': {LossFunction.VALUE: 0, LossFunction.METHOD: F.l1_loss},
              'L2_AVG': {LossFunction.VALUE: 0, LossFunction.METHOD: F.mse_loss}}
    vector_operations = {'STANDARD_DEVIATION_AVG': {LossFunction.VALUE: 0, LossFunction.METHOD: Tensor.std}}
    # 'L1_90%': 0, 'L1_75%': 0, 'L1_50%': 0,
    # 'L2_90%': 0, 'L2_75%': 0, 'L2_50%': 0,

    with torch.no_grad():
        for source, destination, actual_distance in test_loader_pr:
            # Move tensors to relevant devices, and handle distances tensor.
            source, destination, actual_distance = \
                source.to(device), destination.to(device), actual_distance.float().to(device).unsqueeze(1)
            model_distance = model_pr(source, destination)
            for loss_type in losses:
                losses[loss_type][LossFunction.VALUE] += \
                    losses[loss_type][LossFunction.METHOD](actual_distance, model_distance,
                                                           reduction='sum').item()
            for vector_operation in vector_operations:
                vector_operations[vector_operation][LossFunction.VALUE] += \
                    vector_operations[vector_operation][LossFunction.METHOD](actual_distance, 0)

    # Options used for printing dataset summaries and statistics
    pd.set_option('display.max_columns', 10)
    pd.set_option('precision', 2)
    statistics_df = pd.DataFrame(columns=['Method', 'Result'])
    size_of_dataset = len(test_loader_pr.dataset)
    for loss_type in losses:
        statistics_df = statistics_df.append({
            'Method': loss_type, 'Result': losses[loss_type][LossFunction.VALUE] / size_of_dataset},
            ignore_index=True)
    for vector_operation in vector_operations:
        statistics_df = statistics_df.append({
            'Method': vector_operation,
            'Result': vector_operations[vector_operation][LossFunction.VALUE] / size_of_dataset},
            ignore_index=True)

    # Print out statistics to file
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')
    print(
        tabulate.tabulate(statistics_df, headers='keys', tablefmt='grid', floatfmt='.2f', showindex=False), )
    with open(statistics_path_pr, 'w', encoding='utf8') as f:
        f.write(tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='grid',
                                  floatfmt='.2f'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to the model file')
    parser.add_argument('-t', '--test', help='Path to testing file')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
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

    test_loader = torch.utils.data.DataLoader(DistanceDataset(args.test, embedder),
                                              batch_size=args.batch_size)
    get_model_accuracy_statistics(model, test_loader, args.statistics)
