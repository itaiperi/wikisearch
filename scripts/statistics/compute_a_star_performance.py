import argparse
import json
import time
from collections import defaultdict
from importlib import import_module
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
import torch.utils.data

from scripts.utils import print_progress_bar, timing
from wikisearch.astar import Astar
from wikisearch.consts.mongo import WIKI_LANG, PAGES, CSV_SEPARATOR
from wikisearch.consts.nn import EMBEDDING_VECTOR_SIZE
from wikisearch.consts.statistics_column_names import *
from wikisearch.costs.uniform_cost import UniformCost
from wikisearch.embeddings import EMBEDDINGS_MODULES
from wikisearch.graph import WikiGraph
from wikisearch.heuristics import BFSHeuristic
from wikisearch.heuristics.nn_archs import EmbeddingsDistance
from wikisearch.heuristics.nn_heuristic import NNHeuristic
from wikisearch.strategies import DefaultAstarStrategy
from wikisearch.utils.clean_data import tokenize_title


def print_path(path_pr):
    return "->".join([f"'{node.title}'" for node in path_pr])


def calculate_averages(distances_times: dict, indent):
    distances = []
    average_values = []
    std = []
    for distance, distance_times in distances_times.items():
        distance_times = np.array(distance_times)
        average_time = distance_times.mean()

        distances.append(distance)
        average_values.append(average_time)
        std.append(distance_times.std())

        plt.text(distance+indent, average_time, str(round(average_time, 5)))
    return np.array(distances), np.array(average_values), np.array(std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to the model file. When running from linux - '
                                              'notice to not put a \'/\' after the file name')
    parser.add_argument('-df', '--dataset_file', help='Path to a dataset file')
    args = parser.parse_args()

    model_dir_path = path.dirname(args.model)
    model_file_name = path.splitext(path.basename(args.model))[0]

    # Loads dynamically the relevant embedding class
    with open(path.join(model_dir_path, f"{model_file_name}.meta")) as f:
        model_metadata = json.load(f)
    embedding = model_metadata['embedder']['type']
    embedding_module = import_module(
        '.'.join(['wikisearch', 'embeddings', EMBEDDINGS_MODULES[embedding]]),
        package='wikisearch')
    embedding_class = getattr(embedding_module, embedding)
    embedder = embedding_class(WIKI_LANG, PAGES)

    # Loads the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(EMBEDDING_VECTOR_SIZE).to(device)
    model.load_state_dict(torch.load(args.model, map_location=None if torch.cuda.is_available() else 'cpu'))
    model.eval()

    # Loads the dataset file
    dataset = pd.read_csv(args.dataset_file, sep=CSV_SEPARATOR).values

    # Prepare the statistics table
    statistics_df = pd.DataFrame(columns=[SRC_NODE, DST_NODE,
                                          BFS_DIST, BFS_TIME, BFS_DEVELOPED, BFS_PATH,
                                          NN_DIST, NN_TIME, NN_DEVELOPED, NN_PATH])
    cost = UniformCost()
    strategy = DefaultAstarStrategy()
    graph = WikiGraph(WIKI_LANG)
    astar_bfs = Astar(cost, BFSHeuristic(), strategy, graph)
    astar_nn = Astar(cost, NNHeuristic(model, embedder), strategy, graph)
    dataset_len = len(dataset)
    bfs_distance_times = defaultdict(list)
    bfs_distance_developed = defaultdict(list)
    nn_distance_times = defaultdict(list)
    nn_distance_developed = defaultdict(list)
    with torch.no_grad():
        start = time.time()
        for idx, (source, destination, _) in enumerate(dataset, 1):
            tokenized_source = tokenize_title(source)
            tokenized_destination = tokenize_title(destination)
            bfs_path, bfs_dist, bfs_developed, bfs_time = \
                timing(astar_bfs.run, tokenized_source, tokenized_destination)
            nn_path, nn_dist, nn_developed, nn_time = \
                timing(astar_nn.run, tokenized_source, tokenized_destination)
            bfs_distance_times[bfs_dist].append(bfs_time)
            bfs_distance_developed[bfs_dist].append(bfs_developed)
            nn_distance_times[bfs_dist].append(nn_time)
            nn_distance_developed[bfs_dist].append(nn_developed)
            statistics_df = statistics_df.append(
                {
                    SRC_NODE: source,
                    DST_NODE: destination,
                    BFS_DIST: bfs_dist,
                    BFS_TIME: bfs_time,
                    BFS_DEVELOPED: bfs_developed,
                    BFS_PATH: print_path(bfs_path),
                    NN_DIST: nn_dist,
                    NN_TIME: nn_time,
                    NN_DEVELOPED: nn_developed,
                    NN_PATH: print_path(nn_path)
                }, ignore_index=True)
            print_progress_bar(idx, dataset_len, time.time() - start, prefix=f'Progress: ', length=50)

    # Print out the statistics as tabulate
    statistics_file_path = path.join(model_dir_path, f"{model_file_name}.a_star_stats")
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')
    statistics_df_tabulate = \
        tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='fancy_grid', floatfmt='.5f')
    print(statistics_df_tabulate)
    with open(statistics_file_path, 'w', encoding='utf8') as f:
        f.write(statistics_df_tabulate)

    # Creates the distance-time statistics
    width = 0.3

    bfs_distances_time, bfs_average_times, bfs_std_time = \
        calculate_averages(bfs_distance_times, -width / 2)
    nn_distances_time, nn_average_times, nn_std_time = \
        calculate_averages(nn_distance_times, width / 2)

    plt.title("A* running times")
    plt.xlabel("Distance")
    plt.ylabel("Time")
    plt.bar(bfs_distances_time - width / 2, bfs_average_times, yerr=bfs_std_time, width=width, align='center')
    plt.bar(nn_distances_time + width / 2, nn_average_times, yerr=nn_std_time, width=width, align='center')
    plt.legend([BFS_TIME, NN_TIME])
    plt.savefig(path.join(model_dir_path, "a_star_running_time.jpg"))
    plt.show()

    plt.figure()

    # Creates the distance-#developed statistics
    bfs_distances_developed, bfs_average_developed, bfs_std_developed = \
        calculate_averages(bfs_distance_developed, -width / 2)
    nn_distances_developed, nn_average_developed, nn_std_developed = \
        calculate_averages(nn_distance_developed, width / 2)

    plt.title("A* developed")
    plt.xlabel("Distance")
    plt.ylabel("#Developed")
    plt.bar(bfs_distances_developed - width / 2, bfs_average_developed,
            yerr=bfs_std_developed, width=width, align='center')
    plt.bar(nn_distances_developed + width / 2, nn_average_developed,
            yerr=nn_std_developed, width=width, align='center')
    plt.legend([BFS_DEVELOPED, NN_DEVELOPED])
    plt.savefig(path.join(model_dir_path, "a_star_#developed.jpg"))
    plt.show()
